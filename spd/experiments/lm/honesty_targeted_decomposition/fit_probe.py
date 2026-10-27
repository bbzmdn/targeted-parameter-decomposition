"""Leave-one-domain-out truth probes on CI features of every run, against activation baselines.

python -m spd.experiments.lm.honesty_targeted_decomposition.fit_probe
python -m spd.experiments.lm.honesty_targeted_decomposition.fit_probe --runs tpd-liu-false

Per run, the CI probe is also fit on subsets of the components: the run's top components (ranked
on its own target statements, see find_top_components.py), size-matched random subsets, and the
top principal components of the CI.

Writes summary.md, auc.csv, accuracy.csv, auc_table.png (per-domain AUC of the main methods) and
predictions_<run>.png (held-out probe scores per statement) to results/probe/.
"""

import argparse
import csv
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import Config
from spd.experiments.lm.honesty_targeted_decomposition.find_top_components import (
    TOP_K_PER_LAYER,
    ComponentScores,
    layer_index,
    score_components,
    split_target_statements,
    top_components_per_layer,
    write_component_scores,
)
from spd.experiments.lm.honesty_targeted_decomposition.tpd_runs import (
    EVAL_DOMAINS_DIR,
    RESULTS_DIR,
    RUN_NAMES,
    build_component_model,
    load_tokenizer,
    load_tpd_config,
    load_trained_weights,
    prompts_file,
    read_lines,
    results_dir,
)
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import bf16_autocast

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Some Liu et al. arithmetic negatives use the placeholder answer "banana", so the answer token
# differs in kind rather than in truth value.
EXCLUDE_DOMAINS = ["arithmetic"]
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_PCA = 300
N_RANDOM_DIRECTIONS = 20
SEED = 0
TOP_KS = [10, 40, 80, 160]
CI_PCA_SIZES = [10, 50]
N_RANDOM_SUBSETS = 5
OUT_OF_DOMAIN_PREFIXES = ("got_", "truthfulqa")
PREDICTIONS_PLOT_RUNS = ["tpd-liu-false"]
DOMAIN_LABELS = {
    "hotpot_qa_re": "HotpotQA",
    "nq_re": "NQ",
    "sciq": "SciQ",
    "triva_qa_re": "TriviaQA",
    "got_larger_than": "larger than",
    "got_smaller_than": "smaller than",
    "got_sp_en_trans": "sp-en trans",
    "got_neg_sp_en_trans": "neg sp-en trans",
    "got_neg_cities": "neg cities",
    "got_likely": "likely",
    "truthfulqa": "TruthfulQA",
}

# sklearn deprecates penalty="l1" in favour of l1_ratio, which liblinear does not accept
warnings.filterwarnings("ignore", message=".*penalty.*")
warnings.filterwarnings("ignore", message=".*l1_ratio.*")


@dataclass
class Dataset:
    texts: list[str]
    y: np.ndarray  # 1 = false statement
    domains: np.ndarray
    domain_names: list[str]


@dataclass
class ProbeResult:
    accuracy: dict[str, float]  # held-out domain -> accuracy (nan when the method has no threshold)
    auc: dict[str, float]
    concat_auc: float  # one AUC over the held-out scores of all domains concatenated
    scores: np.ndarray  # [n_texts] held-out score per statement (nan for averaged results)


def load_dataset() -> Dataset:
    domains = sorted(p.name.removesuffix("_true.txt") for p in EVAL_DOMAINS_DIR.glob("*_true.txt"))
    domains = [d for d in domains if d not in EXCLUDE_DOMAINS]
    texts, labels, text_domains = [], [], []
    for domain in domains:
        for label, polarity in ((0, "true"), (1, "false")):
            lines = read_lines(EVAL_DOMAINS_DIR / f"{domain}_{polarity}.txt")
            texts += lines
            labels += [label] * len(lines)
            text_domains += [domain] * len(lines)
    return Dataset(texts, np.array(labels), np.array(text_domains), domains)


@torch.no_grad()
def ci_features(
    component_model: ComponentModel,
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
) -> np.ndarray:
    """[n_texts, n_modules * C]: CI averaged over the tokens of each statement."""
    ci_dtype = next(component_model.ci_fn.parameters()).dtype
    rows = []
    for text in texts:
        ids = torch.tensor([tokenizer.encode(text)]).to("cuda")
        with bf16_autocast(enabled=config.autocast_bf16):
            out = component_model(ids, cache_type="input")
            acts = {k: v.to(ci_dtype) for k, v in out.cache.items()}
            ci = component_model.calc_causal_importances(acts, sampling=config.sampling).lower_leaky
        rows.append(
            torch.cat([ci[m][0].float().mean(0) for m in component_model.target_module_paths])
            .cpu()
            .numpy()
        )
    return np.stack(rows)


@torch.no_grad()
def activation_features(
    component_model: ComponentModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    layers: list[int],
) -> np.ndarray:
    """[n_texts, n_layers * d_model]: residual stream at the last token, at the input of each layer."""
    rows = []
    for text in texts:
        ids = torch.tensor([tokenizer.encode(text)]).to("cuda")
        hidden = component_model.target_model(ids, output_hidden_states=True).hidden_states
        rows.append(torch.cat([hidden[layer][0, -1].float() for layer in layers]).cpu().numpy())
    return np.stack(rows)


def fit_l1(x: np.ndarray, y: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=c, max_iter=3000, random_state=SEED
    )
    return scaler, clf.fit(scaler.transform(x), y)


def choose_c(x: np.ndarray, y: np.ndarray, domains: np.ndarray) -> float:
    """C with the best accuracy on the last training domain, skipping C's that zero every weight."""
    inner_held = sorted(set(domains))[-1]
    tr, te = domains != inner_held, domains == inner_held
    best_c, best_acc = C_GRID[-1], -1.0
    for c in C_GRID:
        scaler, clf = fit_l1(x[tr], y[tr], c)
        if not clf.coef_.any():
            continue
        acc = accuracy_score(y[te], clf.predict(scaler.transform(x[te])))
        if acc > best_acc:
            best_c, best_acc = c, acc
    return best_c


Scorer = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""(x_train, y_train, domains_train, x_test) -> (scores, predictions); higher score = false."""


def logistic_regression(n_pca: int | None) -> Scorer:
    def scorer(
        x_tr: np.ndarray, y_tr: np.ndarray, d_tr: np.ndarray, x_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if n_pca is not None:
            pca = PCA(
                n_components=min(n_pca, x_tr.shape[0] - 1, x_tr.shape[1]), random_state=SEED
            ).fit(x_tr)
            x_tr, x_te = pca.transform(x_tr), pca.transform(x_te)
        scaler, clf = fit_l1(x_tr, y_tr, choose_c(x_tr, y_tr, d_tr))
        return clf.predict_proba(scaler.transform(x_te))[:, 1], clf.predict(scaler.transform(x_te))

    return scorer


def mass_mean(
    x_tr: np.ndarray, y_tr: np.ndarray, d_tr: np.ndarray, x_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Marks & Tegmark: project onto mean(false) - mean(true), threshold at the training mean."""
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_te = np.asarray(scaler.transform(x_tr)), np.asarray(scaler.transform(x_te))
    direction = x_tr[y_tr == 1].mean(0) - x_tr[y_tr == 0].mean(0)
    scores = x_te @ direction
    return scores, (scores > (x_tr @ direction).mean()).astype(int)


def leave_one_domain_out(x: np.ndarray, data: Dataset, scorer: Scorer) -> ProbeResult:
    accuracy: dict[str, float] = {}
    auc: dict[str, float] = {}
    all_scores = np.full(len(data.y), np.nan)
    for held in data.domain_names:
        tr, te = data.domains != held, data.domains == held
        scores, predictions = scorer(x[tr], data.y[tr], data.domains[tr], x[te])
        all_scores[te] = scores
        auc[held] = float(roc_auc_score(data.y[te], scores))
        accuracy[held] = float(accuracy_score(data.y[te], predictions))
    assert not np.isnan(all_scores).any()
    return ProbeResult(accuracy, auc, float(roc_auc_score(data.y, all_scores)), all_scores)


def random_directions(x: np.ndarray, data: Dataset) -> ProbeResult:
    rng = np.random.default_rng(SEED)
    directions = rng.normal(size=(N_RANDOM_DIRECTIONS, x.shape[1]))
    auc: dict[str, float] = {}
    for held in data.domain_names:
        te = data.domains == held
        auc[held] = float(
            np.mean([float(roc_auc_score(data.y[te], x[te] @ d)) for d in directions])
        )
    concat_auc = float(np.mean([float(roc_auc_score(data.y, x @ d)) for d in directions]))
    return ProbeResult(
        {d: float("nan") for d in data.domain_names},
        auc,
        concat_auc,
        np.full(len(data.y), np.nan),
    )


def probe_components(x: np.ndarray, data: Dataset, keys: list[tuple[str, int]], path: Path) -> None:
    """Non-zero weights of an L1 probe fit on all domains (> 0: predicts false)."""
    scaler, clf = fit_l1(x, data.y, choose_c(x, data.y, data.domains))
    beta = clf.coef_[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "component", "beta"])
        for i in np.argsort(-np.abs(beta)):
            if beta[i] != 0:
                w.writerow([keys[i][0], keys[i][1], f"{beta[i]:.6g}"])


def mean_result(results: list[ProbeResult]) -> ProbeResult:
    domains = results[0].auc.keys()
    return ProbeResult(
        {d: float(np.mean([r.accuracy[d] for r in results])) for d in domains},
        {d: float(np.mean([r.auc[d] for r in results])) for d in domains},
        float(np.mean([r.concat_auc for r in results])),
        np.full(len(results[0].scores), np.nan),
    )


def probe_feature_sets(
    x: np.ndarray, data: Dataset, scores: ComponentScores
) -> dict[str, tuple[int, ProbeResult]]:
    """{feature set: (n features, LODO result)} for subsets and projections of one run's CI features."""
    assert x.shape[1] == len(scores.keys)
    by_score = np.argsort(-np.abs(scores.scores))
    per_layer = top_components_per_layer(scores, TOP_K_PER_LAYER)
    per_layer_idx = np.array([scores.keys.index((m, c)) for m, cs in per_layer.items() for c in cs])
    rng = np.random.default_rng(SEED)
    lr = logistic_regression(None)

    def random_subsets(k: int) -> ProbeResult:
        return mean_result(
            [
                leave_one_domain_out(x[:, rng.choice(x.shape[1], k, replace=False)], data, lr)
                for _ in range(N_RANDOM_SUBSETS)
            ]
        )

    sets: dict[str, tuple[int, ProbeResult]] = {}
    for k in TOP_KS:
        sets[f"top {k}"] = (k, leave_one_domain_out(x[:, by_score[:k]], data, lr))
        sets[f"random {k}"] = (k, random_subsets(k))
    sets[f"top {TOP_K_PER_LAYER} per layer"] = (
        len(per_layer_idx),
        leave_one_domain_out(x[:, per_layer_idx], data, lr),
    )
    for n in CI_PCA_SIZES:
        sets[f"PCA {n}"] = (n, leave_one_domain_out(x, data, logistic_regression(n)))
    return sets


def domain_groups(data: Dataset) -> dict[str, list[str]]:
    """In domain: Liu domains, whose statements are the target data of the liu and
    capitals-liu-tqa runs. Out of domain: Geometry of Truth and TruthfulQA, unseen by every run."""
    out_of_domain = [d for d in data.domain_names if d.startswith(OUT_OF_DOMAIN_PREFIXES)]
    in_domain = [d for d in data.domain_names if d not in out_of_domain]
    liu_targets = {line for p in ("true", "false") for line in read_lines(prompts_file("liu", p))}
    in_targets = {
        d for d, text in zip(data.domains, data.texts, strict=True) if text in liu_targets
    }
    assert in_targets == set(in_domain), (in_targets, in_domain)
    return {"in domain": in_domain, "out of domain": out_of_domain, "pooled": data.domain_names}


def write_tables(
    results: dict[str, ProbeResult], features: dict[str, int], data: Dataset, out_dir: Path
) -> None:
    groups = domain_groups(data)
    for metric in ("auc", "accuracy"):
        with open(out_dir / f"{metric}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", *results])
            for domain in data.domain_names:
                w.writerow(
                    [domain, *(f"{getattr(r, metric)[domain]:.4f}" for r in results.values())]
                )
            for group, domains in groups.items():
                w.writerow(
                    [
                        f"MEAN {group}",
                        *(
                            f"{np.mean([getattr(r, metric)[d] for d in domains]):.4f}"
                            for r in results.values()
                        ),
                    ]
                )
            if metric == "auc":
                w.writerow(
                    ["CONCATENATED pooled", *(f"{r.concat_auc:.4f}" for r in results.values())]
                )

    def pct(values: list[float]) -> str:
        return "–" if np.isnan(np.mean(values)) else f"{100 * np.mean(values):.1f}"

    header = " | ".join(f"Acc {g} | AUC {g}" for g in groups)
    lines = [
        f"| Method | Features | {header} | AUC pooled, concatenated |",
        "| --- | --- |" + " --- | --- |" * len(groups) + " --- |",
    ]
    for name, r in results.items():
        cells = " | ".join(
            f"{pct([r.accuracy[d] for d in ds])} | {pct([r.auc[d] for d in ds])}"
            for ds in groups.values()
        )
        lines.append(f"| {name} | {features[name]:,} | {cells} | {100 * r.concat_auc:.1f} |")
    domain_lists = "\n".join(f"- {g} ({len(ds)}): {', '.join(ds)}" for g, ds in groups.items())
    summary = (
        "\n".join(lines)
        + "\n\nMeans over held-out domains (leave-one-domain-out). In-domain statements are the"
        + " target data of the liu and capitals-liu-tqa runs; out-of-domain ones were seen by no run."
        + " The concatenated AUC pools the held-out scores of all 18 domains into one ranking.\n\n"
        + domain_lists
        + "\n"
    )
    (out_dir / "summary.md").write_text(summary)
    print(summary)


def table_columns(runs: list[str]) -> list[tuple[str, str, str]]:
    """(method name, header, subheader) of the per-domain AUC table."""
    columns = [
        ("Logistic regression, activations", "Raw", "logistic"),
        (f"Logistic regression, activations (PCA {N_PCA})", "Raw", f"PCA {N_PCA}"),
        ("Mass-mean, activations", "Mass-mean", "raw"),
        ("Logistic regression, CI of untrained decomposition", "Untrained", "CI"),
    ]
    for run in runs:
        target, polarity = run.removeprefix("tpd-").rsplit("-", 1)
        header = "combined" if target == "capitals-liu-tqa" else target
        columns.append((f"Logistic regression, CI ({run})", header, polarity))
    return columns


def plot_auc_table(
    results: dict[str, ProbeResult], runs: list[str], data: Dataset, path: Path
) -> None:
    """Per-domain AUC of the main methods, best method per row in bold."""
    groups = domain_groups(data)
    columns = table_columns(runs)
    sections = [
        ("Liu et al.\n(in domain)", groups["in domain"]),
        ("Geometry\nof Truth", [d for d in groups["out of domain"] if d.startswith("got_")]),
        ("Adversarial", ["truthfulqa"]),
    ]
    summary_rows: list[tuple[str, list[float]]] = [
        (f"Mean {g}", [float(np.mean([results[m].auc[d] for d in ds])) for m, _, _ in columns])
        for g, ds in groups.items()
    ]
    summary_rows.append(("Pooled, concatenated", [results[m].concat_auc for m, _, _ in columns]))

    n_rows = sum(len(ds) for _, ds in sections) + len(summary_rows)
    col_w, label_w, row_h = 1.15, 3.6, 0.32
    width = label_w + col_w * len(columns)
    height = row_h * (n_rows + len(sections) + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    plt.rcParams["font.family"] = "serif"
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    x_cols = [label_w + col_w * (i + 0.5) for i in range(len(columns))]

    def bold_row(y: float, values: list[float]) -> None:
        best = max(values)
        for x, v in zip(x_cols, values, strict=True):
            weight = "bold" if v >= best - 1e-9 else "normal"
            ax.text(x, y, f"{100 * v:.1f}", ha="center", va="center", fontsize=10, weight=weight)

    def rule(y: float, lw: float, ls: str = "-") -> None:
        ax.plot([0.1, width - 0.1], [y, y], color="black", lw=lw, ls=ls)

    rule(0.15, 2.0)
    ax.text(0.2, row_h * 1.2, "Method", va="center", fontsize=11, weight="bold")
    for x, (_, header, sub) in zip(x_cols, columns, strict=True):
        ax.text(x, row_h * 0.85, header, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x, row_h * 1.6, sub, ha="center", va="center", fontsize=10)
    y = row_h * 2.3
    rule(y, 1.0)
    for i, (section, domains) in enumerate(sections):
        y += row_h * 0.5
        top = y
        for d in domains:
            y += row_h
            ax.text(1.7, y, DOMAIN_LABELS.get(d, d), va="center", fontsize=10)
            bold_row(y, [results[m].auc[d] for m, _, _ in columns])
        ax.text(0.2, (top + y) / 2 + row_h / 2, section, va="center", fontsize=10.5, weight="bold")
        y += row_h * 0.5
        rule(y, 0.8, "--" if i < len(sections) - 1 else "-")
    for label, values in summary_rows:
        y += row_h
        ax.text(0.2, y, label, va="center", fontsize=10.5, weight="bold")
        bold_row(y, values)
    rule(y + row_h * 0.6, 2.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(result: ProbeResult, data: Dataset, title: str, path: Path) -> None:
    """Held-out probe score of every statement, one row per domain, worst domain on top."""
    domains = sorted(data.domain_names, key=lambda d: result.auc[d])
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(domains) + 1.5))
    for row, domain in enumerate(domains):
        sel = data.domains == domain
        jitter = rng.uniform(-0.15, 0.15, sel.sum())
        colors = np.where(data.y[sel] == 1, "tab:orange", "tab:blue")
        ax.scatter(result.scores[sel], row + jitter, c=colors, s=16, alpha=0.65)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([f"{d}  ({result.auc[d]:.2f})" for d in domains], fontsize=8)
    ax.set_ylim(len(domains) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probe score  =  P(statement is false)")
    ax.set_title(
        f"{title}: held-out predictions per domain (AUC in brackets)\n"
        "blue = actually TRUE, orange = actually FALSE, dashed line = decision threshold"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", choices=RUN_NAMES, default=RUN_NAMES)
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    print(f"{len(data.domain_names)} domains, {len(data.texts)} statements")
    print({g: len(ds) for g, ds in domain_groups(data).items()})

    config = load_tpd_config(args.runs[0])
    torch.manual_seed(SEED)
    component_model = build_component_model(config, "cuda")
    tokenizer = load_tokenizer(config)
    modules = component_model.target_module_paths
    keys = [(m, c) for m in modules for c in range(component_model.components[m].C)]
    layers = sorted(layer_index(m) for m in modules)

    results: dict[str, ProbeResult] = {}
    features: dict[str, int] = {}

    def add(name: str, n_features: int, result: ProbeResult) -> None:
        results[name], features[name] = result, n_features
        print(
            f"{name:45s} acc={np.mean(list(result.accuracy.values())):.3f} auc={np.mean(list(result.auc.values())):.3f}"
        )

    x_act = activation_features(component_model, tokenizer, data.texts, layers)
    add("Mass-mean, activations", x_act.shape[1], leave_one_domain_out(x_act, data, mass_mean))
    add(
        "Logistic regression, activations",
        x_act.shape[1],
        leave_one_domain_out(x_act, data, logistic_regression(None)),
    )
    add(
        f"Logistic regression, activations (PCA {N_PCA})",
        N_PCA,
        leave_one_domain_out(x_act, data, logistic_regression(N_PCA)),
    )

    x_untrained = ci_features(component_model, config, tokenizer, data.texts)
    add(
        "Logistic regression, CI of untrained decomposition",
        x_untrained.shape[1],
        leave_one_domain_out(x_untrained, data, logistic_regression(None)),
    )

    for run in args.runs:
        assert load_tpd_config(run).all_module_info == config.all_module_info, run
        load_trained_weights(component_model, run)
        x_ci = ci_features(component_model, config, tokenizer, data.texts)
        add(
            f"Logistic regression, CI ({run})",
            x_ci.shape[1],
            leave_one_domain_out(x_ci, data, logistic_regression(None)),
        )
        probe_components(x_ci, data, keys, out_dir / f"components_{run}.csv")
        if run == args.runs[0]:
            add(f"Random direction, CI ({run})", x_ci.shape[1], random_directions(x_ci, data))

        scores = score_components(
            component_model, config, tokenizer, split_target_statements(run).rank
        )
        results_dir(run).mkdir(parents=True, exist_ok=True)
        write_component_scores(scores, results_dir(run) / "component_scores.csv")
        for set_name, (n_features, result) in probe_feature_sets(x_ci, data, scores).items():
            add(f"Logistic regression, CI {set_name} ({run})", n_features, result)

    write_tables(results, features, data, out_dir)
    plot_auc_table(results, args.runs, data, out_dir / "auc_table.png")
    for run in PREDICTIONS_PLOT_RUNS:
        if run in args.runs:
            plot_predictions(
                results[f"Logistic regression, CI ({run})"],
                data,
                f"Logistic regression on CI ({run})",
                out_dir / f"predictions_{run}.png",
            )


if __name__ == "__main__":
    main()
