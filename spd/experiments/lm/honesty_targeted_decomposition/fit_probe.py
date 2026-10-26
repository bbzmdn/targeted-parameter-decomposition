"""Leave-one-domain-out truth probes on CI features of every run, against activation baselines.

python -m spd.experiments.lm.honesty_targeted_decomposition.fit_probe
python -m spd.experiments.lm.honesty_targeted_decomposition.fit_probe --runs tpd-liu-false
"""

import argparse
import csv
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import Config
from spd.experiments.lm.honesty_targeted_decomposition.find_top_components import layer_index
from spd.experiments.lm.honesty_targeted_decomposition.tpd_runs import (
    EVAL_DOMAINS_DIR,
    RESULTS_DIR,
    RUN_NAMES,
    build_component_model,
    load_tokenizer,
    load_tpd_config,
    load_trained_weights,
    read_lines,
)
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import bf16_autocast

# Some Liu et al. arithmetic negatives use the placeholder answer "banana", so the answer token
# differs in kind rather than in truth value.
EXCLUDE_DOMAINS = ["arithmetic"]
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_PCA = 300
N_RANDOM_DIRECTIONS = 20
SEED = 0

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
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=c, max_iter=3000)
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
    for held in data.domain_names:
        tr, te = data.domains != held, data.domains == held
        scores, predictions = scorer(x[tr], data.y[tr], data.domains[tr], x[te])
        auc[held] = float(roc_auc_score(data.y[te], scores))
        accuracy[held] = float(accuracy_score(data.y[te], predictions))
    return ProbeResult(accuracy, auc)


def random_directions(x: np.ndarray, data: Dataset) -> ProbeResult:
    rng = np.random.default_rng(SEED)
    directions = rng.normal(size=(N_RANDOM_DIRECTIONS, x.shape[1]))
    auc: dict[str, float] = {}
    for held in data.domain_names:
        te = data.domains == held
        auc[held] = float(
            np.mean([float(roc_auc_score(data.y[te], x[te] @ d)) for d in directions])
        )
    return ProbeResult({d: float("nan") for d in data.domain_names}, auc)


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


def write_tables(
    results: dict[str, ProbeResult], features: dict[str, int], data: Dataset, out_dir: Path
) -> None:
    for metric in ("auc", "accuracy"):
        with open(out_dir / f"{metric}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", *results])
            for domain in data.domain_names:
                w.writerow(
                    [domain, *(f"{getattr(r, metric)[domain]:.4f}" for r in results.values())]
                )
            w.writerow(
                [
                    "MEAN",
                    *(
                        f"{np.mean(list(getattr(r, metric).values())):.4f}"
                        for r in results.values()
                    ),
                ]
            )
    lines = ["| Method | Features | Acc | AUC |", "| --- | --- | --- | --- |"]
    for name, r in results.items():
        acc = np.mean(list(r.accuracy.values()))
        acc_str = "–" if np.isnan(acc) else f"{100 * acc:.1f}"
        lines.append(
            f"| {name} | {features[name]:,} | {acc_str} | {100 * np.mean(list(r.auc.values())):.1f} |"
        )
    summary = (
        "\n".join(lines)
        + f"\n\nMean over {len(data.domain_names)} held-out domains: {', '.join(data.domain_names)}\n"
    )
    (out_dir / "summary.md").write_text(summary)
    print(summary)


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

    write_tables(results, features, data, out_dir)


if __name__ == "__main__":
    main()
