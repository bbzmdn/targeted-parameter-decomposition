"""Do a run's top components write along the residual-stream truth direction of their layer?

python -m spd.experiments.lm.honesty_targeted_decomposition.truth_direction_alignment --run tpd-liu-false
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import Config
from spd.experiments.lm.honesty_targeted_decomposition.find_top_components import (
    TOP_K_PER_LAYER,
    ComponentScores,
    answer_position,
    layer_index,
    score_components,
    split_target_statements,
    top_components_per_layer,
)
from spd.experiments.lm.honesty_targeted_decomposition.tpd_runs import (
    RUN_NAMES,
    Statements,
    load_tokenizer,
    load_tpd_model,
    results_dir,
)
from spd.models.component_model import ComponentModel


@torch.no_grad()
def answer_token_activations(
    component_model: ComponentModel, tokenizer: PreTrainedTokenizerBase, texts: list[str]
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """At the answer token: down_proj inputs {module: [n, d_mlp]} and residual stream after each
    decomposed layer {layer: [n, d_model]}."""
    modules = component_model.target_module_paths
    device = next(component_model.ci_fn.parameters()).device
    inputs: dict[str, list[np.ndarray]] = {m: [] for m in modules}
    resid: dict[int, list[np.ndarray]] = {layer_index(m): [] for m in modules}
    for text in texts:
        token_ids = tokenizer.encode(text)
        pos = answer_position([tokenizer.decode([t]) for t in token_ids])
        ids = torch.tensor([token_ids]).to(device)
        cache = component_model(ids, cache_type="input").cache
        hidden = component_model.target_model(ids, output_hidden_states=True).hidden_states
        for m in modules:
            inputs[m].append(cache[m][0, pos].float().cpu().numpy())
        for layer in resid:
            resid[layer].append(hidden[layer + 1][0, pos].float().cpu().numpy())
    return {m: np.stack(v) for m, v in inputs.items()}, {k: np.stack(v) for k, v in resid.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, choices=RUN_NAMES)
    ap.add_argument("--top-k", type=int, default=TOP_K_PER_LAYER, help="components per layer")
    args = ap.parse_args()

    out_dir = results_dir(args.run) / "truth_direction"
    out_dir.mkdir(parents=True, exist_ok=True)
    component_model, config = load_tpd_model(args.run, "cuda")
    tokenizer = load_tokenizer(config)

    split = split_target_statements(args.run)
    scores = score_components(component_model, config, tokenizer, split.rank)
    top = top_components_per_layer(scores, args.top_k)
    rows, summary = alignment_rows(component_model, config, tokenizer, split.held_out, scores, top)

    write_csv(out_dir / "components.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    plot(rows, out_dir / "alignment.png", args.run)
    print(f"Wrote {out_dir}")


def alignment_rows(
    component_model: ComponentModel,
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
    held_out: Statements,
    scores: ComponentScores,
    top: dict[str, list[int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Per component: cosine of its output direction U_c with the layer's truth direction
    (mean(false) - mean(true) of the residual stream after the layer, on held-out statements),
    signed by the mean sign of x.V_c on false statements, so > 0 means it pushes towards "false"."""
    acts = {p: answer_token_activations(component_model, tokenizer, held_out[p]) for p in held_out}
    score_of = {key: s for key, s in zip(scores.keys, scores.scores, strict=True)}
    d_model = next(iter(acts["true"][1].values())).shape[1]
    random_abs_cos = float(np.sqrt(2 / (np.pi * d_model)))
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for module in component_model.target_module_paths:
        layer = layer_index(module)
        direction = acts["false"][1][layer].mean(0) - acts["true"][1][layer].mean(0)
        direction /= np.linalg.norm(direction)
        comps = component_model.components[module]
        u = comps.U.detach().float().cpu().numpy()  # [C, d_model]
        v = comps.V.detach().float().cpu().numpy()  # [d_mlp, C]
        cos = (u / np.linalg.norm(u, axis=1, keepdims=True)) @ direction
        sign = np.sign((acts["false"][0][module] @ v).mean(0))
        abs_cos = np.abs(cos)
        for c in range(len(cos)):
            rows.append(
                {
                    "layer": layer,
                    "component": c,
                    "score": float(score_of[(module, c)]),
                    "top": c in top[module],
                    "cos_towards_false": float(sign[c] * cos[c]),
                    "abs_cos": float(abs_cos[c]),
                    "abs_cos_percentile_in_layer": float((abs_cos < abs_cos[c]).mean()),
                }
            )
        is_top = np.isin(np.arange(len(cos)), top[module])
        summary.append(
            {
                "layer": layer,
                "mean_abs_cos_top": float(abs_cos[is_top].mean()),
                "mean_abs_cos_rest": float(abs_cos[~is_top].mean()),
                "max_abs_cos_layer": float(abs_cos.max()),
                "expected_abs_cos_random_direction": random_abs_cos,
            }
        )
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path, run: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for is_top, color, label in (
        (False, "0.7", "other components"),
        (True, "C3", "top components"),
    ):
        pts = [(r["layer"], r["abs_cos"]) for r in rows if r["top"] == is_top]
        ax.scatter(*zip(*pts, strict=True), s=10, color=color, label=label)
    ax.set_xlabel("layer")
    ax.set_ylabel("|cos(U_c, truth direction)|")
    ax.set_title(run)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
