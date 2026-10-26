"""Token-level CI of a run's top components on its own target statements.

python -m spd.experiments.lm.honesty_targeted_decomposition.visualize_token_firing --run tpd-liu-false
"""

import argparse
import csv
import html

import numpy as np

from spd.experiments.lm.honesty_targeted_decomposition.find_top_components import (
    RANK_FRAC,
    TOP_K_PER_LAYER,
    answer_position,
    answer_token_ci,
    layer_index,
    score_components,
    split_target_statements,
    token_ci,
    top_components_per_layer,
    write_component_scores,
)
from spd.experiments.lm.honesty_targeted_decomposition.tpd_runs import (
    RUN_NAMES,
    load_tokenizer,
    load_tpd_model,
    results_dir,
)

STYLE = """body { font-family: monospace; font-size: 14px; line-height: 2.2; }
h2 { margin-top: 40px; border-bottom: 2px solid #333; }
table { border-collapse: collapse; line-height: 1.4; } td, th { padding: 2px 10px; text-align: right; }
.example { margin-bottom: 6px; padding: 4px; border: 1px solid #ddd; }
.true { color: #1f6feb; font-weight: bold; } .false { color: #e8590c; font-weight: bold; }"""


def render_tokens(tokens: list[str], values: np.ndarray, vmax: float, answer_pos: int) -> str:
    spans = []
    for i, (tok, v) in enumerate(zip(tokens, values, strict=True)):
        shade = int(255 * (1 - min(max(v / vmax, 0.0), 1.0))) if vmax > 0 else 255
        border = "border-bottom:2px solid #000;" if i == answer_pos else ""
        spans.append(
            f'<span style="background-color:rgb(255,{shade},{shade});{border}" title="{v:.4f}">'
            f"{html.escape(tok)}</span>"
        )
    return "".join(spans)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, choices=RUN_NAMES)
    ap.add_argument("--top-k", type=int, default=TOP_K_PER_LAYER, help="components per layer")
    ap.add_argument("--n-examples", type=int, default=10, help="statements shown per polarity")
    args = ap.parse_args()

    out_dir = results_dir(args.run) / "token_firing"
    out_dir.mkdir(parents=True, exist_ok=True)
    component_model, config = load_tpd_model(args.run, "cuda")
    tokenizer = load_tokenizer(config)

    split = split_target_statements(args.run)
    scores = score_components(component_model, config, tokenizer, split.rank)
    write_component_scores(scores, results_dir(args.run) / "component_scores.csv")
    top = top_components_per_layer(scores, args.top_k)
    held_out_ci = {
        p: answer_token_ci(component_model, config, tokenizer, lines)
        for p, lines in split.held_out.items()
    }

    shown = [
        (polarity, text)
        for pair in zip(
            split.held_out["true"][: args.n_examples],
            split.held_out["false"][: args.n_examples],
            strict=False,
        )
        for polarity, text in zip(("true", "false"), pair, strict=True)
    ]
    shown_ci = [token_ci(component_model, config, tokenizer, text) for _, text in shown]

    score_of = dict(zip(scores.keys, range(len(scores.keys)), strict=True))
    summary_rows = []
    for module, components in top.items():
        layer = layer_index(module)
        parts = [
            f"<html><head><meta charset='utf-8'><style>{STYLE}</style></head><body>",
            f"<h1>{args.run}: layer {layer} down_proj, top {len(components)} components</h1>",
            f"<p>Components ranked by |mean CI(false) - mean CI(true)| at the answer token "
            f"(underlined) on {RANK_FRAC:.0%} of the run's target statements; the examples and "
            f"held-out means are on the other half.</p>",
            "<table><tr><th>component</th><th>score</th><th>true (rank)</th><th>false (rank)</th>"
            "<th>true (held out)</th><th>false (held out)</th></tr>",
        ]
        for c in components:
            i = score_of[(module, c)]
            row = {
                "layer": layer,
                "component": c,
                "score": float(scores.scores[i]),
                "mean_ci_true_rank": float(scores.mean_ci_true[i]),
                "mean_ci_false_rank": float(scores.mean_ci_false[i]),
                "mean_ci_true_held_out": float(held_out_ci["true"][module][:, c].mean()),
                "mean_ci_false_held_out": float(held_out_ci["false"][module][:, c].mean()),
            }
            summary_rows.append(row)
            parts.append(
                "<tr>"
                + "".join(
                    f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                    for k, v in row.items()
                    if k != "layer"
                )
                + "</tr>"
            )
        parts.append("</table>")
        for c in components:
            values = [ci[module][:, c] for _, ci in shown_ci]
            vmax = float(max(v.max() for v in values))
            parts.append(f"<h2>component {c} (max CI shown {vmax:.3f})</h2>")
            for (polarity, _), (tokens, _), v in zip(shown, shown_ci, values, strict=True):
                parts.append(
                    f'<div class="example"><span class="{polarity}">[{polarity}]</span> '
                    f"{render_tokens(tokens, v, vmax, answer_position(tokens))}</div>"
                )
        parts.append("</body></html>")
        (out_dir / f"L{layer}.html").write_text("\n".join(parts))

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(
            {k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()}
            for r in summary_rows
        )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
