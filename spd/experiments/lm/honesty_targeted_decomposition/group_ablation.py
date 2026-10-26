"""Ablate groups of a run's top components (same selection as token firing) against size-matched random groups.

python -m spd.experiments.lm.honesty_targeted_decomposition.group_ablation --run tpd-liu-false
"""

import argparse
import csv
import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset
from scipy.stats import binomtest
from torch import Tensor, nn

from spd.configs import Config, LMTaskConfig
from spd.experiments.lm.honesty_targeted_decomposition.find_top_components import (
    TOP_K_PER_LAYER,
    ComponentKey,
    ComponentScores,
    layer_index,
    score_components,
    split_target_statements,
    top_components_per_layer,
    write_component_scores,
)
from spd.experiments.lm.honesty_targeted_decomposition.tpd_runs import (
    EVAL_DOMAINS_DIR,
    HERE,
    RUN_NAMES,
    Statements,
    load_tokenizer,
    load_tpd_model,
    read_lines,
    results_dir,
)
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos

CAPITALS_PAIRS = HERE / "data" / "capitals_contrast_pairs.jsonl"
CONTEXTS = ("bare", "honest", "dishonest")


# -- Data -----------------------------------------------------------------------


@dataclass
class Question:
    question: str
    correct: str
    wrong: str
    prefix: str  # assistant text before the answer, e.g. "The capital of France is"
    honest_systems: list[str]
    dishonest_systems: list[str]


@dataclass
class EvalItem:
    question_idx: int
    context: str
    prompt_text: str  # chat-templated prompt + assistant prefix


@dataclass
class Baseline:
    """Measurements of the unablated model."""

    margin: Tensor
    logp_correct: Tensor
    first_logprobs: Tensor
    nontarget: list[Tensor]
    statements: dict[str, tuple[float, float]]  # domain -> (log-prob gap, truth AUC)
    truthful: dict[str, np.ndarray]  # "lenient" / "strict" -> per generation


def truthful_hits(
    eval_qs: list[Question], gen_items: list[EvalItem], gens: list[str]
) -> dict[str, np.ndarray]:
    """lenient: the correct answer appears in the generation; strict: the generation starts with it."""
    correct = [eval_qs[it.question_idx].correct.lower() for it in gen_items]
    return {
        "lenient": np.array([c in g.lower() for c, g in zip(correct, gens, strict=True)]),
        "strict": np.array(
            [g.strip().lower().startswith(c) for c, g in zip(correct, gens, strict=True)]
        ),
    }


def load_questions(jsonl_path: Path) -> list[Question]:
    by_question: dict[str, Question] = {}
    for line in jsonl_path.read_text().splitlines():
        p = json.loads(line)
        assert p["correct_answer"] in p["fact_sentence"], p
        q = by_question.get(p["question"])
        if q is None:
            prefix = p["fact_sentence"][: p["fact_sentence"].index(p["correct_answer"])].rstrip()
            q = Question(p["question"], p["correct_answer"], p["wrong_answer"], prefix, [], [])
            by_question[p["question"]] = q
        q.honest_systems.append(p["honest_instruction"])
        q.dishonest_systems.append(p["dishonest_instruction"])
    return list(by_question.values())


def build_eval_items(questions: list[Question], tokenizer: Any) -> list[EvalItem]:
    items = []
    for i, q in enumerate(questions):
        user = {"role": "user", "content": q.question}
        contexts: list[tuple[str, list[dict[str, str]]]] = [("bare", [user])]
        contexts += [("honest", [{"role": "system", "content": s}, user]) for s in q.honest_systems]
        contexts += [
            ("dishonest", [{"role": "system", "content": s}, user]) for s in q.dishonest_systems
        ]
        for context, messages in contexts:
            chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            items.append(EvalItem(i, context, chat + " " + q.prefix))
    return items


# -- Ranking ---------------------------------------------------------------------


def build_groups(
    scores: ComponentScores, ks: list[int], per_layer_k: int, n_random: int, seed: int
) -> dict[str, list[ComponentKey]]:
    rng = random.Random(seed)
    keys, s = scores.keys, scores.scores
    order_abs = np.argsort(-np.abs(s))
    order_false = [i for i in np.argsort(-s) if s[i] > 0]
    order_true = [i for i in np.argsort(s) if s[i] < 0]
    alive_idx = [i for i in range(len(keys)) if scores.alive[i]]

    groups: dict[str, list[ComponentKey]] = {"none": []}
    for k in ks:
        groups[f"top{k}_abs"] = [keys[i] for i in order_abs[:k]]
        if len(order_false) >= k:
            groups[f"top{k}_false"] = [keys[i] for i in order_false[:k]]
        if len(order_true) >= k:
            groups[f"top{k}_true"] = [keys[i] for i in order_true[:k]]
        pool = alive_idx if len(alive_idx) >= k else list(range(len(keys)))
        for i in range(n_random):
            groups[f"random{k}_s{i}"] = [keys[j] for j in rng.sample(pool, k)]

    for module, components in top_components_per_layer(scores, per_layer_k).items():
        groups[f"L{layer_index(module)}_top{per_layer_k}_abs"] = [(module, c) for c in components]

    groups["all_alive"] = [keys[i] for i in alive_idx]
    groups["all_components"] = list(keys)
    return groups


# -- Ablation --------------------------------------------------------------------


@contextmanager
def ablate(
    component_model: ComponentModel, group: list[ComponentKey], mask_value: float
) -> Iterator[None]:
    """Set the masks of `group` to `mask_value` (all other masks and the delta stay at 1).

    With masks m and delta on, a decomposed module computes
    x V diag(m) U + x (W - V U)  =  x W - sum_c (1 - m_c) x V_c U_c,
    so the hook only has to subtract the ablated rank-one pieces.
    """
    by_module: dict[str, list[int]] = {}
    for module, c in group:
        by_module.setdefault(module, []).append(c)

    handles = []
    for module, idx in by_module.items():
        comps = component_model.components[module]
        v = comps.V[:, idx].detach().float()
        u = comps.U[idx, :].detach().float()
        scale = 1.0 - mask_value

        def hook(
            _mod: nn.Module,
            args: tuple[Tensor, ...],
            output: Tensor,
            v: Tensor = v,
            u: Tensor = u,
            scale: float = scale,
        ) -> Tensor:
            removed = (args[0].float() @ v) @ u
            return output - (scale * removed).to(output.dtype)

        target = component_model.target_model.get_submodule(module)
        handles.append(target.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def check_hook_matches_component_path(
    component_model: ComponentModel,
    tokenizer: Any,
    text: str,
    group: list[ComponentKey],
    mask_value: float,
    device: str,
    autocast_bf16: bool,
) -> None:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    module_to_c = {m: component_model.components[m].C for m in component_model.target_module_paths}
    masks = {m: torch.ones((*ids.shape, c), device=device) for m, c in module_to_c.items()}
    for m, c in group:
        masks[m][..., c] = mask_value
    deltas = component_model.calc_weight_deltas()
    delta_masks = {m: (deltas[m], torch.ones(ids.shape, device=device)) for m in deltas}
    all_ones = {m: torch.ones_like(t) for m, t in masks.items()}

    original = component_model.target_model(ids).logits.float()
    with ablate(component_model, group, mask_value):
        via_hook = component_model.target_model(ids).logits.float()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_bf16):
        via_components = component_model(
            ids, mask_infos=make_mask_infos(masks, weight_deltas_and_masks=delta_masks)
        ).float()
        recon = component_model(
            ids, mask_infos=make_mask_infos(all_ones, weight_deltas_and_masks=delta_masks)
        ).float()

    effect = (via_hook - original).abs().max().item()
    hook_err = (via_hook - via_components).abs().max().item()
    recon_err = (recon - original).abs().max().item()
    print(
        f"[check] max|logit diff| for a {len(group)}-component group: ablation effect={effect:.4f}, "
        f"hook vs ComponentModel masked forward={hook_err:.4f}, "
        f"ComponentModel all-ones vs original (numerical noise)={recon_err:.4f}"
    )
    assert hook_err <= 2 * recon_err + 1e-2 * max(effect, 1.0), (
        "forward-hook ablation disagrees with ComponentModel masked forward"
    )


# -- Measurement -----------------------------------------------------------------


@dataclass
class Scored:
    sum_logp: Tensor  # [n] sum log-prob of the continuation
    first_logprobs: Tensor | None  # [n, vocab] next-token log-probs at the first continuation slot


@torch.no_grad()
def score_continuations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    continuations: list[str],
    batch_size: int,
    device: str,
    keep_first_logprobs: bool,
) -> Scored:
    assert len(prompts) == len(continuations)
    sums, firsts = [], []
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None
    for start in range(0, len(prompts), batch_size):
        rows, spans = [], []
        for prompt, cont in zip(
            prompts[start : start + batch_size],
            continuations[start : start + batch_size],
            strict=True,
        ):
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = tokenizer(prompt + cont, add_special_tokens=False)["input_ids"]
            assert full[: len(p_ids)] == p_ids and len(full) > len(p_ids), (prompt, cont)
            rows.append(full)
            spans.append((len(p_ids), len(full)))
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            attn[i, : len(r)] = 1
        logits = model(ids.to(device), attention_mask=attn.to(device)).logits.float()
        logprobs = F.log_softmax(logits, dim=-1)
        for i, (a, b) in enumerate(spans):
            targets = ids[i, a:b].to(device)
            sums.append(logprobs[i, a - 1 : b - 1].gather(-1, targets[:, None]).sum())
            if keep_first_logprobs:
                firsts.append(logprobs[i, a - 1].cpu())
    return Scored(torch.stack(sums).cpu(), torch.stack(firsts) if keep_first_logprobs else None)


@torch.no_grad()
def statement_mean_logp(
    model: Any,
    tokenizer: Any,
    statements: list[str],
    batch_size: int,
    device: str,
) -> Tensor:
    """Mean per-token log-prob of each raw statement (no chat template, as in tPD target data)."""
    out = []
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(statements), batch_size):
        encoded = [
            tokenizer(s, add_special_tokens=False)["input_ids"]
            for s in statements[start : start + batch_size]
        ]
        width = max(len(e) for e in encoded)
        ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for i, e in enumerate(encoded):
            assert len(e) >= 2
            ids[i, : len(e)] = torch.tensor(e)
            attn[i, : len(e)] = 1
        logprobs = F.log_softmax(
            model(ids.to(device), attention_mask=attn.to(device)).logits.float(), -1
        )
        token_lp = logprobs[:, :-1].gather(-1, ids[:, 1:, None].to(device))[..., 0]
        valid = attn[:, 1:].to(device).float()
        out.append(((token_lp * valid).sum(1) / valid.sum(1)).cpu())
    return torch.cat(out)


def load_statement_sets(
    held_out: Statements, max_statements: int, seed: int
) -> dict[str, tuple[list[str], list[str]]]:
    """The run's held-out target statements plus every Geometry of Truth domain."""
    sets = {"target_held_out": (held_out["true"], held_out["false"])}
    for path in sorted(EVAL_DOMAINS_DIR.glob("got_*_true.txt")):
        domain = path.name.removesuffix("_true.txt")
        sets[domain] = (read_lines(path), read_lines(EVAL_DOMAINS_DIR / f"{domain}_false.txt"))
    rng = random.Random(seed)

    def subsample(lines: list[str]) -> list[str]:
        return lines if len(lines) <= max_statements else rng.sample(lines, max_statements)

    return {d: (subsample(t), subsample(f)) for d, (t, f) in sets.items()}


def truth_auc(true_scores: Tensor, false_scores: Tensor) -> float:
    """P(score of a random true statement > score of a random false one); ties count 1/2."""
    diff = true_scores[:, None] - false_scores[None, :]
    return ((diff > 0).float() + 0.5 * (diff == 0).float()).mean().item()


def load_nontarget(config: Config, tokenizer: Any, n: int) -> Tensor:
    cfg = config.nontarget_task_config
    assert isinstance(cfg, LMTaskConfig) and cfg.dataset_name is not None
    ds = load_dataset(cfg.dataset_name, split=cfg.train_data_split, streaming=True)
    assert isinstance(ds, IterableDataset)
    rows: list[list[int]] = []
    for ex in ds:
        assert isinstance(ex, dict)
        ids = tokenizer(ex[cfg.column_name], add_special_tokens=False)["input_ids"]
        if len(ids) >= cfg.max_seq_len:
            rows.append(ids[: cfg.max_seq_len])
        if len(rows) == n:
            break
    return torch.tensor(rows)


@torch.no_grad()
def nontarget_logprobs(model: Any, ids: Tensor, batch_size: int, device: str) -> list[Tensor]:
    return [
        F.log_softmax(model(ids[s : s + batch_size].to(device)).logits.float(), -1)
        .to(torch.bfloat16)
        .cpu()
        for s in range(0, len(ids), batch_size)
    ]


def kl(p_logprobs: Tensor, q_logprobs: Tensor) -> Tensor:
    """KL(p || q) over the last dimension."""
    p_logprobs, q_logprobs = p_logprobs.float(), q_logprobs.float()
    return (p_logprobs.exp() * (p_logprobs - q_logprobs)).sum(-1)


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    device: str,
) -> list[str]:
    tokenizer.padding_side = "left"
    outputs = []
    for s in range(0, len(prompts), batch_size):
        enc = tokenizer(
            prompts[s : s + batch_size], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        outputs += tokenizer.batch_decode(
            gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
    tokenizer.padding_side = "right"
    return outputs


# -- Main ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, choices=RUN_NAMES)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--per-layer-k", type=int, default=TOP_K_PER_LAYER)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument(
        "--mask-value",
        type=float,
        default=0.0,
        help="0 = ablate. Other values scale the group (e.g. 2 = amplify, -1 = flip).",
    )
    ap.add_argument(
        "--max-statements",
        type=int,
        default=200,
        help="Per domain and polarity; a fixed random subset is used when a file is longer",
    )
    ap.add_argument(
        "--n-nontarget", type=int, default=16, help="Pile sequences for the specificity KL"
    )
    ap.add_argument(
        "--generate-groups",
        nargs="*",
        default=["top4_false", "top8_false", "random8_s0", "all_components"],
        help="Groups to also run greedy generation for ('none' is always added).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = results_dir(args.run) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    device = "cuda"

    component_model, config = load_tpd_model(args.run, device)
    model = component_model.target_model
    tokenizer = load_tokenizer(config)
    tokenizer.padding_side = "right"

    # --- select components on the ranking half of the run's target statements ---
    split = split_target_statements(args.run)
    scores = score_components(component_model, config, tokenizer, split.rank)
    print(f"{int(scores.alive.sum())} of {len(scores.keys)} components alive on the ranking split")
    write_component_scores(scores, out_dir / "component_scores.csv")
    groups = build_groups(scores, args.ks, args.per_layer_k, args.n_random, args.seed)
    (out_dir / "groups.json").write_text(
        json.dumps({g: [f"{m}:{c}" for m, c in v] for g, v in groups.items()}, indent=1)
    )
    missing = [g for g in args.generate_groups if g not in groups]
    assert not missing, f"groups this run does not have: {missing}"
    print(f"{len(groups)} groups")

    check_hook_matches_component_path(
        component_model,
        tokenizer,
        split.rank["false"][0],
        groups[f"top{max(args.ks)}_abs"],
        args.mask_value,
        device,
        config.autocast_bf16,
    )

    # --- evaluation data: capital questions whose statements were not used for ranking ---
    rank_lines = set(split.rank["true"]) | set(split.rank["false"])
    eval_qs = [
        q
        for q in load_questions(CAPITALS_PAIRS)
        if f"{q.prefix} {q.correct}." not in rank_lines
        and f"{q.prefix} {q.wrong}." not in rank_lines
    ]
    print(f"{len(eval_qs)} capital questions for evaluation")
    items = build_eval_items(eval_qs, tokenizer)
    prompts = [it.prompt_text for it in items]
    correct_conts = [" " + eval_qs[it.question_idx].correct for it in items]
    wrong_conts = [" " + eval_qs[it.question_idx].wrong for it in items]
    statement_sets = load_statement_sets(split.held_out, args.max_statements, args.seed)
    print(
        "statement domains: "
        + ", ".join(f"{d} ({len(t)}/{len(f)})" for d, (t, f) in statement_sets.items())
    )
    nontarget_ids = load_nontarget(config, tokenizer, args.n_nontarget)
    gen_items = [it for it in items if it.context != "bare"]
    gen_prompts = [it.prompt_text for it in gen_items]

    baseline: Baseline | None = None
    summary_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []
    statement_rows: list[dict[str, object]] = []
    generation_rows: list[dict[str, str]] = []
    group_names = ["none"] + [g for g in groups if g != "none"]
    for gi, name in enumerate(group_names):
        group = groups[name]
        with ablate(component_model, group, args.mask_value):
            correct = score_continuations(
                model, tokenizer, prompts, correct_conts, args.batch_size, device, True
            )
            wrong = score_continuations(
                model, tokenizer, prompts, wrong_conts, args.batch_size, device, False
            )
            stmt = {
                domain: (
                    statement_mean_logp(model, tokenizer, true_s, args.batch_size, device),
                    statement_mean_logp(model, tokenizer, false_s, args.batch_size, device),
                )
                for domain, (true_s, false_s) in statement_sets.items()
            }
            nt = nontarget_logprobs(model, nontarget_ids, args.batch_size, device)
            gens = (
                generate(
                    model, tokenizer, gen_prompts, args.max_new_tokens, args.batch_size, device
                )
                if name == "none" or name in args.generate_groups
                else None
            )

        margin = correct.sum_logp - wrong.sum_logp
        assert correct.first_logprobs is not None
        stmt_metrics = {
            d: ((t.mean() - f.mean()).item(), truth_auc(t, f)) for d, (t, f) in stmt.items()
        }
        truthful = truthful_hits(eval_qs, gen_items, gens) if gens is not None else None
        if name == "none":
            assert truthful is not None
            baseline = Baseline(
                margin, correct.sum_logp, correct.first_logprobs, nt, stmt_metrics, truthful
            )
        assert baseline is not None
        answer_kl = kl(baseline.first_logprobs, correct.first_logprobs)

        row: dict[str, object] = {
            "group": name,
            "size": len(group),
            "n_layers": len({m for m, _ in group}),
        }
        for context in CONTEXTS:
            sel = torch.tensor([it.context == context for it in items])
            m, m0 = margin[sel], baseline.margin[sel]
            row[f"{context}_margin"] = m.mean().item()
            row[f"{context}_dmargin"] = (m - m0).mean().item()
            row[f"{context}_acc"] = (m > 0).float().mean().item()
            row[f"{context}_flip_rate"] = ((m > 0) != (m0 > 0)).float().mean().item()
            row[f"{context}_dlogp_correct"] = (
                (correct.sum_logp[sel] - baseline.logp_correct[sel]).mean().item()
            )
            row[f"{context}_answer_kl"] = answer_kl[sel].mean().item()
        for domain, (true_lp, false_lp) in stmt.items():
            gap, auc = stmt_metrics[domain]
            gap0, auc0 = baseline.statements[domain]
            row[f"{domain}_gap"], row[f"{domain}_dgap"] = gap, gap - gap0
            row[f"{domain}_auc"], row[f"{domain}_dauc"] = auc, auc - auc0
            statement_rows.append(
                {
                    "group": name,
                    "size": len(group),
                    "domain": domain,
                    "n_true": len(true_lp),
                    "n_false": len(false_lp),
                    "gap": gap,
                    "d_gap": gap - gap0,
                    "auc": auc,
                    "d_auc": auc - auc0,
                }
            )
        row["nontarget_kl"] = (
            torch.cat([kl(a, b).flatten() for a, b in zip(baseline.nontarget, nt, strict=True)])
            .mean()
            .item()
        )
        if gens is not None:
            assert truthful is not None
            dishonest = np.array([it.context == "dishonest" for it in gen_items])
            for context in ("honest", "dishonest"):
                sel = dishonest if context == "dishonest" else ~dishonest
                for metric, hits in truthful.items():
                    row[f"{context}_truthful_{metric}"] = float(hits[sel].mean())
            for metric, hits in truthful.items():
                hits0 = baseline.truthful[metric]
                gained = int((hits & ~hits0 & dishonest).sum())
                lost = int((~hits & hits0 & dishonest).sum())
                row[f"dishonest_truthful_{metric}_sign_p"] = (
                    binomtest(gained, gained + lost).pvalue if gained + lost else 1.0
                )
            for it, g in zip(gen_items, gens, strict=True):
                generation_rows.append(
                    {
                        "group": name,
                        "context": it.context,
                        "question": eval_qs[it.question_idx].question,
                        "correct": eval_qs[it.question_idx].correct,
                        "generation": g,
                    }
                )
        summary_rows.append(row)
        for i, it in enumerate(items):
            item_rows.append(
                {
                    "group": name,
                    "question": eval_qs[it.question_idx].question,
                    "context": it.context,
                    "logp_correct": correct.sum_logp[i].item(),
                    "logp_wrong": wrong.sum_logp[i].item(),
                    "margin": margin[i].item(),
                    "answer_kl": answer_kl[i].item(),
                }
            )
        print(
            f"[{gi + 1}/{len(group_names)}] {name:>22s} size={len(group):4d} "
            + " ".join(f"{c}:dmargin={row[f'{c}_dmargin']:+.3f}" for c in CONTEXTS)
            + f" answer_kl={row['bare_answer_kl']:.4f} nontarget_kl={row['nontarget_kl']:.4f}"
        )
    (out_dir / "generations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in generation_rows)
    )

    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "per_item.csv", item_rows)
    write_csv(out_dir / "statement_domains.csv", statement_rows)
    plot_dose_response(summary_rows, args.ks, out_dir / "dose_response.png")
    plot_statement_domains(
        summary_rows, list(statement_sets), args.ks, out_dir / "statement_domains.png"
    )
    print(f"\nWrote results to {out_dir}")


def plot_statement_domains(
    rows: list[dict[str, object]], domains: list[str], ks: list[int], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_name = {str(r["group"]): r for r in rows}
    fig, axes = plt.subplots(1, len(domains), figsize=(3.6 * len(domains), 3.4), squeeze=False)
    for ax, domain in zip(axes[0], domains, strict=True):
        metric = f"{domain}_dauc"
        for kind, color in (("false", "C3"), ("true", "C2")):
            pts = [
                (k, by_name[f"top{k}_{kind}"][metric]) for k in ks if f"top{k}_{kind}" in by_name
            ]
            if pts:
                ax.plot(*zip(*pts, strict=True), "o-", color=color, label=f"top-k {kind}")
        rand = [
            (k, [r[metric] for n, r in by_name.items() if n.startswith(f"random{k}_s")]) for k in ks
        ]
        rand = [(k, v) for k, v in rand if v]
        if rand:
            xs = [k for k, _ in rand]
            ax.fill_between(
                xs,
                [min(v) for _, v in rand],
                [max(v) for _, v in rand],
                color="0.6",
                alpha=0.3,
                label="random (min–max)",
            )
        if "all_components" in by_name:
            ax.axhline(
                by_name["all_components"][metric],
                color="0.3",
                ls="--",
                lw=1,
                label="all components",
            )
        ax.axhline(0, color="0.2", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("group size k")
        ax.set_title(f"{domain}\nbaseline AUC {by_name['none'][f'{domain}_auc']:.2f}", fontsize=9)
    axes[0][0].set_ylabel("Δ truth AUC vs. unablated")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_dose_response(rows: list[dict[str, object]], ks: list[int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_name = {str(r["group"]): r for r in rows}
    metrics = [f"{c}_dmargin" for c in CONTEXTS] + ["bare_answer_kl"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, metric in zip(axes, metrics, strict=True):
        for kind, color in (("abs", "C0"), ("false", "C3"), ("true", "C2")):
            pts = [
                (k, by_name[f"top{k}_{kind}"][metric]) for k in ks if f"top{k}_{kind}" in by_name
            ]
            if pts:
                ax.plot(*zip(*pts, strict=True), "o-", color=color, label=f"top-k {kind}")
        rand = [
            (k, [r[metric] for n, r in by_name.items() if n.startswith(f"random{k}_s")]) for k in ks
        ]
        rand = [(k, v) for k, v in rand if v]
        if rand:
            xs = [k for k, _ in rand]
            ax.fill_between(
                xs,
                [min(v) for _, v in rand],
                [max(v) for _, v in rand],
                color="0.6",
                alpha=0.3,
                label="random (min–max)",
            )
            ax.plot(xs, [np.mean(v) for _, v in rand], color="0.4", lw=1)
        ax.axhline(0, color="0.2", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("group size k")
        ax.set_title(metric)
    axes[0].set_ylabel("change vs. unablated model")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
