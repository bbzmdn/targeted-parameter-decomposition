"""Select a run's top components from its own target statements (shared by token firing and ablation)."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import Config, LMTaskConfig
from spd.experiments.lm.honesty_targeted_decomposition.tpd_runs import (
    POLARITIES,
    Statements,
    target_statements,
)
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import bf16_autocast

SPLIT_SEED = 0
RANK_FRAC = 0.5
TOP_K_PER_LAYER = 8
PUNCT = {".", ",", "!", "?"}

ComponentKey = tuple[str, int]  # (module name, component index)


@dataclass
class TargetSplit:
    rank: Statements  # used to select components
    held_out: Statements  # used to evaluate them


@dataclass
class ComponentScores:
    keys: list[ComponentKey]
    mean_ci_true: np.ndarray  # [n_components] mean answer-token CI on true statements
    mean_ci_false: np.ndarray
    alive: np.ndarray  # [n_components] bool

    @property
    def scores(self) -> np.ndarray:
        """> 0: fires more on false statements."""
        return self.mean_ci_false - self.mean_ci_true


def split_target_statements(name: str) -> TargetSplit:
    """Seeded split of each polarity. Equal-length (paired) lists get the same permutation."""
    rank: Statements = {}
    held_out: Statements = {}
    for polarity, lines in target_statements(name).items():
        order = list(range(len(lines)))
        random.Random(SPLIT_SEED).shuffle(order)
        n_rank = round(RANK_FRAC * len(lines))
        rank[polarity] = [lines[i] for i in sorted(order[:n_rank])]
        held_out[polarity] = [lines[i] for i in sorted(order[n_rank:])]
    return TargetSplit(rank, held_out)


@torch.no_grad()
def token_ci(
    component_model: ComponentModel, config: Config, tokenizer: PreTrainedTokenizerBase, text: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Tokens of `text` and the CI of every component at every token: {module: [seq, C]}."""
    ids = torch.tensor([tokenizer.encode(text)])
    assert isinstance(config.task_config, LMTaskConfig)
    assert ids.shape[1] <= config.task_config.max_seq_len, text
    device = next(component_model.ci_fn.parameters()).device
    ci_dtype = next(component_model.ci_fn.parameters()).dtype
    with bf16_autocast(enabled=config.autocast_bf16):
        out = component_model(ids.to(device), cache_type="input")
        acts = {k: v.to(ci_dtype) for k, v in out.cache.items()}
        ci = component_model.calc_causal_importances(acts, sampling=config.sampling).lower_leaky
    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]
    return tokens, {m: v[0].float().cpu().numpy() for m, v in ci.items()}


def answer_position(tokens: list[str]) -> int:
    """Last token that is not punctuation: where a statement's truth value is decided."""
    pos = len(tokens) - 1
    if tokens[pos].strip() in PUNCT:
        pos -= 1
    return pos


def answer_token_ci(
    component_model: ComponentModel,
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
) -> dict[str, np.ndarray]:
    """{module: [n_texts, C]} CI at the answer token of each text."""
    per_module: dict[str, list[np.ndarray]] = {m: [] for m in component_model.target_module_paths}
    for text in texts:
        tokens, ci = token_ci(component_model, config, tokenizer, text)
        pos = answer_position(tokens)
        for m in per_module:
            per_module[m].append(ci[m][pos])
    return {m: np.stack(v) for m, v in per_module.items()}


def score_components(
    component_model: ComponentModel,
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
    statements: Statements,
) -> ComponentScores:
    mean_ci = {
        polarity: answer_token_ci(component_model, config, tokenizer, statements[polarity])
        for polarity in POLARITIES
    }
    modules = component_model.target_module_paths
    keys = [(m, c) for m in modules for c in range(mean_ci["true"][m].shape[1])]
    mean_true = np.concatenate([mean_ci["true"][m].mean(0) for m in modules])
    mean_false = np.concatenate([mean_ci["false"][m].mean(0) for m in modules])
    alive = np.maximum(mean_true, mean_false) > config.ci_alive_threshold
    return ComponentScores(keys, mean_true, mean_false, alive)


def top_components_per_layer(scores: ComponentScores, k: int) -> dict[str, list[int]]:
    """{module: the k components with the largest |score| in that module}."""
    order = np.argsort(-np.abs(scores.scores))
    per_layer: dict[str, list[int]] = {}
    for i in order:
        module, c = scores.keys[i]
        if len(per_layer.setdefault(module, [])) < k:
            per_layer[module].append(c)
    return per_layer


def layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    return int(parts[parts.index("layers") + 1])


def write_component_scores(scores: ComponentScores, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "component", "score", "mean_ci_true", "mean_ci_false", "alive"])
        for i in np.argsort(-np.abs(scores.scores)):
            w.writerow(
                [
                    scores.keys[i][0],
                    scores.keys[i][1],
                    f"{scores.scores[i]:.6g}",
                    f"{scores.mean_ci_true[i]:.6g}",
                    f"{scores.mean_ci_false[i]:.6g}",
                    bool(scores.alive[i]),
                ]
            )
