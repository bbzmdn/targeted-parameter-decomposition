"""The 8 honesty tPD runs: names, where they live, and how to load them.

Run names are tpd-<target>-<polarity>, e.g. tpd-capitals-true or tpd-capitals-liu-tqa-false.
Each run is saved locally at SPD_OUT_DIR/spd/<run name>/ and uploaded to the Hugging Face
repo <HF_USER>/qwen2.5-7b-instruct-<run name>.
"""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.settings import SPD_OUT_DIR
from spd.utils.general_utils import resolve_class
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import parse_config

HERE = Path(__file__).parent
PROMPTS_DIR = HERE / "prompts"
EVAL_DOMAINS_DIR = HERE / "liu_per_domain_prompts"
RESULTS_DIR = HERE / "results"
HF_USER = "mdnml"
CHECKPOINT_NAME = "model_10000.pth"

TARGETS = ["capitals", "liu", "tqa", "capitals_liu_tqa"]
POLARITIES = ["true", "false"]


def run_name(target: str, polarity: str) -> str:
    assert target in TARGETS and polarity in POLARITIES, (target, polarity)
    return f"tpd-{target.replace('_', '-')}-{polarity}"


RUN_TARGETS = {run_name(t, p): t for t in TARGETS for p in POLARITIES}
RUN_NAMES = list(RUN_TARGETS)

Statements = dict[str, list[str]]  # polarity ("true" / "false") -> one statement per entry


def prompts_file(target: str, polarity: str) -> Path:
    return PROMPTS_DIR / f"{target}_{polarity}.txt"


def read_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    assert lines, f"no statements in {path}"
    return lines


def target_statements(name: str) -> Statements:
    """True and false statements of the run's target set (the run itself trained on one polarity)."""
    return {p: read_lines(prompts_file(RUN_TARGETS[name], p)) for p in POLARITIES}


def results_dir(name: str) -> Path:
    assert name in RUN_NAMES, f"unknown run {name!r}; expected one of {RUN_NAMES}"
    return RESULTS_DIR / name


def local_run_dir(name: str) -> Path:
    assert name in RUN_NAMES, f"unknown run {name!r}; expected one of {RUN_NAMES}"
    return SPD_OUT_DIR / "spd" / name


def hf_repo_id(name: str) -> str:
    assert name in RUN_NAMES, f"unknown run {name!r}; expected one of {RUN_NAMES}"
    return f"{HF_USER}/qwen2.5-7b-instruct-{name}"


def run_files_dir(name: str) -> Path:
    """Directory holding a run's checkpoint and final_config.yaml, downloaded from Hugging Face
    if the run isn't on disk."""
    local = local_run_dir(name)
    if (local / CHECKPOINT_NAME).exists():
        return local
    return Path(
        snapshot_download(
            hf_repo_id(name),
            allow_patterns=[CHECKPOINT_NAME, "final_config.yaml"],
        )
    )


def checkpoint_path(name: str) -> Path:
    return run_files_dir(name) / CHECKPOINT_NAME


def build_component_model(config: Config, device: str) -> ComponentModel:
    """ComponentModel with freshly initialised components and the target model in bf16.

    Loading the target model in bf16 straight onto the device keeps Qwen2.5-7B at ~15GB and
    avoids a full fp32 copy in CPU memory.
    """
    assert config.pretrained_model_name is not None
    target_model = resolve_class(config.pretrained_model_class).from_pretrained(  # pyright: ignore[reportAttributeAccessIssue]
        config.pretrained_model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    target_model.eval()
    target_model.requires_grad_(False)
    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=expand_module_patterns(target_model, config.all_module_info),
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )
    component_model.to(device)
    component_model.eval()
    return component_model


def load_trained_weights(component_model: ComponentModel, name: str) -> None:
    """Load a run's components and CI function into a ComponentModel of the same architecture."""
    missing, unexpected = component_model.load_state_dict(
        torch.load(checkpoint_path(name), map_location="cpu", weights_only=True), strict=False
    )
    assert not unexpected, f"Unexpected keys in checkpoint: {unexpected}"
    assert all(k.startswith("target_model.") for k in missing), (
        f"Missing trained weights: {missing}"
    )


def load_tokenizer(config: Config) -> PreTrainedTokenizerBase:
    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    return tokenizer


def load_tpd_config(name: str) -> Config:
    return parse_config(run_files_dir(name) / "final_config.yaml", None)


def load_tpd_model(name: str, device: str) -> tuple[ComponentModel, Config]:
    """Load a run by name (e.g. "tpd-liu-false") for analysis."""
    config = load_tpd_config(name)
    component_model = build_component_model(config, device)
    load_trained_weights(component_model, name)
    return component_model, config
