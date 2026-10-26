"""Train the honesty tPD runs one after another and upload each to the HF_USER account (private repos).

python run_tpd.py                     # all 8 runs; a run with a checkpoint on disk is only uploaded
python run_tpd.py --runs tpd-tqa-true --no-upload
"""

import argparse
import json
import subprocess
import sys

import yaml
from huggingface_hub import HfApi

from spd.experiments.lm.honesty_targeted_decomposition.tpd_runs import (
    CHECKPOINT_NAME,
    HERE,
    HF_USER,
    POLARITIES,
    RUN_NAMES,
    TARGETS,
    hf_repo_id,
    local_run_dir,
    prompts_file,
    run_name,
)
from spd.log import logger

TARGET_DESCRIPTIONS = {
    "capitals": "35 capital-city statements",
    "liu": "statements from the 12 domains of Liu et al. (2024)",
    "tqa": "180 TruthfulQA question-answer statements (questions disjoint from the TruthfulQA eval set)",
    "capitals_liu_tqa": "the capitals, liu and tqa statements combined",
}


def train(target: str, polarity: str) -> None:
    config = yaml.safe_load((HERE / "tpd_config.yaml").read_text())
    name = run_name(target, polarity)
    config["label"] = name
    config["task_config"]["prompts_file"] = str(prompts_file(target, polarity))
    cmd = [
        sys.executable,
        "-m",
        "spd.experiments.lm.lm_decomposition",
        "--config_json",
        "json:" + json.dumps(config),
        "--run_id",
        name,
    ]
    logger.info(f"Training {name}")
    subprocess.run(cmd, check=True)


def model_card(target: str, polarity: str) -> str:
    name = run_name(target, polarity)
    return f"""---
base_model: Qwen/Qwen2.5-7B-Instruct
tags: [targeted-parameter-decomposition, interpretability]
---
# {name}

Targeted parameter decomposition (tPD) of the `mlp.down_proj` matrices of layers 12-21 of
Qwen2.5-7B-Instruct (96 rank-one components per layer, 960 total), targeting **{polarity}**
statements: {TARGET_DESCRIPTIONS[target]}. Non-target data: the Pile (monology/pile-uncopyrighted).

- `{CHECKPOINT_NAME}`: components and causal-importance function (the frozen Qwen weights are not
  included; they are loaded from `Qwen/Qwen2.5-7B-Instruct`)
- `final_config.yaml`: full training config
- `{prompts_file(target, polarity).name}`: the target statements, one per line
- `metrics.jsonl`, `figures/`: training metrics

Load with the SPD codebase:

```python
from spd.models.component_model import ComponentModel
from huggingface_hub import snapshot_download

path = snapshot_download("{hf_repo_id(name)}")
model = ComponentModel.from_pretrained(f"{{path}}/{CHECKPOINT_NAME}")
```
"""


def upload(target: str, polarity: str) -> None:
    name = run_name(target, polarity)
    run_dir = local_run_dir(name)
    assert (run_dir / CHECKPOINT_NAME).exists(), f"no checkpoint for {name} at {run_dir}"
    repo_id = hf_repo_id(name)
    api = HfApi()
    logged_in_as = api.whoami()["name"]
    assert logged_in_as == HF_USER, (
        f"HF login is {logged_in_as!r}, not {HF_USER!r}; run `hf auth login`"
    )
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=run_dir,
        allow_patterns=[
            CHECKPOINT_NAME,
            "final_config.yaml",
            "run_metadata.json",
            "metrics.jsonl",
            "figures/*",
        ],
    )
    api.upload_file(
        path_or_fileobj=prompts_file(target, polarity),
        path_in_repo=prompts_file(target, polarity).name,
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=model_card(target, polarity).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    logger.info(f"Uploaded {name} to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", choices=RUN_NAMES, default=RUN_NAMES)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    for target in TARGETS:
        for polarity in POLARITIES:
            name = run_name(target, polarity)
            if name not in args.runs:
                continue
            if not (local_run_dir(name) / CHECKPOINT_NAME).exists():
                train(target, polarity)
            if not args.no_upload:
                upload(target, polarity)


if __name__ == "__main__":
    main()
