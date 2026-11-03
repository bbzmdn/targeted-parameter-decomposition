# Honesty targeted decomposition

tPD of the `mlp.down_proj` matrices of layers 12-21 of Qwen2.5-7B-Instruct (96 components per
layer, 960 total), targeting bare true or false statements, with the Pile as non-target data.
Eight runs, then four analyses (token firing, group ablation, logistic probe, truth-direction
alignment). Human-facing docs are [README.md](README.md) (how to run, file layout) and
[RESULTS.md](RESULTS.md) (write-up of all results).

## Branch

This folder only exists on `feature/honesty-targeted-decomposition`. Check with
`git branch --show-current` and run `git checkout feature/honesty-targeted-decomposition` if
needed. Branch off it, not off `main`, for work on this experiment.

## Running

From the repo root, after `source .venv/bin/activate`:

```bash
python -m spd.experiments.lm.honesty_targeted_decomposition.<script> [args]
```

| Script | Args | Writes |
| --- | --- | --- |
| `build_prompts` | none (needs internet) | `prompts/<target>_<polarity>.txt`, `prompts/tqa_split.json` |
| `run_tpd` | `--runs ...`, `--no-upload` | `SPD_OUT_DIR/spd/<run>/`, HF repo |
| `visualize_token_firing` | `--run <run>` | `results/<run>/token_firing/`, `results/<run>/component_scores.csv` |
| `group_ablation` | `--run <run>` | `results/<run>/ablation/` |
| `fit_probe` | `--runs ...` (default all 8) | `results/probe/` |
| `truth_direction_alignment` | `--run <run>` | `results/<run>/truth_direction/` |
| `export_geometry_of_truth`, `export_truthfulqa` | none | `liu_per_domain_prompts/got_*`, `truthfulqa_*` |

- Every analysis needs a CUDA GPU (device is hard-coded to `cuda`). The bf16 target model is ~15 GB.
- Analyses overwrite their `results/` files. Check `git diff results/` before committing.
- Uploads assert the HF login is `mdnml`. Anyone else trains with `--no-upload`.
- Training logs to W&B (`.env` with `WANDB_API_KEY`). Analyses do not use W&B.

## Code map

- `tpd_runs.py`. Run names (`tpd-<target>-<polarity>`, targets `capitals`, `liu`, `tqa`,
  `capitals_liu_tqa`), paths, and `load_tpd_model(name, device)`. Loads from
  `SPD_OUT_DIR/spd/<run>/model_10000.pth` if present, else downloads the private HF repo
  `mdnml/qwen2.5-7b-instruct-<run>`. Checkpoints hold only trained weights; Qwen is reloaded
  from `Qwen/Qwen2.5-7B-Instruct`.
- `find_top_components.py`. Shared component selection. Splits the run's own true and false
  target statements in half (seeded; capital pairs kept together), scores each component by mean
  CI(false) − mean CI(true) at the answer token (last non-punctuation token) on the ranking
  half. Token firing, ablation, alignment and top-k probes all use this ranking.
- `group_ablation.py`. Sets masks of a group to 0 with the delta component on. Measures lying
  under honest/dishonest system prompts (`data/capitals_contrast_pairs.jsonl`), answer margins,
  statement-truth AUC and Pile KL.
- `fit_probe.py`. L1 logistic regression on sequence-mean CI, leave-one-domain-out over the 18
  domains in `liu_per_domain_prompts/` (11 Liu = in domain, 6 `got_*` + TruthfulQA = out of
  domain). Activation baselines and an untrained-decomposition control.
- `truth_direction_alignment.py`. Cosine of each U\_c with the mass-mean truth direction.

## Facts that are easy to get wrong

- **Component indices are aligned across runs.** All runs use seed 0; same (layer, index) has
  mean |cos| ≈ 0.81 between runs' U vectors vs ≈ 0.08 for any other index. Comparing a
  component across runs by index is valid; their CI functions still differ.
- **capitals-true has no ranked components.** Nothing fires at its answer token, so all its
  scores are 0 and its top-k sets and token firing pages are arbitrary.
- **`component_scores.csv` is sorted by score.** Join runs on `(module, component)`, never by row.
- **liu arithmetic is excluded** from evaluation (placeholder "banana" answers).
- **The tqa targets exclude the 50 TruthfulQA eval questions** (`prompts/tqa_split.json`).
- `generations.jsonl` has groups in blocks of equal size (`none` first, `all_components`
  last), so the unablated answer to line L of a group is at line `(L - 1) % block + 1`.

## Writing in RESULTS.md

Match the existing style. Bold lead sentence per paragraph, short declarative sentences,
numbers with their comparison, tables for anything with more than three numbers. No em dashes,
avoid colons in prose, no filler ("interestingly", "it's worth noting"). Link quoted
generations to their line (`results/<run>/ablation/generations.jsonl#L<n>`). Keep in-domain
and out-of-domain numbers separate.
