# Honesty targeted decomposition

Can targeted parameter decomposition (tPD) find the mechanisms a language model uses to tell
true from false statements, and to lie? Activation probes show that LLMs linearly represent
whether a statement is true (Marks and Tegmark, 2024; Liu et al., 2024), but they do not say
which weights compute that verdict. tPD decomposes weight matrices into rank-one components and
learns, for every input, which components the model actually uses. Targeting it at true or
false statements should isolate the weights involved in processing truth.

We decompose the `mlp.down_proj` matrices of layers 12-21 of Qwen2.5-7B-Instruct (96 components
per layer, 960 total), targeting bare true or false statements, with the Pile as non-target data
so the rest of the model's behaviour is preserved. We then ask three questions of each
decomposition:

1. **Detection.** Do the components' causal importances (CI) tell true from false statements,
   also on datasets the decomposition never saw? (logistic probe, token firing)
2. **Causality.** Does ablating the most truth-selective components change whether the model
   lies when told to? (group ablation)
3. **Representation.** Do those components write along the residual-stream truth direction?
   (truth-direction alignment)

## Results

The full write-up, with methods and tables, is in [RESULTS.md](RESULTS.md). In short:

- **Detection works.** A logistic probe on the 960 CI values reaches 83.3 mean AUC over 18
  held-out domains, close to the best probe on raw activations (86.3) with about 37 times
  fewer features. It does well on some domains where activation probes are near chance.
- **Causality only shows up all at once.** Ablating every component raises truthful answers
  under a lying instruction from 17% to 46-54%. No small selected group beats random groups of
  the same size; small groups change how the model lies, not whether it lies.
- **The components do not look like a truth representation.** Their output directions are not
  aligned with the activation truth direction, and the truth signal is spread over all
  decomposed layers.

tPD is useful as a compact truth detector, but so far it does not isolate a causal truth
mechanism.

## The 8 runs

| Run name | Target statements | Hugging Face repo (private) |
| --- | --- | --- |
| `tpd-capitals-true` / `tpd-capitals-false` | 35 capital-city facts | `mdnml/qwen2.5-7b-instruct-tpd-capitals-{true,false}` |
| `tpd-liu-true` / `tpd-liu-false` | ~180 statements from the 12 Liu et al. (2024) domains | `mdnml/qwen2.5-7b-instruct-tpd-liu-{true,false}` |
| `tpd-tqa-true` / `tpd-tqa-false` | 180 TruthfulQA question + best (in)correct answer | `mdnml/qwen2.5-7b-instruct-tpd-tqa-{true,false}` |
| `tpd-capitals-liu-tqa-true` / `-false` | all of the above combined | `mdnml/qwen2.5-7b-instruct-tpd-capitals-liu-tqa-{true,false}` |

All runs share `tpd_config.yaml` (seed 0, 10k steps, `max_seq_len` 128) and differ only in the
target prompts file. The TruthfulQA training questions exclude the 50 TruthfulQA evaluation
questions in `liu_per_domain_prompts/`, so TruthfulQA stays held out (`prompts/tqa_split.json`).

## Pipeline

Run everything from the repo root with `python -m spd.experiments.lm.honesty_targeted_decomposition.<script>`.

| Step | Script | Output |
| --- | --- | --- |
| Build target prompts | `build_prompts` | `prompts/<target>_<polarity>.txt` |
| Train + upload | `run_tpd [--runs ...] [--no-upload]` | `SPD_OUT_DIR/spd/<run>/`, HF repo |
| Token firing | `visualize_token_firing --run <run>` | `results/<run>/token_firing/` |
| Group ablation | `group_ablation --run <run>` | `results/<run>/ablation/` |
| Logistic probe | `fit_probe [--runs ...]` | `results/probe/` |
| Truth-direction alignment | `truth_direction_alignment --run <run>` | `results/<run>/truth_direction/` |

Analyses load a run from `SPD_OUT_DIR/spd/<run>/` if it is there and from Hugging Face otherwise
(`tpd_runs.load_tpd_model`). Uploads refuse to run under any HF login other than `mdnml`.

**Component selection** (`find_top_components.py`, shared by token firing and ablation): the
run's target set, in both its true and its false version, is split in half. On one half, each
component is scored by mean CI(false) − mean CI(true) at the answer token (the last
non-punctuation token). The other half is used for evaluation. A liu run is therefore always
ranked on liu statements, a capitals run on capital statements, and so on.

**Token firing**: CI per token of held-out statements for the top 8 components of each layer.

**Group ablation**: masks of a group set to 0 (delta component kept on). Groups are global
top-k false / true / |score|, the per-layer top 8 (the same components as the token firing),
size-matched random groups, all alive components, and all 960 components. It measures:
- lying behaviour on held-out capital questions with honest/dishonest system prompts
- answer log-prob margins
- statement truth AUC on held-out target statements and Geometry of Truth
- Pile KL

**Logistic probe**: L1 logistic regression on sequence-averaged CI, leave-one-domain-out over
the 18 domains in `liu_per_domain_prompts/`:
- 11 Liu et al., without arithmetic
- 6 Geometry of Truth `got_*`
- TruthfulQA

It is compared with activation baselines (mass-mean, logistic regression, PCA-300), an
untrained decomposition, and random directions.

**Truth-direction alignment**: cosine between each component's output direction U_c and the
mass-mean truth direction (mean(false) − mean(true)) of the residual stream after its layer.
The top 8 per layer are compared with the rest of the layer and with a random direction.

## Results layout

```
results/
├── probe/
│   ├── summary.md             mean accuracy / AUC per method (the RESULTS.md table)
│   ├── auc.csv, accuracy.csv  per held-out domain x method
│   └── components_<run>.csv   non-zero probe weights per component
└── <run>/
    ├── component_scores.csv   all 960 components, ranked on the ranking half
    ├── token_firing/
    │   ├── L<layer>.html      token-level CI of the layer's top components
    │   └── summary.csv        ranking vs. held-out mean CI per shown component
    ├── truth_direction/
    │   ├── components.csv     cosine with the truth direction, every component
    │   ├── summary.csv        per layer: top vs. rest vs. random direction
    │   └── alignment.png
    └── ablation/
        ├── summary.csv        one row per group: all metrics
        ├── generations.jsonl  greedy answers under honest / dishonest prompts
        ├── per_item.csv, statement_domains.csv, groups.json, component_scores.csv
        └── dose_response.png, statement_domains.png
```

## Data

- `data/`: source data (`capitals_contrast_pairs.jsonl` with honest/dishonest system prompts, `liu_{true,false}.txt`)
- `prompts/`: tPD target data, written by `build_prompts.py`
- `liu_per_domain_prompts/`: evaluation domains. The Geometry of Truth and TruthfulQA files are
  written by `export_geometry_of_truth.py` and `export_truthfulqa.py`

## Credits

Targeted parameter decomposition is the method of Vigouroux and Sharkey (2026), "Targeted
Recovery of Weight-Space Mechanisms From Neural Networks". This experiment runs on their
codebase ([Antovigo/targeted-parameter-decomposition](https://github.com/Antovigo/targeted-parameter-decomposition)),
itself a fork of [goodfire-ai/spd](https://github.com/goodfire-ai/spd) (Stochastic Parameter
Decomposition; Bushnaq, Braun and Sharkey, 2025). Evaluation data comes from Liu et al. (2024),
Marks and Tegmark (2024, Geometry of Truth) and Lin et al. (2022, TruthfulQA).
