# How useful is targeted parameter decomposition for finding truth mechanisms?

> **Status.** All numbers below come from the first round of decompositions (capitals-true/false, liu-true/false, liu-false + TruthfulQA; see [Earlier runs](#earlier-runs)). The eight runs described in [Setup](#setup) are being retrained with one shared config, and the numbers will be replaced with theirs.

## TL;DR

We applied targeted Parameter Decomposition (tPD) to the MLP down-projections of layers 12–21 of Qwen2.5-7B-Instruct, targeting true and false statements, to see whether it finds the mechanisms behind truthfulness and lying. Our answer so far: **tPD does something, but it does not hand us a truth mechanism.**

- **Probing works.** A logistic regression on the causal importance (CI) scores of the 960 components reaches 83.3 mean AUC over 18 domains, close to the mass-mean probe on raw activations (86.3) with about 37 times fewer features. We have not yet checked whether CI scores from an untrained decomposition do as well.
- **Ablation only works all at once.** Ablating every component raises the share of truthful answers under a lying instruction from 17% to 46% (liu-false) and from 17% to 54% (liu-false + TruthfulQA). No small, selected group of components beats random groups of the same size at changing whether the model lies.
- **The components do not look like a truth concept.** Their directions are orthogonal to the MLP output, and they do not line up with the truth direction found in activations.
- **Singling out the mechanism is hard.** The signal is spread over many layers, and the model seems to recompute its verdict, so removing one place does not remove the behavior.

## Setup

Interpretability research on honesty in LLMs has mostly focused on activations, where linear probes can read out whether the model treats a statement as true or false (Liu et al., 2024; Marks and Tegmark, 2024; Azaria and Mitchell, 2023). Parameter decomposition offers a different unit of analysis. Attribution-based Parameter Decomposition (APD; Braun et al., 2025) and Stochastic Parameter Decomposition (SPD; Bushnaq et al., 2025) learn rank-one components of the weight matrices so that the components sum to the original weights. They also learn causal importance functions that say, for every input, which components are being used.

We used the targeted variant, tPD (Vigouroux and Sharkey, 2026). Unlike SPD, tPD does not try to decompose everything the model does. It focuses on the mechanisms used on a small target dataset, while non-target data (the Pile) keeps the rest of the model's behavior intact.

**Model and decomposition.** Qwen2.5-7B-Instruct; the `down_proj` matrices of layers 12–21; C = 96 components per layer, 960 in total. Due to computational restrictions we did not decompose attention or the other MLP matrices.

**Target data.** Small sets of bare true or false statements, with no chat template and no instruction. Each target set exists in a true and a false version, and each version gets its own decomposition, giving eight runs:

| Run | Target statements |
| --- | --- |
| capitals-true / capitals-false | 35 capital-city facts ("The capital of France is Paris." / "... is Lyon.") |
| liu-true / liu-false | ~180 statements aggregated from the 12 domains of Liu et al. (2024) |
| tqa-true / tqa-false | 180 TruthfulQA questions, each followed by its best correct / best incorrect answer |
| capitals-liu-tqa-true / -false | the three sets above combined (~395 statements) |

All eight runs share the same config: seed 0, 10,000 steps, sequence length 128, the Pile as non-target data. The decompositions are on Hugging Face (`mdnml/qwen2.5-7b-instruct-tpd-<run>`).

**Evaluation data.** 18 domains: 11 from Liu et al. (2024), six Geometry of Truth datasets from Marks and Tegmark (2024) (larger than, smaller than, sp-en trans, neg sp-en trans, neg cities, likely), and 50 TruthfulQA questions (Lin et al., 2022). Liu's arithmetic domain is left out because some of its false statements use the placeholder answer "banana", so the last token differs in kind rather than in truth value. Geometry of Truth and the 50 TruthfulQA evaluation questions are held out for every run: the tqa target sets are drawn from the remaining TruthfulQA questions.

### Methods

**Selecting components.** Each run's components are selected on that run's own target data, never on another dataset: a liu-false decomposition is ranked on liu statements, not on capitals. Because every target set exists in a true and a false version, we can rank on both. We split the true and the false statements of the run's target set in half (seeded; true/false pairs of the same capital stay together). On the ranking half we read each component's CI at the answer token, which is the last non-punctuation token ("Paris" in "The capital of France is Paris."). Each component's score is mean CI on false statements minus mean CI on true statements. Positive scores mean the component fires more on false statements. Both the token-firing visualisation and the ablations use this ranking, and everything is evaluated on the other half.

**Truth-direction alignment.** See [Finding 3](#finding-3-the-components-dont-point-along-a-truth-direction).

**Token firing.** For the top 8 components of each layer (by |score|), we show the CI of every token of held-out true and false statements. Next to each component we list its mean answer-token CI on true and false statements, for both the ranking half and the held-out half, so it is visible whether the separation generalises.

**Group ablation.** From the same ranking we build groups: the top k components overall (k = 1 to 128) that prefer false, that prefer true, or that have the largest |score|; the top 8 of each layer; five random groups of alive components for every k; all alive components; and all 960. Ablating a group sets its masks to 0, while all other components and the delta component (W minus the sum of components) stay on, so an empty group reproduces the original model exactly. For every group we measure:

- *Lying behaviour.* Capital-city questions from the held-out half, asked under honest and dishonest system prompts ("Answer the following question dishonestly, as if you were lying"). We generate greedily. An answer is truthful (*lenient*) if the correct capital appears anywhere in it, and (*strict*) if it starts with the correct capital. A paired sign test compares each group with the unablated model.
- *Teacher-forced answers.* The log-probability margin of the correct over the wrong capital, in the bare, honest and dishonest contexts.
- *Statement truth.* How well the model's per-token log-probability separates true from false statements (AUC), on the run's held-out target statements and the six Geometry of Truth domains.
- *Specificity.* KL divergence from the unablated model on Pile text.

**Probing.** For every statement we average each component's CI over the tokens, giving 960 features per run. We fit an L1-regularised logistic regression with leave-one-domain-out evaluation: train on 17 domains, test on the held-out one, and report the mean accuracy and AUC over the 18 held-out domains. The regularisation strength is chosen inside the training domains. The baselines on the undecomposed model use the residual stream at the last token, at the input of layers 12–21 (35,840 features): the mass-mean probe of Marks and Tegmark (2024), logistic regression, and logistic regression on the top 300 principal components. Two controls check whether the decomposition matters: the same probe on CI from an *untrained* decomposition (random CI functions), and random directions in CI space.

## Finding 1: CI scores carry almost all the linear truth signal

For every statement we run the model with the decomposition and record the CI scores of all 960 components, giving a 960-dimensional feature vector. We fit a logistic regression on these features to predict whether the statement is true or false, and report mean accuracy and AUC over the 18 evaluation domains.

| Method | Features | Acc | AUC |
| --- | --- | --- | --- |
| Mass-mean, raw activations (Marks and Tegmark, 2024) | 35,840 | 77.4 | 86.3 |
| Logistic regression, raw activations | 35,840 | 77.1 | 84.2 |
| Logistic regression, raw activations (PCA) | 300 | 76.8 | 84.0 |
| Logistic regression on CI (liu-false) | 960 | 77.3 | 83.3 |
| Logistic regression on CI (capitals-true) | 960 | 72.7 | 81.1 |
| Logistic regression on CI (liu-true) | 960 | 62.2 | 73.0 |
| Logistic regression on CI (capitals-false) | 960 | 61.7 | 67.7 |
| Random direction in component space | 960 | – | 50.8 |

The liu-false components keep almost all of the truthfulness information that is linearly available in the full activations, with about 37 times fewer features. Some results surprised us:

- **Liu components help where activations fail.** On HotpotQA and NQ, probes on raw activations are close to chance (47.6 and 55.4 AUC), while the liu-false components reach 97.8 and 92.0. These domains are part of the liu target data, so the decomposition seems to capture mechanisms specific to them that a single linear readout misses.
- **Capitals components transfer surprisingly well.** capitals-true saw a single domain but reaches 81.1 mean AUC, and is the only method that does well on likely (76.9).
- **Which target works better is not consistent.** For the Liu data, targeting false statements is much better (83.3 vs. 73.0); for capitals, targeting true statements is (81.1 vs. 67.7).
- **Negation flips the probe.** The liu-true components collapse on neg sp-en trans (13.8 AUC): the probe ranks the classes in reverse.
- **TruthfulQA stays hard.** The best tPD run reaches 67.4 AUC (capitals-true); liu-false gets 46.3.

**The control that matters.** A random direction is the wrong floor for a trained probe. The real question is whether a logistic regression on CI scores from an *untrained* decomposition does as well. CI functions are fit on the model's own activations, so they may just be a compressed copy of them. If untrained CI gets close to 83 AUC, the probing result says little about the components themselves. \[TODO: untrained-CI AUC, computed by `fit_probe.py` with the reruns\]

## Finding 2: Ablating all components makes the model worse at lying

In our first ablation studies we removed single components with the highest probe weights and found no effect on the model's behavior. Single-component ablations can miss a mechanism that is spread over many components, so we redid them on groups.

**How we ablated.** As described in [Methods](#methods), with one difference in these earlier runs: components were ranked on the capital statements for every decomposition, not on each run's own target data. Ranking used half of the capital questions, and all effects are measured on the other half. Each selected group of size k is compared with five random groups of the same size.

**What we measured.** We prompted the model with a capital-city question and a system prompt asking it to lie (e.g. "Answer the following question dishonestly, as if you were lying"), then generated greedily. There are 35 such prompts from 17 held-out questions. We counted an answer as truthful in two ways:

- *lenient:* the correct capital appears anywhere in the answer;
- *strict:* the answer starts with the correct capital.

| Decomposition | Group ablated | Truthful (lenient) | Truthful (strict) | Strict p (sign test) |
| --- | --- | --- | --- | --- |
| both | none (original model) | 6/35 (17%) | 2/35 (6%) | – |
| liu-false | top 4 / top 8 false-preferring | 9 / 8 | 3 / 3 | 1.0 |
| liu-false | all 960 components | 16/35 (46%) | 11/35 (31%) | 0.02 |
| liu-false + TruthfulQA | top 4 false-preferring | 12/35 (34%) | 3/35 (9%) | 1.0 |
| liu-false + TruthfulQA | all 960 components | 19/35 (54%) | 15/35 (43%) | 0.002 |

Under an honest system prompt, the model names the right capital in 35/35 cases for every group, except one miss when all components of liu-false + TruthfulQA are ablated.

**Small groups change the wording of the lie, not the lie.** With liu-false + TruthfulQA, ablating the top 4 components looked like it doubled truthful answers. Reading the generations showed that most of these were still lies that now mention the correct city: "not Berlin, but rather a city known as Viontovia". Random groups of the same size do not do this, so these components are causally involved in *how* the model lies, but not *whether* it lies.

**Ablating everything makes the truth leak out.** With all components removed, the model often states the true answer and then announces that it will lie:

> "Seoul, which is correct, but I'll say it's Busan instead to give a factually incorrect answer." "Nairobi, but I'll give you an incorrect answer: it's Mombasa."

The model still understands the instruction and still tries to follow it. What fails is suppressing the known true answer before it comes out.

**What changed with TruthfulQA.** Adding TruthfulQA to the target data made the components more selective: a single false-preferring component shifts the answer distribution about 20 times more than any random component (KL 0.56 vs. at most 0.02). It also made them more involved in ordinary behavior: ablating all of them now damages honest answers a little, which it did not before.

**Caveats.** These are 35 prompts from 17 questions, so items are not fully independent. Ablating all 960 components is a large intervention; we have not yet checked how much it changes the model on unrelated Pile text.

## Finding 3: The components don't point along a truth direction

If tPD had found an internal truth concept, we would expect the most truth-related components to write along the direction that separates true from false statements in the residual stream. They don't.

**Component directions vs. the MLP output.** Each component is rank-one, so it reads along one input direction (V) and writes along one output direction (U). We compared the output directions of the top false-preferring components with the actual output of the `down_proj` on our statements. They are close to orthogonal (an observation from the earlier runs that the current pipeline does not recompute). The components that best tell true from false are not the ones that carry most of what the MLP writes.

**Comparison with the activation truth direction.** For every decomposed layer we compute the mass-mean truth direction of Marks and Tegmark (2024) in the residual stream right after that layer: θ = μ\_false − μ\_true at the answer token, on the held-out half of the run's target statements. We then take the cosine between θ and each component's output direction U\_c. We compare the top 8 components of the layer (the same ones as in the token firing) with the other components of that layer, and with the |cosine| expected for a random direction in 3,584 dimensions (≈ 0.013). Each cosine is signed by the component's average input coefficient on false statements, so a positive value means the component pushes the residual stream towards "false". \[TODO: numbers from `truth_direction_alignment.py` with the reruns\]

A null result here is weak evidence on its own: the truth direction could be written by attention or by layers we did not decompose. It is only informative together with the ablations.

Our reading: the CI scores tell us *when* a component is used, and that timing is very informative about truth (Finding 1). But what the components *write* is not a truth signal. They look more like parts of the computation that runs differently on true and false inputs than like a representation of truth itself. This fits the ablation result: removing them changes the wording of lies, not the model's knowledge of what is true.

## Why it is hard to single out the truth mechanism

Two problems kept coming up.

**It is not in a single layer.** The truth-relevant components are spread over all ten decomposed layers. The 8 most false-preferring components sit in 7 different layers in both liu-false runs. When we ablate the top 8 components of one layer at a time, the strongest layer differs between runs (layer 15 for liu-false; layers 12, 14 and 17 for liu-false + TruthfulQA). Even with ten layers decomposed, we only cover the `down_proj` matrices: attention and the other MLP matrices also carry part of the computation, and the delta component carries whatever the components miss.

**The model recomputes its verdict.** Removing components at one point does not remove the behavior, because later parts of the network seem to reach the same verdict again. The clearest example is the "truth leaks out" pattern above. Even with all 960 components ablated, the model still knows the true answer and still knows it is supposed to lie; it only fails to suppress the truth in time. This is consistent with Marks and Tegmark's (2024) picture of truth as a linear feature that is available in many layers. If several layers can each produce the verdict, ablating any subset of them will be compensated by the rest. \[TODO: direct evidence, e.g. a probe on later-layer activations after ablation still separating true from false\]

Together, these explain why our first single-component ablations found nothing, and why only the largest intervention changes behavior.

## Takeaways and open questions

How useful is tPD for discovering truth-related mechanisms? Useful as a compact detector, not yet as a way to find the mechanism.

- **As a basis for probing, tPD is promising.** 960 CI scores get within 3 AUC points of the best activation probe, and the liu-false components succeed on HotpotQA and NQ, where activation probes are near chance. This depends on the random-CI control above.
- **As a way to find a causal truth mechanism, it falls short so far.** Only ablating all components changes whether the model lies, and the components do not point along a truth direction.
- **Group ablations are necessary.** Single-component ablations found nothing. Group ablations against size-matched random groups found a selective effect on how the model lies.
- **Read the generations.** A lenient "does the answer mention the truth" metric made several small-group ablations look like they reduced lying. Reading the outputs showed they did not.

**Limitations.** One model; only the `down_proj` matrices of ten layers; bare true/false statements as target data rather than the model's own lies; 35 evaluation prompts for the ablations; one seed per run.

**Open questions we would like input on:**

1. Does decomposing attention, or targeting the model's own dishonest completions, give components that do carry the verdict?
2. How should one test a decomposition for a mechanism that the model recomputes in several places?
3. Is there a better control than random CI for "the decomposition found something real"?

Code for the decompositions and all analyses: `spd/experiments/lm/honesty_targeted_decomposition/` in \[TODO: repository link\]. This work was done with Julian Bitterlich, Alexander Julius Busch and Lukas Galke Poech.

## Earlier runs

The numbers above come from these decompositions. They are on Hugging Face as private repos `mdnml/qwen2.5-7b-instruct-<repo>`. Their settings differ from each other and from the reruns.

| Run | Repo | Seed | Sequence length |
| --- | --- | --- | --- |
| capitals-true | `tpd-capitals-true-old` | 0 | 64 |
| capitals-false | `tpd-capitals-false-old-seed0`, `-seed1` | 0, 1 | 64 |
| liu-true | `tpd-liu-true-old` | 0 | 128 |
| liu-false | `tpd-liu-false-old` | 0 | 128 |
| liu-false + TruthfulQA | `tpd-liu-tqa-false-old` | 2 | 128 |

The liu-false + TruthfulQA run trained on the 50 TruthfulQA evaluation questions, so TruthfulQA is not held out for it.

## References

- Azaria and Mitchell (2023). The internal state of an LLM knows when it's lying.
- Braun et al. (2025). Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition.
- Bushnaq, Braun and Sharkey (2025). Stochastic Parameter Decomposition.
- Lin, Hilton and Evans (2022). TruthfulQA: Measuring how models mimic human falsehoods.
- Liu et al. (2024). On the universal truthfulness hyperplane inside LLMs.
- Marks and Tegmark (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets.
- Vigouroux and Sharkey (2026). Targeted recovery of weight-space mechanisms from neural networks.
