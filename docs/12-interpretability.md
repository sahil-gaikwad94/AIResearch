# 12 — Mechanistic Interpretability: Opening the Black Box

*Read time: 55 minutes. Prerequisites: `01` §1.4–1.6, `02`.*

---

## 12.1 What "interpretability" means here

Two different projects get called interpretability:

- **Explainability (post-hoc)**: given an input, say why the model produced an output (saliency maps, attention visualisation, SHAP, natural-language rationales). Useful for users and auditors; often unfaithful.
- **Mechanistic interpretability**: reverse-engineer the *internal computation* — find the circuits, features, and algorithms the network actually implements. This chapter is about the second one.

The aspiration is worth stating plainly: if we can read the computation, we can **audit** models for deception, **debug** failures, **steer** behaviour, and eventually **verify** safety properties. If we can't, we're shipping systems whose behaviour we can only sample.

The honest status in 2026: enormous genuine progress on small/medium models and narrow behaviours; no demonstrated ability to fully explain a frontier model. The gap between the ambition and the achievement is the interesting part.

---

## 12.2 The two founding insights

**(1) The residual stream as a communication bus.** Every layer reads from and writes to the same vector. Attention heads and MLPs are *independent readers/writers* on that bus. This means the network has a shared workspace with a privileged basis, and computations can be decomposed as "which component wrote what".

**(2) Features, not neurons.** Individual neurons are **polysemantic** — one neuron fires for "dog", "car", and "sentences ending in a period". So you cannot explain a model neuron-by-neuron. The hypothesis is that the model represents concepts as **directions in activation space** (features), and neurons are arbitrary linear combinations of many features.

---

## 12.3 Superposition: why the features are hidden

**Superposition hypothesis** (Anthropic, 2022, "Toy Models of Superposition"): a model has `d` dimensions but wants to represent `n ≫ d` features. It can do so approximately if the features are **sparse** (each rarely active): pack many nearly-orthogonal directions into a low-dimensional space, tolerating some interference.

They demonstrated this with a toy model: compress a sparse high-dimensional input into a low-dimensional bottleneck, and the model spontaneously learns to represent more features than dimensions, arranged in geometric patterns (digons, triangles, pentagons) — a phase diagram with a clear transition from "no superposition" to "full superposition" as sparsity increases.

**Why this matters:** it explains *why* neurons are polysemantic, and it says the right unit of analysis is a **sparse dictionary of directions**, not neurons. That single conclusion launched the SAE programme.

---

## 12.4 Sparse autoencoders (SAEs)

**Idea:** learn an overcomplete, sparse decomposition of activations.

```
f = ReLU(W_enc · x + b_enc)        # sparse feature activations (dim ≫ d_model)
x̂ = W_dec · f + b_dec              # reconstruction
L = ‖x − x̂‖² + λ‖f‖₁              # reconstruction + sparsity
```

Train this on millions of activations from one layer. Each row of `W_dec` is a **feature direction**; the corresponding `f_i` is "how much feature i is present".

**Why it's exciting:** individual features turn out to be strikingly interpretable — "the Golden Gate Bridge", "code with SQL injection", "sycophantic praise", "the concept of a deferred answer", "base64 strings". Anthropic's 2024 scaling work trained SAEs on Claude 3 Sonnet, finding millions of features including safety-relevant ones (deception, bias, dangerous content) that could be **clamped on/off** to change behaviour.

**Variants:**
- **TopK SAEs**: enforce exactly k active features (no L1 tuning).
- **JumpReLU SAEs**: learned thresholds, better reconstruction-sparsity frontier.
- **Transcoders**: instead of reconstructing an activation, learn the *input→output function* of an MLP sublayer as a sparse map. This gives you features that are causally connected across layers, which SAEs don't (`Transcoders Find Interpretable LLM Feature Circuits`, NeurIPS 2024).
- **Crosscoders**: features shared between two models (e.g. base vs. chat), for comparing what fine-tuning changed.
- **Cross-layer transcoders (CLTs)**: share features across layers with layer-specific decoders → compact attribution graphs (CLT-Forge, 2026).
- **Temporal SAEs** (2025–26): add a contrastive loss so features stay consistent across adjacent tokens, fixing the "features are token-local and noisy" problem.

---

## 12.5 Circuits and attribution graphs

If features are the nouns, **circuits** are the sentences.

**The classic result: induction heads.** In two-layer attention-only models, a specific circuit implements in-context pattern completion: head 1 ("previous token head") copies the previous token into the current position; head 2 ("induction head") attends to whatever followed that token earlier in the context. This is the mechanism behind in-context learning of repeated patterns — found by Olsson et al. (2022) and confirmed to appear across model sizes.

**Methodology for finding circuits:**
- **Activation patching / causal tracing**: run the model on a clean input and a corrupted input; patch an activation from clean into corrupted and see if the output flips. This localises causal responsibility. **Attribution patching** is a linear approximation that makes it cheap.
- **Path patching**: restrict patching to specific edges to test a hypothesised path.
- **Ablation**: zero out a head/feature and measure the effect.
- **Edge pruning**: faithfully extract a minimal subgraph (HAP, 2025: hybrid attribution then pruning, 46% faster than baseline with faithful circuits).

**Attribution graphs / circuit tracing** (Anthropic, 2025) is the flagship application: use **cross-layer transcoders** to replace MLPs, then compute a graph of feature→feature→feature contributions for a specific prompt. The published examples are remarkable — e.g. a model writing "Dallas" when asked about a city, with a graph showing the intermediate "Texas" feature and a "say a fact about the capital of Texas" pathway. The June 2026 update describes **turn-averaged SAEs** (averaging the residual stream over a whole assistant turn) to cut the number of features to interpret from millions to thousands.

**Automation is now the bottleneck-solver:** LLMs annotating attribution graphs (2026) group features into "supernodes" as well as human annotators, and recover the intermediate hop in 97/100 two-hop prompts. Circuit analysis is becoming a pipeline rather than an art.

---

## 12.6 Steering, probing, and representation engineering

Three adjacent techniques you'll see everywhere:

**Probing.** Train a linear classifier on activations to predict a property (sentiment, truthfulness, part of speech, whether the model is being tested). A high probe accuracy means the information is *linearly present*; it does not mean the model *uses* it. **Probing finds correlates, not causes.** The rigorous version is **causal probing**: intervene on the direction and see if behaviour changes.

**Steering vectors / activation addition.** Find a direction `v` (from a probe, a contrastive prompt pair, or an SAE feature), and add `α·v` to the residual stream during generation. This reliably changes behaviour: sentiment, refusal, honesty, tone, "concise vs verbose". **Representation engineering** (Zou et al. 2023) systematised this.

Practical caveats: steering often has side effects (steering "honesty" can reduce capability), and the effect is dose-dependent and unstable across prompts. It's a diagnostic tool more than a control mechanism.

**Logit lens / tuned lens.** Project intermediate residual streams through the unembedding to see what the model "would say" at each layer. Crude but informative: predictions often converge in the last third of the network, and fact recall often localises in specific mid-layers.

---

## 12.7 The critiques (this is the important section)

Mechanistic interpretability is a field with a real, healthy internal argument. The strongest critiques:

**(1) SAE features may not be the model's features.** SAEs find *a* sparse decomposition, not *the* one. Different seeds/architectures yield different feature sets with similar reconstruction. If the decomposition is under-determined, "the feature for X" is a fact about your SAE, not the model.

**(2) Aggregate metrics don't measure interpretability.** ICLR 2026 work ("Automated Interpretability Metrics Do Not…") shows standard SAE quality metrics (reconstruction loss, L0, auto-interpretability scores) can be high while features fail to correspond to computation, and argues for **randomized baselines** and measures of **feature abstractness**. Without baselines, you can't attribute anything to learning.

**(3) Features are often trivial.** Many SAE features fire on syntax, specific tokens, or the surface form of a decision. Anthropic's own June 2026 update acknowledges many features are "boring" and safety-irrelevant, motivating turn-averaged SAEs.

**(4) Scaling is unproven.** A circuit for "Dallas" in a small model does not generalise to "explain the model's foreign policy reasoning". Human brains are also mechanistically interpretable at the neuron level in principle, and we still can't explain them.

**(5) The faithfulness problem.** Any explanation is a *model of the model*. There is no ground truth, and no accepted test that an explanation is complete rather than merely plausible. (This is the "interpretability illusions" critique.)

**The defensible 2026 position:** interpretability has produced real, causal, replicable findings (induction heads, superposition, feature steering) and is genuinely useful for safety auditing of *specific behaviours* (e.g. finding whether a model represents "I am being evaluated"). It is not close to a general account of frontier model computation.

---

## 12.8 Why safety teams care anyway

Even partial interpretability changes the safety toolkit:

- **Finding evaluation-awareness**: SAEs and probes can detect whether a model represents "this is a test" (`13` §13.5).
- **Detecting deception/deceptive alignment**: features for "intentionally misleading" can be monitored.
- **Pre-deployment auditing**: "does this model represent a plan to do X?" is a question you can partly answer internally, not just behaviourally.
- **Debugging**: knowing which feature drives a failure tells you what data to fix.
- **Monitoring**: run a feature monitor on production traffic (cheaper than an LLM judge).

The uncomfortable dependency: all of this assumes the interpretability tools are faithful, which §12.7 questions. That circularity is unresolved.

---

## 12.9 Common misconceptions

- **"Attention weights explain the model."** Attention shows where information *can* flow, not what is computed. Many heads are redundant; the MLPs do a lot of the work.
- **"Neurons are the units of meaning."** They're polysemantic. Directions are the better unit.
- **"SAEs solved superposition."** They're a promising dictionary-learning approach with serious measurement problems.
- **"A probe that predicts X proves the model represents X."** Correlation in activation space is not use in computation.
- **"Interpretability is just for safety people."** It's also the best debugging tool for data and architecture problems.

---

## 12.10 If you remember five things

1. Superposition explains polysemanticity: models pack sparse features into fewer dimensions than features.
2. SAEs learn sparse overcomplete dictionaries → interpretable feature directions; transcoders add causal cross-layer structure.
3. Induction heads are the canonical circuit and the mechanistic basis of in-context learning.
4. Steering vectors work but are dose-dependent and diagnostic rather than surgical.
5. The field's central unresolved problem is *faithfulness measurement*: no accepted test that an explanation is the real computation.

---

## 12.11 Self-check

1. Explain superposition with the toy-model setup and why sparsity is required.
2. Write the SAE loss and say what each term does.
3. Why does an SAE's feature set depend on its seed, and why is that a problem?
4. What's the difference between a probe and a causal intervention?
5. Give a safety question that interpretability could answer and behaviour alone could not.

**Next:** [`13-safety-and-alignment.md`](13-safety-and-alignment.md)
