# Experiment 2 — Where Should Position Live?

**A fractional NoPE ablation at fixed hybrid ratio**

> **Thesis.** Positional encoding is a per-layer resource, not a global property of the model. There is an interior optimum at roughly 25–40% of layers carrying no position at all — and most of the benefit of "no position" can be recovered by "very slow position", which means Flint's no-RoPE global layers and Qwen3-Next's RoPE-only-on-globals are two points on one curve rather than two philosophies.

| | |
|---|---|
| **Question** | Should layers that see the whole context carry positional encoding? |
| **Intervention** | NoPE vs RoPE vs slow-RoPE, applied per layer, at fixed 3:1 hybrid ratio |
| **Primary metric** | Content retrieval (NIAH) and order-sensitive accuracy, measured separately |
| **Decisive statistic** | Interior optimum in the NoPE fraction for retrieval, with monotone loss on ordering |
| **Compute** | ~4,300 GPU-hours (~$9k) |
| **Duration** | 8–10 weeks |
| **Kill condition** | Retrieval is monotone in NoPE fraction (optimum at 0% or 100%) |

---

## 1. Where the question comes from

Two 2025–26 frontier open models made **opposite** choices about the same layers:

| | Qwen3-Next (Sep 2025) | Flint (Oct 2026) |
|---|---|---|
| Global-attention layers | 12 of 48 | 14 of 49 |
| Share | 25% | 29% |
| Position on those layers | **RoPE** | **none** |
| Position on the other layers | none (recurrence) | RoPE (sliding window) |

Both work well enough to ship. The ratio is nearly identical. The positional choice is inverted, and neither model card argues for it.

There is a large literature on positional encoding, but it does not answer this question:

- **Kazemnejad et al. (NeurIPS 2023)** showed that a decoder-only transformer with **no** positional encoding (NoPE) *outperforms* APE, ALiBi and RoPE on length generalization across ten reasoning tasks. Theoretically NoPE can represent both absolute and relative position; trained with SGD it behaves like T5-style relative bias. That is evidence for removing position everywhere — not for removing it from *specific* layers.
- **A follow-up** (ACL 2024 Findings) found NoPE still has a limited context and traced the failure to "distraction" of the attention distribution — a hint that position-free attention degrades as the number of candidates grows.
- **"A Systematic Analysis of Hybrid Linear Attention"** (2025) swept the linear:full ratio (24:1, 12:1, 6:1, 3:1, pure) on RULER and found long-context quality "rises steeply once a few full-attention blocks are present, then plateaus", with DeltaNet/Gated-DeltaNet peaking at 3:1.

So the **ratio** is established. The **positional scheme on those layers** is not, and the two production designs disagree.

## 2. The mechanism: why removing position from global layers could help, and could hurt

**Why it could help.** A global layer doing retrieval wants content matching: "find the definition of `parse_config`". Applying RoPE at 200k tokens forces the model to represent enormous relative offsets it rarely saw in training, and spreads attention mass thinly over distance (the "distraction" effect). Remove position and the layer sees a bag of tokens; retrieval becomes a pure key match. This is exactly Flint's stated design: local layers handle order, global layers handle identity.

**Why it could hurt.** A position-free layer cannot distinguish "the first error" from "the last error", cannot implement "the token before the colon", and cannot copy a span in order. Causal masking gives it *some* implicit order information (a token can only attend leftward), which is why NoPE is not catastrophically bad — but implicit order is coarse and degrades with length.

**The unifying hypothesis.** Position is not binary. With RoPE base `θ`, the slowest-rotating channel has wavelength `≈ 2π·θ^{1−2/d}`. At `θ = 10,000` that is ~54k tokens; at `θ = 500,000` (Llama 3) it is ~2.5M; at `θ = 10⁷` it exceeds any context you will ever train on. **A model with a large enough base cannot distinguish positions within its window** — which is functionally NoPE for content matching, while retaining a small position signal for ordering. If that is right, the "right" answer is not a discrete choice but a frequency, and both frontier designs are approximations of the same optimum.

## 3. Predictions

| # | Prediction | Quantitative form | If it fails |
|---|---|---|---|
| **P1** | Interior optimum for retrieval | NIAH accuracy peaks at NoPE fraction 0.25–0.50, ≥ 8 points above both 0% and 100% | Removal is monotone; the design question is trivial |
| **P2** | Ordering degrades monotonically | Order-sensitive score falls ≈ linearly, −0.4 to −0.6 relative per 25% NoPE | Order is robust; position matters less than assumed |
| **P3** | Slow position ≈ no position | `θ = 10⁷` recovers ≥ 50% of NoPE's retrieval gain and < 20% of its ordering loss | Position presence and position frequency are separable — a stronger, more interesting result |
| **P4** | Mechanism is visible | NoPE layers show a flat attention-mass-versus-distance profile; RoPE layers show a distance-decaying profile | The effect is not attention-shaped; look elsewhere |
| **P5** | Length generalization peaks in the middle | Train at 2k, test at 8k: best extrapolation at intermediate NoPE fraction | Length generalization is monotone in NoPE (as the 2023 paper implies) |

P1 + P3 together are the contribution: they turn a binary architectural argument into a tunable parameter with a recommended setting.

## 4. Design

### 4.1 Base model

A 1B-parameter dense decoder in the Llama style (RMSNorm, SwiGLU, GQA 4:1), trained **from scratch** for 20B tokens per arm. From scratch, not continued pretraining: position scheme is learned, and fine-tuning an existing RoPE model would confound "the model never learned to use position" with "the model unlearned it".

One hybrid arm is included so the result transfers: same 1B model with 3:1 Gated-DeltaNet:attention layers (which is where the real design question lives).

### 4.2 Arms

**Fraction sweep (dense), NoPE applied to every k-th layer:**

| Arm | NoPE fraction | Layers (of 24) |
|---|---|---|
| A0 | 0% | 0 |
| A1 | 15% | 4 |
| A2 | 25% | 6 |
| A3 | 40% | 10 |
| A4 | 60% | 14 |
| A5 | 100% | 24 |

**Frequency sweep (the P3 test), all layers with RoPE:** `θ ∈ {10k, 500k, 1e6, 1e7}`.

**Hybrid arm:** 3:1 GDN:attention, with NoPE on the attention layers only (the Flint configuration) versus RoPE on them (the Qwen configuration). Two arms, this is the direct replication of the disagreement.

Total: 6 + 4 + 2 = 12 arms × 2 seeds = **24 runs** at 83 GPU-hours = 2,000 GPU-hours.

### 4.3 Metrics, deliberately separated

The whole point is that "long context" hides two different abilities. Measure them apart:

**Content retrieval** (position-free-able):
- single-needle NIAH at 1/4/16/64k
- multi-key NIAH (m = 8)
- "which document mentions X" with 20 distractors

**Order** (position-requiring):
- first/last of k occurrences ("which error appeared first")
- sort 10 items by the order they appear
- copy a span preserving order
- multi-hop chain where the order of steps matters
- code diff: "which function was modified after the import change"

**General**:
- perplexity on held-out text, bucketed by token offset (does NoPE flatten the offset curve?)
- a standard reasoning suite, to check nothing else broke
- length generalization: train 2k, evaluate 8k

### 4.4 Confirmation arm at scale

Two arms at 8B / 50B tokens (0% and 40% NoPE) to check the optimum is not a small-model artefact. 1,667 GPU-hours each.

## 5. Budget

| Item | GPU-hours |
|---|---|
| 24 × 1B runs (20B tokens) | 2,000 |
| 2 × 8B confirmation runs (50B tokens) | 3,334 |
| **Total** | **~5,334** (≈ $11k) |

## 6. Analysis plan

1. **The main plot**: retrieval score and ordering score on the same axes against NoPE fraction. P1 and P2 predict an X-shape: one curve up then flat, one curve down. The intersection is the recommended setting.
2. **The P3 plot**: retrieval and ordering against `log θ`, with the NoPE point plotted at `θ = ∞`. If P3 holds, the four points fall on a smooth curve ending at the NoPE value.
3. **Mechanism**: attention mass versus distance, per layer type. A NoPE layer's profile should be roughly flat; a RoPE layer's should decay. Report the slope, not an impression.
4. **Decision rules, pre-registered**:
   - P1 confirmed if the peak is ≥ 0.08 absolute above both endpoints with non-overlapping bootstrap CIs.
   - P2 confirmed if a linear fit of ordering versus fraction has slope < 0 and R² > 0.8.
   - P3 confirmed if the `θ = 1e7` arm recovers ≥ 50% of the retrieval gap and retains ≥ 80% of ordering.
5. **Report the whole curve**, including the 8B confirmation, whether or not it matches. If the optimum moves with scale, that is the finding.

## 7. What would kill it

- **Monotone retrieval in NoPE fraction** (optimum at an endpoint): the design question has a trivial answer, and the two frontier models differ for reasons unrelated to modelling (kernel support, parallelism, training stability).
- **Ordering is unaffected by NoPE**: then position is doing much less than believed, and the 2023 NoPE result generalises further than expected.
- **The optimum does not replicate at 8B**: the effect is a small-model artefact, and the practical recommendation should be withdrawn.

## 8. If it works, what changes

- **A design rule** replaces a binary argument: put position where order is computed, not where content is matched, and tune the fraction.
- **A cheaper alternative to architecture changes**: if `θ` tuning recovers most of the benefit, models can get the gains with a config change instead of a new layer type — which matters enormously for anyone without a kernel team.
- **A better long-context eval**: separating retrieval from ordering would immediately explain why models that ace NIAH still fail at "which came first".

## 9. Threats to validity

- **Position scheme interacts with training length.** A 20B-token run may be too short for NoPE layers to learn implicit position from the causal mask. Mitigation: one 60B-token arm at the best fraction.
- **The 3:1 hybrid's global layers have a different job** than a dense model's layers, so the dense sweep may not transfer. That is why the hybrid arm exists and is reported separately.
- **Kernel differences.** NoPE is cheaper to compute (no rotation), so arms are not perfectly FLOP-matched. Report wall-clock and FLOPs; the difference is < 1%.

## 10. The one-week pilot

Train 4 × 100M models for 2B tokens: 0%, 25%, 50%, 100% NoPE, at 2k context, and evaluate NIAH at 2k and 8k. Cost ~30 GPU-hours. If retrieval is monotone in the fraction at 100M, the interior-optimum hypothesis is probably wrong and the full study should be redesigned rather than run.
