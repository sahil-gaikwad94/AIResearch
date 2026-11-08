# Experiment 3 — Is the Router the Ceiling?

**Why MoE sparsity stopped at ~50×, and whether better routers move it**

> **Thesis.** At fixed active compute and fixed expert width, adding experts stops helping because a single linear router cannot reliably discriminate between hundreds of near-tied candidates. The saturation point is set by router capacity, not by memory or data — so upgrading the router should move the sparsity ceiling by 2–4×.

| | |
|---|---|
| **Question** | Does loss keep improving as the expert count grows at fixed active compute, or does it saturate — and why? |
| **Intervention** | Sweep E from 32 to 2048 at fixed expert width, G, and active FLOPs; then swap router architectures at the best E |
| **Primary metric** | Validation loss per active FLOP, plus the router's margin distribution |
| **Decisive statistic** | Location of E* (where marginal loss improvement → 0) and its shift under a higher-capacity router |
| **Compute** | ~6,200 GPU-hours (~$12k) |
| **Duration** | 8–10 weeks |
| **Kill condition** | Loss improves monotonically to E = 2048 (§9) |

---

## 1. The observation that needs explaining

Every open frontier MoE converged on almost the same sparsity:

| Model | Experts | Active | Expert-level sparsity | Year |
|---|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 4× | 2023 |
| DeepSeek-V3 | 256 | 8 | 32× | 2024 |
| Kimi K2 | 384 | 8 | 48× | 2025 |
| Flint | 256 | 6 | 43× | 2026 |
| Qwen3-Next | 512 | 10 | 51× | 2025 |

Four teams, three years, a narrow band. That is either a deep optimum or a shared blind spot.

The published scaling laws explain *part* of it, and it is worth being precise about what they do and do not cover:

- **Krajewski et al. (2024)** introduced "granularity" (expert size relative to the FFN) and showed that making experts the same size as a dense FFN is "not optimal at almost any computational budget" — smaller experts win. This explains why 14,336-wide experts disappeared.
- **Ludziejewski et al. (ICML 2025)**, joint MoE scaling laws: at a fixed compute budget, **more experts always improves** performance, provided model size and tokens are adjusted. This predicts no ceiling at all.
- **The comprehensive MoE scaling law** (2025, ~450 controlled experiments over five factors: data `D`, total size `N`, active size `N_a`, active experts `G`, shared-expert ratio `S`) finds an **optimal `G ≈ 7`**, recommends 1–2 shared experts, and finds that the optimal `N_a/N` ratio becomes sparser as `N` grows.
- **"Towards Greater Leverage"** (ICLR 2026, 300+ models up to 28B) finds MoE efficiency is driven by activation ratio and compute, with granularity acting as "a non-linear modulator with a clear optimal range".

`G ≈ 7` explains why every model activates 6–10 experts. "Sparser with scale" explains the 4× → 50× drift. **Nothing explains why the expert *count* stops at a few hundred** — the joint law says it should keep helping.

That is the gap. And it has a clean economic consequence: if the joint law extrapolates, the field is leaving performance on the table because of *serving* costs (all-to-all traffic, memory), not because of learning. If it does not extrapolate, the router is the binding constraint and is the thing to fix.

## 2. The mechanism

A token-choice router is one linear map:

```
s = W_r x           W_r ∈ R^{E × d}
top-k select, renormalise
```

Two facts make this fragile at large `E`:

**(a) Rank.** `W_r` has rank ≤ min(E, d). At `E = 2048` and `d = 2048` that is fine in principle, but measured routers are strongly low-rank: most of the score variance lies in a handful of directions. With `r` effective dimensions, `E` experts are being placed into an `r`-dimensional score space, so their pairwise angular separation shrinks like `1/√r` regardless of how many experts you add.

**(b) Order statistics.** The router selects the top `k` of `E` scores. The *margin* between the `k`-th and `(k+1)`-th score is what decides whether a token's assignment is stable. Under a rank-limited (hence correlated) score distribution, that margin shrinks as `E` grows: many experts become near-ties, and the selection becomes sensitive to noise in `x`.

When the margin falls below the token-to-token variation in `x` within a semantic class, two things follow:

1. **Inconsistent assignment.** Two near-identical tokens get routed to different experts, so neither expert sees enough of that pattern to specialise.
2. **Gradient dilution.** Each expert's gradient is a noisy subsample of the tokens it should own, which raises the variance of its update and slows learning.

Both predict the same observable: **specialisation saturates at the same `E` where loss saturates**, and **the margin distribution collapses first**.

That gives the experiment its mechanistic intermediate — the thing that must move for the explanation to be right.

## 3. Predictions

| # | Prediction | Quantitative form | If it fails |
|---|---|---|---|
| **P1** | Loss saturates in `E` | Marginal Δloss per doubling falls below 0.1% by `E* ≈ 256–512` at `d = 512` | Memory is the only constraint; the field is under-exploring |
| **P2** | The margin collapses first | Mean margin between k-th and (k+1)-th score falls as `E^{−α}`, α ∈ [0.3, 0.7] | The router is not the mechanism |
| **P3** | Specialisation saturates at the same point | Pairwise expert JS divergence plateaus within one octave of `E*` | Loss saturation is caused by something else (data, capacity) |
| **P4** | A better router moves the ceiling | Bilinear or 2-layer router shifts `E*` right by ≥ 2× at equal active FLOPs | Router capacity is not the limit |
| **P5** | The shared expert absorbs the head of the frequency distribution | Shared-expert output correlates with token frequency; a second shared expert extends `E*` | The shared expert is doing something else |

P1 and P2 together are the contribution. P4 is the intervention that would make it useful.

## 4. Design

### 4.1 Main sweep: `E` at fixed everything else

The critical design choice is what is held fixed. Most published sweeps vary `E` while adjusting size and tokens (confounding `E` with `N`) or vary expert width (confounding `E` with granularity). This sweep fixes:

- **expert width** at 256 (granularity constant)
- **active experts** `G = 4`
- **active FLOPs per token** (attention + 4 experts + shared) constant
- **training tokens** constant (20B)
- **model width** `d = 512`, 6 layers

and varies only `E ∈ {32, 64, 128, 256, 512, 1024, 2048}`. Total parameters grow from ~0.15B to ~3B; active stays ~50M. That is exactly the axis the joint scaling law extrapolates along, so the experiment is a direct test of its extrapolation.

**Arms:** 7 values × 3 seeds = 21 runs at ~4 GPU-hours each = 84 GPU-hours. Cheap enough to add a second width (`d = 1024`, 12 layers, `E ≤ 512`) as a scaling check.

### 4.2 Router variants at the best `E`

Held at `E = max(E*)` found above, same active compute:

1. **Linear (baseline).**
2. **Bilinear / low-rank**: `s = xᵀ A_e x` approximated as `(x U) ⊙ (x V)` per expert, or a small learned projection before the linear router (d → 4d → E). Tests rank directly.
3. **Two-layer MLP router** with a nonlinearity. Tests whether rank or linearity is the limit.
4. **Margin-regularised linear**: add a term that maximises the top-k margin on training tokens. Tests whether the *training signal* rather than capacity is the issue.

4 variants × 3 seeds = 12 runs.

### 4.3 Measurements

- validation loss (held-out, per active FLOP)
- **margin distribution**: `s_(k) − s_(k+1)` per token, by layer and by token frequency bucket
- **effective rank** of `W_r` (participation ratio of singular values)
- **expert utilisation**: normalised entropy of assignment counts
- **specialisation**: pairwise Jensen–Shannon divergence between experts' token distributions, and CKA between expert weight matrices (are experts converging to copies?)
- **per-expert gradient norm** variance
- **shared-expert traffic**: correlation between shared-expert activation magnitude and token unigram frequency
- downstream: a small suite (a few reasoning/code tasks) at the best and worst arms, to confirm loss gains are not purely distributional

## 5. Budget

| Item | GPU-hours |
|---|---|
| Main sweep: 21 runs (300M active, 20B tokens) | 3,948 |
| Router variants: 12 runs | 2,256 |
| Width-scaling check: 6 runs at d = 1024 | ~1,100 |
| **Total** | **~7,300** (≈ $15k) |

(Runs are deliberately small. The risk is discussed in §8.)

## 6. Analysis plan

1. **The main curve**: loss versus `log₂ E`, one line per seed, with the dense-equivalent baseline as a horizontal reference. P1 predicts a knee, not a plateau from the start.
2. **The mechanism plot**: mean margin versus `E` on log-log axes, with the fitted exponent. Then overlay the loss curve — if the knee in loss coincides with the margin crossing the token-noise level, the mechanism is supported.
3. **The intervention**: the same loss-versus-`E` curve for all four routers, with `E*` estimated by segmented regression. P4 predicts a rightward shift of at least one octave.
4. **Decision rules, pre-registered**:
   - P1 confirmed if the segmented-regression breakpoint has a CI excluding both 128 and 1024.
   - P2 confirmed if the fitted α ∈ [0.3, 0.7] with R² > 0.9 and the margin knee is within ±1 octave of `E*`.
   - P3 confirmed if specialisation's plateau begins within one octave of `E*`.
   - P4 confirmed if the best router's `E*` is ≥ 2× the baseline's with non-overlapping CIs.
5. **Report effect sizes in loss, not just significance.** A breakpoint that moves but a loss that does not is not a result.

## 7. What would kill it

**If loss improves monotonically to `E = 2048`** — the joint scaling law extrapolates further than anyone has tested, and the observed 32–51× band is an economic equilibrium (memory + all-to-all + kernel support), not a learning limit. That is arguably the *more* interesting result: it says the frontier of MoE research is systems engineering, and anyone with cheap interconnect should be running 2000-expert models.

**If the margin collapses but loss does not saturate**, the router is a red herring: assignment noise is being absorbed by the shared expert and by depth.

**If a better router helps at every `E`** (not just at large `E`), the linear router was simply a bad default all along, and the finding is about router design rather than about a ceiling.

## 8. Threats to validity

The biggest risk is scale. A 50M-active model may saturate for reasons that do not bind at 30B active: at small scale, each expert sees few tokens, so *data per expert* may be the real constraint rather than router discrimination. That alternative predicts saturation at a fixed **tokens-per-expert** ratio, not at a fixed `E`.

This is testable inside the same experiment: hold `E` fixed and vary tokens (20B vs 100B) at one or two values of `E`. If `E*` moves with tokens, the constraint is data per expert; if it does not, it is the router. **Include this arm** — it is the single most important control, and its absence is why most MoE ablations are uninterpretable.

Second risk: expert parallelism is absent at these sizes, so the all-to-all cost that dominates production is invisible. That is fine for a learning question, but it means the paper cannot speak to the *practical* optimum — only to whether a learning ceiling exists.

## 9. If it works, what changes

- **A design rule**: choose `E` from router capacity (effective rank × margin), not from memory budget.
- **A cheap win if P4 holds**: router upgrades are a few million parameters on a model with hundreds of billions. If a 2-layer router buys two octaves of sparsity, that is the best parameter-for-parameter trade in the architecture.
- **A redirect if P1 fails**: effort moves from routing research to interconnect and memory, and the 32–51× band is explained as an economic equilibrium that better hardware will dissolve.

## 10. The one-week pilot

Train 3 seeds × `E ∈ {64, 256, 1024}` at 100M active / 5B tokens (~2 GPU-hours each), and plot margin versus `E`. If the margin does not fall measurably across that range, the mechanism is wrong and the full sweep should not be run. Cost: ~20 GPU-hours.
