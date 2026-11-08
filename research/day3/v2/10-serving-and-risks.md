# 10 — Serving, cost, and what would make me abandon this

## What runs where

| | |
|---|---|
| Weights, 4-bit | **11.5 GB** |
| 24 GB card | fits, **12.5 GB** spare for KV, activations, and the runtime |
| Always-on weights per step | 0.551 GB (everything that is not a routed expert) |
| One expert | 2.60 MB at 4-bit |

The 24 GB story is not a stretch: 11.5 GB of weights leaves half the card. The constraint is bandwidth, not capacity, which is the MoE serving tax in its usual form.

## Latency at batch 1

| | |
|---|---|
| Step time, r=1 (0.57 GB at 3.35 TB/s) | **170 µs** |
| One additional workspace pass (65 MB) | **+19 µs** |
| r=2 | 189 µs |
| r=8 | 303 µs |
| r=1..8 gap | **1.6×** |

On a part where the 65 MB workspace stays in L2 (126 MB on B300-class), the extra passes cost arithmetic rather than bytes: about **2.4 µs** each, so r=8 is 187 µs against r=1's 170 µs — **a 10 % penalty for 4× the depth.** That is the difference the narrow workspace was designed to make.

Stated plainly: **on an H100, depth costs 1.6× latency at batch 1. On a B300, it costs 1.1×.** Anyone deploying this on older hardware should expect the ladder's r=8 rung to be expensive and should set the default to r=2.

The blind spot: at batch ≥ 16 the step becomes FLOP-bound, and the bandwidth-ceiling numbers in `tools/count_params_v31.py` (70k, 200k tok/s) stop being predictions. Those rows are labelled as ceilings rather than included in any claim.

## Cost per task

An agent task: ~40 decode steps, ~300 tokens each, r≈2.

| | |
|---|---|
| GPU time | 2.4 s |
| At $2/GPU-h | $0.0013 |
| Environment (90 s CPU core) | $0.001 |
| **Per task** | **~$0.002** |

Ten thousand training attempts a day costs about $20 in inference and $24 in environments. That is the loop being affordable, and it is why the plan spends 12,000 CPU core-hours without flinching.

## The expert hot set

At batch 16, **80.2 of 256 experts** are touched (computed as `E·(1−(1−1/E)^(b·k))`). At batch 64, 199.

Consequence for deployment: a serving system needs ~31 % of expert weights resident at batch 16, and the remaining 69 % can live in host memory or on disk with prefetch. This is where Mira's predictive prefetch (5.71× on constrained GPUs) applies directly — the router's logits are available one step early, so the next step's experts can be requested during the current one.

We do not implement prefetch in the reference model. It is a serving-layer optimisation that the architecture makes possible, not a property of the model.

## Risks, with probabilities

These are my estimates, stated so they can be marked wrong.

| # | Risk | P | Consequence if true | Detect by |
|---|---|---|---|---|
| R1 | Latent recurrence does not transfer from math to code | **40 %** | workspace is 118 M wasted parameters; revert to v1 with latent MoE | B2, 1,000 GPU-h |
| R2 | Verifier independence unachievable on our tasks | **25 %** | self-improvement loop cannot run; v2 = v1 + smaller experts | B5, one week, no GPU |
| R3 | Sparsity 42.7 does not beat 32 on code | **30 %** | latent width is a serving optimisation, not a capability one | B1, 1,200 GPU-h |
| R4 | Collapse in rounds 2–3 despite gates | **50 %** | two productive rounds, then stop | per-round held-out eval |
| R5 | Environment cost makes the loop uneconomic | **35 %** | fewer, larger rounds; batch verification | core-hour accounting |
| R6 | Batch-1 latency unacceptable | **30 %** | depth capped at r=2 for interactive use | serving prototype |
| R7 | Latent-width experts lose fine-grained specialisation | **25 %** | revert to full width, sparsity 32 | B1 expert-usage analysis |
| R8 | Teacher distillation transfers style, not search | **40 %** | drop distillation, keep execution RL | trajectory-vs-answer ablation |
| R9 | The 24 GB story is a demo, not a deployment | **20 %** | 48 GB class only; cost per task doubles | real batch-16 profiling |

Three of these (R1, R2, R4) each have a cheap test that runs before any large training job. That is deliberate: **the plan is ordered so the design's riskiest assumptions are tested by the cheapest experiments.**

## When to abandon the design

Not "if metrics are disappointing" — specific conditions.

1. **R1 confirmed and R2 confirmed.** Latent depth does nothing *and* the verifier can't be made independent. Then v2 has no thesis: it is v1 with a different expert shape. Ship v1's design with v2's MoE and say so.
2. **R4 confirmed twice.** Two consecutive runs collapsing in round 2 means the guardrails are not working, and the honest conclusion is that self-improvement for code at this scale is not yet safe — not that the gates need more knobs.
3. **Verifier FPR above 5 % after the commit-first protocol.** If the one intervention with a 60× measured effect doesn't work here, the loop is unverifiable and should not run.

What would *not* be grounds for abandoning: a slower-than-hoped improvement curve, a stage needing more compute, or the MoE needing a different sparsity. Those are tuning.

## What this design is not

- **Not a reasoning model in the o-series sense.** 2.0 B active is a 2.0 B reasoning ceiling. Depth is rented, but the thing being deepened is small.
- **Not safe without tests.** Every capability claim in [05](05-execution-driven-learning.md)–[08](08-verification.md) depends on an executable oracle. On tasks without one, this is v1 with a latency penalty.
- **Not proven.** Nothing here is trained. Every mechanism is an argument from someone else's measurement plus arithmetic from our own implementation.
- **Probably wrong about its most novel part.** R1 at 40 % is not a hedge; latent CoT underperforming explicit CoT beyond 1 B is a measured result, and we are betting that code with a verifier is the exception.

## The one-paragraph version

Kestrel-3.1 stores 20.9 B parameters and reads 2.0 B per token. Capacity is bought with 256 latent-width experts at 1:42 sparsity; depth is rented with a 118 M-parameter workspace that costs 12.3× less per reasoning step than a decoded token; the signal comes from test execution rather than a judge, because the 2026 record shows judges optimise plausibility and plausibility is not correctness; verified trajectories are distilled back into the weights on-policy; and the whole thing repeats only while a verifier that commits its own answer first keeps saying yes. It trains for 28,761 GPU-hours and 12,000 CPU core-hours, fits a 24 GB card, and would be falsified in about 2,200 GPU-hours and one week of auditing.

Start at [01 — Verification is the scarce resource](01-thesis.md).
