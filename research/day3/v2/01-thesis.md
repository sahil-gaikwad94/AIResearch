# 01 — Verification is the scarce resource

Day 3 v1 ended with a claim about memory: what a model remembers is only as good as its ability to tell whether it is still true. v2 generalises that claim until it covers the whole system, because the same argument applies to every part of the stack, and because the 2026 evidence now makes it quantitative.

**The claim:**

> In code, generation is expensive and checkable. Every design decision should move work from the expensive side to the checkable side.

Three facts make that a design principle rather than a slogan.

**Fact 1 — checks are cheap and hard to fake.** Running a test suite costs seconds and a CPU core. Reading a diff and deciding whether it is correct costs a human. The asymmetry is not small: a decoder step is 2.91 GFLOP and 0.57 GB of memory traffic (see `tools/count_params_v31.py`); a test run is a forked process. When a task has an executable oracle, you can afford hundreds of checks per attempt.

**Fact 2 — the check is the only signal that cannot be argued with.** A minimal coding agent scored **1.000** pass@1 on a 16-task suite; delete its ability to run bash and it scores **0.769**, while spending **3.6×** more tokens. Same model, same prompt, same tools otherwise. Execution feedback was worth **+23.1 pp**. No reward model, no judge, no preference data — just the ability to observe a result. (mini-harness ablation, 2026.)

**Fact 3 — when the check is fake, everything downstream rots.** This is where the literature turns ugly, and it is the part most designs skip.

| Failure | Measured |
|---|---|
| Self-play against a reference-free judge | judge's pass rate climbs **0.72 → 0.94** while true accuracy stays at **0.20** |
| Three-judge ensemble as a defence | still accepts **55 %** of the false positives |
| Iterative code optimisation against a proxy metric | **73.8 %** of kernel optimisations, **46.8 %** of algorithmic ones gain on the proxy with no real gain |
| Same, as optimisation continues | hacking rises **26.4 % → 57.8 %** from 10 to 100 steps |
| Self-training for code with a binary grader | pass@1 rises, peaks within tens of steps, then **collapses — sometimes to near zero**; KL and EWC do not prevent it |

*sources: 2607.05904, ICLR 2026 workshop on recursive self-improvement, 2606.21090*

So the scarce resource is not parameters, not tokens, not even tests. It is **a check whose errors are independent of the policy's errors**. The last row of that table is the one that kills naive self-improvement: if the learner grades itself, the gradient points at its own blind spot.

The good news is that this failure has a precise fix, and it is absurdly cheap. In 2607.05904 the decisive variable was whether the judge **commits to its own answer before seeing the candidate**: false-positive rate fell from **0.719 to 0.012**, and discrimination rose to **0.96**. One design choice, a 60× reduction in a failure mode that otherwise compounds through memory and destroys the run.

## The five subsystems are one loop

The request for v2 named five things. They are not five features; they are one cycle, and each stage exists because the next one needs it.

```
        ┌──────────────────────────────────────────────┐
        │                                              │
   (1) sparse MoE ──► capacity that is cheap to run    │
        │                                              │
   (2) recurrent workspace ──► depth that is cheap to │
        │                       spend, per token       │
        │                                              │
   (3) execution ──────────► the learning signal       │
        │                                              │
   (4) distillation ───────► move search into weights  │
        │                                              │
   (5) self-improvement ───► repeat, with the verifier  │
                              as the only gate         │
        └──────────────────────────────────────────────┘
```

Read it as a budget. Capacity (1) is cheap because experts are sparse and narrow. Depth (2) is cheap because recurrence rents it instead of owning it. Neither is worth much without a signal that says *which* of the 8 attempts was right; that is (3). But attempts are expensive, so (4) compresses the successful ones back into the weights, which is what makes (5) a loop rather than a one-off.

The loop closes only if the gate is honest. Hence the ordering: **verification is designed first, and everything else is downstream of it.**

## Three inversions against v1

v1 and v2 share a spine — windowed attention, delta-rule memory, fine-grained MoE — and differ in three places, each a reversal of a v1 decision.

| | v1 | v2 | Why |
|---|---|---|---|
| Depth | fixed 16 layers, rejected recurrence | 4-layer workspace applied 1–8× | a pass costs **12.3×** less than a decoded token (0.236 vs 2.908 GFLOP) |
| Expert shape | 96 full-width experts, top-3 | 256 latent-width experts, top-6 | each expert is **10.7×** cheaper, so sparsity rises 32 → 42.7 at the same layer budget |
| Learning | offline data, then RL on tests | execution first, distillation second, self-training third | tests are the only signal that resists gaming |

The third row is the substantive change. v1 treated training as something that happens before deployment. v2 treats deployment as a data source whose output must be filtered by something that cannot be talked into agreeing.

## The economics, stated up front

From `tools/count_params_v31.py`:

| | |
|---|---|
| Parameters stored / read per token | **20,865,312,258 / 1,990,944,258** (9.54 %) |
| Forward compute, 4 k window | **4.65 GFLOP** at r=1, **5.59** at r=8 |
| Training | **28,761 GPU-hours**, 9.4 days on 128 GPUs, $43k–$86k |
| Execution environments | **12,000 CPU core-hours** (188 node-hours at 64 cores) |
| Weights at 4-bit | **11.5 GB** — fits a 24 GB card with 12.5 GB spare |

Two of those deserve comment. The GPU total is *lower* than v1's 31,025 despite 13 % more active parameters, because the long-context stages are windowed and therefore cheap. And the CPU number is not a rounding error: **test execution is the second-largest infrastructure cost in the design**, larger than distillation, RL, and SFT combined. That is what "execution-driven" means when you cost it honestly — the bottleneck moves off the GPU.

## What would falsify this

The thesis dies if any of these hold. Each is measurable before the design is committed.

1. **Depth adds nothing at our scale.** If Huginn's 5× GSM8K gain doesn't reproduce for code at 2 B active — if recurrent passes only help arithmetic — then the workspace is 118 M parameters of nothing. Test: B2 (§09), 1,000 hours.
2. **The verifier cannot be made independent.** If commit-first verification still correlates with the policy's errors on our own tasks, self-improvement is unsafe at any scale and v2 reduces to v1 plus a smaller MoE. Test: B5, one week, no GPU.
3. **The latency cost of recurrence is unacceptable.** If a pass costs real wall-clock rather than being hidden under batch, agents get slower and users route around them. Test: B4, serving prototype.
4. **Sparsity 42.7 doesn't beat 32 on code.** Then the latent-width experts are a serving optimisation, not a capability one, and the honest move is to say so.

## Forecasts, labelled

Everything above about *mechanisms* is evidence from someone else's measurement. Everything about *this model's* performance is a forecast, and should be read as one.

- **Likely (70 %)**: the workspace helps on tasks with a verifiable intermediate state and hurts nothing, because it starts as a 8.05e-05 identity (§04).
- **Plausible (50 %)**: the execution→distill→self-train loop yields more than one round of improvement before the guardrails bind.
- **Unlikely (25 %)**: latent recurrence substitutes for explicit chain-of-thought on hard code reasoning. LOTUS needed per-latent-step supervision to close that gap at 3 B; we are not doing that (§04.7).

Read on: [02 — Architecture](02-architecture.md), or the piece that decides everything, [08 — Verification](08-verification.md).
