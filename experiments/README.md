# Three Experiments

Three research programmes that came out of reverse-engineering six models. Each one targets a specific loose end in the audit — a number I could not explain, a design choice nobody has tested, or a convergence with no agreed mechanism.

They are written the way I would actually run them: a falsifiable thesis, the mechanism that would make it true, a pre-registered decision rule, a compute budget, and the observation that would kill it. Two of them can be run for under $10,000.

| # | Question | Thesis in one line | Compute | Cost |
|---|---|---|---|---|
| [1](01-state-capacity.md) | Why do hybrid linear models fail at recall, and when exactly? | Failure is set by the number of *distinct associations*, not by context length — and the cliff scales with head dimension. | ~3,000 GPU-hr | ~$6k |
| [2](02-nope-fraction.md) | Should global attention layers carry positional encoding? | Position is a per-layer resource with an interior optimum around 25–40% NoPE layers — and "no position" is mostly "very slow position". | ~4,300 GPU-hr | ~$9k |
| [3](03-router-ceiling.md) | Why did MoE sparsity stop at ~50×? | The router's discrimination margin collapses as experts multiply; better routers move the ceiling. | ~6,200 GPU-hr | ~$12k |

## Why these three

They are not chosen to be broad. They are chosen because each one **explains a measured convergence** in the 2023–2026 record, and because each has a cheap decisive experiment.

1. **Sparsity converged to 32–51×** across four independent teams (Mixtral 4×, DeepSeek 32×, Kimi K2 48×, Flint 43×, Qwen3-Next 51×). Existing scaling laws explain the number of *active* experts (G ≈ 7) and the trend toward sparser activation, but not why the *expert count* stops growing at a few hundred. Experiment 3 tests the obvious candidate mechanism: the router.

2. **Hybrid attention converged to 25–29% global layers** (Qwen3-Next 12/48, Flint 14/49), and one of the two puts **no positional encoding at all** on exactly those layers. The ratio has been ablated publicly. The positional scheme has not. Experiment 2 isolates it.

3. **Linear attention keeps being adopted while everyone admits it recalls worse.** The field's working assumption is "state size sets recall". That is true at small scale on synthetic tasks, but nobody has turned it into a *task-level invariant* that predicts failure on real workloads. Experiment 1 does that, and gets a sharp number out of it.

## Shared method

All three follow the same protocol, because the failure mode of this kind of work is a pretty curve with no mechanism:

- **One variable at a time, with the confound named.** E2 has the θ-sweep to separate "no position" from "slow position". E1 has the m × T × distance factorial to separate capacity from distance. E3 has the margin measurement to separate "routers are the ceiling" from "we ran out of data".
- **A mechanistic intermediate.** Not just "loss went down" but *which quantity moved*: state saturation in E1, attention-versus-distance profile in E2, router margin in E3. If the intermediate does not move, the explanation is wrong even if the headline result is good.
- **A scaling prediction, not just an effect.** Each experiment predicts how the effect changes with a parameter (d_v, NoPE fraction, E). A single-point result can be an artefact; a curve that matches a predicted exponent is much harder to fake.
- **Pre-registered decision rules.** The threshold for "confirmed" is written before running, including the sample sizes and the seed count.
- **A one-week pilot that can kill it.** Every file ends with the cheapest experiment that would show the idea is dead.

## What none of these test

They are all about *mechanism at the model level*. None of them tests whether the resulting models are more useful — that requires task-level evaluation (agents, coding, science), which is a different programme with different costs. Where a task-level claim appears, it is a prediction to be checked, not a result.
