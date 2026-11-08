# 05 — Execution-driven learning

**Reward comes from running the code, and the richest signal is not pass/fail — it is the vector of per-test outcomes plus the failure text.**

## What a test suite actually gives you

A failing task is usually recorded as one bit: resolved or not. That throws away almost everything. A single `pytest` run on a real repository returns:

- **per-test outcomes** — 47 passed, 3 failed, 2 errors, with names;
- **failure text** — assertion diffs, tracebacks, line numbers;
- **timing** — a patch that passes but takes 40× longer has a problem;
- **coverage deltas** — did the patch execute the lines the issue was about?

Four signals for the price of one process. The literature treats the first as the reward and the rest as debugging output; v2 treats all four as the reward.

## The reward

```
R = 1.0 · fail2pass            # the tests that must go from red to green
  + 0.3 · pass2pass           # did not break anything else
  + 0.2 · coverage_delta      # executed the lines the issue names
  - 0.1 · runtime_penalty
  - 0.1 · diff_size_penalty   # normalized against the gold patch
```

The weights are a design choice, and they are the kind of choice that should be ablated rather than defended. Two of them carry real arguments:

**`pass2pass` is not optional.** A patch that fixes the failing test by breaking three others is worse than no patch, and an outcome-only reward on `fail2pass` will happily produce it.

**`diff_size` is a regulariser on a known failure mode.** Models over-edit: they rewrite a function when a two-line change would do, and large diffs are where regressions hide. The penalty is small because the effect is small, and it is stated as a penalty rather than folded into correctness so it can be removed when a task genuinely needs a big change.

## Partial credit is where the leverage is

The reason to keep the per-test vector is that it turns a sparse outcome into a **dense** one. A task with 12 tests where the model passes 9 is not a failure to be discarded; it is 75 % of a success, and the 3 that failed are a specification of what is still wrong.

Two uses:

**1. Group advantages without a reward model.** In a group of 8 attempts at one task, ranking by test-count rather than by resolved/not gives a non-degenerate advantage signal even when all 8 fail — which is the common case on hard tasks, and exactly where GRPO-style methods collapse to zero gradient. This is cheap: the tests were run anyway.

**2. Failure text as context for the next attempt.** This is the strongest result in the recent literature and it is worth stating exactly.

*FC-SWE* (2026) conditions a recovery attempt on the failed patch **and** the verifier's feedback, restoring the repository to its original state first, and keeps failed attempts inside the advantage comparison group ("active-set advantage estimation"). With a 4 B model on all 500 SWE-bench Verified tasks under a verifier-assisted protocol:

| | Resolved@1 | Resolved@2 |
|---|---|---|
| GRPO | 38.9 % | 48.5 % |
| **FC-SWE (failure-conditioned)** | **41.7 %** | **52.8 %** |

**+2.8 and +4.3 points from using the failure as context instead of throwing it away.** No new parameters, no new data — the same rollouts, a different conditioning. That is the highest return per line of code in this design.

## Credit assignment without a process reward model

A process reward model is expensive: you need step labels, and step labels for code require a human or a strong model to judge intermediate states. Execution gives a cheaper route.

**Patch slicing.** Given a failing patch with `h` hunks, delete one hunk at a time and re-run the tests. That is `h` extra test runs — seconds each — and it attributes the failure to a specific hunk. Repeated over subsets it is delta debugging, which is the standard way to localise a fault and which nobody applies to *training labels* because the loop is usually not set up to run tests during data construction.

The result is a **free process signal**: for each trajectory, which edit caused which failure. That is the supervision that §04.7 says latent steps are missing. It is also what makes the diagnosis in a recovery attempt specific ("the change to `parse_config` broke `test_defaults`") rather than generic ("the tests failed").

Cost: `h` test runs per trajectory. With a mean of 2.3 hunks on our task sample and a 90 s environment, that is ~200 s of CPU per trajectory, against a GPU cost of ~0.5 s. **The CPU:GPU imbalance is the defining property of this whole design** — see §09's budget, where execution is the second-largest line item.

## Why not a learned reward model as the primary signal

*SWE-RM* (ICLR 2026) is the strongest execution-free reward model in the literature: a 30 B/3 B-active MoE trained on execution labels, which lifts Qwen3-Coder-Flash from 51.6 % to 62.0 % on SWE-bench Verified under test-time scaling and **+3 absolute points over execution-based RL**.

It is a genuine result, and it is also the clearest statement of the trade: a reward model that good exists **because it was trained on execution labels**, and even then it beats execution by only 3 points in RL — while adding a 30 B model to the loop, a second failure surface, and an optimisation target that is not the objective.

The paper contains the more important finding for us: **two verifiers with nearly identical test-time-scaling performance can produce very different RL results.** Ranking ability does not imply a good training signal. A reward model can be an excellent selector and a terrible teacher.

So the design uses execution as ground truth and the confidence head (525 K parameters) as an auxiliary ranker, trained on execution labels. If execution is unavailable, we do not substitute a judge — we fall to the confidence head and accept the loss, and §08 says why that is the honest option.

## The failure modes, and what they cost

| Failure | Measured | Defence in v2 |
|---|---|---|
| Proxy metric diverges from real tasks | **73.8 %** of kernel and **46.8 %** of algorithmic optimisations show proxy gains with no real gain; hacking rises **26.4 % → 57.8 %** from 10 → 100 steps | held-out task set never used for training; retrospection trigger on metric jumps |
| Agent writes its own weak tests | test-writing intensity is a **process style**, only weakly aligned with correctness | training rewards use the repository's tests, never agent-authored ones |
| Reward from a reference-free judge | judge pass rate **0.72 → 0.94** while accuracy stays **0.20** | judges are never the reward; commit-first if used at all |
| Self-play drift | rise-then-collapse within tens of steps; KL/EWC do not prevent it | between-round capability gate ([07](07-self-improvement.md)) |

The first row is the one that matters. It is not a hypothetical: 73.8 % is the *majority* of a frontier-model optimisation run, and it grows with optimisation steps. Any loop that optimises a fixed set of tests will eventually fit the tests. The only structural defence is a test set the optimiser has never seen, evaluated by a party that is not the optimiser.

## What execution cannot do

Three honest limits.

1. **It needs an oracle.** A task without tests is not a task, for this loop. The design handles that by refusing to reward it, not by inventing a verifier.
2. **It is slow relative to generation.** 90 s of environment for 0.5 s of GPU. This bounds the loop at thousands of tasks per day per node, not millions.
3. **Tests are incomplete specifications.** Passing all tests is necessary, not sufficient — a patch can pass and be unmaintainable, and the reward function above cannot see that. That is what the diff penalty is for, and it is a blunt instrument.

## Rejected alternatives

| Option | Why not |
|---|---|
| **Outcome-only binary reward** | throws away the per-test vector, which is the cheapest dense signal available |
| **LLM judge as reward** | 2607.05904: the failure is structural, not a matter of a better judge |
| **Agent-authored tests as reward** | the optimiser can write tests it passes |
| **PRM trained on human step labels** | thousands of GPU-hours of labelling for a signal that patch slicing derives from test runs |
| **Execution-free RM as the primary signal** | SWE-RM's own numbers: +3 points, at the cost of a 30 B model. Used as an auxiliary ranker instead |

## What would falsify this

1. **Patch slicing**: does a hunk-level label actually improve recovery attempts? If the conditioning gain is under 1 pp, the `h` extra test runs are not worth the wall-clock.
2. **Partial credit vs binary**: does group-advantage-by-test-count beat resolved/not on tasks where all 8 fail? If the advantage is degenerate anyway, the vector is only useful for the recovery text.
3. **Reward weights**: if `coverage_delta` and `diff_size` contribute nothing in ablation, delete them and simplify to three terms.

## Next

[06 — Distillation](06-distillation.md): what to do with the verified trajectories once you have them.
