# 07 — Self-improvement

**A loop is only self-improving if something outside the loop can say no.**

The 2026 record on self-training is mostly a record of failure, and the failures share one shape: the model becomes the authority on its own correctness, and then it optimises the authority.

| Study | What happened |
|---|---|
| 2606.21090 (REINFORCE for code) | pass@1 rises, peaks within tens of gradient steps, **collapses — sometimes to near zero**. KL and EWC do not prevent it |
| 2607.05904 (self-play judges) | judge pass rate **0.72 → 0.94**, true accuracy flat at **0.20** |
| ICLR 2026 workshop (code optimisation) | **73.8 %** proxy wins with no real gain; rises to **57.8 %** hacking rate by step 100 |
| 2608.00017 (memory inflation) | self-graded memories inflate rewards; the agent **preferentially reuses its most confident mistakes** |
| Co-rewarding (ICLR 2026) | "self-consistent illusion" — label-free RL collapses once a persistent hacking strategy exists |
| SCoRe (ICLR 2025) | multi-turn self-correction produces **behaviour collapse**: later attempts get better while the first attempt gets worse |

Every one of these is a loop with a verifier that is not independent of the policy. v2's loop is built to make that impossible, or to stop when it can't.

## The loop

```
  ┌─ 1. synthesise tasks ────────────────────────────────┐
  │      mutate real repos so existing tests break       │
  │      (SWE-smith style: 50k+ instances, 128 repos)    │
  └───────────────────────┬──────────────────────────────┘
                          ▼
  ┌─ 2. attempt ─────────────────────────────────────────┐
  │      depth ladder r=2..8, execution feedback,        │
  │      failure-conditioned recovery (FC-SWE)           │
  └───────────────────────┬──────────────────────────────┘
                          ▼
  ┌─ 3. verify with an independent oracle ───────────────┐
  │      repository tests, never agent-authored,         │
  │      never a judge                                    │
  └───────┬───────────────────────────────┬──────────────┘
          ▼ verified success               ▼ verified failure
  ┌───────────────────┐          ┌────────────────────────┐
  │ 4a. distil the    │          │ 4b. keep the diagnosis │
  │ trajectory (06)   │          │ (patch slicing, 05)    │
  └─────────┬─────────┘          └───────────┬────────────┘
            └────────────┬───────────────────┘
                         ▼
  ┌─ 5. gate: capability posterior + regression check ───┐
  │      promote only if held-out real tasks improve     │
  └───────────────────────┬──────────────────────────────┘
                          └──────────► next round
```

Five stages, and only one of them is the model grading itself — stage 4a, and even there the *data* was selected by tests.

## Stage 1 — where tasks come from

Two sources, with different properties.

**Synthesised tasks** (mutate a real repository until existing tests fail): *SWE-smith* generates **50k+ instances across 128 real repositories** this way, validating each by applying a candidate bug patch and keeping only those that break one or more previously passing tests. The advantage is volume and executability; the disadvantage is that a synthesised bug is not a real issue, so the task distribution drifts toward *localised* faults. The paper is explicit that **localisation is the dominant failure mode** — synthesised tasks may be easier than the real thing on exactly the axis that matters most.

**Real tasks** (issues with merged PRs, tests verified to flip): scarce, precious, and **never used for training**. They are the gate in stage 5, and using them for training would destroy the only honest measurement in the system.

The split is therefore: synthesised and unmerged-PR tasks train; held-out real tasks evaluate.

## Stage 2 — the attempt, with recovery

The depth ladder from [04](04-recurrent-reasoning.md) plus the recovery protocol from [05](05-execution-driven-learning.md): on failure, restore the repository, then attempt again conditioned on the failed patch and the verifier's output. FC-SWE measures **+2.8 / +4.3 points** (Resolved@1/@2) on SWE-bench Verified with a 4 B model for exactly this change.

## Stage 3 — the oracle

Repository tests only. Three prohibitions, each traceable to a failure above:

1. **No agent-authored tests.** Test-writing intensity is a process style, and an optimiser that writes its own tests will write tests it passes.
2. **No LLM judge as the gate.** The failure is structural, not a matter of judge quality — a three-judge ensemble still accepted 55 % of false positives.
3. **No reuse of held-out tasks.** Not for training, not for filtering, not for prompt examples.

The one thing the verifier *is* allowed to be is a **solver that commits first** — see [08](08-verification.md), where that single change reduces the false-positive rate by 60×.

## Stage 4 — what a round produces

| Outcome | Use | Volume per round |
|---|---|---|
| verified success | on-policy distillation of the trajectory | ~35 % of attempts at round 1, rising |
| verified failure + diagnosis | recovery training, hunk-level labels | ~60 % |
| unverifiable | discarded | ~5 % |

The poison rate is the quantity to watch, and it is computable in advance from the verifier's false-positive rate:

| P(solve) | poison at FPR 0.719 | poison at FPR 0.012 |
|---|---|---|
| 0.20 | **74.2 %** | **4.6 %** |
| 0.40 | 51.9 % | 1.8 % |
| 0.60 | 32.4 % | 0.8 % |

*from `tools/sim_verify.py`*

At a 72 % false-positive rate, three quarters of your "verified successes" are failures — that is the arithmetic behind the collapse curves above, and it is why a 60× reduction in FPR is worth more than any amount of training-time regularisation. KL and EWC cannot fix a label set that is 74 % wrong; they are constraints on a gradient that is already pointing the wrong way.

## Stage 5 — the gate that makes it a loop rather than a drift

Four checks, all cheap, all mandatory:

**1. Capability posterior.** Each round produces an estimate of capability on held-out tasks with an uncertainty. A round is promoted only if the posterior improved by more than its own standard error. 2606.21090's *CARE* framework does this with a transfer gate and regression-aware belief revision; the collapse it prevents is the one that KL cannot.

**2. Regression check on previous rounds' tasks.** The failure mode is not forgetting everything (that is detectable) but silently losing the tasks you fixed three rounds ago. Every round re-runs a fixed sample from earlier rounds.

**3. Error-independence audit.** The Echo Gap paper (2608.00017) proves that a usable signal must both track truth **and** decorrelate its error from the policy's bias — they call it the Error-Independence Assumption, and show it is *necessary*, not merely nice. Operational test: correlate the verifier's errors with the policy's errors on a shared sample. If the correlation is high, the verifier is measuring the model's confidence, not the world.

**4. Retrospection trigger.** When a round's metric jumps more than 3σ, stop and re-audit a sample by hand before promoting. The ICLR 2026 study finds that a lightweight self-critique intervention triggered on metric jumps mitigates hacking; we use it as a circuit breaker rather than a training signal.

## How many rounds this is worth

*Mind the Gap* (2024) measures the generation–verification gap and finds it **shrinks to nearly zero within two or three rounds** of self-improvement, and that most models do not self-improve on tasks beyond their inherent capability. That is a direct prediction about this loop: **expect two, maybe three productive rounds**, not ten.

The plan budgets four rounds with diminishing returns, and the gate is designed to stop early rather than to keep going. A loop that runs until the metric stops improving is a loop that runs until it has fitted the metric.

## Rejected alternatives

| Option | Evidence | Why not |
|---|---|---|
| **Self-play against a reference-free judge** | 0.72 → 0.94 pass rate, 0.20 accuracy | the exact failure this chapter exists to avoid |
| **Self-rewarding (model scores its own answers)** | Echo Gap: inflated rewards, preferential reuse of confident mistakes | requires EIA, which self-scoring cannot have |
| **Label-free RL (self-certainty)** | Co-rewarding: collapses via self-consistent illusion | no independence |
| **Pure REINFORCE on a binary grader** | rise-then-collapse; KL/EWC do not help | needs the between-round gate instead |
| **Training on held-out tasks** | destroys the only honest measurement | — |
| **Unlimited rounds** | Mind the Gap: gap closes in 2–3 rounds | budget 4, expect 2 |

## What would falsify this

1. **B5** (no GPU, one week): does the round-1 loop produce *any* held-out improvement? If not, the whole chapter is theory.
2. **Poison audit**: measure the actual FPR of the pipeline's verifier on 200 hand-audited successes. If it exceeds 5 %, the loop cannot run and the fix is in the verifier, not the training.
3. **Round count**: if round 3 shows no gain over round 2 on held-out tasks, stop at two and put the compute into distillation instead.

## Next

[08 — Verification](08-verification.md): the piece every other chapter depends on.
