# 08 — Verification

**The chapter every other chapter depends on. If the verifier is not independent of the policy, nothing downstream is safe — and the fix is not a better verifier, it is a different protocol.**

## Three verifiers, ranked by independence

| Verifier | Independence | Cost | Failure mode |
|---|---|---|---|
| **Execution** (repo test suite) | total — it doesn't know the model exists | ~90 s environment + ms per run | incomplete spec; flaky tests |
| **Commit-first solver** (a model that answers the task itself, then compares) | high, if it is a different model or a differently-seeded one | one full solve per judgement | correlated with policy if same weights |
| **Confidence head** (525 K params) | none — it is part of the policy | 0.0001 GFLOP | systematic bias, masked as calibration |

Only the first is a gate. The second is a tie-breaker. The third is an inference-time controller and is never used to decide whether a training example is correct.

## The single most important measurement in this document

*More Convincing, Not More Correct* (2607.05904) runs self-play where the model judges its own answers, then audits against a held-out anchor the judge never sees.

| | Judge's verdict | True accuracy |
|---|---|---|
| 5 self-play iterations | **0.72 → 0.94** | **0.20, flat** |
| Strict 3-judge ensemble | accepts **55 %** of the false positives | — |
| **Judge commits its own answer first** | FPR **0.719 → 0.012** | discrimination **0.96** |

The last row is the whole design. A judge that scores a shown candidate is scoring *plausibility*; a judge that commits to an answer first and then compares is scoring *agreement with a solution*. Same model, two protocols, a **60× difference in false-positive rate**.

Consequence for this design: **no verifier in v2 is ever shown a candidate before it has produced its own.** The confidence head is exempt only because it never gates a label.

## What that difference is worth, in arithmetic

From `tools/sim_verify.py` — assumptions stated in the file, results are arithmetic, not measurement. `p = 0.30` per sample, verifier accuracy `1 − FPR`.

| k | commit-first (FPR 0.012) | blind judge (FPR 0.719) | pass@k ceiling |
|---|---|---|---|
| 1 | 0.296 | **0.084** | 0.300 |
| 4 | 0.751 | 0.214 | 0.760 |
| 16 | **0.985** | **0.280** | 0.997 |
| 64 | 0.988 | 0.281 | 1.000 |

Read the blind-judge column. **Best-of-16 with a blind judge scores 0.280 — below the 0.300 you get by not searching at all.** Even k=1 loses 22 points, because the judge discards correct answers that look unlikely.

This is the mechanism behind every "search didn't help" result in the literature: it is not that search fails, it is that the selector is worse than the generator. Generation and verification have different difficulty profiles, and a 0.72 false-positive rate means verification is *harder* than generation on this task distribution.

With commit-first, k=16 recovers 0.985 of a 0.997 ceiling. **Search only pays when the verifier is better than the sampler**, which is the generate–verify asymmetry stated as a number.

## The asymmetry, and how it is measured

*Mind the Gap* (2024) formalises this as the generation–verification (GV) gap: a model can self-improve only where verification is easier than generation. Their findings that constrain this design:

- the gap **increases with verifier capability** and **decreases with generator capability** — so a strong generator with a weak verifier cannot self-improve, no matter how many samples you draw;
- the gap **shrinks to nearly zero within two or three rounds** of self-improvement;
- most models do **not** self-improve on tasks that exceed their inherent capability.

*ReVeal* (ICLR 2026) takes the complementary move: instead of hoping the gap exists, **optimise self-verification directly** to widen it. Their model, trained on only three turns, sustains **20+ turns** of self-verified evolution at inference on LiveCodeBench, and improves Pass@k — evidence that verification is a *trainable* capability rather than a fixed property of scale.

That is the strongest argument for spending training compute on the confidence head: it is not just a controller, it is the thing that makes search profitable.

## Error independence is necessary, not desirable

*Memory Reward Inflation* (2608.00017) proves the condition. For a verifier's signal to correct a policy's bias, its errors must both track truth **and** decorrelate from the policy's error. They call this the **Error-Independence Assumption** and show it is necessary — the recoverable payoff is a closed-form function of exactly those two quantities. Their LUCID de-inflation reaches **56.9 %** on BIRD against **54.0 %** for a self-graded memory agent.

Operationally, this gives a test that costs one afternoon:

> Take 200 attempts. Record, for each, whether the policy thought it was right and whether it was right. Compute the correlation between *policy confidence* and *verifier error*. If it is high, the verifier is re-measuring the model's confidence and the loop cannot run.

This test is B5 in §09, and it runs before any GPU is booked.

## Where verification is actually cheap — and where it isn't

The cost structure is counterintuitive and it shapes the entire training plan.

| Step | Cost |
|---|---|
| Build the environment (image, deps, fixture) | **~60–90 s** |
| Run the test suite once | ~1–10 s |
| Run it again after a patch | ~1–10 s |
| Slice the patch into `h` hunks and re-run each | `h × 10 s` |

**Verification is nearly free conditional on a warm environment, and the environment is expensive.** That is why the budget in §09 has 9,000 CPU core-hours for RL and only 84 GPU-hours: the check is cheap, the ability to check is not.

It also explains why "just verify more" is bad advice for LLM judges — a judge that commits an answer first costs a full solve (2.9 GFLOP and 0.57 GB), which is the same order as generating the candidate. **A judge is only cheap when it is a program.**

## What verification cannot do

1. **Incomplete specifications.** Passing the tests is necessary, not sufficient. A patch can pass and be unmaintainable, and no amount of test running detects it. This is a permanent hole, and the diff-size penalty is a blunt, honest patch over it.
2. **Flaky tests.** A test that fails 1 % of the time makes a 1 % poison rate indistinguishable from noise — and 1 % is the safety threshold in [07](07-self-improvement.md). Mitigation: run each selected success twice before keeping it.
3. **Environment drift.** A dependency update can turn a verified success into a failure without the code changing. Mitigation: pin images per round; re-verify a sample from round *n* in round *n+1*.
4. **The tasks nobody wrote tests for.** Design, architecture, ambiguous specs. §10 reports these separately; the design does not pretend to improve them.

## Rejected alternatives

| Option | Why not |
|---|---|
| **LLM-as-judge without commit-first** | 0.719 FPR; best-of-16 lands *below* greedy |
| **Three-judge ensemble** | accepts 55 % of false positives; ensembling plausibility does not produce correctness |
| **Confidence threshold as the gate** | zero independence; the Echo Gap is exactly this |
| **Verifier trained on the policy's own labels** | circular; EIA cannot hold |
| **Majority vote over samples** | self-consistency measures agreement, not truth; a confidently wrong model is consistent |
| **Human review of everything** | correct but not a loop; reserved for the 4 % where automated checks disagree |

## What would falsify this

1. **FPR audit on our own distribution.** If the pipeline's commit-first verifier has FPR above 5 % on 200 hand-audited successes, the loop stops. This is measured, not assumed.
2. **GV gap on code.** If verification is *harder* than generation for our tasks (verifier accuracy below sampler accuracy), then search is not worth doing at all and [04](04-recurrent-reasoning.md)'s ladder is pointless.
3. **Commit-first's gain does not reproduce.** The 0.719 → 0.012 result is one paper on GSM8K with Qwen3 policies. If it does not hold on code, the verification budget moves to more test runs and fewer judges.

## Next

[09 — Training](09-training.md) for the budget, and [10 — Serving and risks](10-serving-and-risks.md) for what happens when you run it.
