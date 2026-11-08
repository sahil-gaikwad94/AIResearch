# 5 — Reasoning

> A 1.76 B active model cannot out-think a 20 B dense model by being cleverer per forward pass. It can out-think it by spending forwards where they matter — and the published evidence says that is worth **4×** the compute of the naive alternative, or **14×** the parameters, on tasks where you can check the answer.

---

## 5.1 The three results this chapter is built on

**Compute-optimal allocation beats best-of-N by 4×.** *Scaling LLM Test-Time Compute Optimally* (ICLR 2025) compares PRM-guided search and distribution adaptation across difficulties and finds that the best strategy *depends on how hard the problem is* — easy problems are best served by sampling and stopping, hard ones by deep search. Allocating adaptively recovers the same accuracy with **more than 4× less** test-time compute than a fixed best-of-N policy, and on problems where a smaller model has non-trivial success rates, test-time compute lets it **outperform a 14× larger model** at matched FLOPs.

The paper's negative results are as important as its positive one: several popular search strategies *lose* to plain sampling when applied uniformly.

**Verifier-free scaling is suboptimal.** *Scaling Test-Time Compute Without Verification or RL is Suboptimal* (2025) proves and measures it: at a fixed compute/data budget, methods that use a verifier (RL or search) beat methods that distil search traces, and the gap **widens** as the test-time budget grows. The condition is a base model with a heterogeneous distribution over correct solutions — exactly a coding model.

**An imperfect verifier caps the return.** *roc-n-reroll* (ICLR 2026) shows the gain from rejection sampling or best-of-N is governed by the verifier's ROC curve, not by generation quality alone. With a verifier whose true-positive rate is flat near a threshold, extra samples buy almost nothing.

Taken together: **reasoning is a budget-allocation problem around a verifier, and the verifier's quality is the binding constraint.**

That reframes what a small model should be good at. It does not need to be right on the first try. It needs to be **cheap to sample**, **calibrated about difficulty**, and **good at ranking its own candidates**, because those are the three levers that turn a verifier into accuracy.

---

## 5.2 The budget: samples × turns, not latent depth

The obvious way to buy reasoning in a small model is to think longer inside the forward pass — latent recurrence, extra layers, more compute per token. This design does not do that, and the reason is measured: the previous design's recurrent block rests on `φ = 0.46`, an exchange rate measured at other scales, and the evidence for it at 4 B is thin.

Instead the reasoning budget is **external and inspectable**:

| Knob | Range | Cost | Best for |
|---|---|---|---|
| samples `n` | 1–8 | linear | independent attempts at a fix |
| turns `t` | 1–30 | linear | acting on test feedback |
| candidate depth | 1 | — | — |

And allocation is a learned function of the task, not a constant:

```
p̂ = difficulty_head(h_first_token)        # 2-layer MLP on the prelude state
if p̂ > 0.85:  n=1, t=3        # looks easy: commit
elif p̂ > 0.35: n=4, t=12      # worth trying hard
else:          n=8, t=20       # or ask / explore
```

The threshold policy is not hand-tuned: it is fit on a held-out set to maximise solved tasks per verifier call, which is the quantity that actually costs money ([06](06-data.md)). The ICLR 2025 result is precisely that this kind of per-prompt allocation recovers most of the achievable gain.

### Why this is better than making the model deeper

Because the model is already the cheapest part. At batch 16 one decode step is 0.68 ms for the whole model; a `pytest` run is 10–60 seconds. Spending 4× more model compute is nearly free relative to spending 1.5× more verifier calls. The optimal ratio of thinking to testing for a coding agent is **wildly** in favour of thinking — and the model is sized so that it can afford to.

---

## 5.3 Self-verification: rank, don't score

Sampling more helps only if you can pick the winner. Absolute scoring of candidates is weak — models are overconfident and their scores are miscalibrated across difficulty. *V1* (ICLR 2026) finds models are **substantially stronger at pairwise comparison than at pointwise scoring**, and uses that:

- generate `n` candidates,
- run a **tournament** on pairs, spending verification compute where the pair is most uncertain,
- pass the winner to the real verifier.

Reported: **+10 % Pass@1** over pointwise verification, and a joint generation/verification training (V1-PairRL) giving **7–9 %** test-time scaling gains over standard RL.

For code this is unusually well-founded: two candidate patches can be compared on *static* grounds (does one touch fewer files? does one add a test? does one change a signature callers use?) without running anything. The tournament is cheap, and it converts `n` samples into a better-than-random choice before the expensive step.

The architecture supports this directly: a **pairwise head** on the final hidden state, trained with a margin loss on (correct, incorrect) pairs harvested from RL rollouts. Cost: one extra 4096×1 projection. It is the cheapest reasoning improvement in the design.

---

## 5.4 The verifier is a resource to be scheduled

Most treatments of verification assume it is free. For code it is the dominant cost, and it is noisy (flaky tests, environment drift, timeouts). Three rules follow:

1. **Cheap checks first.** Syntax, imports, type-check, and diff-size sanity before running any test. A candidate that fails `ruff` should never reach `pytest`.
2. **Tiered tests.** Fast unit subset (2 s) → affected tests (20 s) → full suite (minutes). Escalate only on pass. This is standard CI practice and it maps exactly onto a search loop.
3. **Budget the verifier, not the model.** Cap test-minutes per task. When the cap is hit, fall back to the pairwise ranking and emit the best candidate with a confidence estimate, rather than looping.

A coding agent that runs the full suite on every one of 8 candidates is 8× slower than one that triages. The measured difference in solved tasks is usually zero — [09](09-evals-and-risks.md) makes this an explicit measurement, because it is the kind of thing teams assume without checking.

---

## 5.5 Training: what RL is actually for

Two 2026 results discipline the RL recipe.

**Most of RL's gain may be redistribution.** *From Base Rollouts to RL Reasoning: A Budgeted Search Perspective* (2026) compares paired base/RL checkpoints across ten models and four families, and finds that much of the measured RL improvement corresponds to a **change in sampling efficiency** — the RL policy reaching operating points the base model already contained, reachable with more search. The paper offers a "qualified internalized-search reading."

If that holds here, RL's value is making the *good* samples more likely, not creating new capability. That changes the recipe: optimise for the policy being *right more often per sample*, not for longer chains of thought.

**Length is a bias you have to choose.** *GFPO* (ICLR 2026) shows that token-level normalisation amplifies rewards for long correct outputs — reinforcing verbosity — while removing it (Dr. GRPO) lets long trajectories dominate the gradient instead. GFPO's resolution is to **sample more and filter the group**, keeping the shortest correct completions: +1–3 % accuracy over Dr. GRPO on AIME/GPQA/Omni-Math with **10–70 % more** length reduction.

For a model whose output is consumed by a diff and a test suite, conciseness is not cosmetic — it is tokens the user pays for and code a reviewer must read. So:

| Component | Choice | Evidence |
|---|---|---|
| Importance ratio | **sequence-level (GSPO)** | token-level has a unit mismatch with sequence reward; stabilises long CoT |
| Group construction | **filtered (GFPO)** | shortest-correct sampling beats length-penalty tricks |
| Advantage | group-relative, **no std division** | std division upweights low-variance groups |
| KL | 0.02 to SFT reference | unchanged |
| Reward | tests pass + coverage delta − cost | [06](06-data.md) |

The honest note: GSPO, GFPO and Dr. GRPO disagree in the literature about which normalisation is correct, and a 2026 impossibility result argues that **no objective can be simultaneously unbiased and length-invariant** under outcome rewards. That means the choice is a value judgement about which bias you want. Here: shorter outputs, because the product is a diff.

---

## 5.6 What a 1.76 B model still cannot do

Stated plainly, because the budget numbers are seductive.

- **No verifier, no advantage.** On tasks where correctness cannot be checked — API design, prose, ambiguous requests — this model is a 1.76 B model, and will lose to 20 B dense. The design is honest about being a *verifiable-work* model.
- **Compositional depth is capped by active parameters.** MoE adds knowledge, not sequential reasoning depth. Twenty steps of deduction that require holding five constraints are harder at 1.76 B active than at 4 B.
- **Search multiplies, it does not create.** If pass@1 on a task class is 0, no allocation policy helps. The ICLR 2025 result explicitly conditions on "non-trivial success rates".

These are the reasons [09](09-evals-and-risks.md) includes a *no-verifier* eval slice, so the regression is measured rather than discovered by users.

---

## 5.7 Falsifiers

1. **Allocation policy saves <2× over fixed best-of-N** on held-out tasks → the difficulty head is not informative and should be replaced by a constant.
2. **Pairwise head gives <3 pp** over pointwise → drop it, keep the tournament on static heuristics.
3. **RL moves pass@1 but not pass@8** → the gain is capability, not sampling efficiency; increase RL, weaken the "redistribution" assumption.
4. **Cost per solved task rises** as `n` grows beyond 4 → the verifier is saturated; cap `n` at 4.

**Next:** [06 — Data](06-data.md).
