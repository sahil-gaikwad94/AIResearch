# 06 — Distillation

Three different things get called distillation here, and confusing them is how people end up distilling a teacher's bad habits into a student that was already right.

| Target | What is transferred | When |
|---|---|---|
| **Search → policy** | the trajectory that passed, not the answer | after execution RL, every round |
| **Verifier → head** | pass probability from execution labels | continuously, from the rollout buffer |
| **Teacher → data** | gold traces for SFT and PR mid-training | once, early |

## Why distill the search and not the answer

A single correct patch is one sample from a distribution. A *search that found it* contains strictly more information: which files were opened first, which hypothesis was abandoned, which test failure redirected the work. If you distil only the final answer you teach the student what to say; if you distil the trajectory you teach it where to look.

The mechanism is **on-policy distillation**: the student generates a prefix, the teacher scores the continuation. Because the student's own prefixes are used, the student learns to recover from *its own* states rather than from the teacher's clean ones — which is exactly the distribution it will face at inference.

The evidence is consistent on two points:

- **On-policy beats off-policy.** GKD (ICLR 2024) shows on-policy generation with reverse KL substantially outperforming sequence-level and supervised KD across student sizes, and identifies the cause as the train–inference distribution mismatch that offline distillation cannot see.
- **Reverse KL beats forward KL for this.** Reverse KL is mode-seeking: the student concentrates on the teacher's high-probability regions rather than spreading mass to cover everything the teacher might say. For code, where one answer is correct and the rest are wrong, mode-covering is the wrong objective. GKD reports reverse KL and JSD(0.9) as the best settings; MiniLLM makes the same argument formally.

Cost is real: on-policy sampling adds **1.8–2.2×** over distilling from a fixed dataset (GKD's own measurement), because you pay for student generation. The 2026 prefix-truncation result (ACL 2026) cuts this by progressively allocating compute to longer prefixes, reaching comparable quality at **~50 % of the FLOP**.

Our stage: **30 B student tokens, 1,773 GPU-hours**, teacher assumed at 25 B active. That is 6 % of the total budget for the step that converts search into weights.

## Which teacher, and why not to train one

The *Distillation Scaling Laws* result (ICML 2025) is the one that decides the architecture of this stage, and it cuts against the intuition that a bigger teacher is always better:

> When a teacher already exists, or when many students are distilled, distillation beats supervised learning up to a compute level that scales predictably with student size. **When only one student is to be distilled and the teacher must also be trained, supervised learning is generally preferable.**

We distil one student. So the law says: **do not train a teacher.** Use one that exists. This is why the plan borrows a frontier coding model for the SEARCH→policy and data stages rather than training a 70 B teacher — the compute-optimal move for a single student is to skip that entirely.

It also bounds the ambition: we are not trying to surpass the teacher. We are trying to put a *specific* capability — the ability to search under a verifier — into 2 B active parameters, which is a much narrower target and one where the teacher's size helps rather than hurts.

## Verifier distillation

The confidence head (1024 → 512 → 1, **525,313 parameters**) is trained to predict `P(pass)` from the workspace state, on labels from actual test runs. This is distillation from a program into a network.

Two properties matter more than accuracy:

**Calibration, not ranking.** The head is used to decide *whether to spend another rung*, which is a threshold decision on a probability. A head with excellent AUC and bad calibration will over- or under-spend systematically. We fit it with a proper scoring rule and report expected calibration error on held-out tasks, not just AUC.

**Validate on RL outcome, not on selection.** SWE-RM's warning applies directly: two verifiers with nearly identical test-time-scaling performance produced very different RL results. If the head is only ever evaluated as a reranker, it can be excellent at picking winners and useless as a training signal. So the head is accepted only if B5's RL run improves, not if its ranking metric does.

## Teacher → data

The teacher generates traces for the SFT and PR mid-training stages. Three rules, each from a failure we can name:

1. **Never distil the teacher's verbosity.** Long explanations score well on human preference and cost tokens on every call. The student is trained on the trajectory *minus* its narration, keeping only tool calls, diffs, and test output.
2. **Never distil the teacher's uncertainty.** A teacher that says "this might be wrong" teaches hedging, and hedging is a token tax with no information. If the teacher's confidence is not calibrated, discard it rather than transfer it.
3. **Never distil a trajectory the teacher did not verify.** No verification, no data. This is the same rule as [07](07-self-improvement.md), applied one step earlier.

The third rule is what separates this from ordinary SFT: **every distilled token is downstream of a test run.** That property is worth more than the teacher's quality.

## What is not distilled, and why

| Not distilled | Why |
|---|---|
| **Full-vocabulary logits** | 131,072 floats per token × 30 B tokens is not a data pipeline, it is a storage project. Top-k or sampled-token KL only |
| **Forward KL** | mode-covering; the wrong objective when correctness is nearly one-hot |
| **A self-trained teacher** | the scaling law above; one student does not justify training a teacher |
| **The teacher's chain-of-thought at inference** | we are distilling the ability to search, not the habit of narrating; latent depth ([04](04-recurrent-reasoning.md)) is the inference-time mechanism |

## Rejected alternatives

| Option | Evidence | Why not |
|---|---|---|
| **Sequence-level KD (SeqKD)** | GKD: consistently below on-policy | off-policy; the student never sees its own mistakes |
| **GKD with forward KL** | GKD: forward KL performs poorly for generation | mode-covering |
| **Hybrid KL (HPD, ICML 2026)** | balances mode coverage and seeking, improves stability | plausible, and the closest thing to a competitor. Costs an extra hyperparameter and its gains are on math/dialogue; parked for B1 |
| **Distilling a bigger student from a smaller teacher** | scaling law: student size dominates | we are already at the small end |
| **Training a 70 B teacher to distil into 2 B** | scaling law: not compute-optimal for one student | the single most expensive mistake available here |

## What would falsify this

1. **Search→policy vs answer-only**: same token budget, two students. If distilling trajectories does not beat distilling answers by ≥1.5 pp, the extra complexity is unjustified.
2. **Reverse vs forward KL on code**: if forward KL matches reverse within noise, the mode-seeking argument is wrong for this domain and the simpler objective wins.
3. **Prefix truncation at 50 % FLOP**: if quality drops more than 0.5 pp, pay the full sampling cost instead.

## Next

[07 — Self-improvement](07-self-improvement.md): the loop that runs distillation repeatedly, and the guardrails that stop it eating itself.
