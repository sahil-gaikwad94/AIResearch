# 09 — Training and budget

## Compute per token

From `tools/count_params_v31.py`, same conventions as v1 (`6 · active + attention + memory-TTT + workspace`):

| | GFLOP/token |
|---|---|
| `6 × active` | 11.95 |
| attention (window 4,096, fwd+bwd) | 3.22 |
| memory TTT | 0.03 |
| workspace (mean r = 1.9) | 1.35 |
| **total** | **16.54** |

v1 was 13.81. The **+20 %** is the workspace: training at multiple depths costs more than training at one. That is the price of C2 in [02](02-architecture.md), and it is paid once per token rather than once per deployment.

## The stages

| Stage | Tokens | Window | GPU-h |
|---|---|---|---|
| Pretrain | 1,200 B | 4 k | 17,418 |
| Window extension | 150 B | 8 k | 2,601 |
| Long-context (64 k context, 4 k window) | 60 B | 4 k | 871 |
| PR mid-training (32 k context) | 60 B | 4 k | 871 |
| SFT | 3 B | 4 k | 45 |
| Distillation (on-policy, 25 B-active teacher) | 30 B | 4 k | 1,773 |
| Execution RL (60 k tasks × 6 rollouts × 8 k tokens) | 2.88 B | 4 k | 84 |
| Self-improvement (4 rounds × 6 k tasks × 5 rollouts) | 0.60 B | 4 k | 17 |
| Ablations | — | — | 2,000 |
| **Subtotal** | | | **25,680** |
| Restarts (12 %) | | | 3,082 |
| **Total** | **1,505 B** | | **28,761** |

At $1.5 / $2.2 / $3.0 per GPU-hour: **$43k / $63k / $86k**. On 128 GPUs: **9.4 days**.

Two things to notice in that table, because they are the design showing up in the budget.

**Long-context stages cost the same as short ones.** 60 B tokens at a 64 k context costs 871 GPU-hours — identical to 60 B at 32 k — because attention is windowed at 4,096 and the memory tier is O(1). In a dense attention model, that stage would cost 16× more. This is the single largest structural saving in the plan.

**Execution RL costs 84 GPU-hours and 9,000 CPU core-hours.** The GPU line is almost a rounding error; the environment line is larger than distillation, SFT, and RL combined. **The bottleneck in execution-driven learning is the environment, not the model.** 9,000 core-hours is 188 node-hours at 64 cores, and it is the part of this plan that is hardest to buy, hardest to parallelise, and most likely to be the actual constraint on how many rounds you can run.

## Data

| Source | Share | Purpose |
|---|---|---|
| Code, permissive licences | 40 % | syntax, idiom, API surface |
| PRs, issues, commit messages | 18 % | the repo-level task distribution |
| Tests and assertions | 12 % | the artefact the loop is scored on |
| Math and formal reasoning | 10 % | the capability recurrent depth amplifies |
| Documentation and prose | 8 % | instruction following, explanation |
| Long-context repository snapshots | 7 % | the memory tier's training signal |
| Synthetic bug injection | 5 % | volume for the self-improvement loop |

The 12 % on tests is the unusual line and it is deliberate: a model that cannot read a test cannot use the loop that this entire design is built around. v1's *Clean-PR* mid-training result (+13.6 / +12.3 pp on SWE-bench Lite/Verified, agentless) is the reason PR data gets a dedicated stage rather than being folded into pretraining.

## Stability

Five mechanisms, each for a specific failure that this architecture invites.

| Mechanism | Prevents |
|---|---|
| **Muon** for the trunk, AdamW for embeddings and router | the router's gradient scale differs by orders of magnitude from the experts' |
| **Aux-loss-free bias update** (`b ← b + η·sign(mean − n_i)`) | routing collapse without fighting the LM objective |
| **Depth curriculum** (60 % r=1, 25 % r∈{2..4}, 15 % r∈{5,6,8}) | the loop being trained at one depth and used at another |
| **Layer-scale 1e-4 on the workspace output** | a random recurrent path injecting noise into a converged trunk |
| **QK-norm in attention** | logit growth at 64 k context, which the window mask makes easy to miss |

The workspace initialisation deserves a note because it is the least standard: at step 0 the recurrent path contributes **8.05e-05** relative to the logits (test [6]). If a run starts with the loop already at scale 1.0, the loss curve looks fine for a few hundred steps and then diverges — the trunk is being asked to absorb an untrained recurrence.

## Gates

A stage does not advance unless all four pass. Each is a number, not a judgement.

| Gate | Threshold | What it catches |
|---|---|---|
| **Router entropy** | ≥ 4.0 effective experts per token | silent collapse to dense routing |
| **Workspace drift** | 16-pass norm ratio ∈ [0.5, 2.0] | exploding or vanishing recurrence |
| **Verifier calibration** | ECE ≤ 0.05 on held-out | a confidence head that is a reranker but not a probability |
| **Held-out real tasks** | no regression vs previous round | the collapse curve in [07](07-self-improvement.md) |

The first two are cheap and run every 1,000 steps; the last two run per stage.

## Restart triggers

12 % of the budget is reserved for them, which is an admission that some runs will die.

1. **Loss spike above 3σ** that does not recover within 200 steps → restart from the last checkpoint with a 0.5× LR and a fresh data shard offset.
2. **Router entropy below 3.5** for 2,000 steps → restart with a stronger bias learning rate. This is not recoverable in-run; the router has collapsed.
3. **Workspace norm outside the band** for 500 steps → restart with layer-scale 1e-5 and a slower depth curriculum.
4. **Verifier FPR above 5 %** at any round → stop the loop. The fix is in [08](08-verification.md), not in the training config.

Trigger 4 is the one that matters. A run that continues past a bad verifier does not fail loudly; it improves on the wrong thing, and the 2026 literature is a catalogue of exactly that.

## Rejected alternatives

| Option | Why not |
|---|---|
| **Dense attention for the long-context stage** | 16× the compute for a capability the memory tier provides |
| **Training at a single depth** | the loop must see the depths it will be used at |
| **Ponder loss / learned halting** | 2607.20519; see [04](04-recurrent-reasoning.md) |
| **Training the router with an auxiliary load-balancing loss** | fights the LM objective at 256 experts |
| **No restart reserve** | the 12 % is not pessimism; it is the observed rate of a first attempt at a new architecture |

## What would falsify this

1. **Stage ordering**: if distillation before PR mid-training beats distillation after, the pipeline is wrong by one stage and the budget moves by ~1,700 GPU-hours.
2. **The workspace share of compute**: if `p(r)` turns out flat (B2), the 1.35 GFLOP/token spent on recurrence is pure waste and r should be fixed at 1.
3. **The 12 % restart reserve**: if the first three runs need no restarts, the reserve is too conservative and 3,000 GPU-hours move to ablations.

## Next

[10 — Serving and risks](10-serving-and-risks.md).
