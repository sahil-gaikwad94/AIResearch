# 10 — Day 5 preview: three directions for Kestrel-5

**This chapter is a forecast, not a design.** Nothing here is implemented, nothing is
asserted, and every number is a first-order estimate with its assumptions shown. The label
matters: days 1–4 are constructive, day 5 is a bet.

The three directions below are chosen because each one attacks a limit that Kestrel-4 hits for
a *structural* reason, not because it needs more parameters or more data.

## 10.1 Block-diffusion factor-graph unmasking

**The limit.** Kestrel-4 decodes one token per step, and at batch 1 a step streams 0.84 GB of
weights to produce 4.37 GFLOP of work — a 28× memory/compute imbalance (§09). Every trick in
§07 reduces the *number* of steps (0-DoF splicing removes 28 %) but never increases the tokens
produced per weight stream.

**The direction.** Decode a block of `B = 32` positions jointly, denoising from a fully masked
block, and let the **constraint factor graph** decide which positions may be committed at each
denoising step. A position is committed when every completion of the remaining positions is
still tier-1 feasible — the same soundness rule as §04, applied to a partial assignment rather
than a prefix. Deterministic fields are spliced without sampling, as now.

This is not standard diffusion-LM decoding, and the difference is the whole idea: a diffusion
model commits on *confidence*, which is a judge, which is 0.719. Committing on *feasibility of
all completions* is a proof obligation, so the acceptance rate is a property of the factor
graph rather than of the model's calibration.

**Why it could work, in arithmetic.** With per-block acceptance `a`, expected committed tokens
per step is `a(1 − a^B)/(1 − a)`. At `B = 32`:

| `a` | Tokens/step | Step cost (ms) | tok/s | vs 3,984 |
|---:|---:|---:|---:|---:|
| 0.60 | 1.5 | 0.53 | 2,830 | 0.71× |
| 0.70 | 2.3 | 0.53 | 4,400 | 1.10× |
| 0.90 | 8.7 | 0.53 | 16,400 | 4.1× |
| 0.97 | 20.1 | 0.53 | 37,900 | 9.5× |

Step cost is `max(weights 0.251 ms, compute 32 × 4.37 GFLOP / 494 TFLOP/s = 0.283 ms) ≈ 0.45–0.53
ms`. **Break-even is `a ≈ 0.68`**; above 0.90 the win is 4× and above 0.97 it is 9.5×. The entire
question is whether a trained model plus a factor graph can commit 90 % of a 32-token block
soundly, and that is an empirical question worth one 400-GPU-h experiment.

**Rejected alternatives.** Diffusion-LM with confidence-based unmasking (a judge, 0.719);
speculative decoding with a draft model (reduces latency per token, not weight streams per
token, and the draft cannot see the constraint system); Medusa-style extra heads (same problem
— the heads learn to guess, and a guess is not a proof).

**Falsifier.** On a 600-token patch benchmark: `a ≥ 0.90` and ≥ 3× wall-clock at equal task
success, or the mechanism is dead. A secondary kill: any committed block that later fails tier
2 invalidates the soundness argument and the direction, not just the schedule.

## 10.2 Contract-decomposed multi-agent swarms

**The limit.** A 21 B model working alone has one decode stream and one context. Multi-file
patches that are *independent* — two modules with disjoint interfaces — are serialized by the
decoder for no reason other than the API.

**The direction.** Decompose the task into the **connected components of its constraint factor
graph**. Components that share no variables can be emitted by separate decode sessions in
parallel; components that share variables are joined at a meeting node with an explicit
interface contract. The merge is accepted only if tier 2 proves the *combined* system feasible,
so a swarm cannot accept an invalid patch by majority vote.

The parallelism available is a property of the graph, not of the prompt: for a patch with
components of sizes `n_1 … n_k`, wall-clock is `max(n_i)` plus merge cost, against `Σ n_i` for
a single session. Two independent 300-token modules take 300 tokens of wall-clock instead of
600 — a 2× that no amount of model quality buys.

**Why it is not "multi-agent chat".** Free-form agents exchange natural language and merge by
consensus, which is a judge with extra latency. Here the decomposition is derived from the
constraint graph, the interface between agents is a typed contract (a set of shared variables
with domains), and the merge is a tier-2 feasibility check on the union. The swarm is a
parallel solver, not a conversation.

**Rejected alternatives.** Free-form multi-agent debate (no merge soundness; the 0.719 failure
distributed across `k` agents); speculative parallel sampling with a selector (accept rate is
again a judge); one agent with a longer context (sum, not max, and the global KV cost from §09
grows linearly).

**Falsifier.** 200 real multi-file patches: ≥ 2.5× wall-clock at equal accept rate, with zero
cross-component merge failures. If the average graph has fewer than 1.8 components, the
decomposition is not worth the interface machinery and the direction dies on the data rather
than on the model.

## 10.3 Null-space continual weight consolidation

**The limit.** Kestrel-4 is trained once and then adapted. Every adaptation — a customer's
schema dialect, a repository's conventions — either needs replay (tokens, latency, and a
retention policy) or risks catastrophic forgetting. The model has 21.25 B parameters and no
notion of "this subspace is already spoken for".

**The direction.** Protect the directions previous tasks actually used, and project every update
into the null space of that protected subspace:

```
ΔW ← P⊥ ΔW_raw,     P⊥ = I − U_k U_kᵀ
```

where `U_k` holds the top-`k` eigenvectors (per weight matrix) of the accumulated
Fisher/activation covariance from previous tasks. With `k = 8` per matrix, the storage cost is
`8 × (d_in + d_out)` per matrix — for the 16 layers' six matrices, 768 vectors of ≤ 4096 floats,
a few megabytes — and the per-step cost is two small matvecs per matrix, well under 1 % of the
forward pass.

This is strictly stronger than EWC, which adds a quadratic penalty `λ Σ F_i (θ_i − θ*_i)²`:
a penalty makes drift *expensive*, a projection makes it *zero* in the protected subspace. It
is strictly cheaper than replay, which pays tokens for the same guarantee. It is different from
LoRA, which adds capacity per task instead of protecting capacity in the base.

**The bet.** Code-domain adaptation is low-rank. If the top-8 directions per matrix capture the
task-relevant subspace — the same assumption that makes LoRA work — then projection preserves
old tasks exactly and leaves most of the gradient intact. If code adaptation is *not* low-rank,
the projection destroys the update and the direction fails, which is a clean, cheap experiment.

**Rejected alternatives.** EWC (soft, drifts, and the penalty coefficient is a hyperparameter
that trades old against new); full replay (tokens and privacy); LoRA-per-task (capacity grows
without bound; a router over adapters is a new failure mode); frozen trunk (no adaptation at
all).

**Falsifier.** 12 sequential code-domain tasks: forgetting on task 1 after task 12 ≤ 1 point,
while retaining ≥ 90 % of single-task gain, at < 5 % throughput cost. If forgetting exceeds 2
points, the subspace is too small and `k` must rise — at which point the update is mostly
projected away and the direction is not viable.

## 10.4 What this preview does not solve

The finding in §09.2 — **4.35 GB of KV at 1 M, 7.3× the always-on weights** — is not addressed by
any of the three directions. Sparse global attention makes compute constant in context and
leaves residency linear. A fourth direction exists (hierarchical summarization of global blocks
into a second-level index, so blocks can be evicted), but it is not designed here, and it is the
one that a production deployment would need first.

## 10.5 How to read a forecast

Three directions, three falsifiers, three costs: 400 GPU-h, 120 GPU-h, 80 GPU-h. Any of them
can be killed for less than 2 % of the Day 4 budget. That is the standard the constructive
chapters are held to, and a forecast should not be held to a lower one — the only thing a
forecast owes the reader is a way to find out it was wrong.
