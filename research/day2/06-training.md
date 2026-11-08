# 6 — Training

> 1,275 B tokens, 43,097 GPU-hours — *less* than day 1 while training a longer-context, deeper-at-runtime model. The reason is worth reading: sparse attention makes the expensive stages cheap, and depth is only paid for when it is used.

---

## 6.1 The budget

| Stage | Tokens | Context | R | GPU-hours |
|---|---|---|---|---|
| Pretrain phase 1 | 800 B | 4 k | 1 | 17,872 |
| Pretrain phase 2 | 160 B | 16 k | 1 | 3,599 |
| Pretrain phase 3 | 60 B | 64 k | 1 | 1,387 |
| **Loop adaptation** | 200 B | 32 k | 1–3 | 8,173 |
| Long context | 12 B | 256 k | 2 | 553 |
| Mid-train, event-log | 40 B | 128 k | 2.5 | 2,108 |
| SFT | 3 B | 128 k | 3 | 187 |
| RL (generation-bound) | 24 B generated | 128 k | 1–6 | 3,000 |
| Ablations (26 arms) | 208 B | 8 k | 1 | 1,600 |
| Restarts (12%) | — | — | — | 4,618 |
| **Total** | **1,275 B** | | | **43,097** |

At $1.5 / $2.2 / $3.0 per GPU-hour: **$65k / $95k / $129k**.

Sanity check printed by the cost model: the same efficiency assumption applied to day 1's shape gives **24,508 GPU-h** for 1 T tokens at 4 k, against day 1's stated 24,238 — a 1.1% difference from the attention term. The two budgets are measured on the same ruler.

### Why this is cheaper than day 1 despite doing more

Two effects, and they are worth separating because only one of them is architectural:

1. **Sparse attention makes long context cheap.** Day 1's 64 k stage cost 3,846 GPU-h for 40 B tokens. Kestrel-2's costs 1,387 for 60 B — and its 256 k stage (12 B tokens) costs 553. Long-context training went from 8% of the budget to 5% while reaching 4× the context.
2. **Depth is only paid for when it is sampled.** 63% of tokens train at `R = 1`, where the model is simply a 20-layer transformer. The loop tax applies to the last third.

The second is the more important principle: **a runtime dial should not be paid for at training time unless you are turning it.**

---

## 6.2 Data

Unchanged from day 1 in composition ([05 §5.1](../day1/05-training.md)) — 42% code, 18% code-adjacent, 15% natural language, 10% structured, 8% math, 4% multilingual, 3% verifier-filtered synthetic — with one addition and one reprioritisation:

- **+ event-log replay (mid-train, 40 B tokens):** commit histories replayed as read/patch streams in the [04](04-context-protocol.md) format. Supervision is free and exactly matches serving.
- **+ anchor supervision:** the parser's outlines are included in the input for 100% of code samples, so the model learns to *use* anchors rather than to ignore a decoration.

The event-log data deserves emphasis because it is nearly free and buys the single largest systems gain in the project:

```
read(f.py) ──► patch(f.py, 40 tokens) ──► exec(pytest) ──► failure ──► patch ──► pass
```

Filtering: only keep sequences where the final state passes, and only where the patch stream is ≤ 8 patches (longer chains are rare and mostly noise). Expected yield: ~40 B tokens from a 2 M-commit corpus, which is why 40 B is the mid-train allocation.

---

## 6.3 Phase 1–3: the dense-to-sparse warmup

Sparse attention cannot be trained from random initialisation without care, because the indexer has no signal about which blocks matter until attention has one. Kestrel-2 uses a three-step warmup, modelled on DSA's approach:

| Step | Tokens | Attention | Objective |
|---|---|---|---|
| 1. Dense | 0 – 40 B | full causal (mask disabled) | standard LM loss |
| 2. Distil | 40 – 60 B | dense forward, sparse forward in parallel | LM loss + KL(indexer scores ‖ dense attention mass) |
| 3. Sparse | 60 B + | sparse | LM loss + 0.01 × load-balancing on block selection |

The distillation target is the **dense attention mass per 64-token block**, averaged over heads — exactly the quantity the indexer must approximate. This is a cheaper and better-conditioned signal than trying to learn selection from the task loss alone, because the task loss gradient through a top-k is zero for unselected blocks.

That last point is the subtle one: **top-k is not differentiable in its selection**. Gradients only flow to blocks that were chosen, so an indexer trained purely on LM loss can lock onto a bad pattern early and never receive a signal to leave it. The distillation loss gives every block a target.

---

## 6.4 Loop adaptation

The last 200 B tokens of pretraining turn a 20-layer model into a model with a depth dial. Adapting Ouro's recipe to this budget:

- **Sample `R` per batch** from `{1, 2, 3}` (mean 2.0), so every depth is trained.
- **Loss at every depth**, weighted by an exit distribution `p(R)` trained with a uniform-prior entropy term:
  `L = Σ_R p(R) · L_R  −  β · H(p)`,  β = 0.01
- **Exit gate:** a two-layer head on the pre-coda hidden state predicting whether another iteration lowers loss. Trained in the last 10% of the phase, with the trunk frozen — the gate is a cheap calibrator, not a co-adapted component.
- **Loop embedding:** a learned per-iteration vector added to the residual stream (`e_i`, 8 × 4096 = 32,768 parameters). This is the cheap half of the hyperconnection result; the full version is B4.

**Why the phase is only 200 B tokens (16%).** Because the model is already a competent 20-layer transformer after 800 B, and the loop's job is to reorganise computation, not to learn language. Ouro trains looped from scratch on 7.7 T tokens; at a 1.3 T budget, adaptation is the affordable version — and B3 tests whether it costs anything.

**Risk, stated:** a model adapted to loop late may not use the extra depth as well as one trained looped from the start. The mitigation is the per-depth loss (every depth must work) rather than only the final one.

---

## 6.5 Optimiser: Muon, then AdamW

| Phase | Optimiser | Why |
|---|---|---|
| Pretrain (1,220 B) | **Muon** on 2-D hidden weights, AdamW on embeddings/norms/head | ~2× pretraining efficiency; used by Kimi K3 and DeepSeek V4 |
| Loop adaptation onward | Muon (trunk) + AdamW (gate) | the gate is not a matrix worth orthogonalising |
| SFT, RL | **AdamW** | the advantage reverses after large pretraining budgets |

Hyperparameters:

| | Muon | AdamW |
|---|---|---|
| LR (peak) | 0.02 (orthogonalised step), matched to AdamW by update RMS | 2.5 × 10⁻⁴ |
| Momentum | 0.95, Nesterov | β₁ 0.9, β₂ 0.95 |
| Newton–Schulz steps | 5 | — |
| Weight decay | 0.1 on matrices | 0.1 on matrices |
| Grad clip | 1.0 | 1.0 |
| Batch | 4 M tokens, ramped from 0.5 M over 5 B | same |

**The reversal is the most important line in this section.** *Pretraining Data Scale Reverses the Muon–AdamW Fine-Tuning Gap* (Nov 2026) reports that past a large pretraining budget, AdamW-pretrained models fine-tune better than Muon-pretrained ones, with the crossover arriving at smaller token/parameter ratios as models grow. A 4 B model trained on 1.2 T tokens (309 tokens/param) sits near that boundary.

So the design does the obvious safe thing — Muon for the base where the gain is measured, AdamW from SFT where the reversal is measured — and pre-registers B6 to check whether even the *pretraining* advantage holds at this scale and budget, because the honest answer is that nobody has published this exact configuration.

---

## 6.6 What the loop does to stability

Two failure modes are specific to this architecture, and neither exists in a dense model:

**Indexer collapse.** The learned selection can degenerate to a position prior: always select block 0, or always the most recent 12 blocks. Training loss barely notices, because the window and compressed branches cover for it — but recall dies silently.

*Detection:* measure the entropy of the *selected block offset distribution* (normalised by context length) on a fixed probe set every 2 B tokens. A healthy indexer's selections spread across the prefix; a collapsed one concentrates at 0 or at `n-1`.
*Response:* if entropy falls below 25% of its value at 100 B tokens, raise the distillation weight and re-warm.

**Loop divergence.** Repeated application of the same 16 layers can amplify certain directions in the residual stream. With `R = 3` and no normalisation between iterations, the norm can grow geometrically.

*Detection:* track `‖h‖` after each iteration on a fixed batch; a healthy model plateaus, a diverging one grows >5% per iteration.
*Response:* the loop embedding plus pre-norm residual keeps this bounded in practice; if not, a per-iteration RMSNorm on the residual stream is the fallback (adds 16 × 4096 parameters).

Both are logged from step 0 of the loop phase, because both are invisible in the loss curve until they have already cost a week.

---

## 6.7 General stability playbook

Unchanged from day 1, because loss spikes are the same problem: detect on a smoothed derivative (3 consecutive steps above +0.2 nats from the running median), skip the batch, roll back to the last checkpoint with the data order advanced, then lower LR 20% for 500 steps.

Structural defences already in the design: QK-norm, zero-init residual projections, no biases, fp32 master weights, attention sinks.

---

## 6.8 Infrastructure

- **128–512 H100/H200s**, FSDP/ZeRO-3. At 4.1 B, no tensor parallelism is needed.
- **Sequence parallelism for the 256 k stage.** Activations, not weights, are the memory constraint. Activation checkpointing on every recurrent iteration except the last.
- **The recurrent block is cheap to checkpoint** — 16 layers of state, reused `R` times — so the loop phase can use a smaller microbatch than phase 1 with the same memory.
- **Data pipeline versioned and hashed** into the run manifest; a mismatch aborts.
- **Checkpoint every 30 min**, keep last 5 plus every 10th.

---

## 6.9 What would make me restart

1. Indexer selection entropy collapsing before 200 B tokens and not recovering with re-warm → the sparse objective is wrong, not the hyperparameters.
2. Loop adaptation showing *worse* loss at `R = 1` after the phase than before it → the adaptation is forgetting, and the loop must be trained from the start instead.
3. bpb at 4 k tracking more than 3% worse than the 300 M proxy's prediction by 100 B tokens → the μP transfer failed.
4. Attention sinks saturating (all mass on sinks in >10% of heads) → the sparse mask is starving some head of context it needs.

**Next:** [07 — Post-training](07-post-training.md).
