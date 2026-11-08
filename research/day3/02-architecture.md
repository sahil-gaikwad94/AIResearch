# 2 — Architecture

> **20,485,187,584 parameters stored. 1,761,814,528 read per token.** A 4,096-token window, a 4.2 MB learned memory, 96 experts per layer of which 3 fire — and a context cost that stops growing at 4 k.

---

## 2.1 The shape

```
tokens ─► embed (131072 × 4096, tied)
             │
             ▼   × 16 blocks
   ┌─────────────────────────────────────────────┐
   │  x = x + MLA(window=4096, sinks=1)          │  exact recent memory
   │  x = x + NeuralMemory(d_m=512, r=256)       │ compressed long-term memory
   │  x = x + MoE(96 experts, top-3, d_ff=1024)  │ capacity, not compute
   └─────────────────────────────────────────────┘
             │
             ▼
        RMSNorm ─► lm_head (tied) ─► logits
```

Three layers of memory, one layer of capacity. No recurrence over depth, no sparse indexer, no anchors, no KV cache beyond the window.

---

## 2.2 Why a window and not long attention

The alternative — an attention mechanism that reaches 256 k tokens — was the previous design's answer, and it works, at a price: every token pays for the reach, and the cache grows linearly forever.

The alternative taken here is the one Titans (NeurIPS 2025) argues for explicitly: **attention is short-term memory, neural memory is long-term memory.** Attention is exact but bounded; the learned state is lossy but constant. Combine them and neither has to do the other's job.

The consequence is the striking number in [04](04-memory.md):

| Context | Attention cost | Memory cost | Total model FLOPs/token |
|---|---|---|---|
| 4 k | 3.22 GFLOP | ~0.02 GFLOP | **13.82 GFLOP** |
| 64 k | 3.22 GFLOP | ~0.02 GFLOP | **13.82 GFLOP** |
| 256 k | 3.22 GFLOP | ~0.02 GFLOP | **13.82 GFLOP** |

Training a 64 k-context model costs the same per token as a 4 k one, because the window never moved. That is what makes the memory-training stage in [06](07-training.md) affordable: 100 B tokens at 64 k for **1,213 GPU-hours** ([07](07-training.md)).

The honest counter-argument: a fixed 4 k window cannot do exact retrieval at 64 k the way attention can. The memory has to carry it, and the memory is lossy. Whether that is acceptable is the single most important experiment in the project (B2, [09](09-evals-and-risks.md)).

---

## 2.3 Dimensions and the parameter ledger

`d = 4096`, 16 blocks, 32 heads × 128, MLA latent 512 + RoPE 64.

| Component | Per layer | × 16 |
|---|---|---|
| Attention (MLA, 576 B/token/layer) | 27,787,552 | 444,600,832 |
| Neural memory (5 projections + `A`) | 10,616,832 | 169,869,312 |
| MoE (96 × 12,582,912 + router) | 1,208,352,864 | 19,333,645,824 |
| 3 pre-norms | 12,288 | 196,608 |
| **Body** | **1,246,769,536** | **19,948,312,576** |
| Embeddings (131,072 × 4096, tied) | | 536,870,912 |
| Final norm | | 4,096 |
| **Total** | | **20,485,187,584** |

Active per token — the weights actually read:

| | Active |
|---|---|
| Attention | 444,600,832 |
| Neural memory | 169,869,312 |
| MoE: 3 of 96 experts + router | 1,145,070,144 |
| Norms | 196,608 |
| Embeddings / head | 536,870,912 |
| **Active total** | **1,761,814,528** (8.60 %) |

Both numbers are asserted against `torch`'s own count to the parameter — `20,485,187,584` on both sides. They disagreed by 2,162,688 on the first run, which is how the missing memory matrix `A` and the third pre-norm were found.

---

## 2.4 The activation ratio is the design

`1,761,814,528 / 20,485,187,584 = 8.6%`. One routed expert fires per 32 stored.

This is not a compromise forced by budget; it is the configuration the 2026 MoE study says to pick. Over 2,000 pretraining runs up to 6.6 B, quality **improves monotonically with total parameters even at activation ratios of 128:1**, and the optimal *expert* size is nearly invariant to total size — it depends on **active** parameters ([03](03-moe.md)).

So the model is deliberately two things at once:

- a **1.76 B** model, in every respect that costs time — FLOPs, latency, KV, rollouts, training;
- a **20.5 B** model, in the only respect that costs nothing — storage.

---

## 2.5 Three memory tiers, three jobs

| Tier | Mechanism | Holds | Bytes per session | Cost per token | Fails by |
|---|---|---|---|---|---|
| **L1** | window attention | last 4,096 tokens, exactly | 36.9 MB | 3.22 GFLOP | losing the distant past |
| **L2** | delta-rule fast weights | everything older, compressed | **4.2 MB** | 0.02 GFLOP | blurring detail |
| **L3** | versioned store (external) | claims with provenance | disk | 0 FLOPs | never — it revalidates |

L1 and L2 are in the forward pass. **L3 is not a neural mechanism at all**, and that is deliberate: the evidence in [01](01-the-validity-problem.md) says the failure is a *bookkeeping* failure, and bookkeeping does not improve with a larger matrix.

Sizes, for scale:

```
dense KV at 256k, 16 layers, FP8     2.36 GB
this model's L2 state, FP16          4.2 MB      <- 562x smaller
this model's L1 window, FP8         36.9 MB      <- fixed
```

The 4.2 MB is the number to remember. It does not change with context length, with turn count, or with session length. A 500-turn session and a 5-turn session hold the same state.

---

## 2.6 Config, annotated

[`kestrel3/config.json`](kestrel3/config.json) is canonical. Fields that carry a decision:

```jsonc
"attention": { "window": 4096, "d_c": 512, "d_rope": 64, "sinks_per_head": 1 },
"memory":    { "d_m": 512, "rank": 256, "read_before_write": true, "eta": 0.05 },
"moe":       { "experts_per_layer": 96, "active_experts": 3,
               "expert_d_ff": 1024, "shared_expert": false,
               "routing": "dropless_top_k",
               "balancing": "auxiliary_loss_free_bias" }
```

Three of those are contested choices with abandoned alternatives listed in [03](03-moe.md) and [04](04-memory.md): no shared expert, 1:32 activation, and dropping the KV cache entirely in favour of fast weights.

---

## 2.7 What the shape is betting on

| Bet | Wins if | Loses if | Test |
|---|---|---|---|
| Fixed 4 k window + learned memory ≥ long attention at 64 k | B2 | B2 | recall + repo tasks |
| 1:32 activation keeps quality | B1 | B1 | bpb vs 1:8 |
| 20.5 B of experts beats 4 B dense at equal active FLOPs | B1 | B1 | 6 ablations, 4 scales |
| L3 store with validity keys beats text memory | B3 | — | [01 §1.5](01-the-validity-problem.md) |
| Search + verification substitutes for active parameters | B5 | B5 | pass@1 vs 4× larger |

Five bets, five pre-registered tests, 2,500 GPU-hours for the model ablations and a week of harness work for the store. Detailed in [09](09-evals-and-risks.md).

**Next:** [03 — The mixture of experts](03-moe.md).
