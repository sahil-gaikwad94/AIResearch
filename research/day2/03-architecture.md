# 3 — Architecture

> Twenty unique layers that can compute like sixty-eight. Four attention branches, three of which guarantee exactness where code needs it. Every number is produced by `tools/count_params_v2.py` and asserted against `kestrel2/model.py`.

---

## 3.1 The block

```
 ids ──► Embedding (131072 × 4096, tied)
           │
           ▼
    ┌──────────────────────────┐
    │ PRELUDE  × 2 (unshared)  │   dense-ish: window 2048 + compressed global
    └────────────┬─────────────┘
                 │
                 ├──────────────► KV projected ONCE here ◄─────┐
                 ▼                                             │
    ┌──────────────────────────┐                               │
    │ RECURRENT BLOCK  × 16    │  ←── repeated R times ────────┘
    │  + loop-index embedding  │      (R = 1 … 8, default 3)
    └────────────┬─────────────┘
                 ▼
    ┌──────────────────────────┐
    │ CODA  × 2 (unshared)     │
    └────────────┬─────────────┘
                 ▼
             RMSNorm ──┬──► lm_head (tied) ──► logits
                       └──► MTP heads ×2 (training only) ──► t+2, t+3
```

Three departures from a standard 2026 block, each defended below:

1. **Every layer is sparse**, with a *deterministic* anchor branch (§3.3).
2. **Depth is recurrent**, with a fixed-memory cache so the loop is free in cache terms (§3.5).
3. **The same 20 layers are all there is** — 4.126 B parameters, none of them duplicated.

---

## 3.2 Dimensions

| Symbol | Value | Note |
|---|---|---|
| `vocab` | 131,072 | tied, unchanged from day 1 ([04 §4.3](../day1/04-tokenizer.md)) |
| `d_model` | 4,096 | 32 heads × 128, no padding |
| `prelude` / `recur` / `coda` | 2 / 16 / 2 | 20 unique |
| `R` (recurrent steps) | 1–8, default 3 | effective depth 20–68 |
| `d_c` (kv latent) | 512 | `d_c/d_model = 0.125` |
| `d_qc` (q latent) | 512 | |
| `d_rope` | 64 | decoupled, rotated pre-cache |
| `d_ff` | 12,288 | `3.0 × d_model` |
| `window` | 2,048 | exact local |
| compression stride | 64 | one summary key per block |
| selection | 64-token blocks, top-12 | 768 tokens |
| anchor budget | 256 | parser-selected |
| indexer dim | 128 | hierarchical |
| context | 256 k native, 1 M extended | |

| Parameter group | Count | Share |
|---|---|---|
| FFN (per layer) | 150,994,944 | 84.1 % |
| Attention (per layer) | 27,787,524 | 15.5 % |
| Indexer (per layer) | 655,360 | 0.4 % |
| **Per layer** | **179,446,020** | |
| Body (20 layers) | 3,588,920,400 | |
| Embeddings (tied) | 536,870,912 | |
| Final norm + loop embeddings | 36,864 | |
| **Inference total** | **4,125,828,176** | 4.126 B |
| MTP (training only) | 33,562,624 | |
| **Training total** | **4,159,390,800** | 4.159 B |

The FFN share rose from 77% (day 1) to 84%, because there are half as many attention layers per unit of depth and MLA's query path is cheap. That is the correct shape for a model whose job is *manipulation* rather than *storage* — the loop literature's finding is that recurrence buys reasoning, not knowledge, so the capacity should sit where knowledge lives.

---

## 3.3 The four attention branches

Every layer computes:

```
attended = window(2048) ∪ compressed(n/64) ∪ selected(12×64) ∪ anchors(256)
```

| Branch | Cost at 256 k | Selected by | Failure mode |
|---|---|---|---|
| Sliding window | 2,048 | position | none — exact, but local |
| Compressed global | 4,096 | pooling (stride 64) | coarse; good for gist, not for copying |
| Selected blocks | 768 | learned indexer | may miss a block |
| Structural anchors | 256 | **parser** | stale outline if the file changed |

### Why all four

The window handles the thing attention is best at anyway (recent, local syntax) at a cost that does not grow. The compressed branch is what gives every layer a view of the *whole* prefix — this is the branch that stops sparse attention from being myopic, and NSA's ablation credits it with preserving global awareness. The selected branch is the learned retrieval. The anchor branch is the code-specific guarantee.

Total at 256 k: **7,168 tokens per layer, 2.7% of the context.** At 1 M: 19,456, or 1.9%.

### The indexer, and why it is hierarchical

A naive DSA-style indexer scores every block for every query. At 1 M tokens with 64-token blocks, that is 16,384 scores per query per layer — and the 2026 HISA paper identifies the resulting problem precisely: *the indexer itself becomes the bottleneck*, because it must scan the entire prefix even though the attention it feeds is sparse.

Kestrel-2 uses two stages, per HISA:

1. **Coarse:** score pooled block representations (one 128-dim vector per 64 tokens) against the query; keep the top 32 blocks. Cost: `n/64` dot products.
2. **Fine:** rescore at token granularity *inside* those 32 blocks; take top-12 blocks.

Two stages, ~16× less indexer work than a flat scan at 1 M, with the same token-level selection the attention kernel consumes. The indexer is FP8 (DSA's choice) and shares one query/key pair across head groups.

### Anchors: what exactly is always attended

| Source | Budget |
|---|---|
| Current file outline (def/class/signature lines) | 128 |
| Outlines of the 8 most recently opened files | 8 × 16 = 128 |

Produced by a regex/AST pass at preprocessing time and cached per file content hash. **Not learned, not gradients.** If the parser is wrong, the model sees a stale outline — a graceful failure that costs quality on one task, versus the indexer's failure mode, which is silence.

This is the branch that makes the "what if the indexer misses it" objection survivable for code specifically, and it costs 0.4% of parameters.

---

## 3.4 MLA, retained

Day 1's MLA argument survives intact, so the mechanism is unchanged: `d_c = 512` latent plus `d_rope = 64` decoupled rope key, fused into a single `W_DKV` that emits `c_kv ‖ k_pe`. The cache is 576 floats per layer per token, rotated before caching.

| Design | Params/layer | KV/token (FP16) | @256 k |
|---|---|---|---|
| MHA | 67.2 M | 640 KiB | 160 GiB |
| GQA-4 | 33.5 M | 80 KiB | 20 GiB |
| **MLA (512+64)** | **27.8 M** | **45 KiB** | **11.25 GiB** |

The domination still holds at the new width: fewer parameters *and* less cache than GQA-4. Kestrel-2 simply has half as many layers, so the total cache halves again: **11,520 B/token at FP8.**

Day 1's latency claim about `R` is worth restating because it is now load-bearing: the recurrent block's K/V is projected once, from the prelude output, and never recomputed. The cache is therefore 20 layers' worth regardless of `R` — the property asserted as test [4] in the reference implementation, where `r=1` and `r=4` produce byte-identical cache sizes.

---

## 3.5 The recurrent block: depth as a dial

```
h ← prelude(x)
(c_kv, k_pe) ← W_DKV(h)               # once
for i in 1..R:
    h ← h + e_i                        # learned per-iteration embedding
    for block in recur:                # 16 layers
        h ← block(h, kv=(c_kv, k_pe))  # queries refine; the record is fixed
h ← coda(h)
```

### Why the cache is fixed-memory, and what it costs

In Huginn/Ouro-style recurrence, each iteration recomputes K/V from the *updated* hidden state, so iteration `i` writes its own memory. That is more expressive and it is what the literature does.

Kestrel-2 deliberately does not, for a systems reason and a conceptual one:

- **Systems:** a prefix cache is shared across sessions, and sessions choose different `R`. If the cache depended on `R`, one repository would need up to eight cached copies — which destroys [04](04-context-protocol.md)'s entire economy.
- **Conceptual:** this model is a *reader*. Its attention output is a retrieval, and the thing worth refining across iterations is the **question**, not the record. A fixed key-value memory that gets asked better questions is exactly the right shape for the workload in [02](02-the-reading-problem.md).

The cost is real: later iterations cannot write new associations for *future* tokens to read. That is B3 in [09](09-evals-and-ablations.md), with a pre-registered fallback (recompute K/V per iteration and accept per-`R` cache variants).

### The exchange rate

The iso-depth result gives this design its price: `φ = 0.46` means one recurrence recovers about `2^0.46 ≈ 1.38×`... more usefully, it means a looped layer is *not* worth a real layer — it is worth roughly half of one in parameter terms, at equal compute.

So the honest claim is not "68 layers for the price of 20". It is:

> At equal **parameters** and equal **cache**, a 20-layer model with 3 recurrent steps behaves like something between a 20-layer and a 52-layer model, and costs 1.6× the FLOPs per token to do it.

Which is a good trade *if* FLOPs are the cheap resource. In agent serving they are: a turn is 0.5 s of model time inside a 30 s wall clock, and the scarce resources are VRAM and parameters.

### Training the dial

`R` is sampled per batch from `{1,2,3}` (mean 2.0) during the loop-adaptation phase, with an entropy-regularised exit gate trained on the last 10% of tokens. At serving, the harness sets `R` per step: 1 for a formatting fix, 2–3 for normal edits, 4–8 for a failing test that has already resisted two attempts. The gate can override upward.

That last sentence is the design's real product claim: **compute per decision, not per token.**

---

## 3.6 Sinks, norm, init

Unchanged from day 1 because they were right: 4 attention sinks per head (they make eviction and long-context drift safe), QK-norm, zero-init on `w_o` and `down_proj`, no biases, no dropout, RMSNorm ε = 1e-6.

One addition: **hyperconnections** between recurrent iterations are the iso-depth paper's single largest improvement (`φ` 0.46 → 0.65). Kestrel-2 keeps the cheap form — a learned per-iteration embedding added to the residual stream — and pre-registers the full hyperconnection as B4.

---

## 3.7 Multi-token prediction

Two MTP heads, kept for the same two reasons as day 1: gradient density, and free speculative decoding. On code, draft acceptance is high (closing braces, `return`, repeated identifiers), and decode is the part a human waits for.

At 1 M context, decode is 1.29 ms/token against 4.3 GB of weights — MTP's ~1.6× is the difference between 0.62 s and 1.0 s for a 500-token patch.

---

## 3.8 Rejected, with reasons

| Alternative | Why not | Revisit if |
|---|---|---|
| **Hybrid linear (3:1 KDA)** | Compresses the record irreversibly; 75% cache saving is real but Kestrel-2 already reads 96× less at 1 M without losing addressability | B2 shows KDA winning >3 points at equal KV |
| **Full attention** (day 1) | Cache-bound above ~128 k; the whole point of day 2 | Contexts turn out to stay under 32 k (B1) |
| **Recompute K/V per loop iteration** | Breaks the `R`-agnostic shared prefix cache | B3 shows fixed-memory loops lose >2 points |
| **Dense layers interleaved** | The compressed branch supplies the global view at O(n/64); dense layers would dominate cost at 1 M (4 of 20 dense = 4.85× more reads) | Recall at 1 M is poor without them |
| **MoE** | Same reason as day 1: rollout cost, routing stability, single-GPU serving | Not at this budget |
| **Diffusion / parallel decoding** | Tool-call reliability unproven; the win is latency, which MTP already gets | Never for this product |
| **Learned anchors** | A parser is free, deterministic and cannot be trained away | Anchor budget shows no benefit |

---

## 3.9 What would change at 10× the budget

1. **Width 5120, 24 unique layers, R up to 8** — a bigger dial with a bigger base.
2. **A learned compressor for the compressed branch** rather than mean pooling.
3. **Per-head sparse budgets** — some heads want 4 blocks, some want 40.
4. **Anchors extended to call-graph edges**, only if ISSTA's saturation result turns out to be an artefact of small repos.

**Next:** [04 — The context protocol](04-context-protocol.md).
