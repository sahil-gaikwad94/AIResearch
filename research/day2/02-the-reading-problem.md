# 2 — The Reading Problem

> The cost of a coding session has three factors: how many tokens are unique, how much each costs to read, and how many are written. Day 1 attacked factor two and called it done. Kestrel-2 attacks all three, and the arithmetic says factor two was not even the largest.

---

## 2.1 The cost equation, written out

For a session of `T` turns on a repository of `P` tokens, with `d` new tokens per turn and `g` generated tokens per turn:

```
cost ≈ P · c_pre           (prefill the prefix, once, if the cache works)
     + Σ_t d · c_pre       (prefill what changed)
     + Σ_t g · c_dec       (decode)
     + Σ_t r · c_pre       (re-prefill after invalidation, r = invalidated tokens)
```

Day 1 minimised `c_pre` (the per-token prefill cost) with MLA and caching, and modelled `d = 8,000`. It assumed `r = 0` except at compaction, and it never priced `c_dec` at long context.

Kestrel-2's three moves map onto the three remaining terms:

| Term | Day 1 | Day 2 | Mechanism |
|---|---|---|---|
| `c_pre`, `c_dec` | O(context) reads | **O(1) reads** with exact selection | Sparse attention ([03](03-architecture.md)) |
| `d` | 8,000 tokens/turn | **~600 tokens/turn** | Append-only deltas ([04](04-context-protocol.md)) |
| `r` | full context at every compaction | **0** | Deltas never invalidate a prefix |

And the resulting budget is spent on `g` — reasoning tokens — which is the only term that buys capability.

---

## 2.2 Why reading is *bandwidth*, not FLOPs

The single most important table in this document. Decode one token at context length `n`, on an H100 (3.35 TB/s, 989 TFLOPS bf16 dense):

| Context | Day 1: cache streamed | Day 2: cache streamed | Ratio | Day 1 time/token | Day 2 time/token |
|---|---|---|---|---|---|
| 8 k | 168 MB | 36.9 MB | 4.6× | 1.27 ms | 1.24 ms |
| 64 k | 1.34 GB | 47.2 MB | 28× | 1.62 ms | 1.25 ms |
| 256 k | **5.37 GB** | 82.6 MB | **65×** | 2.82 ms | 1.26 ms |
| 1 M | **21.5 GB** | 224 MB | **96×** | 7.63 ms | 1.30 ms |

Read the two right-hand columns together, because that is the whole design. Day 2's decode time is **flat from 8 k to 1 M** — it is dominated by the 4.13 GB of weights at every length. Day 1's grows without bound, overtaking the 4.08 GB of weights at ~200 k.

That is what "bound by cache" actually means, and day 1 never printed it.

### The roofline, properly

| Operation | Arithmetic intensity | Bound by |
|---|---|---|
| Prefill of 800 new tokens against a 256 k prefix | ~1,100 FLOP/byte | **compute** |
| Decode, batch 1, 64 k context (Day 1) | 40 GFLOP / 5.3 GB = 7.5 FLOP/byte | **bandwidth** |
| Decode, batch 1, 1 M context (Day 2) | 26.6 GFLOP / 4.3 GB = 6.2 FLOP/byte | **bandwidth** |

The ridge point on H100 is ~295 FLOP/byte. **Decode is an order of magnitude below the ridge and stays there.** No amount of arithmetic optimisation helps; only moving fewer bytes does. This is why the entire design is about what gets read, and why "make the cache smaller" (day 1's instinct, and MLA already does it well) is a weaker lever than "read less of it".

---

## 2.3 The three ways to read less, and why two are wrong

**Option A: shrink the cache.** Day 1's answer. MLA gets to 20 KiB/token; the absolute floor for a 20-layer model with a 512-dim latent is ~11 KiB. Storage falls by 2×, reads fall by 2× — but they still scale with `n`. At 1 M, 11 KiB × 1 M = 11 GiB streamed per token. Not enough.

**Option B: compress the record into a fixed state.** Hybrid linear attention (3:1 KDA/GDN). Reads become O(1) — genuinely attractive — but every token is folded irreversibly into a state of fixed size. Kimi Linear shows this can *beat* full MLA on aggregate benchmarks with 75% less cache, so it is not a strawman. The problem is what it cannot do: reproduce an arbitrary earlier token exactly, on demand, forever. The 2026 split-prefill analysis ("What Attention Recalls and Recurrence Controls") shows hybrid models lean on their attention channel for exact recall and their recurrent channel for something else — which is an argument for keeping a real KV channel, not for deleting it.

**Option C: keep the record, read selectively.** Every token remains addressable; a learned indexer plus a deterministic structural index chooses what to read. Cost is O(selected), capacity is O(n). This is NSA/DSA's position, and the reported result is the one that matters: **matching or exceeding full attention**, not "acceptable degradation".

Kestrel-2 takes C, with one addition the general literature cannot make.

---

## 2.4 The code-specific addition: deterministic anchors

An indexer is a learned guess about which blocks matter. On code, a large fraction of that guess is *decidable before the model runs*, because code has a symbol table.

If the current window mentions `User`, the definition of `User` matters. If the file opens with `import numpy as np`, the import line matters for every later line. If a function is defined at line 40 and called at line 900, the definition is load-bearing for the call. None of this requires an indexer — it requires a parser.

ISSTA 2026 measured the effect of injecting this structure deterministically (call/inheritance/import tags, no embedding index, no reranker) into a grep-first agent's view: issue localisation improves, and — importantly for a design decision — **the benefit saturates at basic topology**, with denser graphs adding context cost and little gain.

So Kestrel-2 always attends to a **256-token anchor budget**: the current file's outline, plus the outlines of the eight most recently opened files. Selected by regex/AST at preprocessing time, not learned, not trainable away.

| Branch | Tokens read | Selected by | Guarantee |
|---|---|---|---|
| Window | 2,048 | position | exact, recent |
| Compressed global | n/64 | pooling | coarse view of everything |
| Selected blocks | 12 × 64 = 768 | learned indexer | exact on what it picks |
| **Anchors** | 256 | **parser** | **exact on definitions, always** |
| Total at 256 k | **7,168** | | vs 262,144 for full attention |

The anchor branch is the answer to the obvious objection to sparse attention ("what if the indexer misses the one block you needed?"). For the single most common case in code — *a definition referenced far from its use* — the parser cannot miss it.

---

## 2.5 What this costs

Sparse attention is not free, and the honest accounting is a table:

| Cost | Day 1 | Day 2 | Notes |
|---|---|---|---|
| Indexer parameters | 0 | 10.5 M (0.25%) | per sparse layer, 128-dim |
| Indexer compute at 1 M | 0 | 2% of attention FLOPs | one coarse scan per layer |
| Kernel complexity | one path | three paths + gather | the real engineering cost |
| Worst case | full attention | indexer misses a block | mitigated, not eliminated, by anchors |
| Prefill at 4 k | baseline | **1.6× more** | 20 layers × 1.8 iterations |

That last row is why pretraining runs at `R = 1` for most of the budget ([06 §6.4](06-training.md)): at short context, sparse attention buys nothing and the loop costs something.

And the crossover, which is the design's real claim:

| Context | Training FLOPs/token, v1 | v2 (R=2) | v1 / v2 |
|---|---|---|---|
| 4 k | 31 GFLOP | 50 GFLOP | **0.61×** (v2 costs 1.6×) |
| 16 k | 49 | 50 | 0.96× (parity) |
| 64 k | 121 | 52 | 2.3× |
| 256 k | 411 | 57 | **7.2×** |
| 1 M | 1,571 | 80 | **19.7×** |

**Kestrel-2 is more expensive per token below 16 k and dramatically cheaper above 64 k.** Since the product's whole premise is that agents read 200 k–1 M token repositories, that is the correct side of the crossover to be on — but it is a real bet on where the workload lives, and [09 B1](09-evals-and-ablations.md) tests it.

---

## 2.6 The consequence nobody plans for: 1 M context changes the harness

Once reading a million tokens costs 7 k of attention per layer, the constraints that shape agent harnesses invert:

| Harness decision | Rationale at 64 k, O(n) reads | Rationale at 1 M, O(1) reads |
|---|---|---|
| Summarise/compact the context | mandatory, cache dies | **counterproductive** — loses information and invalidates the prefix |
| Retrieve 3–5 files and hope | context is scarce | read the module; selection is cheap |
| Re-render a file after editing | normal | **never** — append the diff |
| Keep the system prompt short | token budget | it is 0.02% of the context |
| One agent per file | parallelism | one agent per repository |

Every row of that table is a habit formed under a constraint that Kestrel-2 removes. The architectural work is only half the change; [04](04-context-protocol.md) is the other half, and it is worth more.

---

## 2.7 What would falsify this

1. **If real agent sessions are mostly under 32 k tokens**, sparse attention buys nothing and costs 1.6× at 4 k. The design's premise collapses to day 1's.
2. **If the indexer's recall on repo-level tasks is materially worse than full attention** at equal training tokens, the third option is not real. This is B1, pre-registered with a 3-point switch rule.
3. **If anchors saturate in the wrong direction** — i.e. if adding deterministic structure does not help once a strong indexer is present — then the fourth branch is dead weight (0.25% of parameters, but kernel complexity).
4. **If the delta protocol cannot be made to work in a real harness** because tools rewrite files or reorder output, `d` stays at 8 k and the 10× gain in §2.1 disappears.

Each is measurable, and three of the four are cheap.

**Next:** [03 — Architecture](03-architecture.md).
