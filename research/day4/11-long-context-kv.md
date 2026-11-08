# 11 — Long-context KV: the 4.35 GB was never the working set

§09 reported a 4.35 GB KV cache at 1 M and called it the binding constraint. That number is
correct and the conclusion was wrong, because it priced *residency* as if the cache were read
whole. It is not: the global layers attend to 4,864 of 1,048,576 keys — **0.46 %**. Kestrel-4
already had sparse compute; the storage was dense, and the mismatch was the bug.

This chapter separates residency from traffic, then fixes both.

## 11.1 What is stored

One 512-float latent per token per global layer (NoPE, so no decoupled rotary key), and
`d_c + d_rope = 576` floats per token in each of the 12 local layers, but only for the window.

| Component | Per token | At 4 k | At 1 M (2²⁰) |
|---|---:|---:|---:|
| 4 global layers × 512 fp16 | 4,096 B | 16.8 MB | 4.295 GB |
| 12 local layers × 576 fp16 (window only) | 13,824 B | 56.6 MB | 56.6 MB |
| **Total fp16** | | **73.4 MB** | **4.35 GB** |

Against 137 GB for the same stack with uncompressed 32 × 128 MHA keys this is a 31.6×
compression — and it is 7.3× the 0.595 GB of always-on weights, which is what made it look like
the wall.

## 11.2 Residency is not traffic

Per decode step the global layers read `window + top-8 blocks + anchors = 4,864` latents per
layer: 9.96 MB at int8 for all four. The other 1.04 M latents are not touched. So at 1 M the
cache is **99.5 % cold by construction**, and the number that matters for a step is 10 MB, not
4.35 GB.

Two things follow, and they are the whole chapter:

1. **Precision and placement should be decided by traffic, not by total bytes.** The hot 33.6 MB
   (window + block summaries) is read every step and should be fast; the 2.14 GB cold tail is read
   a page at a time and can be stored at 8 bits, off-chip, or both.
2. **KV *reads* are part of the serving budget, and §09 omitted them.** At 4 k and batch 64 they
   are 2.55 GB per step against 8.86 GB of weights — **22 % of the byte traffic**. Correcting
   that changes the sustained throughput from 29,627 to 18,784 tok/s (−36.6 %).

That second correction is the one that stings: the serving table in §09 was a weights-only model,
and at large batch the local window is not free.

| Context | Batch | Local KV | Global KV | Index | KV total | Weights | KV share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 k | 1 | 31.5 MB | 8.4 MB | 0.1 MB | 0.04 GB | 0.84 GB | 4.5 % |
| 4 k | 16 | 503 MB | 134 MB | 1.0 MB | 0.64 GB | 3.92 GB | 14.0 % |
| 4 k | 64 | 2,013 MB | 537 MB | 4.2 MB | 2.55 GB | 8.86 GB | 22.4 % |
| 1 M | 1 | 31.5 MB | 10.0 MB | 16.8 MB | 0.06 GB | 0.84 GB | 6.5 % |
| 1 M | 64 | 2,013 MB | 638 MB | 1,074 MB | 3.72 GB | 8.86 GB | 29.6 % |

Note which term dominates at batch 64: **the 12 layer × 4 k local window, not the global cache.**
The long-context machinery is not the expensive part of a 4 k step; the windowed attention is.

## 11.3 Three fixes, each measured

### (a) int8 per-channel latents: 4.30 → 2.16 GB (1.99×)

Keys are quantized **per channel**, not per token, because latent channels carry outliers
(KIVI, KVQuant): one per-token absmax is dominated by a single channel and wastes the range.
Measured on 4,096 latents with two injected outlier channels (×40, ×12), using an asymmetric
8-bit code:

| Scheme | Cosine | Relative error |
|---|---:|---:|
| int8 per-channel | **0.999952** | 0.82 % |
| int8 per-token | 0.9993 | — |
| int4 per-channel | 0.9865 | — |

Scales are calibrated once on 128 k tokens and frozen, which is what makes encoding
streaming-safe: an online per-channel absmax would force re-encoding the whole cache every time
it grew. int4 is rejected at cosine 0.9865 — a 1.4 % direction error on every key compounds
across 4,864 keys per step, and the storage saving buys nothing once the cold tier is off-chip.

The first implementation of this quantizer clamped a 0–254 code range into `int8` and saturated
half the values; assertion [12] caught it at cosine 0.895. That is the audit harness doing its
job on code written the same day, which is the only honest way to report it.

### (b) Sealing the block index: 4.29 GB/step → 16.8 MB (256×)

The training forward recomputes the coarse and fine block summaries from **every** cached latent
on every call — `w_ip` over a pooled 64-token mean, `w_ik` over all 64 latents. That is O(n·d_c)
per step. At 4 k it costs 0.02 GB and nobody notices; at 1 M it re-reads 4.29 GB per layer set
per decoded token, which is **1.28 ms against a 251 µs step**. The indexer would have cost five
times the model it indexes.

Sealing a block when it fills — compute the summary once, store 128 floats, never look at those
64 latents again — reduces the per-step index to one dot product per sealed block:
16.8 MB and 5 µs at 1 M. Assertion [13] checks that a sealed summary is *exactly* the summary the
dense forward computes (max |Δscore| = 4.5e-08); audit A11 checks it on a different seed and size.
This is the difference between "1 M decode is possible" and "1 M decode is not implementable",
and it was invisible because every FLOP table priced the indexer analytically instead of
counting what it reads.

### (c) Cold tier: 33.6 MB on chip instead of 4.35 GB

With the window and summaries resident, a step needs `stream = read_keys − window = 768` cold
latents per layer — 1.57 MB at int8. Paging them from host DRAM over PCIe Gen4 x16 (25 GB/s
effective) costs 63 µs, against a 251 µs weight-streaming step:

| Batch | Stream/step | PCIe time | Step | Share |
|---:|---:|---:|---:|---:|
| 1 | 1.57 MB | 0.06 ms | 0.25 ms | 25 % |
| 2 | 3.15 MB | 0.13 ms | 0.32 ms | 39 % |
| 8 | 12.6 MB | 0.50 ms | 0.72 ms | 70 % |
| 64 | 100.7 MB | 4.03 ms | 2.64 ms | **152 %** |

So the cold tier is a **residency** instrument, not a throughput one. It is worth paying 25 % step
latency to hold 65 MB of HBM per 1 M session instead of 4.35 GB — that is 16× more concurrent
long-context sessions on one device. It is not worth it past batch 2, where paging dominates, and
it is impossible at batch 64. The honest rule: **tier for concurrency, not for speed.**

## 11.4 What it buys, and what it costs

At 1 M, batch 1:

| Design | Resident | On chip | Step traffic |
|---|---:|---:|---:|
| fp16, dense index | 4.35 GB | 4.35 GB | 4.29 GB (index) + 10 MB |
| int8, sealed index | 2.16 GB | 33.6 MB | 16.8 MB |
| + cold tier | 65 MB | 33.6 MB | 16.8 MB + 1.57 MB paged |

Three capability statements follow, each bounded by what was measured:

- **2× the context per device** at equal HBM (2.16 GB vs 4.35 GB per 1 M session), for a 0.82 %
  latent error that has not been shown to move any downstream metric (§11.6 is the test).
- **1 M decode is implementable at all.** Without sealing, the indexer alone exceeds the step
  budget by 5.1×; no amount of quantization fixes an O(n) read per token.
- **16× concurrency at batch ≤ 2** with the cold tier, at 25 % added step latency.

Corrected sustained throughput, KV traffic included (weights-only figures from §09 in brackets):

| Context | Batch 1 | Batch 16 | Batch 64 |
|---|---:|---:|---:|
| 4 k | 3,801 (5,080) | 11,747 (17,633) | **18,784 (29,627)** |
| 64 k | 3,790 | 11,643 | 18,518 |
| 1 M | 3,724 | 11,039 | 17,037 |

The 4 k row is the one to remember: the model was never 29,627 tok/s. It was 18,784, and the
missing third was the local window's KV reads.

## 11.5 Prior art: what was taken, and what was rejected

**Multi-head Latent Attention (DeepSeek-V2/V3, 2024).** Kestrel-4's global layer *is* MLA with the
rotary key removed: `w_dkv` down-projects to 512, `w_uk`/`w_uv` up-project, and the query is
absorbed into the latent space (`q_abs`). DeepSeek caches 576 numbers per token per layer because
its RoPE key cannot be folded into the latent; NoPE means we cache 512. Taken: the whole scheme.

**KIVI (Liu et al., 2024) and KVQuant (Hooper et al., 2024).** Per-channel key quantization
because key channels carry outliers, per-token value quantization because values stream. Taken:
per-channel, static calibration. Rejected: 2-bit. Our latents are *pre-projection* keys feeding a
512-wide up-projection; at int4 the direction error is 1.4 % and the up-projection does not average
it away.

**Quest (Tang et al., 2024).** Page-level selection by per-page min/max key bounds: two vectors per
page, no learned index. Rejected: our indexer already produces a learned 128-dim block score, and
Quest's bound is a heuristic for *unindexed* caches. Its useful idea — never read a page you did
not rank — is exactly what sealing implements.

**NSA (Yuan et al., 2025) and MoBA (Lu et al., 2025).** Block-sparse attention with a compression
branch that scores blocks, a selection branch that reads top-k blocks, and a sliding window.
Taken: the three-branch structure is what the indexer + window already is. Rejected: their
compression branch is trained end-to-end with the attention; ours keeps the indexer as a separate
128-dim head so block summaries can be sealed and cached (a trained compression branch changes
when the context changes and cannot be sealed incrementally).

**StreamingLLM (Xiao et al., 2023).** Keep attention sinks and a recent window, evict the rest.
Taken in spirit: every Kestrel layer has learned `sinks`. Rejected as a policy — eviction is lossy
and this design can afford exactness by tiering instead.

**H2O / SnapKV (Zhang et al., 2023; Li et al., 2024).** Keep heavy-hitter tokens by accumulated
attention. Rejected: a heavy-hitter set is query-independent and therefore wrong for retrieval;
our selection is per-query and re-ranked every step, and the cold tier means we never have to
throw a token away.

## 11.6 Falsifier: C12, three arms, 20 GPU-h

All three are evaluation-only; no retraining.

- **C12a — int8 latents.** Needle-in-a-haystack at 1 M plus a 200-task repo-retrieval set, fp16
  vs int8. Kill if exact recall drops by more than 0.5 points or pass@1 by more than 1 point.
  The measured latent cosine is 0.999952; the open question is whether attention is as forgiving
  as that number suggests.
- **C12b — sealed index.** Compare top-8 block selection against full recomputation on 10⁴
  queries at 1 M. Kill if block-selection agreement falls below 97 % (the summary is exact, so
  any loss is numerical, not structural — a lower number means the seal is wrong).
- **C12c — cold tier.** Measure step latency at batch 1, 2, 8 with the tail on host DRAM. Kill if
  added latency exceeds 30 % at batch 1 (predicted 25 %) or if the paging share at batch 2 exceeds
  50 % (predicted 39 %).

If C12a fails, the fallback is fp16 cold with the sealed index and tiering, which still gives
33.6 MB on chip and 16× concurrency; the precision win is separable from the placement win. That
is deliberate: each arm can fail alone.

## 11.7 What is still not solved

- **Batch 64 at 1 M.** 64 sessions × 2.16 GB = 138 GB. Neither resident nor paged fits: shard the
  sequences across devices, or accept batch 8 as the long-context operating point.
- **The local window at high batch.** 2.0 GB of the 2.55 GB KV traffic at 4 k/batch 64 is the
  12 × 4 k local window. The next lever is the window shape, not the global cache: halving it to
  2,048 would cut that traffic by 1 GB per step, and C3 (local RoPE) is the experiment that would
  tell us whether the window can shrink without losing syntax.
- **Prefill.** All of the above prices decode. A 1 M prefill still writes every latent once
  (4.3 GB of writes at fp16, 2.15 GB at int8) and the indexer costs one pass, not one dot product
  per step.
