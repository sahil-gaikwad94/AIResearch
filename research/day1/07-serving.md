# 7 — Serving

> The model's job is to read fast and remember exactly. That makes serving a cache-management problem with a matmul attached, and it makes the KV cache — not the weights — the thing the deployment is sized around.

---

## 7.1 Memory, from the top

| Component | bf16 | FP8 | 4-bit |
|---|---|---|---|
| Weights (4.083 B) | 8.17 GB | 4.08 GB | 2.04 GB |
| KV per token | 40 KiB | **20 KiB** | — |
| KV @ 16 k | 0.63 GiB | 0.31 GiB | — |
| KV @ 64 k | 2.50 GiB | **1.25 GiB** | — |
| KV @ 256 k | 10.0 GiB | 5.00 GiB | — |

An 80 GB H100 with FP8 weights, ~4 GB of activations and workspace, and 4-bit-quantised **weights** loaded lazily is not the configuration to plan around. The realistic one: FP8 weights (4.08 GB) + 10 GB overhead + KV at 64 k. That leaves ~66 GB of cache, or **~50 concurrent sessions**, less 25% for fragmentation and scheduler slack → **~38**.

The same model with MHA (480 KiB/token) would fit **1.7** sessions. That comparison is the entire justification for MLA, expressed in a number an operations team can act on.

---

## 7.2 Prefix caching is the cost model

An agent re-sends its context every turn. Whether that is cheap depends entirely on cache hits.

| | No cache | Prefix cache |
|---|---|---|
| Unique tokens prefilled per session | 1.04 M (re-sent) | 190 k |
| Prefill FLOPs | 1.26 × 10¹⁷ | 2.30 × 10¹⁶ |
| CPU/GPU seconds at 350 TF/s | ~100 s | ~18 s |
| Input cost at $0.10 / M tokens | $0.104 | $0.019 |
| **Turn latency (prefill only)** | **3.7–7.4 s** | **0.2 s** |

A 5.5× compute saving and a 20–35× latency improvement on the prefill leg.

**The compaction trap.** Prefix caching requires the prefix to be **byte-identical**. A harness that summarises the context every five turns, reorders files, or re-renders a system prompt with a timestamp invalidates the cache and reverts to the uncached column. The design implication is a harness constraint, and it belongs in the model's documentation as much as in the serving code:

1. **Append-only between compactions.** New tool output goes at the end.
2. **Stable prefix ordering.** File contents are inserted in a deterministic order and never reshuffled.
3. **No timestamps or session IDs in the system prompt.** They are the most common silent cache-killer.
4. **Compaction is a planned cache flush**, not an incidental one. Do it once, when the context is full, and accept the one-time 3.7 s re-prefill.

Measured hit rate is the single number to instrument first. Below ~70%, the serving design's assumptions are wrong and hybrid attention becomes attractive again ([01 §1.7](01-thesis.md), falsifier 1).

---

## 7.3 Roofline: prefill is compute-bound, decode is not

On an H100 (989 TFLOPS bf16, 3.35 TB/s HBM), the ridge point is ~295 FLOP/byte.

| Phase | Arithmetic intensity | Bound by | Implication |
|---|---|---|---|
| Prefill (64 k, batched) | ~500–2,000 FLOP/byte | **compute** | use FP8 matmuls, large chunks, high MFU |
| Decode, batch 1 | 8.17 GFLOP / 4.08 GB ≈ **2 FLOP/byte** | **bandwidth** | weights are re-read per token; quantisation is a 2× speedup |
| Decode, batch 32 | ~64 FLOP/byte | bandwidth | batching scales linearly until ~256 |
| Decode, batch 256 | ~500 FLOP/byte | compute | beyond this, decode behaves like prefill |

Two consequences that shape the deployment:

- **Weights must be quantised for decode latency, not just for memory.** At 4-bit, 2.04 GB per step at 3.35 TB/s = 0.61 ms/token → ~1,600 tok/s single-stream ceiling; FP8 gives ~800. This is why on-device 4-bit is not a compromise but the correct choice for latency.
- **Decode throughput scales with batch**, so the economic optimum is to keep the GPU busy with many sessions rather than to serve one fast. Latency SLOs and cost pull in opposite directions here, and the curve is flat enough that 16–32 batch is the usual compromise.

---

## 7.4 A turn, end to end

Per turn, 8 k new tokens in, 600 out, 64 k context, dedicated H100, FP8, prefix cache warm, MTP-2 speculative decoding:

| Step | Work | Time |
|---|---|---|
| Prefill new content | 8 k tokens at ~4 × 10¹⁰ FLOPs/token | **0.20 s** |
| Decode | 600 tokens, ~2.4 tokens/step, ~400 steps/s | **1.5 s** |
| MTP speculation | acceptance ~0.7 at depth 2 → 1.6–1.8× | −0.6 s |
| **Model time per turn** | | **~1.1 s** |
| Cold prefill (first turn, cache miss) | 30 k tokens | +2 s, once |

A 20-turn session: **~25 s of model compute**, ~4 s of cold-prefill.

**And then the tests run.** A realistic agent session spends 5–15 *minutes* executing test suites, installing dependencies, and retrying. The model is 3% of wall-clock. That is worth stating plainly because it reframes where engineering effort should go — in production, the verifier is the latency bottleneck; in *training*, the model is, because RL needs millions of rollouts and cannot afford test execution to dominate.

---

## 7.5 Cost

Self-hosted, one H100 at $2.20/h:

| Scenario | Throughput | Cost per 1 M generated tokens |
|---|---|---|
| Batch 1, FP8 | ~800 tok/s | $0.76 |
| Batch 16, FP8 | ~6,000 tok/s | $0.10 |
| Batch 16, 4-bit | ~11,000 tok/s | $0.055 |

A 20-turn session (20 k generated, 190 k prefilled) costs **$0.002–0.006** of compute at batch 16 — versus $0.03 at API list prices. The gap is utilisation, not capability, which is why self-hosting a small model is attractive exactly when you can keep it busy.

**The crossover.** Below ~15% GPU utilisation, an API is cheaper than owning the hardware, before counting engineering time. Above ~40%, self-hosting wins by 3–5×. Any team deciding between the two should compute utilisation first.

---

## 7.6 Quantisation

| Format | Weights | Quality cost | Use |
|---|---|---|---|
| bf16 | 8.17 GB | — | reference, eval |
| FP8 (E4M3) | 4.08 GB | <0.3% on code benchmarks | serving default |
| 4-bit AWQ/GPTQ | 2.04 GB | 0.5–1.5% | edge, high-batch serving |
| KV FP8 | 1.25 GiB @64k | 0.2–0.5% | **default: halves the cache** |

**KV FP8 is the highest-leverage quantisation in this design**, because it doubles the concurrency ceiling for a sub-1% quality cost. Weight quantisation to 4-bit buys latency and edge deployment, not concurrency.

Calibrate on code, not web text. Activation outliers in code models concentrate in different channels (long identifiers, indentation runs), and an AWQ calibration set of prose can be 1–2 points worse than one of code.

---

## 7.7 Speculative decoding with MTP

The two MTP heads predict `t+2` and `t+3` from the trunk's final hidden state at zero extra forward cost. Used as draft tokens with the main head verifying, this yields:

- acceptance ~0.7 per draft at depth 2 on code (higher than prose, because code is structurally predictable: `)`, `:`, `return`, closing braces, repeated identifiers);
- effective **1.6–1.8×** decode speedup;
- zero quality change (verification is exact).

This is a latency feature, and it is worth ~0.6 s per turn out of 1.7 s — a 35% cut in the time a human waits. It does nothing for throughput, which is why it must not be sold as a cost saving.

---

## 7.8 Deployment shapes

| Shape | Hardware | Capacity | Use |
|---|---|---|---|
| **Server** | 1 × H100 80 GB, FP8 | ~38 concurrent 64 k sessions | the default unit; scale by adding replicas |
| **Node** | 8 × H100 | ~300 sessions | one prefix-cache domain per tenant |
| **Edge** | 1 × 24 GB GPU, 4-bit | 1 session, 32 k context, ~30 tok/s | privacy, offline, zero marginal cost |
| **Laptop** | 16–24 GB unified, 4-bit | 1 session, 16 k context, ~12–20 tok/s | acceptable for single-file work, not for repo-scale |

The edge shape is the one that changes product economics: 2.04 GB of weights plus 0.31 GiB of KV at 16 k fits in a 4 GB budget, which means *every* laptop and phone from 2027 onward can run this model locally. That is a distribution strategy, not just a deployment option.

---

## 7.9 Failure modes to design against

| Failure | Symptom | Fix |
|---|---|---|
| Cache thrash | prefill time 10× expected; GPU-bound on a workload that should be cache-bound | audit harness for prompt mutation; pin prefix ordering |
| Fragmentation | OOM at 40 sessions despite arithmetic saying 50 | paged attention with block size 16–32; preallocate per block size |
| Long-tail sessions | one 256 k session starves 200 short ones | separate pools by context length, or evict the longest-idle first |
| Prefill/decode interference | decode latency spikes during chunked prefill | chunked prefill with a decode-priority scheduler, or disaggregated prefill/decode |
| MTP acceptance collapse | decode only 1.1× faster than baseline | acceptance rate is task-dependent; disable per-request if it drops below 0.4 |

---

## 7.10 What this says about the architecture

If this chapter were the only input, the design would be:

- maximise KV efficiency over everything else → **MLA**, FP8 KV;
- keep recall exact → **full attention**;
- make decode latency cheap without hurting quality → **MTP + 4-bit**;
- keep the model small enough to saturate a single GPU with useful batches → **4 B, not 30 B**.

Which is the design. Serving was not an afterthought here; it is where three of the four main architecture decisions came from.

**Next:** [8 — Evals and ablations](08-evals-and-ablations.md).
