# 8 — Serving

> The deployment unit is no longer the session. It is the repository. A single 400 k-token prefix, cached once and shared by every agent working on that commit, is what makes 64 concurrent sessions fit on one GPU.

---

## 8.1 Decode is flat, and that is the point

One decode step on an H100 (3.35 TB/s, FP8 weights, batch 1):

| Context | Cache streamed | Weights streamed | Total bytes | Time/token | Tokens/s |
|---|---|---|---|---|---|
| 8 k | 34 MB | 4.13 GB | 4.16 GB | 1.24 ms | 806 |
| 64 k | 44 MB | 4.13 GB | 4.17 GB | 1.25 ms | 802 |
| 256 k | 77 MB | 4.13 GB | 4.21 GB | 1.26 ms | 796 |
| 1 M | 209 MB | 4.13 GB | 4.34 GB | 1.30 ms | 771 |

**From 8 k to 1 M, decode slows by 5%.** Day 1's equivalent at 1 M streams 21.5 GB and takes 7.63 ms — 5.9× slower, and diverging.

That flatness has a direct operational consequence: context length stops being a capacity planning variable. A 1 M-token session costs the same VRAM-per-token as a 64 k one, so the question "how long a context can we afford?" disappears and is replaced by "how many distinct prefixes do we hold?".

---

## 8.2 Memory: the shared-prefix arithmetic

An 80 GB H100, FP8 weights (4.126 GB), ~8 GB of workspace and activations:

| Configuration | Shared | Per session | Sessions |
|---|---|---|---|
| 400 k repo prefix + 64 k deltas | 4.29 GiB | 0.70 GiB | **64** (≈61 GiB used, 24% headroom) |
| 400 k prefix + 256 k deltas | 4.29 GiB | 2.81 GiB | 20 |
| No sharing, 256 k each | — | 2.81 GiB | 23 |
| **Day 1** at 256 k, extended | — | 5.00 GiB | 13 |
| **Day 1** at 64 k (native limit) | — | 1.25 GiB | 38, but capped at 64 k |

The row that matters is the first. Day 1's 38 sessions each carried a private 64 k copy of overlapping repository content; Kestrel-2 holds one 400 k prefix and 64 divergent tails.

### Why the prefix can be shared at all

Because it is a pure function of `(repository, commit)`. That is a protocol requirement ([04 §4.6](04-context-protocol.md)), not an optimisation: any session id, timestamp, user id or randomised file ordering in the prefix makes the KV unshareable. The tail is per-session and copy-on-write.

### Tiers

| Tier | Capacity | Holds | Latency |
|---|---|---|---|
| L1 (HBM) | 80 GB | hot sessions + active tails | 1.3 ms/token |
| L2 (host DRAM) | 1–2 TB | repo prefixes for the last ~100 commits | ~40 µs to page a block |
| L3 (NVMe) | 8 TB | prefixes for the last ~2,000 commits | ~200 µs |

SGLang's reported numbers for this exact pattern: multi-turn prefix hit rates near **98%** with an L3 tier, and effective input throughput of **145.5 K tokens/s versus 9.4 K** without it — a 15.5× difference, from a cache.

---

## 8.3 A turn, end to end

800 new tokens in, 600 out, 256 k context, `R = 3`, MTP-2:

| Stage | Work | Time |
|---|---|---|
| Prefill 800 new tokens | 800 × (8.25 GFLOP + sparse attention) ≈ 8.5 TFLOP | **25 ms** |
| Decode 600 tokens at `R = 3` | 600 × 19.15 GFLOP | 600 × 1.26 ms = **0.76 s** |
| MTP-2 (acceptance ~0.7) | 1.6× fewer steps | −0.29 s |
| **Model time per turn** | | **~0.5 s** |
| Test execution | pytest, install, retry | 10–60 s |

A 20-turn session: **~10 s of model compute**, against day 1's ~28 s, and a wall-clock that is 97% verifier. The model is no longer the latency problem — which is why the remaining engineering effort belongs in the sandbox, not the kernel.

### Cost

| | Day 1 (64 k) | Day 2 (256 k) |
|---|---|---|
| GPU-seconds, one session on a dedicated GPU | ~38 s | **~16 s** |
| Cost per session at $2.20/GPU-h | $0.023 | **$0.0095** |
| Cost per 1 M generated tokens, batch 16 | $0.062 | **$0.048** |
| Cost per 1 M generated tokens at 1 M context | cannot run | **$0.048** |

(16 s = 20 turns × [25 ms prefill + 754 ms decode], decode being bandwidth-bound at 1.26 ms/token. Day 1's 38 s is the same computation at 64 k, where its decode is 1.62 ms/token and its prefill is ~0.9 s per turn at matched efficiency — day 1's own document quotes 200 ms by assuming FP8 prefill near peak.)

At batch 16 the weights are read once per step for 16 tokens, so aggregate throughput reaches ~12,800 tokens/s and the marginal cost of generation falls to roughly a twentieth of an API's list price. As in day 1: the gap is utilisation, not capability.

---

## 8.4 Prefill is now the cheap half

| | Day 1 | Day 2 |
|---|---|---|
| Prefill per turn | 200 ms (8 k tokens) | **24 ms** (800 tokens) |
| Cold prefill of a 400 k repo | n/a (64 k limit) | 400 k × 8.25 GFLOP ≈ 3.3 PFLOP → **9.4 s**, once per repo, then cached |
| Re-prefill after an edit | 3.7 s (compaction) | **0** (delta) |

That 9.4 s is paid once per repository per commit, and amortised across every session and every user. It is a build step, and it should be treated as one: scheduled, cached, invalidated by commit, and never on the critical path of a user's first turn.

---

## 8.5 Quantisation

| Format | Weights | Effect |
|---|---|---|
| bf16 | 8.25 GB | reference, eval |
| **FP8 (E4M3)** | 4.13 GB | serving default; <0.3% quality cost |
| 4-bit AWQ | 2.07 GB | edge; 0.5–1.5% cost |
| **KV FP8** | 11,520 B/token | default; halves the cache |

KV FP8 is again the highest-leverage choice: it doubles sessions per GPU for sub-1% quality. Weight 4-bit buys *latency* (weights-bound decode) and edge fit, not concurrency.

One day-2-specific note: **the indexer must stay in a precision where ranking survives.** FP8 is fine for the coarse block scores (they are compared, not summed), but quantising the fine re-scoring to 4-bit flips close top-k decisions and costs recalls. Indexer stays FP8; there is no reason to go lower, since it is 0.4% of parameters.

---

## 8.6 Speculative decoding

MTP-2 still pays: on code, draft acceptance is high and verification is exact. At 1 M context the win is 0.76 s → 0.47 s per turn — 40% of the time a human waits.

It does nothing for throughput, and should not be sold as a cost saving. It is a latency feature whose value grows with context, because decode time is the only part of the turn that is not already negligible.

---

## 8.7 Deployment shapes

| Shape | Hardware | Capacity | Use |
|---|---|---|---|
| **Repo server** | 1 × H100 80 GB, FP8 | 1 shared 400 k prefix + 64 sessions | the default unit |
| **Node** | 8 × H100 | 8 prefixes, ~500 sessions | one tenant per GPU, shared L3 |
| **Edge** | 1 × 24 GB, 4-bit | 1 session, 128 k context, ~60 tok/s | privacy, offline |
| **Laptop** | 16 GB unified, 4-bit | 1 session, 64 k, ~20 tok/s | single-file work |

The edge shape is less comfortable than day 1's (2.07 GB of weights, but 64 k of context is 0.70 GiB and laptops have little to spare), and the honest statement is that Kestrel-2's advantage is a *server* advantage. If the product is on-device, day 1's smaller-context model is the better fit.

---

## 8.8 Failure modes

| Failure | Symptom | Fix |
|---|---|---|
| Prefix not shareable across sessions | L1 fills with near-duplicate prefixes | enforce `(repo, commit)`-pure prefix; hash and assert |
| Indexer mask recomputed per request | prefill 3× expected | cache the selection mask alongside the KV blocks; it is a function of the prefix |
| Tail fragmentation | OOM at 50 sessions when arithmetic says 64 | paged attention, block size 32; preallocate by block size |
| L2 thrash | TTFT spikes when two repos alternate | pin the top-2 prefixes; evict by commit age |
| Anchor outlines stale after edit | worse patches on recently-edited files | outline keyed by content hash; invalidate on write |
| MTP acceptance collapse at deep `R` | decode only 1.1× faster | acceptance is `R`-dependent; disable per-request when <0.4 |

The second row is new and easy to miss: **the sparse selection mask is part of the cache**, not a per-request computation. Recomputing it on every prefill throws away most of the indexer's cost advantage.

---

## 8.9 What the serving model says about the architecture

If serving were the only input, the design would be: maximise addressable context per byte (sparse reading over an exact cache), keep decode weights-bound (quantise weights, keep the cache small and rarely read), make prefixes shareable (append-only, `(repo, commit)`-pure), and spend the freed FLOPs on depth (the loop).

Which is the design. Three of Kestrel-2's four architecture decisions were derived from this chapter rather than from a paper.

**Next:** [09 — Evals and ablations](09-evals-and-ablations.md).
