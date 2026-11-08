# 09 — Training, serving, and what every ablation costs

Two budgets, one ledger. Training is priced in GPU-hours at a fixed MFU; serving is priced in
bytes streamed per step. Neither is a benchmark, and §9.5 says which numbers are measured,
which are modelled, and which are forecast.

## 9.1 Training

Assumptions: 989 TFLOP/s peak per accelerator (dense fp16/bf16), **MFU 0.34**, training FLOPs
= 3 × forward (one forward, two backward), `r_eff` as listed. At 4 k and `r = 1.9` that is
≈ 13.9 GFLOP/token, i.e. 3× the 4.65 GFLOP forward.

| Stage | Tokens | Context | `r_eff` | GPU-hours |
|---|---:|---:|---:|---:|
| Pretrain | 1,200 B | 4 k | 1.9 | 13,827 |
| Window + anchor | 150 B | 16 k | 1.9 | 1,747 |
| Long-context NoPE + memory | 60 B | 256 k | 2.0 | 704 |
| Clean-PR + CWM traces | 80 B | 32 k | 2.2 | 950 |
| SFT + JEPA + VEXX specs | 4 B | 32 k | 2.5 | 48 |
| On-policy GKD distillation | 30 B | 32 k | — | 1,650 |
| VEXX RL + 8× self-improvement | 2.9 B | — | — | 83 |
| Ablations + 12 % restarts | — | — | — | 4,521 |
| **Total** | **1,527 B** | | | **23,530** |

Plus **183 CPU core-hours** for copy-on-write verification: 600,000 rollout and
self-improvement steps at 1.1 s average environment time. That 1.1 s is itself the result of
the runtime: the same loop costs 90 s per step without CoW snapshots and in-flight
verification, an **81×** reduction that is the difference between 183 core-hours and 15,000.

**The budget fell by 1,537 GPU-hours (6.1 %) during the audit**, and no capability was removed.
The first cost model double-counted one workspace pass per token at every `r` (ACTIVE already
contains iteration 1) and over-charged the indexer by scoring 2,048 anchor tokens where the
model selects 512 + 256. Both errors were in the conservative direction, which is why they
survived: a cost model that is wrong against you looks like prudence.

At 1,527 B tokens for 2.07 B active parameters, the token-to-active-parameter ratio is **737:1**
(72:1 against stored parameters). Chinchilla-optimal is ~20:1 per active parameter, so this is
deliberately over-trained: the design bets that inference cost dominates, which is true for a
coding model with a 2.07 B active budget.

## 9.2 Serving: bytes, not FLOPs

Weight bytes at 4-bit are charged at **0.55 B/param** (weights plus scales and metadata), HBM
bandwidth at **3.35 TB/s**, and expert residency per layer. Always-on weights (everything except
routed experts and unselected subspaces) are **0.595 GB**; one expert is **2.60 MB**.

Weights only — §11 corrects this table for KV traffic, and the correction is large.

| Batch | Distinct experts/layer | GB/step | ms/step | tok/s (spliced) | Splice speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.9 | 0.84 | 0.251 | 5,080 | 1.28× |
| 16 | 80.2 | 3.92 | 1.171 | 17,633 | 1.29× |
| 64 | 199.0 | 8.86 | 2.645 | 29,627 | 1.22× |

The distinct-expert curve is `256 × (1 − (1 − 1/256)^(batch × 6))`, which saturates: at batch 64
the router has touched 199 of 256 experts, so per-token byte cost is approaching the worst case
rather than improving. That is the honest shape of MoE serving at large batch — batching
amortizes *weights*, not *expert diversity*.

The splice speedup **falls** with batch, which is the opposite of the first version's table.
The CPU cost of a splice is per token, not per step, so batching does not amortize it; at batch
64 a 600-token patch does 600 splices, not 9.4. With 4 cores splicing in parallel the overhead
is 2.4 ms, and the honest speedup is 1.22× rather than 1.38×.

Stored 11.69 GB at 4-bit, active 1.14 GB per token. At batch 1 the step is 28× memory-bound
(4.37 GFLOP is 8.8 µs of compute against 251 µs of streaming), which is what makes §07's
block-parallel decode argument plausible and what makes 0-DoF splicing profitable at low batch.

### The byte model omits KV traffic, and that is a 37 % error at batch 64

Two omissions in the table above. First, **residency**: at 1 M the global NoPE layers hold 4.29 GB
of latents per sequence, 7.3× the always-on weights. Second, and larger in practice, **traffic**:
the table charges expert streaming only, so it never pays for reading the KV cache. At 4 k and
batch 64 those reads are 2.55 GB per step against 8.86 GB of weights.

With KV traffic included, sustained decode is 3,801 / 11,747 / **18,784** tok/s at batch 1 / 16 /
64 — not 5,080 / 17,633 / 29,627. §11 prices the cache, fixes the residency (2.16 GB int8,
33.6 MB on chip, 1.57 MB paged per step at 1 M) and explains why the local window, not the
global cache, dominates a 4 k step at high batch.

One number from that chapter belongs here: without block sealing, the indexer re-reads 4.29 GB
of latents per decoded token at 1 M — 1.28 ms against a 251 µs step. The indexer would have cost
five times the model it indexes, and no FLOP table could see it.

## 9.3 The ablation programme, priced

| ID | GPU-h | What it can delete |
|---|---:|---|
| C2 MoE two-stage | 1,400 | the subspace machinery (93.8 % of stored params) |
| C3 local RoPE | 900 | the rotary path in 12 layers |
| C5 NoPE global | 400 | the NoPE/indexer design in 4 layers |
| C6 hypothesis codebook | 260 | conditioned retries |
| C9 self-training poison | 180 | unattended distillation |
| C4 Householder memory | 120 | the 203.6 M memory module |
| C10 chunked BPTT | 60 | the 64-token gradient window |
| C11 indexer prior | 40 | the only path from `w_iq` to the logits |
| C8 verifier soundness | 40 | the runtime's authority |
| C12 KV hierarchy | 20 | the 4.35 GB long-context cache |
| C7 0-DoF splicing | 6 | the splicing path |
| C1 absorption | 0.5 | absorption at serving |
| **Total** | **3,427** | 14.6 % of budget |

Ordered by cost ascending, the programme can delete the runtime's authority (C8, 40 GPU-h)
before it deletes the memory (C4, 120) before it deletes the MoE (C2, 1,400). That ordering is
the point: cheap experiments that can invalidate expensive ones run first.

## 9.4 What the audit changed

`tools/audit_v4.py` refuted 13 claims in the first draft of this document set. The ones that
moved a number:

| Claim | Measured before | After |
|---|---|---|
| memory transition has a negative eigenvalue | λ_min = 0.93 (a delta rule) | λ_min = −0.80 at β = 1.8 |
| memory layer preserves the residual stream | out/in RMS = 0.01 | 1.004 |
| indexer weights change the logits | 0.0 | 7.4e-03 |
| forward FLOPs match the documented decomposition | +7.1 % / +11.9 % | 0.00 % at every `r` |
| retry discount is `p(1−ρ)` | `p(1−0.85ρ)` | exact |
| a violating witness is not certified | `verified: True` | `False` |
| bounds survive repeated domain cuts | (0.05, 0.01) — empty | (0.05, 0.50) |
| workspace respects the configured window | 0.876 leaked | 0.000 |

Two of these were **mechanism failures**, not arithmetic: the memory's sign flip and the
indexer's influence existed only in prose. They are the same failure class as day 3.1's
RoPE — a parameter that is counted, initialized, and inert — and they are now pinned by a
harness that measures the mechanism rather than the formula.

## 9.5 Measured, modelled, forecast

| Claim | Status |
|---|---|
| 21,248,719,490 / 2,071,178,882 parameters | **measured** (meta-device instantiation + analytic, assertion [1]) |
| absorption 8.94e-08 (fp32) | **measured** (two forward passes, assertion [3]) |
| routed rank 3,072 collective | **by construction** (assertion [2]) |
| memory β ∈ (0, 2), max 1.52 at init | **measured** (assertion [6]) |
| 4.19 MB session state | **by construction** (2,097,152 floats fp16) |
| 4.37 GFLOP/token at r=1, 4k | **modelled** and cross-checked against an independent decomposition (audit A4) |
| 23,530 GPU-h | **modelled** at 0.34 MFU, 989 TFLOP/s peak |
| 0.84 / 3.92 / 8.86 GB per step | **modelled** at 0.55 B/param, 3.35 TB/s |
| 1.28× / 1.29× / 1.22× splicing speedup | **modelled** with 28 % deterministic steps, 16 µs/token on 4 cores |
| 2.16 GB int8 KV at 1 M, 33.6 MB on chip | **by construction** (`kv_sizes`, assertion [14]) |
| int8 latent cosine 0.999952 | **measured** (assertion [12]) |
| sealed index == dense summary | **measured** (assertion [13], max |Δ| 4.5e-08) |
| 18,784 tok/s at 4 k/batch 64 | **modelled** at 3.35 TB/s, KV reads included |
| FPR ladder 0.719 / 0.012 / 0.00216 / 0.000324 | first two **measured**, last two **derived** (`0.012 × 0.18`, `0.012 × 0.027`) |
| poison 0.049 % at `p_solve = 0.40` | **modelled** from those inputs |
| retry 1,746 → 52.4 GFLOP | **modelled** as 2.91 GFLOP × tokens re-sampled |
| 4.35 GB KV at 1 M | **by construction** from the config |
| any accuracy, pass@1, or loss figure | **not claimed** — no weights were trained |

## Rejected alternatives

- **8-bit weights.** Halves the byte budget's benefit: throughput halves, the splicing crossover
  falls from 6.4 % to 3.2 % deterministic fields, and 11.69 GB becomes 23.4 GB. 4-bit is chosen
  because serving is memory-bound at every batch size in the table.
- **FP8 training.** Would raise MFU from 0.34 to ~0.45 and cut the budget by a quarter, but the
  Householder state and the interval arithmetic both care about mantissa bits; the runtime's
  soundness is not worth 5,900 GPU-h. Revisit for a pure serving distillation.
- **Tensor parallelism inside a layer.** At 0.84 GB/step, batch 1 is already weight-streaming
  bound; TP would split the weights across devices and multiply communication for a step that
  is 251 µs long. Expert parallelism is the only sharding that pays here.
- **Longer pretrain (4 T tokens).** The 737:1 ratio is already 37× beyond Chinchilla; the
  marginal GPU-hour buys less than the same hour spent on the ablation programme.
- **Charging experts once per step (day 3.1).** A 16× under-count. See §01.3.
