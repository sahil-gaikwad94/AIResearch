# 09 — Training, serving, and what every ablation costs

Two budgets, one ledger. Training is priced in GPU-hours at a fixed MFU; serving is priced in
bytes streamed per step. Neither is a benchmark, and §9.4 says which numbers are measured,
which are modelled, and which are forecast.

## 9.1 Training

Assumptions: 989 TFLOP/s peak per accelerator (dense fp16/bf16), **MFU 0.34**, training FLOPs
= 3 × forward (one forward, two backward), `r_eff` as listed. At 4 k and `r = 1.9` that is
≈ 15 GFLOP/token, i.e. 3× the 5.0 GFLOP forward.

| Stage | Tokens | Context | `r_eff` | GPU-hours |
|---|---:|---:|---:|---:|
| Pretrain | 1,200 B | 4 k | 1.9 | 14,924 |
| Window + anchor | 150 B | 16 k | 1.9 | 1,884 |
| Long-context NoPE + memory | 60 B | 256 k | 2.0 | 760 |
| Clean-PR + CWM traces | 80 B | 32 k | 2.2 | 1,027 |
| SFT + JEPA + VEXX specs | 4 B | 32 k | 2.5 | 52 |
| On-policy GKD distillation | 30 B | 32 k | — | 1,650 |
| VEXX RL + 8× self-improvement | 2.9 B | — | — | 83 |
| Ablations + 12 % restarts | — | — | — | 4,686 |
| **Total** | **1,527 B** | | | **25,067** |

Plus **183 CPU core-hours** for copy-on-write verification: 60,000 rollout steps × 6 workers
plus 6,000 × 8 × 5 self-improvement branches, at 1.1 s average environment time. That 1.1 s is
itself the result of the runtime: the same loop costs 90 s per step without CoW snapshots and
in-flight verification, an **81×** reduction that is the difference between 183 core-hours and
15,000.

The 4,686 line deserves its own note, because it is 18.7 % of the budget and it is the only
line that buys no capability: 2,000 GPU-h of pre-registered ablations (§08) plus 12 % of all
other compute for restarts. A model with a 21 B parameter ledger and an 11 assertion reference
implementation should expect to spend a fifth of its budget proving itself wrong; a plan that
does not is a plan that has not been priced.

At 1,527 B tokens for 2.07 B active parameters, the token-to-active-parameter ratio is **738:1**
(and 72:1 against stored parameters). Chinchilla-optimal is ~20:1 per active parameter, so this
is deliberately over-trained: the design bets that inference cost dominates, which is true for
a coding model with a 2.07 B active budget and a 33,330 tok/s serving path.

## 9.2 Serving: bytes, not FLOPs

Weight bytes at 4-bit are charged at **0.55 B/param** (weights plus scales and metadata), HBM
bandwidth at **3.35 TB/s**, and expert residency is charged per layer — the fix from §01.3.
Always-on weights (everything except routed experts and unselected subspaces) are **0.595 GB**;
one expert is **2.60 MB**.

| Batch | Distinct experts/layer | GB/step | ms/step | tok/s (spliced) |
|---:|---:|---:|---:|---:|
| 1 | 5.9 | 0.84 | 0.251 | 5,080 |
| 16 | 80.2 | 3.92 | 1.171 | 18,617 |
| 64 | 199.0 | 8.86 | 2.645 | 33,330 |

The distinct-expert curve is `256 × (1 − (1 − 1/256)^(batch × 6))`, which saturates: at batch 64
the router has touched 199 of 256 experts, so per-token byte cost is approaching the worst case
rather than improving. That is the honest shape of MoE serving at large batch — batching
amortizes *weights*, not *expert diversity*.

Stored 11.69 GB at 4-bit, active 1.14 GB per token. At batch 1 the step is 26× memory-bound
(4.68 GFLOP is 9.5 µs of compute against 251 µs of streaming), which is what makes §07's
block-parallel decode argument plausible and what makes 0-DoF splicing profitable at low batch.

### The byte model does not include KV residency, and at 1 M that is the whole story

The table charges expert streaming only. The global NoPE layers must retain every token's
compressed key to select blocks later, so at 1 M context:

| Cache | Per sequence |
|---|---:|
| 12 local layers, 4 k window × 576 floats fp16 | 56.6 MB |
| 4 global layers, 1 M tokens × 512 floats fp16 | 4.29 GB |
| **Total** | **4.35 GB** |

Against 137 GB for the same 16-layer stack with uncompressed 32 × 128 MHA keys, that is a 31.6×
compression — but against the 0.595 GB of always-on weights it is 7.3× larger. **At long context,
Kestrel-4 is memory-bound by global KV residency, not by experts.** The sparse global layers
make the *compute* constant in context length and leave the *memory* linear; that trade is worth
stating plainly because it is the first thing a production deployment would hit, and because it
is the open problem §10 does not solve.

At batch 64 with a 4 k window instead, KV residency is 64 × 73.4 MB = 4.7 GB, so a real step is
13.6 GB rather than 8.86 GB and sustained throughput is ≈ 21,800 tok/s rather than 33,330. Both
numbers are quoted wherever the figure appears; the tool reports the weight-only model.

## 9.3 The ablation programme, priced

| ID | GPU-h | What it can delete |
|---|---:|---|
| C2 MoE two-stage | 1,400 | the subspace machinery (93.8 % of stored params) |
| C3 local RoPE | 900 | the rotary path in 12 layers |
| C5 NoPE global | 400 | the NoPE/indexer design in 4 layers |
| C6 hypothesis codebook | 260 | conditioned retries |
| C9 self-training poison | 180 | unattended distillation |
| C4 Householder memory | 120 | the 203.6 M memory module |
| C8 verifier soundness | 40 | the runtime's authority |
| C7 0-DoF splicing | 6 | the splicing path |
| C1 absorption | 0.5 | absorption at serving |
| **Total** | **3,306** | 13.2 % of budget |

Ordered by cost ascending, the programme can delete the runtime's authority (C8, 40 GPU-h)
before it deletes the memory (C4, 120) before it deletes the MoE (C2, 1,400). That ordering is
the point: cheap experiments that can invalidate expensive ones run first.

## 9.4 Measured, modelled, forecast

| Claim | Status |
|---|---|
| 21,248,719,490 / 2,071,178,882 parameters | **measured** (meta-device instantiation + analytic, assertion [1]) |
| absorption 8.94e-08 (fp32) | **measured** (two forward passes, assertion [3]) |
| routed rank 3,072 collective | **by construction** (assertion [2]) |
| memory β ∈ (0, 2), max 1.52 at init | **measured** (assertion [6]) |
| 4.19 MB session state | **by construction** (2,097,152 floats fp16) |
| FPR ladder 0.719 / 0.012 / 0.00216 / 0.000324 | **modelled** in `sim_verify_v4.py` from stated gate measurements |
| poison 0.049 % at `p_solve = 0.40` | **modelled** (same) |
| retry 1,746 → 52.4 GFLOP | **modelled** as 2.91 GFLOP × tokens re-sampled |
| 25,067 GPU-h | **modelled** at 0.34 MFU, 989 TFLOP/s peak |
| 0.84 / 3.92 / 8.86 GB per step | **modelled** at 0.55 B/param, 3.35 TB/s |
| 1.28× / 1.36× / 1.38× splicing speedup | **modelled** with 28 % deterministic steps and 16 µs/token CPU |
| 4.35 GB KV at 1 M | **by construction** from the config |
| any accuracy, pass@1, or loss figure | **not claimed** — no weights were trained |

## Rejected alternatives

- **8-bit weights.** Halves the byte budget's benefit: throughput halves, the splicing crossover
  moves from 12 % to ~24 % deterministic fields, and 11.69 GB becomes 23.4 GB. 4-bit is chosen
  because serving is memory-bound at every batch size in the table.
- **FP8 training.** Would raise MFU from 0.34 to ~0.45 and cut the 25,067 GPU-h by a quarter,
  but the Householder state and the interval arithmetic both care about mantissa bits; the
  runtime's soundness is not worth 6,000 GPU-h. Rejected for this design, revisit for a pure
  serving distillation.
- **Tensor parallelism inside a layer.** At 0.84 GB/step, batch 1 is already weight-streaming
  bound; TP would split the weights across devices and multiply communication for a step that
  is 251 µs long. Expert parallelism is the only sharding that pays here, and it is what the
  per-layer residency accounting assumes.
- **Longer pretrain (4 T tokens).** The 738:1 ratio is already 37× beyond Chinchilla; the
  marginal GPU-hour buys less than the same hour spent on the ablation programme.
- **Charging experts once per step (day 3.1).** A 16× under-count. See §01.3.
