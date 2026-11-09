# 09 — Serving model, NANO tier and VRAM budgets

## Batch-1 decode is bandwidth-bound

At 1M context decode reads ~80.7 MB of KV per token plus every active weight.
Arithmetic intensity is **2.59 FLOP/byte** against an H200 ridge point of
~300 — so FLOPs are free and bytes are the budget, by two orders of
magnitude. Any design conversation about decode that starts with FLOPs is
starting in the wrong place.

| profile | ceiling | 12,216 tok/s target |
|---|---|---|
| 8×H200, fp8 weights | 16,312 tok/s | 74.9% MBU |
| 8×H200, fp16 weights | 8,299 tok/s | 147.2% MBU — impossible |
| 1×T4, fp16 weights | 69 tok/s | — |

(The 8×H200 rows use the config-default 6×1 routing; at 2×3 routing the fp8
ceiling is 17,845 tok/s at 68.5% MBU. See doc 02.)

The fp16 row is the important one: it *proves* that fp8 weight quantisation is
a requirement of hitting 12,216 tok/s, not an optimisation. And 12,216 remains
a **target** — at 74.9% MBU it is plausible and unproven, and every place this
directory quotes it, it is quoted as a target.

## KESTREL5_NANO_165M (free tier)

| | |
|---|---|
| stored | **151,495,922** (151.5M) |
| active (6×1) | **80,704,562** (53.3%) |
| fp16 weights | **303 MB** |

The full-model ledger shape scaled down: 12 local + 3 global layers
(`global_layer_indices = (3, 6, 9, 11)`), d 768, and the same subsystems. It is
deliberately a *true* 165M-tier model, not a FULL with the numbers moved.

## Training budget on one 16 GB T4 (batch 16 × 2,048, r = 2)

| component | MB |
|---|---|
| weights (fp16) | 303.0 |
| fp32 master copy | 606.0 |
| Muon/AdamW states | 706.8 |
| activations (fused) | 2,113.9 |
| **total** | **3,729.7 MB = 3.64 GiB** |

Two non-obvious facts in that table:

1. **Muon owns 83.4% of the parameters** (126,293,184 of 151,495,922). Only
   the embedding and the 1-D tensors (norms, sinks, router biases) stay on
   AdamW. The optimiser split is structural: Muon is for 2-D matrices, AdamW
   for everything else, and the embedding is excluded as a matrix.
2. **A fused attention kernel is mandatory, not optional.** Materialising one
   dense score matrix for one layer at batch 16, seq 2,048, window 2,048
   needs **3,221 MB** — more than the entire fused activation budget, and a
   third of the GPU for a single layer's scores. The reference implementation
   runs dense matmuls against an exact sparse mask for *correctness*; serving
   must not copy that strategy.

## What the free tier cannot do

One T4 at 320 GB/s gives a 69 tok/s ceiling at fp16 for the FULL model, and a
NANO model that must be run at small batch. The free tier is for *correctness
and training*, not for the 12,216 tok/s target. The serving story and the
free-tier story are different stories; conflating them is how "12,216 tok/s"
turned into a claim in the first place.
