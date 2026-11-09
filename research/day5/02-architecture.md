# 02 — Architecture, ledger, KV and FLOPs

## Trunk layout

16 layers, 12 local MLA + 4 global-NoPE landmark layers at indices
`(3, 7, 11, 15)`. Each block is `attention → WY Householder memory →
Stiefel subspace MoE`, with three RMSNorms. The embedding is tied with the LM
head. The workspace (`w_layers = 4`, `d_w = 1024`) sits on top.

Config (from `kestrel5/config.json`, generated from `FULL` and round-tripped
byte-exactly): vocab 131,072 · d 4,096 · 32 heads × 128 · latent 512 · rope 64
· window 4,096 · block 64 · top-8 blocks · 256 anchors · d_m 512 / r_m 256 ·
8 groups × 32 experts = 256 · d_lat 384 · d_ff 4,096 · shared 1,024 ·
workspace 4 × (d_w 1024, w_ff 4096) · k_hyp 8 · r_max 8 · mtp_spec_depth 3.

## Parameter ledger (exact integers, recomputed and asserted)

| subsystem | per layer | × | stored |
|---|---|---|---|
| Local MLA layer | 28,574,752 | 12 | 342,897,024 |
| Global NoPE landmark layer | 27,919,392 | 4 | 111,677,568 |
| Compact WY Householder memory | 12,722,176 | 16 | 203,554,816 |
| Stiefel subspace MoE | 1,245,839,624 | 16 | 19,933,433,984 |
| Layer RMSNorms | 12,288 | 16 | 196,608 |
| KV-cached Code-JEPA workspace | — | 1 | 120,084,482 |
| Tied embedding + final norm | — | 1 | 536,875,008 |
| **GRAND TOTAL** | | | **21,248,719,490** |

Active (tested tokens prefer **6 experts of 256**):

| mode | routed groups × experts/group | active parameters | share |
|---|---|---|---|
| 2×3 | 2 × 3 | 2,071,178,882 | 9.75% |
| 6×1 | 6 × 1 | 2,273,293,954 | 10.70% |

The 6×1 mode reads a 3× wider *routed rank* per token (2,304 vs 768 \
per-token latent subspace) for 9.7% more active parameters. 2×3 and 6×1 are
two routing modes over the same expert pool, not two theorems about it.

Every number is produced by `kestrel5/model.py` and re-derived independently in
`count_params_v5.py`; the TINY fixture proves analytic == instantiated ==
323,698 by summing real array sizes.

## Two-tier KV footprint at 1M context (fp16, asserted)

| tier | MB |
|---|---|
| Local windows (12 × 4,096 × 576 × 2) | 56.62 |
| Global block-summary index (4 × 16,384 blocks × 512 × 2) | 67.11 |
| Exact AST landmarks (4 × 4,096 × 512 × 2) | 16.78 |
| WY fast weights (16 × 256 × 512 × 2) | 4.19 |
| **144.70 total** | |

### Honest correction: the specified ledger omits the global-layer windows

The 4 global layers attend over a 4,096-token local window *in addition to*
their block index and landmarks — yet the specified ledger above only charges
one window per **local** layer. Those keys have to live somewhere. Charging
4 × 4,096 × 512 × 2 = 16.78 MB more gives a mechanically complete session
footprint of **161.48 MB**.

We keep both numbers: **144.70 MB** is the specified ledger (asserted in the
assertion suite for conformance), and **161.48 MB** is what a serving stack
must actually allocate. The audit reports the window-complete variant
separately; neither is hidden in a footnote (`count_params_v5.py` §C).

## FLOP model

For one decoded token (decode step, not prefill):

```
flops = 2 · active_params                            (every active weight once)
      + attention_terms                              (visible keys × head dim)
      + indexer_terms                                (dot product per visible block)
      + (r − 1) · 2 · workspace_params               (extra recurrent passes)
```

`workspace_params = 120,084,482`, so one extra recurrent pass is
**0.2181 GFLOP** — the spec's `0.218` reproduces *exactly*
(`2 · 4 · 27,266,048 = 218,128,384`).

| r | GFLOP/token |
|---|---|
| 1 | 5.889 |
| 2 | 6.107 |
| 4 | 6.543 |
| 8 | 7.416 |

The marginal `(r−1)·workspace` term is an *integer identity*, so the identity
`flops(8) − flops(1) = 7 × 218,128,384` is checked without rounding.

### The `0.307 GFLOP` figure does not reproduce

The spec's "pre-cache saving at r=8" of 0.307 GFLOP cannot be produced from any
per-step decomposition we can build. By the natural decomposition (7 avoided
K/V projections per workspace layer: `7 · 2 · 4 · 2·4096·2·1024`) the saving is
**0.940 GFLOP**, not 0.307. The audit asserts the non-reproduction explicitly
(`> 0.2` gap) instead of printing the spec number. We keep the figure that
does reproduce (0.2181) and flag the one that does not.

## Serving model (batch-1 decode, bandwidth-bound)

Decode touches every active weight once and streams the KV read set, so the
ceiling is `aggregate_BW / bytes_moved_per_token`. At 1M context the read set
is ~80.7 MB.

| profile | weights | ceiling (tok/s) | 12,216 target |
|---|---|---|---|
| 8×H200 fp8 | 2.27 GB | 16,312 | 74.9% MBU |
| 8×H200 fp16 | 4.55 GB | 8,299 | 147.2% MBU (impossible) |
| 1×T4 fp16 | 4.55 GB | 69 | — |

These ceilings are computed at the config default routing (6×1, the
`top_groups = 6` of `kestrel5/config.json`). At 2×3 routing the active count
is lower (2,071,178,882), so the fp8 ceiling rises to **17,845 tok/s**
(68.5% MBU) and decode FLOPs drop from 6.107 to 5.703 GFLOP/token. The
conclusion does not change: 12,216 tok/s is only reachable with fp8 weights,
and stays a bandwidth arithmetic result, not a measurement.

Arithmetic intensity is ~2.59 FLOP/byte against an H200 ridge point of ~300, so
decode is bandwidth-bound by two orders of magnitude: FLOPs are nearly free,
bytes are the budget.

**`12,216 tok/s` is a target, never a measurement.** Under fp8 it implies
74.9% MBU — plausible, unproven, and labelled as a target everywhere in this
directory. Under fp16 it would need 147% MBU, which is physically impossible;
that is why fp8 weight quantisation is a *requirement* of the architecture,
not an optimisation.
