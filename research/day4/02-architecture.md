# 02 — Architecture

Kestrel-4 is a 16-layer hybrid: **12 local windowed MLA layers with per-head RoPE, and 4
global NoPE sparse-anchor layers at indices 3, 7, 11, 15.** Every layer carries a Householder
memory (03) and a subspace-grouped MoE (06); the stack is wrapped in a Code-JEPA workspace
(05, 07). This chapter is about the attention skeleton and why it is shaped this way.

## The shape is a pipeline decision as much as a modelling one

Put the four global layers at 3, 7, 11, 15 and the stack factors into **four identical stages
of three local layers plus one global layer**:

```
3 × 1,287,148,840  +  1,286,493,480  =  5,147,940,000 parameters per stage
4 × 5,147,940,000                     = 20,591,760,000 parameters in blocks
```

The two block types differ by 655,360 parameters (a local block is 28,574,752 of attention vs
27,919,392 for a global one, and both carry the same memory and MoE), so a 3:1 pattern is the
only one that makes four *exactly* equal stages. Equal stages mean a 4-way pipeline has no
bubble from parameter imbalance and no stage needs a different micro-batch count; with 16
layers and a 4-stage plan, that is worth more than any 2 % variation in layer placement.

## Local layer: windowed MLA, per-head RoPE from the query latent

Per layer, `d = 4096`, `h = 32`, `dh = 128`, `d_c = 512`, `d_rope = 64`, window 4,096:

| Matrix | Shape | Parameters |
|---|---|---:|
| `w_dq` | 4096 × 512 | 2,097,152 |
| `q_norm` | 512 | 512 |
| `w_uq` | 512 → 32 × 128 | 2,097,152 |
| `w_qr` | 512 → 32 × 64 | 1,048,576 |
| `w_dkv` | 4096 → 512 + 64 | 2,359,296 |
| `ckv_norm` | 512 | 512 |
| `w_uk`, `w_uv` | 512 → 32 × 128 (each) | 4,194,304 |
| `w_o` | 32 × 128 → 4096 | 16,777,216 |
| `sinks` | 32 | 32 |
| **Total** | | **28,574,752** |

Two details are load-bearing and were both wrong in day 3 (see 01):

**Per-head RoPE from `c_q`.** The rotary projection is applied to each head's query slice
(`d_rope = 64` per head, `theta = 5e5`) and to a single shared key slice. Decoupling the
rotary dimensions from the 128-dimensional content head keeps RoPE from consuming capacity
that MLA needs for the compressed latent, and it is what makes the *window* meaningful:
without it (day 3.1's actual state) a local layer is order-blind inside its window and the
4,096-token receptive field is a bag of tokens.

**The latent norm before `w_uk`.** Absorption requires

```
(W_uq c_q) · (W_uk c_kv)  =  c_qᵀ (W_uqᵀ W_uk) c_kv
```

with `c_kv = RMSNorm(w_dkv(x)[:, :512])`. Because the norm is applied to the *latent* and not
per head, `W_abs[h] = W_uq[h]ᵀ W_uk[h]` is a fixed 512 × 512 matrix per head that can be
precomputed after training. Assertion [3] compares the two forward paths in fp32:
**max |standard − absorbed| = 8.94e-08.**

What absorption buys at decode, per token per local layer, against `n` cached keys:

- cache: 512 latent + 64 rotary = **576 floats** instead of 32 × 128 = 4,096 → 7.1× smaller;
- score MACs: `512 × 512` once (262,144) plus `n × 512`, instead of `n × 4,096`; at `n = 4096`
  that is 2.36 M against 16.8 M MACs → **7.1× fewer**.

That ratio is not a coincidence: both are `h · dh / (d_c + d_rope)` = 4096/576.

## Global layer: NoPE, sparse blocks, exact anchors

The global layer drops RoPE entirely and selects keys instead of attending to all of them:

```
keys per query = window (4,096) + top-8 blocks × 64 + 256 anchors = 4,864, at any context
```

Selection is a two-stage indexer over `d_idx = 128`: coarse block scores (`w_iq` × pooled
`w_ip`), then fine token scores (`w_ik`). Assertion [4] checks that all three matrices receive
gradient — the day 3.1 indexer had a scorer that was instantiated and never used, and a
gradient check is the cheapest way to make that class of bug fail loudly.

**Why NoPE here is a mechanism, not a preference.** A rotary layer's score depends on
relative offset, which is what you want for syntax *inside* a window and exactly what you do
not want for retrieval at 1 M tokens: the address of a fact should not decay as a function of
how far away it is. NoPE attention is order-invariant, so the only positional information a
global layer has is the one the indexer hands it (block index, anchor index). That is a
weakness for local order and a strength for address arithmetic, which is why the two layer
types are not interchangeable and the pattern is 3:1 rather than 1:1.

Cost of the global layer at 1 M: 4 × 4 × 4,864 × 32 × 128 MACs = 0.29 GFLOP/token, constant in
context length. The indexer adds `4 × 2 × (n/64 + 32 × 64) × 128`, which grows linearly at
one 128-dim dot product per 64-token block — 0.019 GFLOP at 1 M.

## Rejected alternatives

- **16 local RoPE layers.** Cheapest per token, but the exact-recall argument dies: a 4 k
  window cannot address a fact at token 900,000, and no amount of indexer adds address
  information to a relative-position score. Rejected because it fails the day 3 validity
  thesis, not because of cost.
- **16 global NoPE layers.** Recovers recall but removes the only order-sensitive path; local
  syntax (indentation, bracket nesting, operator precedence in text) becomes a bag-of-tokens
  problem. Also +12 × 0.29 = 3.5 GFLOP/token at 1 M for recall already at ceiling.
- **8 local + 8 global (alternating).** Balanced as a pipeline too, but doubles the global
  attention FLOPs and the indexer calls for a recall gain the 3:1 pattern already reaches in
  C5; and NoPE layers are the expensive ones at long context.
- **Global layers with RoPE + full attention at 256 k.** The day 3 v1 design. Cost scales
  with context (4 layers × 256 k keys), and RoPE's relative addressing is the wrong prior for
  retrieval. Rejected on both axes.

## Falsifiers

- **C3 — RoPE in local layers.** Train a 1 B-token pair of 16-layer models, identical except
  that the local layers are NoPE. Measure needle-in-a-haystack exact recall at 256 k and
  syntax-sensitive pass@1. Kill the rotary path if NoPE is within 2 points on both; if so, the
  parameter was decoration.
- **C5 — NoPE in global layers.** Swap the four global layers to RoPE at 1 M. Kill NoPE if
  RoPE is within 1 point of exact recall, since RoPE is simpler to serve.
- **C1 — absorption after training.** Assertion [3] pins 8.94e-08 at initialization. After
  training, re-measure; if the absorbed path diverges by more than 1e-3 nats or 0.01
  perplexity, the norm placement is wrong and absorption must be dropped at serving.
