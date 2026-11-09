# 04 — Two-tier NoPE landmark attention and the KL-aligned indexer

The context problem: a 1M-token session cannot store a 1M-token KV cache
(4.35 GB at fp16 for the full model). The 4 global layers solve it with a
*dense* mask over a *sparse* visible set, and a learned indexer that is
supervised by the attention itself.

## The visible set (why the budget is structural)

Each query in a global layer attends to exactly three classes of keys:

1. the **local window** (4,096),
2. the **top-8 blocks of 64 tokens** (512), chosen per query by the indexer,
3. **256 exact landmarks** (AST/symbol anchors) — distance-independent.

So the maximum keys per query is `4,096 + 512 + 256 = 4,864`, which is asserted
structurally, not empirically: the audit measures `last_max_keys ≤ 18` at
TINY (budget 18) and `last_outside_mass == 0.0` — *zero attention mass leaks
outside the mask*. Because the selected set is block-aligned, a dense-mask
implementation and an O(n) gather kernel are equivalent; the reference
implementation uses the dense mask over the exact sparse visible set, and the
audit verifies that equivalence through the mask mass, not by running a kernel
(JS: the O(n) gather kernel is a Day 6 deliverable).

## The indexer (fixes F3)

The block indexer is trained with the KL divergence between its pooled block
distribution and the *actual* post-softmax attention mass pooled per block:
`aux_kl = KL(target_blk_mass ‖ idx_prob_block)`. Because the target spans
**every visible block**, the gradient `d_blk_scores ∝ (idx_prob − target)` is
non-zero for blocks that were *not* selected. The audit measures it directly:
`‖grad_w_iq‖` is non-trivial and the unselected-block component is far above
the dead zone (e.g. `4.7795e-02` and `4.6523e-02` in two different seeded
fixtures, both orders of magnitude above `1e-6`). The indexer cannot develop
dead zones because the
supervision is over the whole visible range, not over its own selections.

This is the DeepSeek-V3.2 / DSA "Lightning Indexer" idea (KL supervision,
block size 64, top-2048) reduced to our block size of 64 and top-8 blocks,
plus the NSA-style three-branch visible set (compression / selection /
window).

## The two-tier cache

| tier | stored per global layer | size at 1M ctx |
|---|---|---|
| Block summary index | one 512-float descriptor per 64 tokens | 67.11 MB total |
| Window + landmarks | 4,096 + 4,096 exact keys | 16.78 MB total |

The pooled descriptor replaces 64 keys with 1 mean — that is where the
4.35 GB → 161.48 MB reduction comes from (window-complete, doc 02).

## Why NoPE?

RoPE carries a positional prior that breaks distance-independence: a key
3,000 tokens away is scored as "far" regardless of content. The global layers
carry no RoPE, so a landmark 900k tokens away is retrievable by content alone.
The 12 local layers keep RoPE, where position *is* the prior.

## Falsifiers for this design

- If `last_outside_mass > 0`, the sparse visible set is not being honoured —
  the scheme is a lie.
- If unselected-block gradient is 0, the indexer is dead — F3 is back.
- Neither is true at TINY; both are asserted on real forwards.
