# 06 — Sparse MoE: why 8 × 32 with two-stage routing

Kestrel-4 stores 256 experts per layer and evaluates 6. That ratio is unremarkable. What is
worth arguing about is the *shape* of the 256, because day 3.1's flat layout had a defect that
no parameter count reveals: 256 experts at one latent width add stored capacity but not routed
width.

## The layout

| Piece | Shape | Stored / layer | Active / layer |
|---|---|---:|---:|
| 256 experts × (gate, up, down) | `384 → 4096 → 384` | 1,207,959,552 | 28,311,552 (6) |
| 8 group subspaces × (in, out) | `4096 ↔ 384` | 25,165,824 | 6,291,456 (top 2) |
| Shared expert | `4096 → 1024 → 4096` | 12,582,912 | 12,582,912 |
| Group router | `4096 → 8` | 32,776 | 32,776 |
| Within-group routers (8) | `384 → 32` | 98,560 | 24,640 (top 2 groups) |
| **Total** | | **1,245,839,624** | **47,243,336** |

Across 16 layers: **19,933,433,984 stored** (93.8 % of the model) and **755,893,376 active**
(36.5 % of the 2.07 B read per token). One expert is `3 × 384 × 4096 = 4,718,592` parameters =
**2.60 MB at 4-bit**, which is the unit the serving model in §09 streams.

Routing is:

```
g = softmax(W_g x)                  # 8 group scores
keep top 2 groups
for each kept group: e = softmax(W_e (x W_in^g))   # 32 experts in that subspace
                     y += Σ_{j ∈ top 3} e_j · Expert_j(x W_in^g) W_out^g
y += SharedExpert(x)
```

Dropless: capacity factor 1.0, no token dropping, no token padding. Load balance is an
auxiliary router loss (coefficient 1e-3), not a capacity clamp, because dropping tokens makes
the active parameter count a lie on exactly the batches where routing is most specialized.

## The rank argument, stated precisely

Every expert in a flat 256-expert MoE projects `d = 4096 → d_lat = 384` and back. Summing six
of them with router weights produces a vector in the span of six 384-dimensional outputs —
which is at most **384 dimensions**, identical to one expert. Day 3.1's "256 experts" therefore
bought specialization but never widened the per-token computation: the layer's routed rank was
`d_lat`, not `6 · d_lat`.

Grouping fixes this by giving each group its own 384-dimensional subspace with its own
in/out projection. Per token:

- **collective rank = 8 × 384 = 3,072** — the width the layer *can* address, 8× day 3.1's 384
  (assertion [2] pins 3,072);
- **routed rank ≤ 2 × 384 = 768** — top 2 groups, each contributing its own subspace.

I am stating 768 as an upper bound rather than a guarantee: it is achieved only if the two
selected groups' subspaces are close to orthogonal for that token. That is what C2 measures,
and it is why the honest claim is "2× per-token routed width, 8× collective capacity" rather
than the 8× per token that an earlier draft of this design asserted. The distinction matters:
3,072 is a property of the parameters, 768 is a property of the routing.

**The routing split is a capability knob, not a constant.** At the same six expert evaluations,
`top-2 groups × top-3` gives 768 routed dimensions, while `top-6 groups × top-1` gives
**2,304** — three times the per-token width for the same FLOPs — at the cost of choosing 1 of 32
experts per group instead of 3, i.e. less within-group specialization and a sharper load
balance problem. The configuration can express both (`top_groups`, `top_k_per_group`), and C2b
below is the experiment that decides. This is the most promising unexplored corner of the
architecture: routed rank is free at fixed expert count, and the current 2×3 point was chosen
for routing entropy, not because 768 is the capacity the task needs.

The two-stage router is also cheaper than a flat one. Scoring 256 experts costs `4096 × 256 =
1,048,576` MACs per token; scoring 8 groups then 32 experts inside each of 2 groups costs
`4096 × 8 + 2 × (384 × 32) = 57,344` — **18.3× fewer router MACs**, 0.12 % of the layer's active
FLOPs. The second stage runs on the group's compressed representation, so it is cheap *and*
better conditioned than a 256-way softmax over a shared latent.

## Rejected alternatives

- **Flat 256, top-6.** Day 3.1. Same stored parameters, same expert evaluations, routed rank
  capped at 384. Rejected on the rank argument above; it is the C2 baseline.
- **Flat 256, top-1 group × top-6.** Same 6 expert evaluations, single subspace, routed rank ≤
  384 again — grouping without multiple subspaces buys nothing.
- **Day 3's 96 experts at full width (`d_ff = 4096`, `d_lat = 4096`).** Each expert is 10.7×
  larger in bytes, so the 6-expert working set is 167 MB instead of 15.6 MB at 4-bit; the
  active parameter count would be 1.97 B just for MoE.
- **Dense FFN (`d_ff = 14336`).** `3 × 4096 × 14336 = 176,160,768` per layer, all active:
  +2.8 B to the per-token budget, 3.7× the MoE's active cost, for no specialization.
- **4 groups × 64 experts.** Collective rank halves to 1,536 and the within-group router grows
  to `384 × 64`; no benefit that C2 would detect.
- **Expert choice (token picks experts vs experts pick tokens).** Better balance in theory,
  but it breaks causality in a decoder: an expert that chooses its tokens at position `t` can
  see which tokens arrive later. Rejected for a language model.

## Falsifiers

**C2 — two-stage vs flat, at equal active parameters and equal GPU-hours.** Train both to the
same token count with the same total stored parameters (flat keeps 256 × 384 experts; grouped
keeps 8 × 32 × 384 plus subspaces, so the flat arm gets a slightly larger `d_ff` to match
stored parameters). Metric: pass@1 on HumanEval+ and MBPP+, plus validation loss. **Kill
criterion: if the grouped arm does not beat flat by ≥ 1.5 points pass@1 or ≥ 0.02 nats, the
subspace machinery is decoration** and the flat router wins on simplicity.

A second, cheaper check runs first: measure the effective rank of the routed output (participation
ratio of the singular values of the per-token expert-sum matrix) on 10⁴ tokens. If grouped
routing does not raise it above 384, the subspaces have collapsed and the training recipe needs
an orthogonality term before C2 is worth its GPU-hours.

**C2b — routing split at fixed evaluations.** Compare `top-2 × top-3` (768 rank) against
`top-6 × top-1` (2,304 rank) at equal active parameters and equal expert evaluations, 300
GPU-h. Kill the wide split if it does not beat 2×3 by ≥ 1 point pass@1, or if expert load
imbalance (max/mean tokens per expert) exceeds 2.0 without auxiliary-loss tuning. A win here is
worth more than any parameter-count change in this chapter, because it buys capability at zero
inference cost.
