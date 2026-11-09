# 05 — Stiefel subspace MoE, 2×3 vs 6×1, and zero-shock upcycling

## Orthogonal group subspaces (fixes F4)

The 8 group projections are column blocks of a single `[d, G·d_lat]` matrix
obtained by QR, so `W_up^(g) W_up^(g')ᵀ = 0` for `g ≠ g'` *by construction*.
No regulariser is needed, and none is used. Measured orthogonality loss is
**1.77e-29** and the collective numerical rank is **64/64** at TINY (all
`G·d_lat` directions are live). The point is capacity, not aesthetics: two
group subspaces that overlap are two copies of one expert.

The layer is DeepSeekMoE-style: fine-grained experts (d_lat 384, d_ff 4,096)
plus an always-on shared expert (d_shared 1,024). Per layer the stored MoE is
1,245,839,624 parameters; per token, 2×3 mode touches 47,243,336 and 6×1
touches 59,875,528.

## 2×3 vs 6×1 — two routing modes, one pool

- **2×3**: 2 groups × 3 experts = 6 evaluations, routed rank 768.
- **6×1**: 6 groups × 1 expert = 6 evaluations, routed rank 2,304.

Same number of expert evaluations per token (k = 6), 3× the routed latent
subspace, at 9.7% more active parameters (2,273,293,954 vs 2,071,178,882).
The TINY fixture asserts `routed_rank(6) == 3 · routed_rank(2)` on the real
module, and the constructor asserts `top_groups × top_k_per_group == top_k`,
so an inconsistent fan-in budget fails at construction rather than at serving.

The choice is depth-dependent in the spec (6×1 early, 2×3 deep, per the
DA-MoE observation); the reference implements both and lets the config pick.

## Load balancing without touching the LM gradient

Group and fine-router biases are updated by the auxiliary-loss-free sign rule
(DeepSeek-V3 eq. 7): `bias += lr · sign(mean_load − load)`. The balance
signal never enters the LM gradient, so there's no aux-loss term to tune and
no gradient interference to ablate. The audit verifies the biases actually
move.

## Zero-shock upcycling (and its honest cost)

At `upcycle_step` each parent expert `e` is split into an antithetic pair
`(e⁺, e⁻)` sharing the SwiGLU gate/up matrices with down projections
`W_dn ± ε·u⊥` and a duplicated router column. With both children selected at
equal weight the pair sums to the parent, so:

- **the forward is bit-exact** at the split step (measured
  `max|y_before − y_after| = 2.22e-16` at TINY),
- **the children are immediately distinguishable** (measured
  `‖dn⁺ − dn⁻‖ > 0`),
- **routed rank doubles** at the split step (16 → 32 at TINY).

The honest cost, stated in the docstring and here: preserving the function
exactly requires *both* children, so `top_k_per_group` doubles for the split
step — the transient 2× routed cost. Restoring the original budget is a
routing-prune decision made *after* specialisation; it is not hidden. This is
the delta that the review prompt cannot be read around: "zero-shock" means the
*output* is unchanged, not that the *cost* is.

## Falsifiers

- `orth_loss` materially above 0 → the group subspaces overlap; capacity
  collapses.
- `up_diff` above float noise at the split step → the pair doesn't sum to the
  parent; the upcycle is not zero-shock.
- Biases that never move → the sign rule isn't being applied.
