# 01 — What Day 4 missed

Day 4's prototype shipped seven mechanisms that looked right in prose and were
wrong in code (or unproven). Each is stated as a mechanism, not an anecdote:
what the defect is, why it breaks the design, and where it is fixed.

## F1 — The write operator was not bounded

The delta-write's homogeneous part is

```
B ← B − η · ((B k) ÂᵀÂ) kᵀ,      Â = A / max(1, ‖A‖₂)
```

Its spectrum is `{1} ∪ {1 − η·μᵢ}, μᵢ = eig(ÂᵀÂ)`. Because `‖Â‖₂ ≤ 1`, every
`μᵢ ∈ [0, 1]`, so with `η = 0.5` every eigenvalue lies in **[0.5, 1]**: the
memory is non-expansive over arbitrarily long sequences. Day 4's write had no
spectral cap, so `μᵢ` was unbounded and the state norm could grow without
bound — a silent numerical divergence that only shows at length.

**Verified** (model assertion [3], audit §5): the write operator is probed by
*building its matrix column-by-column* from `apply_write_operator` — the
closed-form `write_spectrum()` is never trusted alone. Measured:
`[0.500000, 1.000000]`, imaginary part `< 7.4e-16`, and the same bound holds
after `‖A‖₂` is inflated 5×, because the cap `Â = A / max(1, ‖A‖₂)` re-imposes
`‖Â‖₂ = 1`.

## F2 — No negative eigenvalues, no parity

Purely contractive memories (spectrum ⊆ [0, 1]) are excellent at stability and
terrible at parity/S5-style tasks that need *sign flips*: a state that can only
lose energy can never count. The Compact-WY reflections
`B ← B (I − β u uᵀ)` carry eigenvalues `1 − β ∈ (−1, 1)` for `β ∈ (0, 2)`, so
the layer is *explicitly* given a reflection channel: two generalized
Householders per token, fused into one WY matmul. Measured lambda at
`β = 1.8` is `−0.800000`.

## F3 — The indexer only learned from blocks it already selected

Day 4 trained the block indexer with the attention gradient restricted to the
*selected* blocks. That is a self-confirming loop: a block that is never
selected never gets a gradient, so it can never become selectable — "dead
zones" in the index. The fix (doc 04) supervises with the KL to the *actual*
post-softmax attention mass pooled per block, over **all visible blocks**.
Audit §4 measures `‖grad_w_iq‖` on *unselected* blocks directly
(`4.779e-02 > 1e-6`) — the gradient provably reaches blocks the router did not
pick.

## F4 — The MoE router was not orthogonal

With independent per-group projections, the group subspaces can collapse onto
each other, and the collective rank falls below `G·d_lat`. The fix (doc 05)
builds one `[d, G·d_lat]` matrix by QR and slices it into the group blocks, so
`W_up^(g) W_up^(g')ᵀ = 0` for `g ≠ g'` *by construction*. Measured:
`orth_loss = 1.77e-29`, numerical rank `64/64` at TINY (16/16 at FULL per
layer: 8 groups × 384 = 3,072 collective rank, not 256·…).

## F5 — A hardcoded `-0.05` cut is scale-fragile and unsound

Day 4's CDCL jumpback did `hi ← hi − 0.05`. Two failures, both measured in the
simulator (doc 07):

- **Explosion at scale.** For violation `Δ`, it needs `Δ/0.05` iterations. At
  unit scale 1e3 it needs **15,001** iterations where the exact cut needs **1**.
- **False UNSAT at small scale.** On `[0, 1e-3]` a single `-0.05` step already
  empties the domain although the system is feasible — it *reports* infeasible
  with a satisfiable instance. That is not a slow check, it is a wrong proof.

The fix is an exact-rational Farkas cut (`fractions.Fraction`) with outward
1-ULP widening, so rounding can never exclude a feasible point.

## F6 — The workspace re-projected its context every step

`CodeJEPAWorkspace` with `r = 8` re-ran the context `K/V` projection in every
recurrent step: 8× the work for a quantity that depends only on the trunk
output. `WorkspaceLayer.project_context_kv` splits projection from `forward`;
the workspace projects once per layer and reuses it. Audit §3 counts
`kv_proj_calls(r=8) = 2` (one per layer, `w_layers = 2`).

## F7 — Delimiter counting on raw text is forgeable

Counting `(` and `)` over the raw prefix lets a string literal or a comment
fake balance — `s = "("` would look like an open call. The Tier-0 automaton
lexes a copy of the prefix, replacing string/comment contents with spaces *at
identical indices*, so a delimiter inside quotes cannot close a real one. The
audit asserts that `s = "("` still rejects a stray `)`.

## Summary

| defect | mechanism | verification |
|---|---|---|
| F1 unbounded spectrum | spectral cap ‖Â‖₂ ≤ 1, η = 0.5 | matrix probe [0.5, 1] |
| F2 no parity channel | Householder reflections, 1 − β ∈ (−1, 1) | λ(1.8) = −0.8 |
| F3 dead indexer zones | KL to actual attention mass, all visible blocks | unselected-block gradient ≠ 0 |
| F4 router collapse | joint QR over `[d, G·d_lat]` | orth_loss 1.77e-29, rank 64/64 |
| F5 magic −0.05 cut | exact-rational Farkas + outward rounding | 1 cut vs 15,001; false-UNSAT removed |
| F6 repeated projection | `project_context_kv` split | kv_proj_calls(r=8) = 2 |
| F7 forgeable balance | lexed copy at identical indices | `s = "("` still rejects `)` |

Every one of these is asserted in code on a real forward pass; none is asserted
by a comment.
