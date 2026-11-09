# 03 — Compact-WY Householder fast-weight memory

The memory subsystem of every block. Its job is inter-token context *beyond*
the 4,096 window, at O(1) per-token state — the alternative to an unbounded KV
cache.

## The mechanism

Two steps per token:

1. **Two generalized Householder reflections** `(k1, βt), (k2, βt)`, fused into
   one Compact-WY matmul. Each reflection applies `B ← B(I − β u uᵀ)` with
   eigenvalue `1 − β ∈ (−1, 1)`. The Compact-WY structure gives, for two
   reflections, a single product: `(max|WY − sequential|) = 8.88e-16`, and `T`
   is upper-triangular with diagonal exactly `β`.
2. **A delta-rule content write** along `k1`:
   `B ← B − η · ((B k) ÂᵀÂ) kᵀ`, with `η = 0.5` and `Â = A / max(1, ‖A‖₂)`.

Reads are causal (read-before-write: `y_t = Â (B_{t−1} q_t)`), gated by a
sigmoid, and projected out.

## Why this is safe at any length (the theorem)

The write's homogeneous part has spectrum `{1} ∪ {1 − η·μᵢ}` where
`μᵢ = eig(ÂᵀÂ)`. Since `‖Â‖₂ ≤ 1`, `μᵢ ∈ [0, 1]`, so every eigenvalue is in
**[0.5, 1]**. The memory cannot amplify anything, ever; the reflections keep
the `−1` end of the spectrum open for parity tasks.

This is not asserted from the formula — it is *probed*: the audit and the
model assertion build the write operator as a real matrix by applying
`apply_write_operator` to all 512 basis vectors of the `[r_m · d_m]` state
space and eigensolving it. Measured `[0.500000, 1.000000]` with imaginary part
`< 7.4e-16`, tolerance `1e-9` (a float64 eigensolve of a non-symmetric 512-dim
matrix cannot do better, and no claim depends on more precision than that).

## Why the cap matters (the falsifier)

The cap is `Â = A / max(1, ‖A‖₂)`. If you multiply `A` by 5, the raw write
spectrum would leave [0.5, 1]. The cap normalises it back. Measured: inflating
`‖A‖₂` by 5× keeps the spectrum in `[0.5, 1]` and caps `‖Â‖₂ = 1.000000`.
Without the cap the same test fails — this is exactly F1 from doc 01.

## Long-horizon behaviour

A 256-step forward pass with random inputs gives `‖B‖_F = 8.867` (TINY) —
finite, small, and not growing with length. The audit re-checks at 64 steps
independently.

## Why Householders and not a plain delta rule

A pure delta rule (spectrum ⊆ [0, 1]) is a perfect integrator and a terrible
parity counter: it can only subtract energy. Grazzi et al. (ICLR 2025) show
that trained recurrent matrices develop *negative* eigenvalues for state
tracking. The Householder channel supplies exactly those eigenvalues
(`1 − β ∈ (−1, 1)`) while staying a projection-like isometry in the reflected
direction, so parity and stability are not in tension. Measured
`λ(β = 1.8) = −0.800000`.

Design groundings: DeltaNet (Yang et al. 2024) for the WY chunkwise product,
DeltaProduct (NeurIPS 2025) for `n_h = 2` Householders per token, Grazzi et al.
(ICLR 2025) for the negative-eigenvalue requirement.

## Cost

Per layer `d_m = 512, r_m = 256`: `6·d·d_m + d·n_h + d_m·r_m = 12,722,176`
parameters; the fast-weight state is `256 · 512 · 2 = 0.26 MB` per layer, and
is part of the 144.70/161.48 MB session footprint (4.19 MB total).
