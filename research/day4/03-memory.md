# 03 — Memory: Householder reflections and the 4.19 MB session state

Each of the 16 layers carries a fast-weight memory with `d_m = 512`, `r_m = 256` slots and
`n_h = 2` generalized Householder reflections.

| Component | Parameters |
|---|---:|
| 6 projections `d = 4096 → d_m = 512` (query, key, value, beta, gate, output) | 12,582,912 |
| `beta` head, `d → n_h` | 8,192 |
| slot embedding `r_m × d_m` | 131,072 |
| **Per layer** | **12,722,176** |
| **All 16 layers** | **203,554,816** (0.96 % of the model) |

Session state is `r_m × d_m` per layer = 131,072 floats → **2,097,152 floats = 4.19 MB in
fp16**, held for the whole session and never re-read from a KV cache.

## The update rule

For a token `x_t`, with `u_i = normalize(W_i x_t)` and `β_i = 2 · sigmoid(b_i)`:

```
H_i = I − β_i u_i u_iᵀ                         (generalized Householder)
S_t = (H_1 H_2) S_{t−1} + g_t ⊗ v_t            (rank-1 write, gated)
y_t = S_tᵀ q_t
```

`H_i` has eigenvalue `1` with multiplicity `d_m − 1` and eigenvalue `1 − β_i` on the
reflection axis. Because `β_i ∈ (0, 2)`, that eigenvalue lives in `(−1, 1)`: the transition
can rotate, can project, and — critically — **can flip sign**.

## Why the sign matters, in one paragraph of math

A state-tracking problem (parity, modular counting, S5 composition) is a word problem over a
finite group. A layer can solve it in one step only if its state transition can represent an
element of order 2. Attention without positional encoding cannot do this at all; a recurrent
layer can only if its transition matrix has a negative eigenvalue. The delta rule

```
S_t = (I − β_t k_t k_tᵀ) S_{t−1} + β_t v_t k_tᵀ,   β ∈ (0, 1)
```

has eigenvalues in `(0, 1]`, so every update contracts along the key direction and no product
of updates can reproduce `−1` without decaying it first. That is the mechanism behind the
"illusion of state" result: softmax attention and gated linear recurrences are both in TC⁰
and fail parity at length. Day 2 had a delta-rule memory and inherited the limit.

Householder with `β ∈ (0, 2)` fixes it with one extra scalar of range: `β = 2` is an exact
reflection, `β < 1` a contraction, `β > 1` a sign flip plus shrink. Assertion [6] checks the
range is actually reachable at initialization: **max β = 1.52 ∈ (0, 2)**.

## Why two reflections and not one, and not a full matrix

- One reflection has a single negative axis. Two reflections compose into a rotation in the
  plane they span (`H_1 H_2` has a complex eigenvalue pair), so `n_h = 2` is the smallest
  count that gives a *rotation*, not just a flip.
- Cost per token per layer: each `H_i` applied to a `d_m` vector is one dot product plus one
  axpy ≈ `3 d_m` MACs, so `n_h = 2` costs ≈ `6 × 512 = 3,072` MACs. A dense `512 × 512`
  transition costs 262,144 — **85× more** — for a strictly less structured operator.
- Parameter cost of the reflection basis is `d × n_h = 8,192`, i.e. 0.06 % of the memory
  module; the memory is a *projection* cost, not a reflection cost.

State capacity, priced against the thing it replaces: the 4 k window KV cache for one local
layer is `4,096 × 576 = 2,359,296` floats. The memory state is 131,072 floats — **18×
smaller** — and it does not reset at token 4,097. That is the whole trade: 18× less state, one
order of magnitude more reach.

## Numerical behaviour

- Reflections are applied as two rank-1 updates, never as a materialized matrix, so no
  inversion and no `O(d_m²)` error accumulation.
- `β = 2` is exactly orthogonal; `β ≠ 2` scales the reflection axis by `1 − β`, so the state
  norm can grow only through the gated write, not through the transition. This is why the
  fp16 state is stable in practice; the master copy is kept in fp32 and cast per step, and
  §08 lists the drift test that would catch it if it were not.
- The gate `g_t` is a per-slot scalar (not per-channel), which keeps the write rank-1 and the
  state interpretable as `r_m` independent key-value memories.

## Rejected alternatives

- **Delta rule / Mamba-2 SSD.** Cheaper by 8,192 parameters per layer and well understood, but
  eigenvalues are confined to `(0, 1]`; C4 predicts it fails parity at length 64 in-context.
  Day 2's memory is exactly this control.
- **Full `d_m × d_m` transition.** 262,144 parameters per layer, 8× the whole memory module,
  and it can represent transitions the task does not need. Rejected as unearned capacity.
- **No memory (attention only).** 203.6 M fewer parameters and 4.19 MB less session state, but
  the state-tracking and 1 M recall arguments both collapse onto the four global layers.
- **Per-channel decay (Mamba-style) instead of Householder.** Adds `d_m × r_m` parameters and
  gives eigenvalues in `(0, 1]` per channel — the same sign problem, at 256× the transition
  parameters.

## Falsifiers

- **C4 — Householder vs delta rule.** In-context parity and S5 composition at length 64, 16
  random seeds. Householder must reach ≥ 95 % exact and beat the delta-rule control by ≥ 3
  points. If it does not, the 203.6 M is paying for an eigenvalue that the optimizer cannot
  find, and the memory reverts to the delta rule.
- **State drift.** fp16 state, 1 M steps, synthetic stationary input: max relative norm drift
  must stay below 1e-2. Above that, the state stays fp32 and the 4.19 MB becomes 8.4 MB.
