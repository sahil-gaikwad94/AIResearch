# 03 — Memory: Householder reflections and the 4.19 MB session state

Each of the 16 layers carries a fast-weight memory with `d_m = 512`, `r_m = 256` slots and
`n_h = 2` generalized Householder reflections.

| Component | Parameters |
|---|---:|
| 6 projections `d = 4096 → d_m = 512` (query, key, value, second key, gate, output) | 12,582,912 |
| `beta` head, `d → n_h` | 8,192 |
| read matrix `A`, `d_m × r_m` | 131,072 |
| **Per layer** | **12,722,176** |
| **All 16 layers** | **203,554,816** (0.96 % of the model) |

Session state is `r_m × d_m` per layer = 131,072 floats → **2,097,152 floats = 4.19 MB in
fp16**, held for the whole session and never re-read from a KV cache.

## The update rule

For token `x_t`, with `u_i = normalize(W_i x_t)`, `β_i = 2 · sigmoid(b_i)` and `w_t = W_v x_t`:

```
H_i = I − β_i u_i u_iᵀ                          (generalized Householder)
B_t = η · w_t ⊗ k_t  +  H_2 H_1 B_{t−1}         (write, then two reflections)
y_t = A (B_{t−1}ᵀ q_t)                          (read before write: causal)
```

`H_i` has eigenvalue `1` with multiplicity `d_m − 1` and eigenvalue `1 − β_i` on the
reflection axis. Because `β_i ∈ (0, 2)`, that eigenvalue lives in `(−1, 1)`: the transition can
rotate, can project, and — critically — **can flip sign**. The write step `η` is a fixed buffer
of 1.0, so the spectrum of the transition is entirely determined by `β`; nothing about the
sign is a learned accident of initialization.

## The audit that made this section true

The first version of this chapter described exactly the rule above. The code did not implement
it: it applied `B ← B − (η β)(B u)uᵀ` with `η = 0.05`, i.e. a delta-rule step of at most 0.1.
The eigenvalue on the reflection axis was therefore `1 − 0.05 β ∈ (0.90, 1]` — **never
negative** — while the assertion in `model.py` checked only that `β` itself exceeded 1, which
is true of any sigmoid with a large enough bias. The mechanism lived in the prose, not in the
tensor.

`tools/audit_v4.py` (check A1) now measures the implemented step instead of trusting the
formula:

```
beta = 1.8  =>  measured eigenvalue on the u axis = -0.800
beta = 2.0  =>  max |H² B − B| = 2.4e-07      (an involution)
```

and the same harness caught the second half of the bug: the memory returned
`o(gate·y) + x`, while `Block` fed it `RMSNorm(x)`, so every memory layer **replaced the
residual stream with a normalized copy of itself**. Feeding an input with RMS 100 produced an
output with RMS 1 (ratio 0.01). The memory now returns a delta and the block adds it; A2
measures out/in RMS ratios of 1.004 and 1.000 at input scales 1 and 100.

## Why the sign matters, in one paragraph of math

A state-tracking problem (parity, modular counting, S5 composition) is a word problem over a
finite group. A layer can solve it in one step only if its state transition can represent an
element of order 2. Attention without positional encoding cannot do this at all; a recurrent
layer can only if its transition matrix has a negative eigenvalue. The delta rule

```
B_t = (I − β_t k_t k_tᵀ) B_{t−1} + β_t v_t k_tᵀ,   β ∈ (0, 1)
```

has eigenvalues in `(0, 1]`, so every update contracts along the key direction and no product
of updates can reproduce `−1` without decaying it first. That is the mechanism behind the
"illusion of state" result: softmax attention and gated linear recurrences are both in TC⁰ and
fail parity at length. Day 2 had a delta-rule memory and inherited the limit; so, silently, did
day 4's first draft.

Householder with `β ∈ (0, 2)` fixes it with one extra scalar of range: `β = 2` is an exact
reflection, `β < 1` a contraction, `β > 1` a sign flip plus shrink.

## Why two reflections and not one, and not a full matrix

- One reflection has a single negative axis. Two reflections compose into a rotation in the
  plane they span, so `n_h = 2` is the smallest count that gives a *rotation*, not just a flip.
- Cost per token per layer: each `H_i` applied to the state is one dot product plus one axpy
  ≈ `3 d_m` MACs, so `n_h = 2` costs ≈ `6 × 512 = 3,072` MACs. A dense `512 × 512` transition
  costs 262,144 — **85× more** — for a strictly less structured operator.
- Parameter cost of the reflection basis is `d × n_h = 8,192`, i.e. 0.06 % of the memory
  module; the memory is a *projection* cost, not a reflection cost.

State capacity, priced against the thing it replaces: the 4 k window KV cache for one local
layer is `4,096 × 576 = 2,359,296` floats. The memory state is 131,072 floats — **18×
smaller** — and it does not reset at token 4,097.

## Gradient flow: chunked, not truncated

The state is detached every 64 tokens (`self.chunk = 64`), not at every step. A per-step
detach makes the memory a fixed recurrence that can never learn *what to store*: the write
receives gradient only through the current token's read. Chunking gives each write credit for
up to 64 subsequent reads at a bounded graph depth (64 × 16 layers of state, not 4,096).

## Numerical behaviour

- Reflections are applied as two rank-1 updates, never as a materialized matrix, so there is no
  inversion and no `O(d_m²)` error accumulation.
- `β = 2` is exactly orthogonal; `β ≠ 2` scales the reflection axis by `1 − β`, so the state
  norm can grow only through the gated write, not through the transition.
- The master state is fp32 and cast per step; §08 lists the drift test that would force the
  state to stay fp32 (8.4 MB instead of 4.19 MB).

## Rejected alternatives

- **Delta rule / Mamba-2 SSD.** Cheaper by 8,192 parameters per layer and well understood, but
  eigenvalues are confined to `(0, 1]`; C4 predicts it fails parity at length 64 in-context.
  Day 2's memory is exactly this control, and day 4's first draft was accidentally this too.
- **Full `d_m × d_m` transition.** 262,144 parameters per layer, 8× the whole memory module,
  and it can represent transitions the task does not need. Rejected as unearned capacity.
- **No memory (attention only).** 203.6 M fewer parameters and 4.19 MB less session state, but
  the state-tracking and 1 M recall arguments both collapse onto the four global layers.
- **Per-channel decay instead of Householder.** Adds `d_m × r_m` parameters and gives
  eigenvalues in `(0, 1]` per channel — the same sign problem, at 256× the transition
  parameters.
- **Learned write step `η`.** One parameter per layer and a nicer story, but it puts the
  spectrum back in the optimizer's hands. The write is fixed at 1.0 and the transition owns the
  sign.

## Falsifiers

- **C4 — Householder vs delta rule.** In-context parity and S5 composition at length 64, 16
  random seeds. Householder must reach ≥ 95 % exact and beat the delta-rule control by ≥ 3
  points. If it does not, the 203.6 M is paying for an eigenvalue the optimizer cannot find,
  and the memory reverts to the delta rule.
- **C10 — chunked BPTT.** Compare chunk 64 against chunk 1 (per-step detach) at equal tokens.
  Kill chunking if it does not improve parity or validation loss by ≥ 1 point, since it costs
  graph memory proportional to the chunk length.
- **State drift.** fp16 state, 1 M steps, stationary input: max relative norm drift below 1e-2.
