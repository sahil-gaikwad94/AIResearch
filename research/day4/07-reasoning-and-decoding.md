# 07 — Reasoning and decoding: CRANE, 0-DoF splicing, CDCL jumpback

Three mechanisms share one budget: the token. This chapter prices them against each other and
against doing nothing.

## The decode budget

Forward FLOPs per token, from `tools/count_params_v4.py`:

| `r` | @4 k | @64 k | @1 M |
|---:|---:|---:|---:|
| 1 | 4.68 GFLOP | 4.73 | 4.75 |
| 2 | 5.06 | 5.11 | 5.12 |
| 4 | 5.80 | 5.86 | 5.87 |
| 8 | 7.30 | 7.35 | 7.37 |

Two things to read from this table. First, **context is nearly free**: going from 4 k to 1 M
adds 0.07 GFLOP/token (1.5 %), because the local layers are windowed and the global layers
read 4,864 selected keys regardless of length. Second, **depth is not**: `r = 8` costs 56 %
more than `r = 1`. The recurrent dial is the expensive axis, which is why it is a dial.

The workspace accounts for most of that slope: one extra iteration costs
`2 × 120,084,482` dense FLOPs plus workspace attention ≈ 0.38 GFLOP/token, and 7 extra
iterations ≈ 2.6 GFLOP — the 4.68 → 7.30 difference. A decoded token at `r = 1` thus costs
12.3× a workspace iteration, which is day 3.1's rent argument, still true.

## CRANE: two modes, one model

`CRANE` (context-independent reasoning and emitting) splits generation into an unconstrained
scratchpad window (` thinking…</think>`) and a constrained emission window. The mask is a
function of the window, not of the content:

| Mode | Tier 0 mask | Tier 1/2 | Purpose |
|---|---|---|---|
| `think` | off (any bytes) | off | let the model compute, plan, and be wrong privately |
| `emit` | exact JSON/schema | on | guarantee the artifact |

The point is not that reasoning should be unconstrained for its own sake; it is that a schema
mask applied during reasoning costs accuracy *and* buys nothing, because intermediate tokens
are not part of the artifact. Day 3.1 applied the mask to the whole stream and paid a
measurable syntax tax on every thought. CRANE moves the mask to the boundary, which is also
where tier 3's checkpoints live: a clause is only checkpointed once the model commits to
emitting a value, so a jumpback never has to reason about scratchpad tokens.

The controller in `kestrel4/model.py` is the reference: it masks with the tier-1 interval
widened by `nextafter`, then applies the future-volume twist.

## The future-volume twist

For a numeric field with domain `[lo, hi]` and a container with remaining capacity `C`, a
candidate value `v` is admissible if `v ∈ [lo, hi]` and `v ≤ C`. Among admissible candidates,
the twist adds

```
β · log( min(hi, C) − v )  /  (hi − lo),     β = 0.5
```

so a value that leaves more residual polytope volume is favoured. Assertion [8] checks both
halves on real logits: the out-of-bounds candidate 0.65 is `-inf`, and the wider-residual
candidate 0.10 outscores 0.45. This is a *soft* preference on top of a *hard* mask — the mask
is what makes it sound, and the twist is only allowed to reorder what the mask already
permitted.

## 0-DoF splicing: skip the token entirely

A field is 0-DoF when every input of its derivation is already committed. Then the value is
not a prediction; it is an evaluation, and the model should not be sampled at all. The runtime
computes the canonical text, heals the BPE boundary against the last emitted token, and splices
the tokens into the stream.

The saving is not per-token arithmetic, it is per-token *steps*: 28 % of decode steps in a
typical structured patch are deterministic. Against a 600-token patch:

| Batch | GB/step | Raw | With 0-DoF | Speedup | tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.84 | 150.71 ms | 118.11 ms | **1.28×** | 5,080 |
| 16 | 3.92 | 43.93 ms | 32.23 ms | **1.36×** | 18,617 |
| 64 | 8.86 | 24.79 ms | 18.00 ms | **1.38×** | 33,330 |

The model charges 16 µs of CPU splicing overhead per token, which is why the speedup is not
1/0.72 = 1.39× at batch 1: at low batch the CPU cost is exposed, and it is repaid only when
the deterministic fraction is high. The crossover is at 16 µs / 251 µs ≈ 6.4 % deterministic
fields (`f · T = 16 µs`); below that, 0-DoF is a loss. That is the honest boundary, and it is why the mechanism is gated on a
runtime count rather than enabled globally.

Note what the table does *not* claim: splicing does not reduce the FLOPs of the remaining
steps. It reduces their number. A patch with 0 % deterministic fields runs at exactly the raw
speed.

## CDCL jumpback: the third use of the same 600 tokens

When tier 1 or tier 2 refutes a partial document, the decoder has three options: rewind one
clause (CDCL), rewind the patch (day 3.1), or ask the model to fix it with a prompt (a judge
loop). The costs are 52.4 GFLOP, 1,746 GFLOP, and an unbounded number of tokens respectively.

Jumpback scores each clause in the unsat core by

```
score = H(clause) · exp(−λ · (k − i)) + γ · conflicts(clause),   λ = 0.25, γ = 0.5
```

where `H` is the mean token entropy over the clause, `k − i` is how far back it is, and
`conflicts` counts previous refutations. High entropy means the model was unsure there, so the
error is more likely to be the model's; recency means the clause is cheap to undo. Assertion
[8] constructs the discriminating case: clause `w1` at token 4 with entropy 2.4, clause `w2` at
token 9 with entropy 0.6, both in the core. The controller rewinds to **token 4**, not 9, and
tightens `w1`'s upper bound to 0.43.

The rewind also snaps the fast-weight states back to the checkpoint, so the memory does not
carry the refuted branch forward — that is the reason the checkpoint stores `states`, not just
token positions. A jumpback that restored tokens but not state would silently duplicate the
error it was repairing.

## Rejected alternatives

- **Mask everything, always (day 3.1).** Costs accuracy in the reasoning window and cannot
  express "this value is determined"; the mask becomes a tax on thought.
- **Sampling every field, then validating.** That is whole-patch retry with extra steps:
  1,746 GFLOP per conflict against 52.4.
- **Greedy rewind-one-clause without entropy weighting.** Chronological backtracking on a
  600-token patch degenerates to whole-patch retry when the conflict is early, and it rewinds
  to the *last* clause, which is usually not the culprit.
- **Speculative decoding with a draft model.** Reduces per-token latency but not per-token
  count, and a draft model that does not know the constraint system will propose tokens that
  the mask then throws away. 0-DoF splicing removes work that is *known*, which is a different
  and better-conditioned 30 %.
- **Unconstrained reasoning with a separate extraction pass.** Two passes over the same tokens
  for a guarantee that the second pass cannot provide (extraction is a model, not a verifier).

## Falsifier

**C7 — 0-DoF splicing at batch 1.** On a 600-token patch benchmark with ≥ 30 % deterministic
fields: net wall-clock speedup must be ≥ 1.2×. Below that, the 16 µs/token overhead is not
repaid and splicing should be limited to patches with ≥ 40 % deterministic fields. A secondary
kill: if the healed splice ever produces a token sequence that the tier-0 automaton rejects,
the healing is unsound and the path must be removed rather than repaired.
