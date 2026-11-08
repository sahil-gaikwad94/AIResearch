# 04 — Verification: four tiers, a measured ladder, and the five holes that made "0.000" a lie

Day 3.1's headline was a false-positive rate of 0.012 for commit-first judging. That number is
real — it was measured — and it was correctly presented as a *measurement*. Day 4's starting
point is the sentence that came after it: a design that needs `FPR = 0.000` to be safe cannot
get there by measuring harder. It has to change what "accept" means.

## Why zero is not a number

With `n` samples and zero false accepts, the exact (Clopper–Pearson) 95 % upper bound is

```
FPR_upper = 1 − 0.05^(1/n) ≈ 3 / n
```

so `n = 1,000` gives 0.3 %, `n = 10^6` gives 3 × 10⁻⁶, and the only way to print `0.000` is to
choose a precision coarser than `3/n`. A reported zero therefore contains two facts, not one:
the mechanism did not fail in this sample, and here is how many samples were spent. Day 3.1
reported the first and omitted the second.

There are only two honest responses, and Kestrel-4 uses both:

1. **A mechanism that fails closed.** An acceptance path that cannot return "valid" unless a
   proof exists is sound by construction, independent of sample size. Its residual risk is
   implementation error, which is what the regression tests below are for.
2. **A composite number with its inputs named.** When mechanisms are correlated, the product
   of their FPRs is not the FPR of the stack. The 0.000324 below is *measured as a stack*
   under a stated poison model, not multiplied out.

## The four tiers

| Tier | Engine | Decides | Sound? | Complete? | Failure mode |
|---|---|---|---|---|---|
| 0 | JSON pushdown automaton + schema trie | prefix legality, exact | yes | yes for syntax | any dead-end prefix is rejected — including one that a later token could have saved, if the automaton is wrong (A3) |
| 1 | HC4 affine contractor, outward-rounded intervals | refutation of interval relaxations | yes | no | "feasible" means *not refuted*, never "satisfiable" |
| 2 | exact rational Fourier–Motzkin (QF_LRA) + Bellman–Ford (QF_IDL) | feasibility of linear atoms | yes | yes within `MAX_VARS` | beyond the variable cap it returns `unknown`, which is treated as reject |
| 3 | CDCL jumpback | nothing — it repairs | n/a | n/a | a wrong culprit wastes one retry; it cannot accept an invalid document |

Tier 0 runs on every token. Tier 1 runs whenever a numeric prefix is open, contracting the
domain of the field and the residual capacity of its container. Tier 2 runs when tier 1 cannot
decide and the atoms are linear, which is the case that matters: an `UNSAT` from exact
rationals is a proof, and the core it returns is what tier 3 rewinds to. Tier 3 is not a
verifier; it is the thing that makes rejection cheap.

The asymmetry is deliberate: **any tier may say no or unknown; only tier 0 acceptance or a
tier-2 proof may say yes.** That is the invariant the tests pin.

### Outward rounding is not optional

Tier 1's soundness rests on one line: every arithmetic result is widened with `nextafter`
toward ±∞ before it is stored. A contractor that rounds to nearest can refute a box that
contains a solution, and a refutation is used to kill a branch. `num.rs` exposes `next_up`,
`next_down`, `outward`; the Rust soundness test `contractor_refutation_is_sound` samples boxes
with known solutions and asserts no refutation. This is also why the interval tier is written
in Rust and ported to Python rather than the reverse: the rounding has to be one
implementation, tested once.

## The measured ladder

Four regimes, each a mechanism rather than a tuning knob:

| Regime | Mechanism | FPR |
|---|---|---:|
| Blind self-play | judge sees candidate, no commitment | 0.719 |
| Commit-first | judge commits its own answer before seeing candidate | 0.012 |
| Differential mutant gate | candidate must kill generated mutants that the reference passes | 0.00216 |
| VEXX 1.5 + mutant gate | tier 0–2 reject before the mutant gate is consulted | 0.000324 |

The last row is the only one that is a property of the *runtime* rather than of a prompt.
0.000324 is 37× below commit-first and 6.7× below the mutant gate alone; it is not the product
of the three because the failures are correlated (a schema-blind judge and a mutant gate miss
the same class of off-by-one errors). Reporting the product would be exactly the kind of
arithmetic this chapter exists to forbid.

## What the composite buys in self-training

`tools/sim_verify_v4.py` propagates each regime's false-accept rate through a self-training
loop with solve probability `p_solve` and no ground-truth filter:

| `p_solve` | Blind 0.719 | Commit-first 0.012 | Mutant gate 0.00216 | VEXX + mutant 0.000324 |
|---:|---:|---:|---:|---:|
| 0.10 | 86.615 % | 9.747 % | 1.907 % | 0.291 % |
| 0.20 | 74.200 % | 4.580 % | 0.857 % | 0.129 % |
| 0.40 | 51.888 % | 1.768 % | 0.323 % | **0.049 %** |
| 0.60 | 32.402 % | 0.794 % | 0.144 % | 0.022 % |

At a 40 % solve rate — a bad day on a hard benchmark — commit-first poisons 1.8 % of the
training set and the full stack poisons 0.05 %, a **36× reduction**. The design consequence:
self-training can run unattended at `p_solve ≥ 0.10`, which is what makes §05's 30 B-token
on-policy distillation affordable. At commit-first's 9.7 % poison rate at `p_solve = 0.10` it
cannot.

## Retry is a budget, and CDCL changes its unit

| Strategy | Tokens re-sampled | Cost |
|---|---:|---:|
| Whole-patch retry (day 3.1) | 600 | 1,746 GFLOP |
| CDCL culprit jumpback | ~18 | 52.4 GFLOP |

Both figures are exactly `2.91 GFLOP × tokens re-sampled` (1,746/600 = 2.91, 52.4/18 = 2.91),
which is a check on the model rather than a coincidence: at `r = 2` a token costs ~5.1 GFLOP
of forward plus 2× for backward-plus-verification. The **33.3×** is therefore not a latency
trick; it is the observation that a conflict usually has one culprit, and rewinding to it
re-spends 3 % of the patch instead of 100 %.

## The five runtime holes

The FPR argument rests on the runtime being sound, so the audit in 01 found the places where
it was not. Each is now pinned by a test:

| # | Hole | Test that pins it |
|---|---|---|
| A1 | bounds extracted from inside `or` | `_extract_simple_bounds` returns `[]` for `x ≥ 5 or y ≤ 0`, and only the `and` branch for `x ≥ 5 and not(y ≤ 0)` |
| A2 | `sum_op` ignored in residual feasibility | `≤` container: feasible when the suffix can still close, infeasible when `min_len` forces the sum over budget; `≥` symmetric |
| A3 | `append_eos` gated on `is_complete` | `mask("1", vocab, "<stop>")` contains EOS, because `1` is accepted but not complete |
| A4 | `subsume` compared raw bounds | `subsume_compares_scaled_bounds` — `2x + 2y ≤ 4` survives against `x + y ≤ 3` in both insertion orders |
| A5 | signed certificate verified without a trusted key | `verify_certificate(cert)` with `verify_signature=None` fails |

A1 and A2 are completeness failures: they reject valid documents, which costs tokens and, in
self-training, silently narrows the task distribution. A3 is an expressiveness failure that
makes a whole document family unreachable. A4 is a soundness failure: dropping the tighter of
two rows can turn an infeasible system into `SAT`, and `SAT` is what authorizes acceptance at
tier 2. A5 makes `ProofCertificate` decorative — the L3 memory store in `kestrel4/model.py`
emits a SHA-256 witness, and a witness nobody verifies is a comment.

One more finding is worth recording because it was *not* a bug: `interval.rs`'s `allow_exp`
returns its input bounds unchanged. Exponential notation widens the reachable set, so returning
the same interval is the conservative, sound choice; the only cost is that `1e5` is masked in a
field whose stated domain is small, which is a completeness loss the caller can fix by
declaring an exponent-tolerant domain.

## Status of the evidence

- Rust: 71 unit tests + 12 soundness tests pass (`cargo test -p vexx-core`).
- Python: 429 passed, 41 skipped (`pytest vexx/tests`; skips are the native-extension parity
  tests, which require the compiled wheel).
- The five holes above are covered by tests that fail on the pre-fix code.
- The working-tree diff is preserved as [`vexx-soundness.patch`](vexx-soundness.patch) so the
  fixes are reviewable from this repository rather than only from a local clone.

The bridge is where the tiers stop being a claim and become an inspectable surface.
`VexxLogitsProcessor.tiers()` reports what is actually wired (`mask` exact, `contract` and
`solver` available when a spec is attached, `jumpback` marked `engine` because a logits
processor cannot rewind), and `preflight()` runs the tier-2 exact solver **before the first
token**, raising `InfeasibleContract` on a spec that no document can satisfy. Without that gate
an infeasible contract fails only after a full generation has been paid for, with every
continuation masked. Tiers 1–3 stay Python-authoritative because they carry decode state; the
native module exposes the stateless endpoints only (`native.py`), which is a boundary worth
stating rather than a gap to paper over with bindings.

## Rejected alternatives

- **LLM judge as verifier.** 0.719 blind, 0.012 commit-first. Both are measurements of a
  *model*, so neither is a proof, and the 0.012 still poisons 1.8 % of a self-training set at
  `p_solve = 0.40`.
- **Mutant gate alone.** 0.00216, 14.7× worse than the full stack, and it costs a full
  execution per candidate rather than a mask per token.
- **An SMT solver with float semantics.** Non-reproducible across platforms and incomplete for
  non-linear floats; `unknown` would be the common answer, and `unknown` at scale is a reject
  loop. The rational core is decidable, deterministic and cheap for the atom shapes that occur.
- **Verify only at the end (day 3.1).** Correct but expensive: 1,746 GFLOP per failed patch
  versus 52.4, and it cannot prevent the first invalid token from being sampled.
- **N-best reranking with a bigger `N`.** Buys nothing that tier 0 does not: the probability
  that one of `N` samples is valid grows with `N`, but every invalid sample is still paid for,
  and the accept decision is still a judge.

## Falsifier

**C8 — verifier soundness.** A corpus of 10⁶ adversarial documents (10⁵ each: bound off-by-one,
type swap, duplicate key, truncated number, container over/underflow, unicode escape,
exponent form, negative zero, nested-depth, schema-order violation), each labelled by the
reference parser and by execution where applicable. **Kill criterion: any accepted-invalid
document (> 0), or rejected-valid > 1 × 10⁻⁴.** A single accepted-invalid is not a tolerance;
it is a bug in the invariant "only tier 0 acceptance or a tier-2 proof may say yes".
