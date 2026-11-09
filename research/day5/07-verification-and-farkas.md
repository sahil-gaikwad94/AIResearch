# 07 — VEXX 2.0 controller, exact Farkas cuts and L3 certificates

The neuro-symbolic layer. It has four tiers, each with a distinct soundness
scope, and one non-negotiable rule: **fail closed**.

## Tier 0 — lexer-aware delimiter automaton

Delimiters are counted on a *lexed* copy of the prefix: string literal and
comment contents are replaced by spaces *at identical indices*, so a `")"`
inside a string or a comment can never close a real delimiter. This is the
smallest thing that makes the automaton sound; a production build uses the
real lexer. Audit: `s = "("` still rejects `)`.

## Tier 1 — symbol-table contractor (L3)

- **Exact arity** for the innermost open call: depth-aware comma counting, so
  `f(g(x), y)` reports 2 arguments for `f`, not 3. A premature `)` and a
  trailing comma are both rejected.
- **Declared identifiers and return type**: unknown callees and wrong-returning
  callees are masked.
- **Outward-rounded interval masking** with the SHIM volume twist (Farkas-2
  volume term).

`expected_return_type` is a real constraint: with `"int"` required, a
`bool`-returning callee is masked. This is the Mündler/SynCode/XGrammar family
of type-constrained decoding, reduced to the part we can verify in NumPy.

## Tier 2/3 — exact-rational Farkas cuts (fixes F5)

`compute_exact_farkas_cut` solves one violated linear constraint for the
culprit variable in `fractions.Fraction` arithmetic and rounds the implied
bound **outward by one ULP** (`+inf` for an upper bound, `−inf` for a lower
bound) — so floating-point rounding can never exclude a feasible point. This
replaces Day 4's hardcoded `-0.05`.

### The ladder (simulator, greedy-max decoder)

| scale | Day-5 exact cut | Day-4 `-0.05` |
|---|---|---|
| 1e-3 | 1 | **domain_empty (false UNSAT)** |
| 1e-2 | 1 | **domain_empty (false UNSAT)** |
| 1e-1 | 1 | 2 |
| 1 | 1 | 15 |
| 10 | 1 | 150 |
| 100 | 1 | 1,501 |
| 1000 | 1 | **15,001** |

And with two binding constraints, the exact method needs exactly 2 cuts at
every scale — iteration count tracks the number of distinct binding
constraints, not the magnitude of the violation. The magic constant's
iteration count is `O(violation/0.05)`, which is the entire problem.

### Two soundness subtleties, both tested

1. **Cut polarity.** If the culprit's coefficient `a_c > 0`, the implied
   bound is an *upper* bound. If `a_c < 0`, it is a *lower* bound — and
   assigning it as an upper bound is unsound (it would exclude feasible
   points). The controller routes by sign; the simulator asserts the
   negative-coefficient case (`−2y ≤ −50` → `bounds = (25, 100)`, sign −1).
2. **The bound is cube-local.** A Farkas cut is implied by one violated
   constraint under the *currently committed* values of the other variables.
   It is a valid tightening of that cube, not a globally valid bound — so a
   rollback of the cube must roll the bound back too. The DPO pair records
   the new bound alongside the rewind index for exactly that reason, and the
   simulator's outward-rounding check asserts the widened bound still
   contains the true optimum (at most 1.5 ULP of slack: 0.5 ULP for
   converting the exact rational to float, plus 1 ULP for the outward step).

`trigger_cdcl_jumpback` picks the culprit by the entropy score
`H(c)·exp(−λ(k − i_c)) + γ·conflicts(c)` over the live unsat core, cuts its
domain, and rewinds **fast weights and KV cache together** to the checkpoint
(causal slicing for `ckv`/`k_pe`/`window_ckv`, block-prefix slicing for
`block_summaries`). The audit asserts the KV comes back at the rewound length
(4, not 12) and the states are bit-identical to the checkpoint.

**Fail-closed:** if the unsat core matches no live checkpoint the controller
raises instead of silently pointing at checkpoint 0. The audit asserts the
`ValueError`.

## L3 proof certificates

`emit_l3_proof_certificate` produces a content-addressed certificate (SHA-256
over the witness, the SMT-LIB assertion and the check predicates). A valid
witness is accepted; a forged one is rejected; and — critically — **an empty
check list fails closed** (`verified = False`), because "no checks" is not the
same as "checked". The audit asserts all three.

## MTP speculative tree pruning

`prune_mtp_speculative_tree` drops draft branches that violate Tier 0/1
before any trunk verification, and enforces `max_spec_depth` (3 at FULL). The
audit shows 4 of 5 draft branches pruned, and a 5-token branch rejected by the
depth cap. Note the cap is implemented as a *length* bound on the branch; the
neural draft heads that would drive it are a Day 6 deliverable (doc 06).

## VEXX-Shell and CRANE-CAS

- **VEXX-Shell**: masks CLI flags absent from the command schema and paths
  absent on disk (`--fake-flag` and `tests/ghost.py` are masked in the audit).
- **CRANE-CAS**: exact rational evaluation of flat linear/rational expressions
  — `"3/7 + 2/5·x − 1/35"` with `x = 3/2` returns `"1"`, `"1/3 + 1/6"` returns
  `"1/2"`. No floats, ever.
