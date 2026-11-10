# 03 — VEXX 3.0: six tiers, six rejected alternatives

VEXX runs *inside* the decode loop, not after it. Each tier owns one class of mistake, and each
tier exists because a cheaper mechanism failed in a way that is easy to state.

## Tier 0 — CRANE dual phase + SWYB budget

`crane_phase()` returns `UNCONSTRAINED_THINK` while the text is inside ` thinking...</think>`,
otherwise `STRICT_VERIFIED`. In the think phase the mask is a no-op: constrained decoding from step
0 degrades chain-of-thought accuracy by up to ~10 points (CRANE, ICML 2025). After `</think>`, the
Tier-1 guard is active.

SWYB adds a distance-to-acceptance check: if `remaining_budget <= open_delimiter_depth + 1`, tokens
that would open another `([{` are masked, because the generator can no longer close them.

**Alternative rejected:** always-on regex masking. **Falsifier:** an unconstrained block that never
reaches a final answer, or a masked generation ending with an open delimiter.

## Tier 1 — stdlib lexer + live `ast` symbol table

`_lex_code_state()` tracks, character by character:

* single, double and triple quotes;
* raw strings (backslashes do not escape quotes there);
* comments;
* f-string `{...}` expression holes, including nested braces;
* bracket stack with the callee name attached to each `(` (and comma count).

`extract_ast_symbol_table()` parses `def` headers with Python's own `ast` and records
`(min_arity, max_arity, return_type)`, including defaults, `*args`/`**kwargs` (`max_arity = None`).

The mask rejects unbalanced brackets, unknown callees, arity violations, and spoofed receivers:
`evil_obj.compute(...)` is blocked while `self.compute(...)` is allowed.

One subtle case is now explicitly rejected: a quote inside an f-string expression hole
(`f"{compute(1"`). Before the fix, the lexer started a new string and accepted the malformed prefix.
**Alternative rejected:** regex brace counting, which cannot see quotes, comments, or holes.
**Falsifier:** a string-smuggled `)` accepted, or a receiver spoof bypassing arity.

## Tier 2 — exact CAS

`crane_cas_evaluate()` parses the expression with `safe_sympy()` — a restricted AST→SymPy builder,
*not* `sympify` — substitutes the environment, simplifies, and returns a rational `p/q` or
`DIV_BY_ZERO` for any `zoo/oo/nan`. The builder accepts only numeric constants, names, `+ - * / **
% //`, unary sign, and a short function allowlist, with 512-character, 200-node, exponent 10,000
and 4096-bit integer caps. `solve_linear_system()` returns the first exact solution tuple;
`integrate_definite()` computes a definite integral; `modular_pow()` uses Python's three-argument
`pow`.

This was not theoretical. The audit executed `sympy.sympify("__import__('os').getcwd()")` and got
the working directory. The replacement raises `ValueError` before SymPy sees an import, attribute,
subscript, comprehension, or lambda; the regression test submits all of those plus `2 ** 1000000`.

**Alternative rejected:** floating-point evaluation, which cannot return exactly `1` for
`(5/6 - 1/4) / (7/12)`, cannot cancel `x + x - x`, and cannot distinguish a symbolic `x` from a
number. **Falsifier:** any of the benchmark's exact-answer items differing from the closed form.

## Tier 3 — single-row bound contractor

This is a *prototype* bound contractor, **not** a general Farkas certificate and **not**
a complete CDCL theory solver. Given one row ``a_c x_c + Σ_{j≠c} a_j x_j ≤ rhs`` it derives
one bound for ``x_c`` along the current assignment slice, widens it outward by one ULP with
``math.nextafter``, and rewinds the hybrid cache to the culprit checkpoint. It is disabled by
default in general reasoning. **Falsifier:** a widened bound excluding a feasible point. The
test asserts the exact Fraction, the outward direction, and that the GDN state survives the
rewind unchanged. This tier does not implement general Farkas certificates or a complete linear
CDCL theory.

## Tier 4 — epistemic calibration (demoted to regression fixture)

``verify_epistemic_calibration()`` inspects the prompt for statistical, temporal, sampling,
contradiction, and premise traps:

| Trap | Verdict |
|---|---|
| 95% CI spanning 0, or ``p >= 0.05`` | ``not established`` |
| Newer dated official notice vs older blog | the newer value (e.g. ``8 units``) |
| Voluntary online poll generalized nationally | ``not justified`` |
| Unverified award or entity | ``INSUFFICIENT EVIDENCE`` |
| Fictional place | ``UNKNOWN`` |
| Table vs paragraph contradiction | ``CONFLICT`` |
| ``0.999... < 1`` premise | ``equal`` (the premise is false) |

The Tier-4 prompt-level **answer bypass was removed** from normal generation (P0.6): a rule
checker may flag a prompt as needing a grounded workflow, but it must not fabricate a completed
natural-language answer on the model's behalf. Tier-4 patterns are kept only as regression
fixtures. The normal factuality pipeline inspects actual generated claims and their evidence.

## Tier 5 — sanitized execution + DPO collector

`safe_execute_python()` walks the AST and rejects `Import`/`ImportFrom`/`Global`/`Nonlocal`, file
and network names (`open`, `os`, `sys`, `subprocess`, `eval`, `exec`, ...), and dunder attributes.
It then executes in a **separate Python process** (`-I -S`, minimal environment, temporary working
directory, `RLIMIT_CPU=2s`, `RLIMIT_AS=512 MiB`, `RLIMIT_NOFILE=64`, 5 s wall timeout) against
**deep-copied** inputs, so an algorithm that mutates its argument is refuted rather than rewarded.
This is process isolation with resource limits, not a hardened container: there is no seccomp,
user namespace, or network namespace, so it is suitable for benchmark-sized pure-Python candidates,
not adversarial native code.

`select_verified()` scores a pool and records a preference pair only when **exactly one** candidate
passes every test and at least one fails; ambiguous pools produce no data. Refuted candidates are
stored as losers. `export_dpo()` writes JSONL. This is a collector, not a DPO training loop.

`execute_capture()` reuses the same sandbox for trace tasks where the answer is observed rather
than asserted.

**Alternative rejected:** in-process `exec` on raw text, which could read files or loop forever in
the evaluator itself. **Falsifier:** an import, file access, infinite loop, or allocation bomb
reaching the main interpreter — the tests submit all four and require a zero score with a
`Forbidden`/`Timeout`/`SandboxError` diagnosis.

## Why six tiers instead of one big verifier

Because the failure modes are different in kind. A statistical claim is not a bracket-matching
problem; an exact rational cut is not an AST problem; and a sandbox cannot tell you whether a
confidence interval crosses zero. Collapsing them into one score would make it impossible to say
*which* mechanism caught *which* error — which is exactly the kind of unverifiable claim this
research log exists to avoid.
