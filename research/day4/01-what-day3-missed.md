# 01 — What day 3 missed

This chapter is an audit of `research/day1` through `research/day3/v2`, written after the
fixes rather than before them, because the interesting part is not the list of bugs. It is
that **all seven defects pointed the same way**: each one either made the design look better
than it was, or left a headline claim with no mechanism behind it. None was caught by the
assertion suites, because those suites recomputed the same analytic formula the bug lived in.
An assertion that derives its expectation from the code under test is a tautology with extra
steps.

The fixes are applied in the repository as of this commit. Numbers below are the post-fix
values produced by the tools in each directory.

## The seven

### 1. RoPE was constructed but never applied (day 3, day 3.1)

`kestrel3/model.py` and `kestrel31/model.py` both created a rotary projection `w_qr`, both
carried `rope_theta=5e5` in `FULL`, and both computed `q_pe`/`k_pe` inside `forward` — and
then summed only the non-rotary scores. The local layers were NoPE, and the parameter count
was *correct*, which is why nothing complained: the matrix existed, was initialized, received
gradients and was counted. It simply had no effect on the forward pass.

Consequence: the 256 k long-context claim rested entirely on the four global NoPE layers and
their block indexer. Within a 4 k window the model had no explicit position mechanism at all,
so the "local exactness" argument in `day3/v2/02-architecture.md` was describing a model that
did not exist.

Fix: apply per-head rotary embedding to `q` and `k` with `d_rope = 64`, `theta = 5e5`, in both
models. The ledger does not move (the parameters were already there), which is precisely why
this survived review: a parameter-preserving bug is invisible to a parameter-counting test.

### 2. The decode path bypassed the causal mask (day 2)

`kestrel2/model.py` built a square causal mask during training and returned `None` when
`m != n`, on the theory that a single-query decode step needs no mask. A single query against
`n` cached keys needs a *rectangular* mask whose causal cutoff is offset by the query's
position; returning `None` made incremental decode attend to every key, including keys to the
right of the query. Training and inference therefore had different receptive fields, and the
4,096-token window advertised in `day2/README.md` was not enforced at decode time.

Fix: `build_mask(m, n)` returns a rectangular mask with query offset `n - m`. The day 2
reference now produces the same attention pattern train and eval.

### 3. Expert bytes were under-counted by 16× (day 3, day 3.1)

Both serving models computed

```python
step_bytes = always + distinct_experts * exp_bytes        # wrong
```

which charges each distinct expert once per *step*. Experts are resident per *layer*: a step
that touches `k` distinct experts in each of 16 layers streams `16k` expert-weight copies, not
`k`. The missing `× L` made day 3.1's serving model claim **0.062 GB and ~54,000 tok/s at
batch 1**; the correct figure is **0.99 GB/step and 3,384 tok/s** — a 16× error in the
direction that flattered the design, in the chapter whose entire purpose was to price the
architecture honestly.

Fix: `step_bytes = always * 0.55 + L * distinct * exp_bytes`, in both `count_params_v3.py`
and `count_params_v31.py`. Day 3's batch-1 step is 1.81 GB (1,846 tok/s).

This is the most serious error in the corpus so far, and the lesson generalizes: **a serving
cost model must be written against the layer loop, not against the router's output.** The
router tells you how many experts; only the loop tells you how many times you pay for them.

### 4. `ckv_norm` was in every diagram and in no module (day 1, 2, 3, 3.1)

All four models normalize the query latent (`q_norm`) before the up-projection, and all four
diagrams show the key/value latent normalized the same way before `w_uk`. None of the four
implementations had the parameter. The absorption identity used to justify MLA —

```
q · k = (W_uq c_q) · (W_uk c_kv) = c_qᵀ (W_uqᵀ W_uk) c_kv
```

— is only the identity that was analysed if `c_kv` is the normalized latent; with an
unnormalized latent the trained model still works, but the matrix that gets absorbed is not
the matrix the documents describe, and the absorption error is unbounded in the latent norm.

Fix: add `ckv_norm` (one `d_c`-vector per layer) before `w_uk` in day 1, 2, 3 and 3.1, and
recompute ledgers (+512 parameters per layer at `d_c = 512`). Day 4's assertion [3] now pins
the property directly: standard and matrix-absorbed attention agree to **8.94e-08** in fp32.

### 5. A scorer that never ran, and a router counted twice (day 2)

`kestrel2/model.py` instantiated `w_ip` (coarse pooled-block scorer) after the indexer was
rewritten to use a single fine-grained scorer, and the accounting tool charged the router
matrix twice. Neither affects a forward pass; both affect a published number. Removed the
dead matrix, fixed the accounting, and recomputed: day 2 moves from **4,125,828,176** to
**4,140,256,336** inference parameters (**4,173,818,960** training), the increase coming from
per-head `w_qr` and `ckv_norm` and partly offset by the removed `w_ip`.

The deeper problem is the tool, not the matrix: `count_params_v2.py` and `model.py` shared a
formula but not a mechanism. Day 4 closes that by instantiating the model on `meta` and
comparing `sum(p.numel())` against the analytic count and against a hard-coded ledger
constant — three numbers that must agree, only one of which is the formula.

### 6. 256 experts that added no rank (day 3.1)

Kestrel-3.1's MoE routed top-6 of 256 experts, each an independent `d = 4096 → d_lat = 384`
bottleneck. Stored capacity scaled with 256; *routed* capacity did not, because every expert
projected into the same 384-dimensional latent. The union of six experts' outputs spans at
most 384 dimensions — exactly what one expert spans. Twenty thousand GPU-hours of expert
parameters bought specialization, but the layer's effective width per token never moved.

Fix: split the latent into 8 orthogonal subspace groups of 384 dimensions; route top-2 groups
then top-3 experts *within* each selected group. Routed rank becomes 2 groups × 3 experts ×
384 = **3,072** at the same 6 expert evaluations per token, an 8× increase in the rank the
token actually sees. Assertion [2] pins the 3,072, and the router is dropless (capacity
factor 1.0) so the rank is a guarantee, not an average.

This one is not a bug in the ordinary sense — it is a design whose stated benefit (fine-grained
capacity) was real and whose unstated cost (rank collapse to one expert's width) was not
priced. §06 gives the falsifier.

### 7. The verifier was unsound in five places (vexx)

The VEXX 1.5 runtime is the part of the stack allowed to say "this document cannot be
extended", so a hole in it is a hole in every number derived from it. Five were found and
closed; each has a regression test, and the Rust and Python suites now pass (71 + 12 Rust,
429 Python).

| # | Hole | Why it is unsound | Fix |
|---|---|---|---|
| A1 | `_extract_simple_bounds` descended into `or` | `x ≥ 5 or y ≤ 0` yields the hard bound `x ≥ 5`, which rejects valid documents | descend only into `and`; `or`/`not` are walls |
| A2 | `residual_feasible` ignored `sum_op` | a `≤` container was treated as `=`, so feasible partials were rejected (and the reverse for `≥`) | operator-aware check over the reachable suffix interval, lenient per `_can_close` |
| A3 | `append_eos` gated on `is_complete()` | a top-level number ends inside its token, so `1` could never terminate — a legal document family was unreachable | gate on `accepts()` |
| A4 | `subsume` compared raw bounds | two rows with the same normalized key but different scales (`x + y ≤ 3` vs `2x + 2y ≤ 4`) kept the *looser* one, which can flip UNSAT to SAT | compare `bound / |pivot|`, overflow → keep both rows |
| A5 | a signed certificate verified with no trusted key | signature verification returned "ok" when `verify_signature=None`, i.e. on every install | absent trusted key ⇒ verification fails |

A1 and A2 are completeness bugs (they reject valid work), A3 is an expressiveness bug, A4 is
a soundness bug in the strict sense (it can return SAT for an infeasible system), and A5 is an
authentication bug that made `ProofCertificate` decorative. A4 is the one that matters for
§04: the whole FPR argument rests on UNSAT being a proof, and a subsume step that silently
drops the tighter row destroys that.

## What day 3 got right

The audit should not read as a demolition. Three things from day 3 and 3.1 survive untouched
and are load-bearing in day 4:

- **Commit-first verification.** Making the judge answer before seeing the candidate drops the
  false-positive rate from 0.719 to 0.012. That is a mechanism, not a prompt trick, and day 4
  keeps it as tier-0 behaviour in §04.
- **The workspace rent argument.** A 4-layer recurrent workspace costs 14.2× less per token
  than a decoded token while adding depth only where the model asks for it. It survives into
  Kestrel-4 as `CodeJEPAWorkspace`.
- **Validity over capacity.** The VibeMemBench result — 11 of 12 memory systems fail to beat no
  memory — is a statement about trust, not size, and it is the reason the verification chapter
  is written before the training chapter.

## The rule that comes out of this

Every defect above was invisible to the check that existed and visible to a check that did
not. Day 4 therefore adds three checks that a formula cannot satisfy:

1. **Instantiate and count.** `meta`-device construction, `sum(p.numel())`, analytic formula,
   hard-coded ledger: four numbers, one assertion.
2. **Price the loop, not the router.** Serving costs are computed by iterating layers and
   charging per-layer residency (fix 3), not by counting the router's distinct experts.
3. **Test the property, not the shape.** Absorption is asserted by comparing two forward
   passes (8.94e-08), not by checking that the matrix exists; routing rank is asserted by
   multiplying out the active groups, not by checking `n_experts`.

## The second pass: auditing day 4 itself

The rules above were written for days 1–3.1, so the first thing to do with them was point them
at day 4. `tools/audit_v4.py` is that harness: 25 checks, each of which tries to refute a claim
in these chapters by measuring the mechanism in `kestrel4/model.py`. The first run refuted
**13 of its original 21**, and two were the same failure class as day 3.1's RoPE — a mechanism
that is counted, initialized, and inert. A second pass aimed at the long-context cache (§11)
refuted three more (A10–A12), including one that made 1 M decode unimplementable:

| # | Claim | Measured | Root cause |
|---|---|---|---|
| A1 | the memory's transition has a negative eigenvalue | `λ_min = 0.93` — never negative | a delta-rule step `η=0.05` was applied where a reflection was described; the assertion checked `β > 1`, which is true of any sigmoid |
| A2 | a memory layer preserves the residual stream | out/in RMS `0.01` | `Block` fed the memory `RMSNorm(x)` and the memory returned `+ x`, replacing the stream with a normalized copy |
| A3 | the indexer changes the forward pass | influence `0.0` | `(bonus − bonus.detach())` is zero in forward and only fabricates a gradient |
| A4 | FLOPs match the documented decomposition | +7.1 % to +11.9 % | one workspace pass double-counted, indexer over-charged by 2.7× |
| A5 | retries are discounted by `p(1−ρ)` | `p(1−0.85ρ)` | a fudge factor with no derivation |
| A6 | a violating witness is not certified | `verified: True` | the certificate was a hash, not a check |
| A7 | domain cuts keep the domain non-empty | `(0.05, 0.01)` | repeated half-space cuts were never bounded |
| A8 | the workspace respects its configured window | 0.876 of attention leaked | `4096` hard-coded instead of the config |
| A9 | per-component parameter counts match | all five match | — (this one held) |
| A10 | the indexer costs 0.019 GFLOP/token | 4.29 GB re-read per step at 1 M | block summaries were recomputed from every latent, not sealed |
| A11 | the cache is 4.35 GB at 1 M | true, but only 0.46 % is read | residency was priced as if it were the working set |
| A12 | per-token int8 is enough for the KV cache | cosine 0.9993 vs 0.99995 per-channel | latent channels carry outliers |

All thirteen are fixed. A1 is now an empirical measurement of the implemented step (`β = 1.8 ⇒
λ = −0.800`, `β = 2 ⇒ H² = I` to 2.4e-07), A3 uses a real 0.1 logit prior instead of a fake
gradient, A6 verifies linear checks with `Fraction`, and the cost model is cross-checked by an
independently written decomposition (0.00 % disagreement at every `r`).

The lesson is the one this chapter opened with, applied one level up: **the formula is not the
mechanism, and an assertion that recomputes the formula cannot tell you which one you built.**
The harness lives in the repository so the next draft has to pass it.
