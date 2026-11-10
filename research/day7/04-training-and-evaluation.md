# 04 — Training recipe, verification gate, and the honest limits

## Four stages (specification, not results)

Architecture sets a ceiling; data and post-training decide where the weights land. The recipe below
is the plan this codebase is built for; none of it is claimed as executed here.

1. **Pretraining — WSD + deduplicated stream packing.**
   Mix: 55% classifier-filtered educational web (FineWeb-Edu / DCLM), 25% educational code
   (Stack-Edu + repository diffs), 15% step-by-step math, 5% synthetic textbooks. Muon on 2-D trunk
   matrices (`lr=0.02`, 5 Newton–Schulz steps) plus AdamW (`lr=3e-4`, β=(0.9, 0.95), wd=0.1) for
   embeddings, LM head, and GDN `A_log`/`dt_bias` with `weight_decay=0.0` so decay gates do not
   collapse.
2. **Decay + long context.** Last 10% linear LR decay, upsample math/code/verified traces to 40%,
   raise `rope_theta` from 500,000 to 1,000,000 on full-attention layers.
3. **Cold-start dual-mode SFT.** 200K–400K verified trajectories: unconstrained ` thinking` with
   inline `<<expr=val>>` CAS splices, strict post-`</think>` output under Tier 1/4 rules.
4. **On-policy reverse-KL + VEXX-gated RLVR.** `G=4` rollouts per prompt with the controller in the
   loop; dense per-token `KL(π_θ || π_teacher)` plus binary execution/verification rewards.

`compute_on_policy_reverse_kl_loss` is implemented natively in torch:

```
KL(s||t) = Σ_v π_s(v) · (log π_s(v) − log π_t(v))       π = softmax(logits / T)
loss     = T² · mean_tokens KL
```

The teacher is detached, the student graph is not. Both distributions are softened by the same
`temperature` and the result is scaled by `T²`, so gradients do not grow with temperature. The
reference implementation round-tripped through NumPy, which detached the student and made the
gradient identically `None`; the test asserts `KL(π||π) < 1e-6` and a nonzero student gradient for
`π_s ≠ π_t`.

**Why reverse KL and not forward KL:** reverse KL is mode-seeking on student-visited states, which is
what on-policy distillation wants; forward KL would force the student to cover teacher mass it will
never visit. **Falsifier:** reverse-KL training failing to improve pass@1 at fixed teacher, or the
loss not reaching zero when student = teacher.

## Trainer invariants

* Tokenizer is trained on the **train split only**; documents are SHA-256-deduplicated *before*
  splitting; overlap is asserted empty, not assumed.
* `model.vocab == tokenizer.vocab_size` is enforced by `build_model`.
* The tail gradient-accumulation group is rescaled and stepped, never dropped.
* Validation runs before a checkpoint is written whenever a validation loader is supplied; the
  end-to-end runner always supplies one. A missing validation split is reported, not silently
  treated as a good result.
* Checkpoints carry `config_hash`, `tokenizer_hash`, the RNG state, and the epoch/batch cursor;
  loading fails closed with `ValueError` on either hash mismatch. `model_status` starts as
  `randomly_initialized`, `checkpoint_loaded` as `False`, and both only change after a verified
  load. Resuming restores weights, optimizer, scheduler, counters, and RNG; the regression test
  compares a resumed run against an uninterrupted one and measures `max|Δw| = 0.0` with identical
  loss history.

## The gate

| Check | Command | Result |
|---|---|---|
| Day 6 unit suite | `python3 -m pytest research/day6/tests/ -q` | **18 passed** |
| Day 7 verification suite | `python3 -m pytest research/day7/tests/ -q` | **35 passed** |
| TempBench | `python3 research/day7/tools/eval_tempbench_v7.py` | **48 / 48 = 100.0%** (33 tasks) |
| End-to-end mechanism run | `python3 research/day7/run_kestrel7.py --preset TINY_HYBRID --steps 50` | trains, validates, verifies checkpoint, generates with VEXX |

The first 20 suites cover: config hygiene; tokenizer round-trip/hash; dedupe-before-split;
single-shift labels and BOS masking; block-diagonal attention (and GDN conv) isolation; hybrid
streaming equivalence; MTP gradients; accumulation/validation/checkpoint fail-closed; reverse KL;
VEXX Tiers 0–5; the computed parameter/KV ledger, the TempBench total, the docs-quote-computed-
numbers check, and the greedy `temperature=None` contract. The 15 audit regressions cover the
initialization scale, sliding-window NaN/streaming, document-isolated convolution chunks, full
tokenizer identity hashing and tamper rejection, unknown-id decoding, mandatory checkpoint tokenizer
verification, config contradiction/deep copies, CAS code injection, sandbox wall-clock and memory
limits, CAS splice cache order, long-prompt positions, independent answer keys, DPO selection, and
exact resume.

## What the 50-step run does and does not show

It shows that the plumbing is real: dedupe → split → tokenizer → hybrid model → CE+MTP loss →
gradient accumulation → validation → hash-verified checkpoint → held-out test evaluation →
VEXX-masked decode completes without error. It does **not** show that the model is good. Fifty
optimizer steps on 38 synthetic documents can only memorize the tiny corpus; the script prints the
initial CE, validation loss, and held-out test loss rather than hiding them. The run starts at CE
5.94 against 5.95 uniform and reaches validation loss 6.80; that is a working optimizer, not a
capability claim.

The benchmark is a pipeline benchmark. Math is solved by the restricted-AST SymPy path, formal
tasks by exact Venn countermodels, code by subprocess-verified candidate pools, research by the
Tier-4 guard. The answer key is produced by `independent_answer()` (separate fraction, integration,
permutation, boolean, and static calibration implementations), and a test re-derives every row. A
score of 48/48 says the *verifier* works; it says nothing about the weights, and the evaluator
prints that caveat on every run.

Implemented: exact CAS, code execution/verification, the six benchmark categories, and the listed
Tier-4 traps. Prototype: the single-row Farkas bound contractor and the DPO data collector.
Planned, not implemented: MTP self-speculative decoding, a DPO training loop, and on-policy RLVR.

## Known limitations

* `BASELINE_100M` is 105,062,040 parameters — inside the documented 100.1M–151.5M band, but no
  pretraining run exists in this repo.
* No GPU path is exercised; all measurements are CPU (`torch 2.15.0+cpu` in `metrics.json`).
* Streaming with packed `doc_ids` across a decode boundary is not supported: decode does not carry
  document IDs, so cached generation assumes a single document. Training and prefill do.
* The Tier-4 guard is rule-based by design. It is a tripwire for the listed traps, not a general
  factuality model, and it must not be read as one.
* The Tier-5 sandbox is a resource-limited subprocess, not a hardened container: no seccomp, user
  namespace, or network namespace. It is adequate for benchmark candidates, not hostile native code.
* The Tier-3 Farkas routine derives one bound along the current assignment slice; it is not a
  general Farkas certificate or an infeasibility proof.

## Reproduce the numbers

```bash
python3 research/day7/tools/param_ledger_v7.py   # rewrites metrics.json (params, KV, bench)
python3 research/day7/tools/eval_tempbench_v7.py # 48/48
python3 -m pytest research/day7/tests/ -q        # asserts metrics.json against live models
```

If a documented number drifts from the code, `test_docs_quote_computed_numbers` fails. That is the
point: the prose is not allowed to remember a number the code no longer produces.
