# 01 — What the audit found, and what changed

The Day-6 audit package and a live run agree: the baseline was a real trainable model with real
bugs. Two rules made the repair tractable.

1. **A repair must be observable.** Every fix below has a test that fails before it and passes
   after it. "It looks better" is not evidence.
2. **A repair must remove the cause, not the symptom.** Removing the assertion that caught a bug is
   how a codebase lies to itself, so the one assertion that no longer described reality (Day 5
   `[19]`) was replaced with three claims that are actually invariant.

## Day 5 — one assertion, three real invariants

Day 6 removed a manual `r >= 8` logit-tie-flip hack from `Kestrel5.forward`. `[19]` still asserted
that two of sixteen argmax tokens flip, which random weights at scale 1.98e-3 cannot guarantee — it
was testing the hack, not the workspace.

Replaced with:

* `max |logits(r=8) − logits(r=2)| > 1e-3` — the recurrent workspace changes the output at all.
* `max |z_exec(r=8) − z_exec(r=2)| > 1.0` — the execution latent evolves across passes.
* `p_pass.shape == (1, 16)` — token-level pass probabilities keep their documented shape.

Measured: `max|dlogit| = 8.51e-02`, `|dz_exec| = 35.43`. All four Day-5 entry points exit 0:
`model.py`, `tools/count_params_v5.py`, `tools/sim_verify_v5.py` (all simulator checks), and
`tools/audit_v5.py` (77/77).

## Day 6 — twelve defects, one table

| # | Defect | Cause | Repair | Test that now passes |
|---|---|---|---|---|
| 1 | `_comment` treated as a preset | JSON loaded wholesale | filter `_`-prefixed keys at load **and** in `get_config` | `test_config_strips_metadata_keys` |
| 2a | `build_model("TINY")` raises `NameError` | `get_config` never imported | explicit `from .config import get_config` | `test_model_build_and_forward` |
| 2b | RoPE `IndexError` on first pass | `torch.empty(0)` is 1-D; `_rope` reads `shape[1]` | 3-D non-persistent buffers `(1, 0, dim//2)`; rebuild on length/device/dtype | all model tests |
| 2c | MLA `w_uk`/`w_uv` shape mismatch | sized to `n_heads`, reshaped to `n_kv_heads` | size to `n_kv_heads * head_dim` | `test_baseline_mla_local_forward_and_cache` |
| 2d | MLA `k_pe` cache corruption | concatenated on `dim=1` (heads) | concatenate on `dim=2` (sequence) | same |
| 2e | MLA sink `cat` fails at batch > 1 | `[1, h, 1, 1]` sink | `expand(b, h, n, 1)` | same |
| 2f | WY memory shape crash | `u2 = k2` instead of `k2[:, t]` | per-token slice | `test_wy_householder_memory_time_slice` |
| 2g | Stiefel impossible when `G*d_lat > d` | stacked `up.t()` on `dim=0` | stack on `dim=1` → `[d, G*d_lat]`, QR, copy blocks both ways; measure `U` as `[G*d_lat, d]` | `test_subspace_moe_stiefel_retraction` |
| 2h | Workspace `x @ nn.Linear`, missing `self.kv`, double residual | five separate slips | call `proj_in(x)`/`proj_out(h)`, use `layer.kv`, return the delta only, enable on final block only | `test_code_jepa_workspace_returns_delta_only`, `test_workspace_only_on_final_block` |
| 3 | Double label shift (predicts *t+2*) | dataset shifted, forward shifted again | dataset stores **unshifted** labels with `-100` padding | `test_single_shift_label_alignment` |
| 4 | Specials key mismatch | `save` wrote `specials`, `load` read `special_tokens` | read both spellings | `test_tokenizer_save_load` |
| 5a | Tail accumulation dropped | `step % grad_accum` never reaches 0 | explicit pending counter, rescale, step | `test_training_step_decreases_loss_on_overfit` |
| 5b | `best_val_loss = inf` | validation interval > corpus | force final validation, pass best/history/seed/run_name | (covered by the same training test) |
| 5c | `temperature=None` sampled | only `== 0.0` was greedy | `None or <= 0` → argmax | `test_generation_determinism` |
| 5d | Corpus leakage | repeated 50× before split; tokenizer on all text | SHA-256 dedupe → split → tokenizer on train only → `cfg["vocab"] = tok.vocab_size` | `test_train_val_split` behaviour in the run script |
| 6 | Test matmul bug | `input_ids @ embed.weight.T` | replicate trunk, compare logits to `rms_norm(x) @ W.T` | `test_model_tied_embedding` |

Day 6 now reports **18 passed** (11 original + 7 added component/label tests). The added tests are
not decoration: the MLA, WY, Stiefel, workspace, and label tests are precisely the ones that would
have caught the audit findings before a run.

## Day 7 — auditing the artifact, not just the prose

The Day-7 audit was run against the code before the remediation report was opened, and each finding
was reproduced with a probe. The defects below were real; every one now has a regression test in
`tests/test_kestrel7.py`.

| # | Defect | Probe | Repair |
|---|---|---|---|
| 1 | A fully masked attention row produced `-inf` after softmax | sliding-window prefill at the first window | exact softmax zeroes fully masked rows; cache eviction only at single-token decode |
| 2 | Initial logits had std 21.55 and CE 252 versus uniform 5.61 | random init forward | `N(0, 1/√d)` embeddings, `N(0, 0.02)` linears, `logit_scale = 1/√d`; measured CE 5.94 vs 5.95 |
| 3 | Tokenizer hash ignored the id→token mapping | swap two ids, hash unchanged | hash covers the sorted full mapping, merges, specials, vocab size, pretokenizer |
| 4 | A checkpoint loaded into a model with no tokenizer verifier | load without `tokenizer=` | `tokenizer_hash` is mandatory and verified; load fails closed |
| 5 | Interrupted training resumed weights but not RNG or batch cursor | resume from step k | checkpoint stores RNG, epoch, batch-in-epoch; exact-resume test gets `max|Δw| = 0.0` |
| 6 | `hybrid_gdn` with a null schedule silently built a dense stack | config probe | shared `validate_config` rejects contradictory mixer/schedule |
| 7 | `sympy.sympify` executed `__import__('os').getcwd()` | CAS injection | restricted AST→SymPy builder (`safe_sympy`) with length/node/exponent/int caps |
| 8 | CAS evaluator ran untrusted code in-process | import/file probes | subprocess worker (`-I -S`, minimal env, temp cwd, RLIMIT_CPU/AS/NOFILE, 5s); process isolation, not a container |
| 9 | CAS splice appended `val>>` but never fed the `=` token back into the cache | instrumented decode | `=` and every spliced value token are fed before normal decoding resumes |
| 10 | A prompt longer than `max_ctx` used the untruncated length as `seq_start` | instrumented decode | truncate first, then use the cached length |
| 11 | Syllogism check called satisfiability “entailment” | `all A are B` ⇒ `all B are A` returned `yes` | exact finite-universe countermodel search returns `no` |
| 12 | Benchmark key was produced by the same path being scored | key regeneration | independent answer implementation plus static calibration labels |
| 13 | `expected_one_of` bypassed the input-mutation check | mutating candidate | mutation is checked on every branch |
| 14 | Tier-4 explanation claimed a CI crossed zero when it did not | `[0.4, 2.1], p=0.8` | reasons are emitted only when the parsed numbers support them |

P0 findings in the supplied remediation report that overlapped this audit were already fixed here:
the GDN convolution leak, NaN from fully masked rows, validation on all-ignored labels, in-process
`exec`, initialization scale, reverse-KL temperature scaling, and the tokenizer hash. The report's
remaining scope items are reflected in the docs rather than overclaimed: MTP self-speculation is a
*target*, the Farkas routine is a single-row bound contractor, and the sandbox is process isolation
with resource limits, not a hardened container.

## Days 5–6 alternative considered and rejected

**Patch only the failing assertions and ship.** Faster, and it would have produced a green suite
with the same twelve defects. Falsified immediately by the label test: a green model that predicts
`t+2` is worse than a red one, because it hides the failure behind a plausible loss curve.
