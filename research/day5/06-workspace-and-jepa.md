# 06 — Code-JEPA recurrent workspace and the cache-once fix

The workspace is the layer that turns trunk features into code-structure
hypotheses. It is the only recurrent part and the only part that runs multiple
passes (`r`) per token — which is why F6 (doc 01) lives here.

## Layout

`w_layers = 4` workspace layers (`d_w = 1024`, `w_ff = 4096`), a projection
`d → d_w`, `k_hyp = 8` orthogonal hypothesis vectors, a residual scale
`1e-4`, and four prediction heads:

- `z_loc` (r=1): localisation,
- `z_exec` (r=2): symbol dependencies / execution trace,
- r=3 VEXX contract,
- r=4..8: patch-pass probability refinement, pooled by `p_pass`.

Parameter total **120,084,482** (includes the hypothesis codebook, the two
JEPA heads and the confidence head). One extra recurrent pass touches
`4 · 27,266,048 = 109,064,192` parameters = **0.2181 GFLOP**, the reproducible
spec figure.

## The cache-once fix (F6)

The context projection `k_ctx, v_ctx` depends only on the trunk output, not
on the recurrent hidden state. `WorkspaceLayer` therefore separates
`project_context_kv(ctx)` from `forward(h, k_ctx, v_ctx)`: the workspace
projects once per layer and reuses the result for all `r` steps.

Measured: `kv_proj_calls(r=8) = 2` at TINY (`w_layers = 2`) instead of
`8·2 = 16`. The uncached scheme would re-project `14` more times per token.

## Two non-obvious behaviours the audit pins

1. **The recurrence is stable, not merely cheap.** `init_drift` (relative
   change in logits between r=1 and r=8) is below `1e-3` at TINY in both the
   model-assertion fixture and the audit fixture — the deep recurrence does
   not blow the trunk output up, and it does not collapse it either (the
   residual scale `1e-4` keeps the initial escape hatch open).
2. **The hypothesis codebook is not decorative.** Replacing `hyp_idx` moves
   the workspace output by tens of units (max abs diff ~36.9 in the model
   fixture, ~62.6 in the audit fixture) — so the `k_hyp` hypothesis mechanism
   measurably changes the computation. A codebook that never moved the output
   would be dead capacity, and the audit would catch it.

## What is NOT in the ledger (declared, not hidden)

The neural MTP draft heads are **not** in the Day 5 parameter ledger. The
draft-tree pruning machinery (doc 07) is implemented and tested, and the
*depth cap* (`mtp_spec_depth = 3`) is asserted, but there are no MTP draft
weights to count. This was a false claim in the uploaded reference draft and
is listed in the README delta list rather than silently corrected. MTP draft
heads land in Day 6.

## Budget sensitivity

The workspace is 0.56% of stored parameters but 0.5–1.5% of decode FLOPs
depending on `r` (6.107 → 7.416 GFLOP). Recurrence depth is the lever:
`r` is a serving-time knob; the trunk is not. Because the context projection
is cached, the *marginal* cost of depth is exactly the workspace pass —
that linear, exact relationship is what the FLOP identity asserts.
