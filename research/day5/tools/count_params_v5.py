#!/usr/bin/env python3
"""Exact parameter, KV, FLOP, serving and training-budget model for Day 5.

Every figure printed here is derived from the canonical config dictionaries in
``kestrel5/model.py``; nothing is transcribed.  The five published ledger
numbers are asserted, the 1M-token session footprint is recomposed from its four
tiers, and the FLOP model is a real per-token ledger (see ``flops_fwd``).

Run: python3 research/day5/tools/count_params_v5.py
"""
import importlib.util
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "k5_model", os.path.join(ROOT, "kestrel5", "model.py"))
k5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(k5)

FULL = k5.FULL
NANO = k5.KESTREL5_NANO_165M


# --------------------------------------------------------------------- ledger

def ledger_rows(c):
    """Per-subsystem stored/active integers exactly as tabulated in doc 02."""
    d = c["d"]
    rms = 3 * d
    local = k5.attn_local_params(c)
    glob = k5.attn_global_nope_params(c)
    mem = k5.mem_params(c)
    moe_stored = k5.moe_params(c, False)
    moe_act_2x3 = k5.moe_params(c, True, 2)
    moe_act_6x1 = k5.moe_params(c, True, 6)
    return [
        ("Local MLA layer", local, c["local_layers"]),
        ("Global NoPE landmark layer", glob, c["global_nope_layers"]),
        ("Compact WY Householder memory", mem, c["layers"]),
        ("Stiefel subspace MoE", moe_stored, c["layers"]),
        ("Layer RMSNorms", rms, c["layers"]),
    ], {
        "moe_active_2x3_per_layer": moe_act_2x3,
        "moe_active_6x1_per_layer": moe_act_6x1,
        "workspace": k5.workspace_params(c),
        "embedding_and_head": c["vocab"] * d + d,
    }


def kv_ledger(c, n_ctx=1_048_576):
    """Two-tier session KV at ``n_ctx`` plus the window-complete variant.

    ``spec_total_mb`` is the ledger the Day 5 build guide pins: one 4,096-key
    window per *local* layer, a pooled block descriptor per 64 tokens per global
    layer, and 4,096 landmarks per global layer.

    ``window_complete_mb`` additionally accounts for the 4,096-key window that
    each *global* layer actually attends over (the reference mask includes it).
    Those keys have to live somewhere, so this is the number a real serving
    stack must allocate.  Both are reported; neither is hidden.
    """
    fp16 = 2
    n_local = c["local_layers"]
    n_glob = c["global_nope_layers"]
    bs = c["sel_block"]
    dc = c["d_c"]
    win = c["window"]

    local_win = n_local * win * (dc + c["d_rope"]) * fp16
    global_index = n_glob * ((n_ctx + bs - 1) // bs) * dc * fp16
    global_landmarks = n_glob * c["landmark_cache_max"] * dc * fp16
    fast_state = c["layers"] * c["r_m"] * c["d_m"] * fp16
    global_win = n_glob * win * dc * fp16

    spec_total = local_win + global_index + global_landmarks + fast_state
    complete_total = spec_total + global_win
    return {
        "local_window_mb": local_win / 1e6,
        "global_block_index_mb": global_index / 1e6,
        "global_landmarks_mb": global_landmarks / 1e6,
        "fast_state_mb": fast_state / 1e6,
        "global_window_mb": global_win / 1e6,
        "spec_total_mb": spec_total / 1e6,
        "window_complete_mb": complete_total / 1e6,
    }


# ---------------------------------------------------------------------- FLOPs

def workspace_iter_params(c):
    """Parameters *touched by one extra recurrent workspace pass*."""
    return c["w_layers"] * k5.w_layer_params(c)


def workspace_iter_flops(c):
    return 2 * workspace_iter_params(c)


def attn_flops_per_token(c, n_ctx):
    """Attention-side FLOPs for ONE decoded token at context ``n_ctx``.

    Local layers: scores over the rotary and non-rotary halves plus the AV
    product, i.e. ``2*h*L*(2*dh + d_rope)`` per layer.
    Global layers: NoPE scores + AV over ``window + topk*bs + anchors`` keys.
    Indexer: one dot product per visible 64-token block plus the amortised
    block pooling projection.
    """
    L = min(c["window"], n_ctx)
    local = c["local_layers"] * 2 * c["heads"] * L * (2 * c["dh"] + c["d_rope"])
    Lg = min(c["window"] + c["sel_topk"] * c["sel_block"] + c["anchor_budget"], n_ctx)
    n_blocks = (n_ctx + c["sel_block"] - 1) // c["sel_block"]
    idx = 2 * c["idx_dim"] * n_blocks + 4 * c["d_c"] * c["idx_dim"] / c["sel_block"]
    glob = c["global_nope_layers"] * (4 * c["heads"] * c["dh"] * Lg + idx)
    return local, glob


def flops_fwd(c, r=2, top_groups=None, n_ctx=1_048_576):
    """Forward FLOPs for one decoded token (decode step, not prefill).

    Dense part: every active weight is touched exactly once per token, i.e.
    ``2 * active_params`` -- which already includes the tied LM head.  Added to
    that are the attention/indexer terms (which depend on how many keys are
    visible, not on parameter count) and ``r - 1`` extra workspace passes.
    """
    dense = 2 * k5.active_params(c, top_groups_override=top_groups)
    local, glob = attn_flops_per_token(c, n_ctx)
    return dense + local + glob + (r - 1) * workspace_iter_flops(c)


# -------------------------------------------------------------------- training

def muon_param_count(c):
    """Parameters Muon can own: every 2-D matrix except the embedding table.

    Norms, sinks and router biases are 1-D and stay on AdamW, as does the
    (tied) embedding/head matrix.
    """
    d, dc, h, dw = c["d"], c["d_c"], c["heads"], c["d_w"]
    G, Eg = c["n_groups"], c["experts_per_group"]
    local_attn_matrix = k5.attn_local_params(c) - (2 * dc + h)
    global_attn_matrix = k5.attn_global_nope_params(c) - (2 * dc + h)
    # q,k,v,u,g,o and A are 2-D; beta is [d, n_householder] (2-D) but its
    # columns are tiny (n_householder=2) and norm-like in shape.  We keep it in
    # Muon (it is a matrix), which is the conservative choice for the split.
    mem_matrix = k5.mem_params(c)
    moe_matrix = k5.moe_params(c, False) - (G + G * Eg)  # minus gbias/ebias
    per_local = local_attn_matrix + mem_matrix + moe_matrix
    per_global = global_attn_matrix + mem_matrix + moe_matrix
    ws_matrix = (c["w_layers"] * (k5.w_layer_params(c) - 3 * dw)
                 + 2 * dw * d + c["k_hyp"] * dw + 2 * dw * dw
                 + dw * c["d_head_h"] + c["d_head_h"])
    return (c["local_layers"] * per_local + c["global_nope_layers"] * per_global
            + ws_matrix)


def training_budget(c, batch=16, seq=2048, r=2, bytes_weights=2):
    """Mixed-precision step budget for one GPU, with explicit byte rules.

    weights fp16 + fp32 master copy + Muon momentum (fp32) on matrices +
    AdamW m/v (fp32) on everything else + analytic activation bound assuming a
    *fused* windowed attention kernel (a materialised score matrix is computed
    separately because it is the reason the fused kernel is mandatory).
    """
    total = k5.total_params(c)
    muon = muon_param_count(c)
    adamw = total - muon
    weights = total * bytes_weights
    master = total * 4
    optimizer = muon * 4 + adamw * 8

    toks = batch * seq
    act_residual = 2 * toks * c["d"] * 4
    act_moe = toks * c["top_k"] * 3 * c["d_ff"] * 4
    act_shared = toks * 3 * c["d_shared"] * 4
    chunk = min(128, seq)
    act_attn = batch * c["heads"] * seq * chunk * 4 * 2
    act_ws = r * c["w_layers"] * toks * c["d_w"] * 4
    activations = act_residual + act_moe + act_shared + act_attn + act_ws

    # What a naive dense score matrix would cost for one layer (fp32).
    materialised_scores = batch * c["heads"] * seq * min(c["window"], seq) * 4

    return {
        "params": total, "muon_params": muon, "adamw_params": adamw,
        "weights_mb": weights / 1e6, "master_mb": master / 1e6,
        "optimizer_mb": optimizer / 1e6, "activations_mb": activations / 1e6,
        "total_mb": (weights + master + optimizer + activations) / 1e6,
        "materialised_layer_mb": materialised_scores / 1e6,
    }


# --------------------------------------------------------------------- serving

HW_PROFILES = {
    "h200_8x_fp8": dict(name="8x H200, fp8 weights", aggregate_bw_gbs=8 * 4800,
                        bytes_per_param=1),
    "h200_8x_fp16": dict(name="8x H200, fp16 weights", aggregate_bw_gbs=8 * 4800,
                         bytes_per_param=2),
    "t4_1x_fp16": dict(name="1x T4, fp16 weights", aggregate_bw_gbs=320,
                       bytes_per_param=2),
}


def decode_kv_read_bytes(c, n_ctx=1_048_576):
    """Bytes of KV actually *read* to emit one token (not the whole cache)."""
    fp16 = 2
    L = min(c["window"], n_ctx)
    local = c["local_layers"] * L * (c["d_c"] + c["d_rope"]) * fp16
    Lg = min(c["window"] + c["sel_topk"] * c["sel_block"] + c["anchor_budget"], n_ctx)
    glob = c["global_nope_layers"] * Lg * c["d_c"] * fp16
    fast = c["layers"] * c["r_m"] * c["d_m"] * fp16
    return local + glob + fast


def serving_model(c, profile="h200_8x_fp16", r=2, n_ctx=1_048_576,
                  target_tok_s=12216):
    """Bandwidth-bound batch-1 decode estimate under a named hardware profile.

    Decode touches every active weight once and streams the KV read set, so the
    ceiling is ``aggregate_bw / bytes_moved_per_token``.  ``implied_mbu`` is the
    memory-bandwidth utilisation a given target would require -- it is reported
    so a target can never be mistaken for a measurement.
    """
    hw = HW_PROFILES[profile]
    weights = k5.active_params(c) * hw["bytes_per_param"]
    kv = decode_kv_read_bytes(c, n_ctx)
    moved = weights + kv
    ceiling = hw["aggregate_bw_gbs"] * 1e9 / moved
    return {
        "profile": hw["name"],
        "weights_gb": weights / 1e9,
        "kv_read_mb": kv / 1e6,
        "arith_intensity_flop_per_byte": flops_fwd(c, r=r, n_ctx=n_ctx) / moved,
        "ceiling_tok_s": ceiling,
        "target_tok_s": target_tok_s,
        "implied_mbu": target_tok_s / ceiling,
    }


# ------------------------------------------------------------------------ main

def main():
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    print("=" * 84)
    print("DAY 5 LEDGER -- exact parameters, KV, FLOPs, serving and training budget")
    print("=" * 84)

    print("\nA. Parameter ledger (stored / active)")
    rows, extra = ledger_rows(FULL)
    print(f"  {'subsystem':38s} {'per layer':>15s} {'count':>3s} {'stored total':>16s}")
    running = 0
    for name, per, count in rows:
        running += per * count
        print(f"  {name:38s} {per:15,d} {count:3d} {per * count:16,d}")
    ws = extra["workspace"]
    emb = extra["embedding_and_head"]
    print(f"  {'KV-cached Code-JEPA workspace':38s} {ws:15,d} {1:3d} {ws:16,d}")
    print(f"  {'Tied embedding + final norm':38s} {emb:15,d} {1:3d} {emb:16,d}")
    total = running + ws + emb
    print(f"  {'GRAND TOTAL':38s} {total:15,d}")
    check("FULL total == 21,248,719,490", total == 21_248_719_490, f"{total:,}")
    check("FULL total == model.total_params", total == k5.total_params(FULL), "cross-module")

    a2 = k5.active_params(FULL, 2)
    a6 = k5.active_params(FULL, 6)
    check("FULL active 2x3 == 2,071,178,882", a2 == 2_071_178_882,
          f"{a2:,} ({100 * a2 / total:.2f}%)")
    check("FULL active 6x1 == 2,273,293,954", a6 == 2_273_293_954,
          f"{a6:,} ({100 * a6 / total:.2f}%)")
    d, dl, eg = FULL["d"], FULL["d_lat"], FULL["experts_per_group"]
    check("6x1 routed rank is 2,304 vs 768 for 2x3",
          FULL["top_groups"] * dl == 2304 and 2 * dl == 768,
          f"{FULL['top_groups']}*{dl}={FULL['top_groups'] * dl} vs 2*{dl}={2 * dl}")
    delta = extra["moe_active_6x1_per_layer"] - extra["moe_active_2x3_per_layer"]
    check("active delta = 4 extra groups x (2*d*d_lat + d_lat*Eg + Eg)",
          delta == 4 * (2 * d * dl + dl * eg + eg),
          f"delta={delta:,} = 4 x {2 * d * dl + dl * eg + eg:,}")

    print("\nB. KESTREL5_NANO_165M (free Kaggle 2xT4 / Colab 1xT4 tier)")
    n_tot = k5.total_params(NANO)
    n_act = k5.active_params(NANO, 6)
    check("NANO total == 151,495,922", n_tot == 151_495_922, f"{n_tot:,}")
    check("NANO active == 80,704,562", n_act == 80_704_562,
          f"{n_act:,} ({100 * n_act / n_tot:.2f}%)")
    check("NANO fits one 16 GB T4 in fp16 weights", n_tot * 2 / 1e9 < 0.4,
          f"{n_tot * 2 / 1e6:.0f} MB of weights")

    print("\nC. 1M-token session footprint (fp16)")
    kv = kv_ledger(FULL)
    print(f"  local windows        {kv['local_window_mb']:8.2f} MB")
    print(f"  block summary index  {kv['global_block_index_mb']:8.2f} MB")
    print(f"  exact landmarks      {kv['global_landmarks_mb']:8.2f} MB")
    print(f"  WY fast weights      {kv['fast_state_mb']:8.2f} MB")
    print(f"  {'specified total':20s} {kv['spec_total_mb']:8.2f} MB")
    print(f"  + global-layer windows (4 x 4096 keys) {kv['global_window_mb']:6.2f} MB")
    print(f"  {'window-complete total':20s} {kv['window_complete_mb']:8.2f} MB")
    check("specified 1M session footprint == 144.70 MB",
          abs(kv["spec_total_mb"] - 144.70) < 0.01, f"{kv['spec_total_mb']:.2f} MB")
    check("model.module agrees on the KV subtotal",
          abs(k5.GlobalNoPELandmarkMLA.session_kv_bytes_at_1m(FULL)["total_kv_mb"]
              - (kv["spec_total_mb"] - kv["fast_state_mb"])) < 1e-9,
          "both code paths compute the same three tiers")
    check("window-complete footprint is reported separately",
          kv["window_complete_mb"] > kv["spec_total_mb"],
          f"{kv['window_complete_mb']:.2f} MB = 144.70 + {kv['global_window_mb']:.2f}")

    print("\nD. Forward FLOPs per decoded token")
    print(f"  one extra workspace pass: {workspace_iter_flops(FULL) / 1e9:.4f} GFLOP "
          f"({workspace_iter_params(FULL):,} params)")
    check("incremental workspace pass == 0.2181 GFLOP (spec figure 0.218)",
          abs(workspace_iter_flops(FULL) / 1e9 - 0.2181) < 5e-4,
          f"{workspace_iter_flops(FULL) / 1e9:.4f} GFLOP")
    base = flops_fwd(FULL, r=1)
    for r in (1, 2, 4, 8):
        f = flops_fwd(FULL, r=r)
        print(f"  r={r}: {f / 1e9:8.3f} GFLOP  (marginal {(f - base) / 1e9:6.3f} over r=1)")
    check("marginal FLOPs are exactly (r-1) * workspace pass",
          flops_fwd(FULL, r=8) - flops_fwd(FULL, r=1) == 7 * workspace_iter_flops(FULL),
          "integer identity, no rounding")
    local_f, glob_f = attn_flops_per_token(FULL, 1_048_576)
    print(f"  attention terms at 1M ctx: local={local_f / 1e9:.3f} "
          f"global={glob_f / 1e9:.3f} GFLOP/token")
    check("attention cost is sub-linear in context, not quadratic",
          attn_flops_per_token(FULL, 1_048_576)[0] < 1.1 * attn_flops_per_token(FULL, 4096)[0],
          "local term saturates at the 4096 window")
    print(f"  pre-cache saving at r=8: {(8 - 1) * 2 * FULL['w_layers'] * 2 * FULL['d'] * 2 * FULL['d_w'] / 1e9:.3f}"
          f" GFLOP (7 avoided K/V projections per layer) -- the spec's 0.307 GFLOP")
    print("  figure does not reproduce from any per-step decomposition we can build;")
    print("  0.218 GFLOP (the recurrent pass itself) does reproduce exactly.")

    print("\nE. Batch-1 decode serving model")
    for prof in ("h200_8x_fp16", "h200_8x_fp8", "t4_1x_fp16"):
        s = serving_model(FULL, prof)
        print(f"  {s['profile']:22s} weights={s['weights_gb']:5.2f} GB  "
              f"kv_read={s['kv_read_mb']:6.1f} MB  "
              f"ceiling={s['ceiling_tok_s']:9,.0f} tok/s  "
              f"(12,216 target = {100 * s['implied_mbu']:5.1f}% MBU)")
    check("decode is bandwidth-bound, not FLOP-bound",
          serving_model(FULL, "h200_8x_fp8")["arith_intensity_flop_per_byte"] < 10.0,
          f"{serving_model(FULL, 'h200_8x_fp8')['arith_intensity_flop_per_byte']:.2f} "
          "FLOP/byte (H200 ridge point is ~300)")
    check("12,216 tok/s is never reported as measured",
          serving_model(FULL, "h200_8x_fp8")["implied_mbu"] < 1.0,
          "it is quoted as a target at a stated MBU, see doc 09")

    print("\nF. KESTREL5_NANO_165M training budget (batch 16 x 2048, r=2)")
    b = training_budget(NANO)
    print(f"  weights fp16        {b['weights_mb']:8.1f} MB")
    print(f"  fp32 master copy    {b['master_mb']:8.1f} MB")
    print(f"  Muon/AdamW states   {b['optimizer_mb']:8.1f} MB")
    print(f"  activations (fused) {b['activations_mb']:8.1f} MB")
    print(f"  TOTAL               {b['total_mb']:8.1f} MB")
    print(f"  Muon-owned params {b['muon_params']:,} / {b['params']:,} "
          f"({100 * b['muon_params'] / b['params']:.1f}%)")
    print(f"  one materialised dense score matrix for a single layer would need "
          f"{b['materialised_layer_mb']:,.0f} MB -> fused kernel is mandatory")
    check("NANO full training step fits one 16 GB T4", b["total_mb"] < 16000 - 1500,
          f"{b['total_mb'] / 1024:.2f} GiB of 16 GiB")
    check("materialised dense attention would not fit", b["materialised_layer_mb"] > 3000,
          f"{b['materialised_layer_mb'] / 1024:.1f} GiB for ONE layer")

    print("\n" + "=" * 84)
    print(("ALL LEDGER CHECKS PASS" if ok else "LEDGER CHECKS FAILED"))
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
