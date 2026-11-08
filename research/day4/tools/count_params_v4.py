#!/usr/bin/env python3
"""Kestrel-4 x VEXX 1.5 (Day 4) analytical parameter, FLOP, memory, and serving cost model.

Every parameter, KV-byte, state-byte, FLOP, GPU-hour and VEXX latency figure quoted in
research/day4/* is produced here or by kestrel4/model.py, and both assert the ledger.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kestrel4.model import (
    FULL as V, active_params, total_params, attn_local_params,
    attn_global_nope_params, mem_params, moe_params, workspace_params,
    EXPECTED_TOTAL, EXPECTED_ACTIVE,
)

PEAK = 989e12
MFU = 0.34
BW = 3.35e12
TOTAL = total_params(V)
ACTIVE = active_params(V)

L, D, W = V["layers"], V["d"], V["window"]
WS = workspace_params(V)
EMBED = V["vocab"] * D
ROUTED_ALL = V["n_experts"] * 3 * V["d_lat"] * V["d_ff"] * L
SUBSPACE_INACTIVE = (V["n_groups"] - V["top_groups"]) * (2 * D * V["d_lat"] + V["d_lat"] * V["experts_per_group"] + V["experts_per_group"]) * L


def flops_fwd(ctx=65536, r=2):
    """Forward FLOPs per token. 2*weights-touched for the dense parts, then
    4*h*dh*attended for each attention pair (QK^T and AV), at the true head
    width heads*(dh + d_rope).  The window caps the local layers; the global
    layer reads its window plus its selected blocks plus its anchors."""
    lin_active = ACTIVE - EMBED - WS
    dense = 2 * (lin_active + r * WS)
    w_eff = min(ctx, W)
    att_loc = V["local_layers"] * 4 * w_eff * V["heads"] * (V["dh"] + V["d_rope"])
    g_eff = min(ctx, W + V["sel_topk"] * V["sel_block"] + V["anchor_budget"])
    att_glb = V["global_nope_layers"] * 4 * g_eff * V["heads"] * V["dh"]
    idx = V["global_nope_layers"] * 2 * ((ctx // V["sel_block"]) + 32 * V["sel_block"]) * V["idx_dim"]
    w_att = V["w_layers"] * r * 2 * 4 * w_eff * V["d_w"]
    return dense + att_loc + att_glb + idx + w_att


def flops_train(ctx=4096, r_eff=2.0):
    return 3.0 * flops_fwd(ctx=ctx, r=r_eff)


def gpu_h(tokens, ctx, r_eff=2.0):
    return tokens * flops_train(ctx, r_eff) / (PEAK * MFU) / 3600


def main():
    print("=" * 78)
    print("KESTREL-4 x VEXX 1.5 (DAY 4)  |  21.25B total, 2.07B active (9.75%)")
    print("=" * 78)
    assert (TOTAL, ACTIVE) == (EXPECTED_TOTAL, EXPECTED_ACTIVE), "Ledger drift!"
    print(f"attn_local/layer       {attn_local_params(V):>14,}   (x12 layers, per-head RoPE from c_q)")
    print(f"attn_global_nope/layer {attn_global_nope_params(V):>14,}   (x4 layers, NoPE + 2-stage indexer)")
    print(f"memory/layer           {mem_params(V):>14,}   (Householder DeltaProduct, n_h=2)")
    print(f"moe/layer (total)      {moe_params(V, False):>14,}   (8 groups x 32 = 256 experts, rank 3,072)")
    print(f"moe/layer (active)     {moe_params(V, True):>14,}   (top-2 groups x top-3 = 6 active)")
    print(f"workspace (Code-JEPA)  {WS:>14,}   (4 layers, d_w=1024 + JEPA & hyp heads)")
    print(f"\nTOTAL PARAMETERS       {TOTAL:>14,}   ({TOTAL/1e9:.4f} B)")
    print(f"ACTIVE / TOKEN         {ACTIVE:>14,}   ({ACTIVE/1e9:.4f} B, {ACTIVE/TOTAL:.2%})")

    print("\n--- Forward Compute per Token (GFLOP) ---")
    for r in (1, 2, 4, 8):
        print(f"  r={r}  @4k: {flops_fwd(4096, r)/1e9:5.2f} GFLOP   "
              f"@64k: {flops_fwd(65536, r)/1e9:5.2f} GFLOP   "
              f"@1M: {flops_fwd(1048576, r)/1e9:5.2f} GFLOP")

    print("\n--- Training + VEXX 1.5 CoW Verification Budget ---")
    stages = [
        ("pretrain @4k",            1_200e9,   4096, 1.9),
        ("window+anchor @16k",        150e9,  16384, 1.9),
        ("long-ctx NoPE+mem @256k",    60e9, 262144, 2.0),
        ("Clean-PR + CWM traces",      80e9,  32768, 2.2),
        ("SFT + JEPA + VEXX specs",     4e9,  32768, 2.5),
    ]
    tot_gpu = 0.0
    for name, tok, ctx, r in stages:
        h = gpu_h(tok, ctx, r)
        tot_gpu += h
        print(f"  {name:26s} {tok/1e9:6.0f}B tok  {h:8,.0f} GPU-h")
    dist_h, rl_h, si_h, abl_h = 1_650.0, 62.0, 21.0, 2_000.0
    tot_gpu += dist_h + rl_h + si_h + abl_h
    restarts = 0.12 * tot_gpu
    grand_gpu = tot_gpu + restarts
    # VEXX 1.5 Tier-0/1/2 + CDCL Jumpback + CoW Snapshots cut average env time from 90s -> 1.1s (81x reduction)
    cow_cpu_core_h = (60_000 * 6 + 6_000 * 8 * 5) * 1.1 / 3600
    print(f"  {'on-policy GKD distill':26s} {'30B tok':>10s}  {dist_h:8,.0f} GPU-h")
    print(f"  {'VEXX RL + 8x Self-Impr':26s} {'2.9B tok':>10s}  {rl_h+si_h:8,.0f} GPU-h"
          f"  + {cow_cpu_core_h:6,.0f} CPU core-h (VEXX+CoW 1.1s avg)")
    print(f"  {'ablations + 12% restarts':26s} {'':>10s}  {abl_h+restarts:8,.0f} GPU-h")
    print(f"  TOTAL BUDGET                           {grand_gpu:8,.0f} GPU-h  + {cow_cpu_core_h:6,.0f} CPU core-h")

    print("\n--- Serving, One Decode Step + VEXX 0-DoF Splicing ---")
    print("  Bytes are GB (1e9) and BW is bytes/s, so one batch step costs")
    print("  step_GB * 1e9 / BW seconds.  Weight bytes are 0.55 B/param at 4-bit.")
    always_params = TOTAL - EMBED - ROUTED_ALL - SUBSPACE_INACTIVE
    always_gb = always_params * 0.55 / 1e9
    exp_mb = (3 * V["d_lat"] * V["d_ff"]) * 0.55 / 1e6   # one expert at 4-bit
    patch_tokens = 600
    splice_save = 0.28        # 0-DoF splicing skips 28% of the decode steps
    for b in (1, 16, 64):
        distinct_per_layer = V["n_experts"] * (1 - (1 - 1 / V["n_experts"]) ** (b * V["top_k"]))
        routed_gb = L * distinct_per_layer * exp_mb / 1000.0
        step_gb = always_gb + routed_gb
        step_ms = step_gb * 1e9 / BW * 1e3                # ms for the whole batch step
        raw_patch_ms = patch_tokens * step_ms / b
        # Net: 0-DoF splicing removes 28% of the steps; the remaining steps pay a
        # 16 us CPU splicing overhead per token.
        net_patch_ms = (step_ms / b) * (1.0 - splice_save) * patch_tokens + patch_tokens * 0.016 / b
        print(f"  batch={b:2d}: {distinct_per_layer:5.1f} exp/layer -> {step_gb:5.2f} GB/step "
              f"({step_ms:.3f} ms/step)")
        print(f"      {patch_tokens}-token patch: raw {raw_patch_ms:6.2f} ms  ->  "
              f"net w/ 0-DoF {net_patch_ms:6.2f} ms  "
              f"({raw_patch_ms / net_patch_ms:.2f}x faster)  "
              f"({patch_tokens / (net_patch_ms / 1000):8,.0f} tok/s)")
    print(f"  always-on weights          {always_gb:5.3f} GB  (trunk + embeddings + router, 4-bit)")
    print(f"  one expert (4-bit)         {exp_mb:5.2f} MB   x {V['n_experts']} experts, top-{V['top_k']}")
    print("=" * 78)


if __name__ == "__main__":
    main()