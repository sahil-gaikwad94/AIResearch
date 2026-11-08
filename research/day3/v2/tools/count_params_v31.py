#!/usr/bin/env python3
"""Kestrel-3.1 cost model. Every number in the v2 documents comes from here.

Same conventions as the v1 tool, so the two are comparable:
  train FLOPs/token = 6 * active + attention + memory-TTT + workspace
  wall hours         = FLOPs / (PEAK * MFU)
"""
import sys

sys.path.insert(0, "/workspace/chats/847e951142ec556fe65be7b2/research/day3/v2")
from kestrel31.model import (FULL as V, active_params, total_params, attn_params,
                             mem_params, moe_params, workspace_params,
                             flops_per_token as fwd_flops)

PEAK = 989e12
MFU = 0.32
BW = 3.35e12            # HBM bytes/s, H100-class
TOTAL = total_params(V)
ACTIVE = active_params(V)
EXPECTED_TOTAL = 21_215_543_938
EXPECTED_ACTIVE = 2_038_003_330

L, D, W = V["layers"], V["d"], V["window"]
WS = workspace_params(V)
ROUTED_ALL = V["n_experts"] * 3 * V["d_lat"] * V["d_ff"] * L
EMBED = V["vocab"] * D


def flops_train(window, r_eff):
    att = 3 * L * 4 * window * D             # qk + av, fwd + bwd
    ttt = 3 * L * 4 * V["d_m"] * V["r_m"]    # read, delta update, grad
    ws = 3 * r_eff * 2 * WS                  # recurrent passes, fwd + bwd
    return 6 * ACTIVE + att + ttt + ws


def flops_fwd(window, r):
    att = L * 4 * window * D
    ws = r * 2 * WS
    return 2 * ACTIVE + att + ws


def gpu_h(tokens, window, r_eff):
    return tokens * flops_train(window, r_eff) / (PEAK * MFU) / 3600


def main():
    print("=" * 78)
    print("KESTREL-3.1  |  capacity 20.9B, compute 1.99B active")
    print("=" * 78)
    assert (TOTAL, ACTIVE) == (EXPECTED_TOTAL, EXPECTED_ACTIVE), "ledger drift"
    print(f"attn/layer      {attn_params(V):>14,}")
    print(f"memory/layer    {mem_params(V):>14,}")
    print(f"moe/layer       {moe_params(V, False):>14,}   "
          f"({V['n_experts']} experts, {V['top_k']} active, d_lat={V['d_lat']})")
    print(f"workspace       {WS:>14,}   ({V['w_layers']} layers, d_w={V['d_w']}, reused r times)")
    print(f"\ntotal           {TOTAL:>14,}")
    print(f"active/token    {ACTIVE:>14,}   ({ACTIVE / TOTAL:.2%} of total)")
    print(f"expert sparsity 1:{V['n_experts'] // V['top_k']}  "
          f"({V['n_experts']}/{V['top_k']})   param ratio 1:{TOTAL / ACTIVE:.1f}")

    print("\n--- memory hierarchy ---")
    state = V["r_m"] * V["d_m"] * L
    kv = L * 256_000 * (V["d_c"] + V["d_rope"])
    print(f"fast state/session   {state:>12,} floats = {state * 2 / 1e6:6.1f} MB fp16")
    print(f"dense KV @256k       {kv:>12,} bytes  = {kv / 1e9:6.2f} GB fp8")
    print(f"workspace state      {V['d_w'] * 2:>12,} bytes/token fp16, independent of r")
    print(f"ratio                              {kv / (state * 2):6.0f}x smaller")

    print("\n--- compute per token, forward (GFLOP) ---")
    print("  from model.flops_per_token: excludes the embedding gather, counts")
    print("  attention at the true head width heads*(dh+dr) = 6,144")
    for r in (1, 2, 4, 8):
        print(f"  r={r}  forward: {fwd_flops(V, r, 4096):6.2f} GFLOP")
    dense_64k = 2 * (V["layers"] * (attn_params(V) + mem_params(V)
                                    + moe_params(V, True) + 3 * V["d"])) / 1e9 \
        + 16 * 4 * 65536 * V["heads"] * (V["dh"] + V["d_rope"]) / 1e9
    print(f"  context 64k: unchanged, the window is 4,096")
    print(f"  dense-attention counterfactual at 64k: {dense_64k:.1f} GFLOP per token")
    cot = 2 * (ACTIVE - EMBED)
    print(f"  one decoding step (no depth): {cot / 1e9:.2f} GFLOP")
    print(f"  one workspace pass:           {2 * WS / 1e9:.3f} GFLOP  "
          f"({cot / (2 * WS):.1f}x cheaper than a CoT step)")

    print("\n--- training budget (windowed attention: context is free) ---")
    print("  training uses the v1 convention 6*active + attention at width d; it")
    print("  overstates by ~15% vs 3*forward, and is kept for comparability with v1")
    stages = [
        ("pretrain @4k",          1_200e9, 4096, 1.9),
        ("window extend @8k",       150e9, 8192, 1.9),
        ("long-context @64k ctx",    60e9, 4096, 1.9),
        ("PR mid-train @32k ctx",    60e9, 4096, 1.9),
        ("SFT @32k ctx",              3e9, 4096, 2.5),
    ]
    tot = 0.0
    for name, tok, win, r in stages:
        h = gpu_h(tok, win, r)
        tot += h
        print(f"  {name:22s} {tok / 1e9:6.0f}B tok  {h:8,.0f} GPU-h")

    teacher_active = 25e9
    dist_tok, dist_r = 30e9, 2.0
    dist_fpt = (2 * ACTIVE) + (2 * teacher_active) + (6 * ACTIVE) + 3 * dist_r * 2 * WS
    dist_h = dist_tok * dist_fpt / (PEAK * MFU) / 3600
    tot += dist_h
    print(f"  {'distillation (on-policy)':22s} {dist_tok / 1e9:6.0f}B tok  {dist_h:8,.0f} GPU-h"
          f"   (teacher {teacher_active / 1e9:.0f}B active)")

    tasks, rollouts, gen_tok = 60_000, 6, 8_000
    gen = tasks * rollouts * gen_tok
    rl_h = gen / 20_000 / 3600 + gpu_h(gen, 4096, 3.0)
    tot += rl_h
    env_core_h = tasks * rollouts * 90 / 3600
    print(f"  {'execution RL':22s} {gen / 1e9:6.2f}B tok  {rl_h:8,.0f} GPU-h"
          f"   + {env_core_h:,.0f} CPU core-h of test execution")

    si_tasks, si_rounds, si_roll = 6_000, 4, 5
    si_gen = si_tasks * si_rounds * si_roll * 5_000
    si_h = si_gen / 20_000 / 3600 + gpu_h(si_gen, 4096, 3.0)
    tot += si_h
    si_env = si_tasks * si_rounds * si_roll * 90 / 3600
    env_core_h += si_env
    print(f"  {'self-improvement x4':22s} {si_gen / 1e9:6.2f}B tok  {si_h:8,.0f} GPU-h"
          f"   + {si_env:,.0f} CPU core-h")

    abl = 2_000.0
    tot += abl
    print(f"  {'ablations':22s} {'':9s}  {abl:8,.0f} GPU-h")
    sub = tot
    restarts = 0.12 * sub
    print(f"  {'subtotal':22s} {'':9s}  {sub:8,.0f} GPU-h")
    print(f"  {'restarts (12%)':22s} {'':9s}  {restarts:8,.0f} GPU-h")
    print(f"  {'TOTAL':22s} {'':9s}  {sub + restarts:8,.0f} GPU-h")
    for rate in (1.5, 2.2, 3.0):
        print(f"     at ${rate}/GPU-h -> ${(sub + restarts) * rate / 1000:.0f}k")
    print(f"  wall clock on 128 GPUs: {(sub + restarts) / 128 / 24:.1f} days")
    print(f"  CPU core-hours for execution: {env_core_h:,.0f} "
          f"({env_core_h / 64:.0f} node-hours at 64 cores)")

    print("\n--- serving, one decode step, 4-bit weights (0.55 B/param) ---")
    always = TOTAL - EMBED - ROUTED_ALL
    exp_bytes = 4.72e6 * 0.55
    print(f"  always-on weights   {always * 0.55 / 1e9:6.3f} GB")
    print(f"  one expert          {exp_bytes / 1e6:6.2f} MB")
    print(f"  batch  bytes/token   experts  tok/s @ {BW / 1e12:.2f} TB/s   "
          f"at r=4 pass cost")
    for b in (1, 16, 64):
        draw = b * V["top_k"]
        distinct = V["n_experts"] * (1 - (1 - 1 / V["n_experts"]) ** draw)
        # routing is per-layer and independent across layers, so the step
        # streams the distinct expert slots touched on EVERY layer: the expert
        # bytes carry a factor of L (an earlier draft left it out -- a 16x
        # under-count of the routed bytes).
        step_bytes = always * 0.55 + L * distinct * exp_bytes
        per_tok = step_bytes / b
        tps = b / (step_bytes / BW)
        print(f"  {b:5d}  {per_tok / 1e9:9.2f} GB  {distinct:7.1f}  {tps:10,.0f}")
    print(f"  4-bit weights total: {TOTAL * 0.55 / 1e9:.1f} GB  "
          f"-> 24 GB card fits with {24 - TOTAL * 0.55 / 1e9:.1f} GB to spare")
    print("  note: batch>=16 is FLOP-bound, not bandwidth-bound; the tok/s above")
    print("        is the bandwidth ceiling, not a prediction")
    print("=" * 78)


if __name__ == "__main__":
    main()
