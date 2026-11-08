#!/usr/bin/env python3
"""Kestrel-3 cost model. Every number in the documents comes from here."""
import math

V = dict(vocab=131072, d=4096, heads=32, dh=128, d_c=512, d_rope=64,
         layers=16, window=4096, d_m=512, r_m=256,
         n_experts=96, top_k=3, d_ff=1024)

PEAK = 989e12
MFU = 0.32          # MoE all-to-all and padding cost real time
TOTAL = 20_485_187_584
ACTIVE = 1_761_814_528


def attn_params():
    d, dc, dr, h, dh = V["d"], V["d_c"], V["d_rope"], V["heads"], V["dh"]
    return (d*dc) + (dc*h*dh) + (d*dr) + (d*(dc+dr)) + 2*(dc*h*dh) + (h*dh*d) + 2*dh + h


def mem_params():
    return 5 * V["d"] * V["d_m"] + V["d_m"] * V["r_m"]


def moe_params():
    d, e, ff = V["d"], V["n_experts"], V["d_ff"]
    return e * 3 * d * ff + d * e + e


def per_layer():
    return attn_params() + mem_params() + moe_params() + 3 * V["d"]


def flops_per_token(window):
    """6N forward+backward, plus quadratic-in-window attention, plus TTT."""
    d, L, w = V["d"], V["layers"], window
    n = ACTIVE
    att = 3 * L * 4 * w * d              # qk + av, fwd+bwd
    ttt = 3 * L * 4 * V["d_m"] * V["r_m"]  # read + delta update + grad
    return 6 * n + att + ttt


def gpu_h(tokens, window):
    return tokens * flops_per_token(window) / (PEAK * MFU) / 3600


def main():
    print("=" * 74)
    print("KESTREL-3  |  capacity 20.5B, compute 1.76B active")
    print("=" * 74)
    print(f"attn/layer      {attn_params():>14,}")
    print(f"memory/layer    {mem_params():>14,}")
    print(f"moe/layer       {moe_params():>14,}   ({V['n_experts']} experts, "
          f"{V['top_k']} active -> {V['top_k']}x{3*V['d']*V['d_ff']:,} active)")
    print(f"norms/layer     {3*V['d']:>14,}   (3 pre-norms: attn, memory, moe)  +A {V['d_m']*V['r_m']:,}")
    print(f"per layer       {per_layer():>14,}   x {V['layers']} = {per_layer()*V['layers']:,}")
    body = per_layer() * V["layers"]
    print(f"\ntotal           {body + V['vocab']*V['d'] + V['d']:>14,}")
    print(f"active/token    {ACTIVE:>14,}   ({ACTIVE/TOTAL:.2%} of total)")
    print(f"activation     1:{V['n_experts']//V['top_k']}   "
          f"tokens per active param @1.7T = {1.7e12/ACTIVE:,.0f}")
    print(f"tokens per total param        = {1.7e12/TOTAL:,.0f}")

    print("\n--- memory hierarchy ---")
    state_floats = V["r_m"] * V["d_m"] * V["layers"]
    kv_256k = V["layers"] * 256_000 * (V["d_c"] + V["d_rope"]) * 1
    print(f"fast state/session   {state_floats:>12,} floats = {state_floats*2/1e6:6.1f} MB fp16")
    print(f"dense KV @256k       {kv_256k:>12,} bytes  = {kv_256k/1e9:6.2f} GB fp8")
    print(f"ratio                              {kv_256k/(state_floats*2):6.0f}x smaller")

    print("\n--- compute per token (GFLOP) ---")
    for w in (4096, 16384, 65536, 262144):
        f = flops_per_token(w)
        print(f"  window {w:>7,}: {f/1e9:7.2f}  "
              f"(attention {3*V['layers']*4*w*V['d']/1e9:5.2f})")
    print("  -> context length no longer changes training cost; only the window does")

    stages = [
        ("pretrain @4k",              1_400e9, 4096),
        ("window extend @8k",           200e9, 8192),
        ("memory training @64k",        100e9, 4096),
        ("PR mid-train @32k",            80e9, 4096),
        ("SFT @32k",                      4e9, 4096),
    ]
    print("\n--- budget ---")
    tot = 0.0
    for name, tok, w in stages:
        h = gpu_h(tok, w)
        tot += h
        print(f"  {name:28s} {tok/1e9:7.0f}B tok  {h:8,.0f} GPU-h")
    rl, abl = 3000.0, 2500.0
    print(f"  {'RL (generation-bound)':28s} {'':12s}  {rl:8,.0f} GPU-h")
    print(f"  {'ablations':28s} {'':12s}  {abl:8,.0f} GPU-h")
    sub = tot + rl + abl
    print(f"  {'subtotal':28s} {'':12s}  {sub:8,.0f} GPU-h")
    print(f"  {'restarts (12%)':28s} {'':12s}  {sub*0.12:8,.0f} GPU-h")
    grand = sub * 1.12
    print(f"  {'TOTAL':28s} {'':12s}  {grand:8,.0f} GPU-h")
    for p in (1.5, 2.2, 3.0):
        print(f"     at ${p}/GPU-h -> ${grand*p/1000:,.0f}k")
    print(f"  wall clock on 128 GPUs: {grand/128/24:.1f} days")

    print("\n--- serving, one decode step ---")
    w = 4096
    kv = V["layers"] * w * (V["d_c"] + V["d_rope"])
    state = state_floats * 1
    for batch in (1, 16):
        distinct = V["n_experts"] * (1 - (1 - V["top_k"]/V["n_experts"])**batch)
        ebytes = distinct * 3 * V["d"] * V["d_ff"]
        total = ACTIVE * 1 + kv + state + ebytes
        print(f"  batch {batch:>2}: experts touched {distinct:5.1f}  "
              f"bytes {total/1e9:5.2f} GB  -> {total/3.35e12*1e3:5.2f} ms/step  "
              f"{batch/(total/3.35e12):8,.0f} tok/s")
    print("=" * 74)


if __name__ == "__main__":
    main()
