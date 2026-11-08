"""
Kestrel-2 (day 2) — parameter and cost model.

Two independent obligations, same as day 1:
  1. Derive every headline number here, from the config, with no dependencies.
  2. kestrel2/model.py must independently reproduce the parameter count.

Day 2 adds: sparse attention cost model (window + compressed global + selected
blocks + structural anchors), a depth-recurrence loop, and a head-to-head
comparison against the day-1 design at equal context length.
"""

# ----------------------------------------------------------------------------
# day 1, for comparison only (values verified in research/day1)
# ----------------------------------------------------------------------------
V1 = dict(name="kestrel-coder-4b", d=3072, layers=40, heads=24, dh=128,
          d_c=448, d_rope=64, d_ff=7680, vocab=131072, params=4_083_410_944)

# ----------------------------------------------------------------------------
# day 2
# ----------------------------------------------------------------------------
V = dict(
    vocab=131_072,
    d=4096, heads=32, dh=128,
    d_c=512, d_qc=512, d_rope=64,
    d_ff=12_288,
    prelude=2, recur=16, coda=2,          # 20 unique layers
    mtp=2,
    # sparse attention geometry
    window=2048,          # exact local window, every sparse layer
    cmp_stride=64,        # compression block: 1 summary key per 64 tokens
    sel_block=64,         # selection block size
    sel_topk=12,          # blocks selected per head-group
    anchor_budget=256,    # ALWAYS-attended structural tokens (definition lines)
    idx_dim=128,          # indexer query/key width
    max_r=8,              # trained/extrapolable recurrent-step ceiling
)

def rms(d): return d

def layer_params(d, d_c, d_qc, d_rope, n_h, dh, d_ff, indexer=False):
    """One MLA + SwiGLU block. No biases. Two RMSNorms."""
    p = {}
    p["W_DQ"]  = d * d_qc
    p["W_UQ"]  = d_qc * n_h * dh
    p["W_QR"]  = d * d_rope              # decoupled rope key, applied pre-cache
    p["W_DKV"] = d * (d_c + d_rope)      # FUSED: emits c_kv || k_pe in one matmul
    p["W_UK"]  = d_c * n_h * dh
    p["W_UV"]  = d_c * n_h * dh
    p["W_O"]   = d * n_h * dh
    p["qk_norm"] = 2 * dh          # per-head RMSNorm on q and k
    p["sinks"] = V.get("sinks", 4)  # learned attention sinks, per head
    p["norms"] = 2 * d
    p["ffn"]   = 3 * d * d_ff            # gate, up, down
    if indexer:
        # hierarchical indexer: coarse block projection + fine query/key
        p["W_IQ"] = d * V["idx_dim"]
        p["W_IK"] = d_c * V["idx_dim"]
        p["W_IP"] = d_c * V["idx_dim"]   # coarse pooled-block projector
    return p

def count():
    d, d_c, d_qc, d_rope = V["d"], V["d_c"], V["d_qc"], V["d_rope"]
    n_h, dh, d_ff = V["heads"], V["dh"], V["d_ff"]
    sparse = layer_params(d, d_c, d_qc, d_rope, n_h, dh, d_ff, indexer=True)
    n_unique = V["prelude"] + V["recur"] + V["coda"]
    per_layer = sum(sparse.values())
    body = per_layer * n_unique
    embed = V["vocab"] * d
    final_norm = d                        # RMSNorm before the head
    loop_emb = V.get("max_r", 8) * d       # per-iteration index embedding
    mtp = V["mtp"] * (d * d + d)          # RMSNorm weight (d) + Linear(d,d) each
    return dict(per_layer=per_layer, attn_only=sum(v for k, v in sparse.items()
                if not k.startswith(("ffn", "norms", "W_IQ", "W_IK", "W_IP"))),
                ffn=sparse["ffn"], body=body, embed=embed, mtp=mtp,
                inference=body + final_norm + loop_emb + embed,
                training=body + final_norm + loop_emb + embed + mtp,
                n_unique=n_unique, sparse_layers=n_unique)

# ----------------------------------------------------------------------------
# attention cost model
# ----------------------------------------------------------------------------
def attended_dense(ctx): return ctx

def attended_sparse(ctx, v=V):
    """Tokens a sparse query actually reads, per layer."""
    if ctx <= v["window"]:
        return ctx
    return v["window"] + ctx // v["cmp_stride"] + v["sel_topk"] * v["sel_block"] + v["anchor_budget"]

def flops_per_token(ctx, cfg, recurrent_steps=1, train=True):
    """2N fwd / 6N train for the linear parts; 4*h*dh*attended for attention."""
    n = cfg["params"] if isinstance(cfg, dict) else cfg
    raise SystemExit("unused")

def flops(ctx, v, v2=False, rr=1.0, train=True):
    """FLOPs per token. rr = mean recurrent executions of the shared block."""
    d, n_h, dh = v["d"], v["heads"], v["dh"]
    if not v2:
        n = v["params"]; layers = v["layers"]; att = ctx
        lin = (6 if train else 2) * n
        a = (3 if train else 1) * 4 * n_h * dh * att * layers
        return lin + a
    # v2: unique weights run `exec` times
    exec_layers = v["prelude"] + v["recur"] * rr + v["coda"]
    e = exec_layers / n_layers(v, v2=True)
    n = count()["inference"]
    lin = (6 if train else 2) * n * e
    att = attended_sparse(ctx, v)
    a = (3 if train else 1) * 4 * n_h * dh * att * exec_layers
    # indexer: coarse + fine scan, per sparse layer per execution
    idx = (6 if train else 2) * (ctx // v["cmp_stride"]) * v["idx_dim"] * 2 * v["recur"] * rr
    return lin + a + idx

def n_layers(v, v2=False):
    """v2 caches one KV entry per UNIQUE layer: the recurrent block shares its
    cache across iterations by construction."""
    if not v2:
        return v["layers"]
    return v["prelude"] + v["recur"] + v["coda"]

def kv_bytes(v, fp8=True, v2=False):
    layers = n_layers(v, v2)
    return layers * (v["d_c"] + v["d_rope"]) * (1 if fp8 else 2)

def kv_read_bytes_per_decode(v, ctx, v2=False):
    """Bytes of cache a single decode step must stream for attention."""
    layers = n_layers(v, v2)
    att = attended_sparse(ctx, v) if v2 else ctx
    return layers * att * (v["d_c"] + v["d_rope"])

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    c = count()
    G = 1e9
    print("=" * 74)
    print("KESTREL-2 — parameter count")
    print("=" * 74)
    for k, val in c.items():
        if isinstance(val, int):
            print(f"  {k:12s} {val:>15,}   ({val/G:.4f} B)")
    print(f"\n  unique layers      : {c['n_unique']}")
    print(f"  effective depth    : R=1 {V['prelude']+V['recur']+V['coda']}, "
          f"R=2 {V['prelude']+2*V['recur']+V['coda']}, "
          f"R=3 {V['prelude']+3*V['recur']+V['coda']}, "
          f"R=4 {V['prelude']+4*V['recur']+V['coda']}")

    print("\n" + "=" * 74)
    print("KV CACHE — v1 vs v2 (FP8)")
    print("=" * 74)
    b1, b2 = kv_bytes(V1), kv_bytes(V, v2=True)
    print(f"  v1: {b1:,} B/token ({b1/1024:.0f} KiB)")
    print(f"  v2: {b2:,} B/token ({b2/1024:.0f} KiB)")
    for ctx in (8_192, 65_536, 262_144, 1_048_576):
        print(f"  ctx {ctx:>9,}: v1 {b1*ctx/2**30:7.2f} GiB   v2 {b2*ctx/2**30:7.2f} GiB")

    print("\n" + "=" * 74)
    print("ATTENTION READS per decode step (cache bytes streamed)")
    print("=" * 74)
    print(f"  {'ctx':>10}  {'v1':>10}  {'v2':>10}  {'ratio':>7}   {'attended/layer v2':>18}")
    for ctx in (4_096, 8_192, 16_384, 65_536, 262_144, 1_048_576):
        r1 = kv_read_bytes_per_decode(V1, ctx)
        r2 = kv_read_bytes_per_decode(V, ctx, v2=True)
        print(f"  {ctx:>10,}  {r1/2**30:9.3f}G  {r2/2**30:9.3f}G  {r1/r2:6.1f}x   "
              f"{attended_sparse(ctx):>18,}")

    print("\n" + "=" * 74)
    print("FLOPS PER TOKEN — training (6N + attention + indexer)")
    print("=" * 74)
    print(f"  {'ctx':>10}  {'v1 (GFLOP)':>12}  {'v2 R=2 (GFLOP)':>15}  {'v1/v2':>7}")
    for ctx in (4_096, 16_384, 65_536, 262_144, 1_048_576):
        f1 = flops(ctx, V1) / G
        f2 = flops(ctx, V, v2=True, rr=2.0) / G
        print(f"  {ctx:>10,}  {f1:12,.0f}  {f2:15,.0f}  {f1/f2:6.2f}x")

    print("\n" + "=" * 74)
    print("FLOPS PER TOKEN — inference (2N + attention)")
    print("=" * 74)
    print(f"  {'ctx':>10}  {'v1 R=-':>10}  {'v2 R=2':>10}  {'v2 R=4':>10}")
    for ctx in (4_096, 65_536, 262_144, 1_048_576):
        print(f"  {ctx:>10,}  {flops(ctx,V1,train=False)/G:10,.2f}  "
              f"{flops(ctx,V,v2=True,rr=2,train=False)/G:10,.2f}  "
              f"{flops(ctx,V,v2=True,rr=4,train=False)/G:10,.2f}")

    # ------------------------------------------------------------------
    # budget
    # ------------------------------------------------------------------
    # H100 bf16 dense peak 989 TFLOPS; 35% MFU. Day 1 used the same 35%, and
    # this model reproduces its 24,238 GPU-h for 1 T tokens @ 4k (see sanity line).
    PEAK = 989e12
    EFF = 0.35 * PEAK
    stages = [
        # name, tokens, ctx, mean recurrent steps
        ("pretrain @4k  (R=1)",           800e9,   4_096, 1.0),
        ("pretrain @16k (R=1)",           160e9,  16_384, 1.0),
        ("pretrain @64k (R=1)",            60e9,  65_536, 1.0),
        ("loop adaptation @32k (R~1-3)",  200e9,  32_768, 2.0),
        ("long-context @256k (R=2)",       12e9, 262_144, 2.0),
        ("mid-train event-log @128k",      40e9, 131_072, 2.5),
        ("SFT @128k (R=3)",                 3e9, 131_072, 3.0),
    ]
    print("\n" + "=" * 74)
    print("TRAINING BUDGET (v2)")
    print("=" * 74)
    total = 0.0
    for name, toks, ctx, rr in stages:
        f = flops(ctx, V, v2=True, rr=rr)
        h = toks * f / EFF / 3600
        total += h
        print(f"  {name:32s} {toks/1e9:7.0f}B tok {ctx:>8,}  {h:9,.0f} GPU-h")
    rl, abl = 3_000, 1_600
    sub = total + rl + abl
    reserve = 0.12 * sub
    print(f"  {'RL (generation-bound)':28s} {'':7s} {'':9s}  {rl:9,.0f} GPU-h")
    print(f"  {'ablations (26 arms @1B proxy)':28s} {'':7s} {'':9s}  {abl:9,.0f} GPU-h")
    print(f"  {'restarts / loss spikes (12%)':28s} {'':7s} {'':9s}  {reserve:9,.0f} GPU-h")
    print("-" * 74)
    print(f"  {'TOTAL':32s} {'':7s} {'':9s}  {sub+reserve:9,.0f} GPU-h")
    for rate in (1.5, 2.2, 3.0):
        print(f"     at ${rate}/GPU-h -> ${(sub+reserve)*rate/1000:.0f}k")
    sanity = 1_000e9 * flops(4_096, V1) / EFF / 3600
    print(f"\n  sanity: v1 @4k for 1T tokens = {sanity:,.0f} GPU-h "
          f"(day 1 budget says 24,238)")

    print("\n" + "=" * 74)
    print("HEADLINE COMPARISON")
    print("=" * 74)
    print(f"  {'':34s} {'v1':>12s} {'v2':>12s}")
    rows = [
        ("parameters (inference)", f"{V1['params']/G:.3f} B", f"{c['inference']/G:.3f} B"),
        ("unique layers", "40", f"{c['n_unique']}"),
        ("effective depth (serve)", "40", "52 (R=3)"),
        ("KV / token (FP8)", f"{b1/1024:.0f} KiB", f"{b2/1024:.0f} KiB"),
        ("cache @256k", f"{b1*262144/2**30:.2f} GiB", f"{b2*262144/2**30:.2f} GiB"),
        ("attended tok/layer @64k", "65,536", f"{attended_sparse(65536):,}"),
        ("attended tok/layer @1M", "1,048,576", f"{attended_sparse(1_048_576):,}"),
        ("cache read / decode @256k", f"{kv_read_bytes_per_decode(V1,262144)/2**30:.2f} GiB",
         f"{kv_read_bytes_per_decode(V,262144,True)/2**30:.2f} GiB"),
    ]
    for r in rows:
        print(f"  {r[0]:34s} {r[1]:>12s} {r[2]:>12s}")
    print("=" * 74)
