"""
Kestrel-4B — analytic cost model.

Pure Python, no dependencies. Every number in research/day1/*.md is produced
by this file, so the prose and the arithmetic cannot drift apart.

    python3 research/day1/tools/count_params.py
"""

VOCAB = 131_072
D = 3072              # d_model
L = 40                # layers
H = 24                # query heads
DH = 128              # head dim  -> h * dh = 3072 = d
D_C = 448             # MLA latent (kv compression) dim
D_ROPE = 64           # decoupled rope key dim
DFF = 7680            # SwiGLU hidden (2.5 x d)
MTP = 2               # extra prediction heads (training only)
TIE = True

# Recurrent-state / hybrid comparison points, for the "why not X" table
GQA_KV_HEADS = 4


def attn_params():
    """MLA as in DeepSeek-V2/V3: low-rank joint KV, decoupled rope key.

    The rope key MUST be decoupled: RoPE cannot be applied to c_kv, because
    c_kv is cached *before* the per-head up-projection. So the rope-carrying
    q/k channels come from their own direct projections of x.
    """
    w_dq = D * D_C                      # down-project queries
    w_uq = D_C * (H * DH)               # up-project queries
    # Fused variant: one down-projection emits c_kv || k_pe. DeepSeek keeps a
    # separate W_KR for k_pe; fusing is identical in FLOPs and saves a matmul.
    # (An earlier draft of this file counted BOTH, over-stating the model by
    # 0.197M/layer. The torch reference caught it — see 09-decisions.md, D7.)
    w_dkv = D * (D_C + D_ROPE)          # down-project kv -> (c_kv || k_pe)
    w_uk = D_C * (H * DH)               # up-project keys
    w_uv = D_C * (H * DH)               # up-project values
    w_o = (H * DH) * D                  # output
    w_qr = D * (H * D_ROPE)             # q rope channels, direct from x
    qk_norm = 2 * DH                    # per-head RMSNorm on q and k
    ckv_norm = D_C                      # latent norm BEFORE w_uk (absorbable)
    return dict(w_dq=w_dq, w_uq=w_uq, w_dkv=w_dkv, w_uk=w_uk, w_uv=w_uv,
                w_o=w_o, w_qr=w_qr, qk_norm=qk_norm, ckv_norm=ckv_norm)


def ffn_params():
    return dict(gate=D * DFF, up=D * DFF, down=DFF * D)


def layer_params():
    a = attn_params()
    f = ffn_params()
    norms = 2 * D                       # pre-attn, pre-ffn RMSNorm
    return sum(a.values()) + sum(f.values()) + norms, a, f


def breakdown():
    per_layer, a, f = layer_params()
    emb = VOCAB * D
    body = per_layer * L
    # MTP: one projection + one norm per extra head; shares the embedding
    mtp = MTP * (D * D + D)
    total = body + (0 if TIE else 2 * emb) + (emb if TIE else 0) + mtp
    infer = body + emb                 # MTP modules are dropped at inference
    return dict(per_layer=per_layer, attn=sum(a.values()), ffn=sum(f.values()),
                body=body, emb=emb, mtp=mtp, total=total,
                infer=infer, attn_parts=a, ffn_parts=f)


def kv_bytes_per_token(dtype_bytes=1):
    """MLA stores only c_kv + k_rope per layer per token."""
    per_layer = (D_C + D_ROPE) * dtype_bytes
    return per_layer, per_layer * L


def gqa_kv_bytes_per_token(n_kv=GQA_KV_HEADS, dtype_bytes=2):
    per_layer = 2 * n_kv * DH * dtype_bytes
    return per_layer, per_layer * L


def flops_per_token(ctx=None):
    """6N per token for training (fwd+bwd); inference prefill ~2N + attention."""
    n = breakdown()["infer"]
    base = 6 * n
    if ctx:
        base += 12 * L * H * DH * ctx   # qk^T and av, fwd+bwd
    return base


def run():
    b = breakdown()
    M = 1e6
    print("=" * 66)
    print("KESTREL-4B  parameter count")
    print("=" * 66)
    print(f"vocab            {VOCAB:>12,}   (tied embeddings: {TIE})")
    print(f"d_model          {D:>12,}")
    print(f"layers           {L:>12,}")
    print(f"heads            {H:>12,} x {DH}  = {H*DH}")
    print(f"MLA latent       {D_C:>12,}  + rope {D_ROPE}")
    print(f"FFN (SwiGLU)     {DFF:>12,}")
    print("-" * 66)
    print("per-layer breakdown")
    for k, v in b["attn_parts"].items():
        print(f"  attn.{k:<9} {v/M:>10.3f} M")
    print(f"  {'attn total':<14} {b['attn']/M:>10.3f} M")
    for k, v in b["ffn_parts"].items():
        print(f"  ffn.{k:<10} {v/M:>10.3f} M")
    print(f"  {'ffn total':<14} {b['ffn']/M:>10.3f} M")
    print(f"  {'LAYER TOTAL':<14} {b['per_layer']/M:>10.3f} M")
    print("-" * 66)
    print(f"body  {L} layers        {b['body']/1e9:>10.3f} B")
    print(f"embeddings (tied)      {b['emb']/1e9:>10.3f} B")
    print(f"MTP modules (train)    {b['mtp']/1e9:>10.3f} B")
    print(f"TOTAL (train)          {b['total']/1e9:>10.3f} B")
    print(f"TOTAL (inference)      {b['infer']/1e9:>10.3f} B")
    print(f"attn share of body     {100*b['attn']/b['per_layer']:>10.1f} %")
    print(f"ffn  share of body     {100*b['ffn']/b['per_layer']:>10.1f} %")

    print()
    print("=" * 66)
    print("KV CACHE (per token)")
    print("=" * 66)
    for name, dt in [("FP16", 2), ("FP8", 1)]:
        pl, tot = kv_bytes_per_token(dt)
        print(f"Kestrel MLA  {name}: {pl:>5} B/layer  {tot:>6} B/token  "
              f"({tot/1024:.1f} KiB)")
        for ctx in (8_192, 65_536, 262_144):
            print(f"    ctx {ctx:>7,}: {tot*ctx/2**30:>7.2f} GiB")
    pl, tot = gqa_kv_bytes_per_token()
    print(f"GQA-4 (FP16)      : {pl:>5} B/layer  {tot:>6} B/token  "
          f"({tot/1024:.1f} KiB)  -> "
          f"{tot/kv_bytes_per_token(1)[1]:.1f}x worse than MLA-FP8")
    pl, tot = 2 * D * 2, 2 * D * 2 * L
    print(f"MHA (FP16)        : {pl:>5} B/layer  {tot:>6} B/token  "
          f"({tot/1024:.1f} KiB)  -> "
          f"{tot/kv_bytes_per_token(1)[1]:.1f}x worse than MLA-FP8")

    print()
    print("=" * 66)
    print("COMPUTE")
    print("=" * 66)
    n = b["infer"]
    for d in (10e9, 100e9, 300e9, 1_200e9, 2_400e9):
        f = 6 * n * d
        print(f"train {d/1e9:>5.0f}B tok : {f:.2e} FLOPs   "
              f"{f/3.5e14/3600:>8,.0f} GPU-h @350 TF/s eff")
    print()
    print("ATTENTION vs PARAMETER FLOPs per token (training, fwd+bwd)")
    print(f"  params: 6N = {6*n:.2e}")
    for ctx in (4_096, 16_384, 65_536, 262_144):
        att = 12 * L * H * DH * ctx
        print(f"  ctx {ctx:>7,}: attention {att:.2e}  = {100*att/(6*n):>6.1f}% of 6N")

    print()
    print("inference, per token (prefill, ctx=64k): "
          f"{flops_per_token(65536):.2e} FLOPs")

    print()
    print("=" * 66)
    print("RUN BUDGET (350 TF/s effective ~ 40% MFU, H100-class)")
    print("=" * 66)
    # stage: (name, tokens, context). attention FLOPs are paid per token, so
    # the long-context stage is far more expensive than its token count suggests.
    stages = [
        ("pretrain @4k", 1000e9, 4_096),
        ("pretrain @16k", 160e9, 16_384),
        ("pretrain @64k", 40e9, 65_536),
        ("mid-train repo-packed @32k", 50e9, 32_768),
        ("SFT @32k", 4e9, 32_768),
    ]
    tot = 0
    for name, tok, ctx in stages:
        f = tok * (6 * n + 12 * L * H * DH * ctx)
        h = f / 3.5e14 / 3600
        tot += h
        print(f"{name:<30} {tok/1e9:>6.0f}B tok  {h:>8,.0f} GPU-h")
    # RL is generation-bound, not 6ND: budget by decode throughput instead.
    rl_tokens = 24e9
    tps_per_gpu = 3_000          # batched fp8 decode, 4B model
    rl_h = rl_tokens / tps_per_gpu / 3600
    print(f"{'RL: 8B tok x 3 rollouts':<30} {'':>6}     {rl_h:>8,.0f} GPU-h")
    tot_rl = tot + rl_h
    abl = 24 * 8e9 * (6 * 1.0e9 + 12 * 16 * 16 * 64 * 8_192) / 3.5e14 / 3600
    print(f"{'ablations: 24 x 8B @1B proxy':<30} {'':>6}     {abl:>8,.0f} GPU-h")
    subtotal = tot_rl + abl
    reserve = subtotal * 0.12
    print(f"{'restarts / loss spikes (12%)':<30} {'':>6}     {reserve:>8,.0f} GPU-h")
    grand = subtotal + reserve
    print("-" * 66)
    print(f"{'TOTAL':<38} {grand:>8,.0f} GPU-h")
    for price in (1.5, 2.2, 3.0):
        print(f"   at ${price}/GPU-h -> ${grand*price/1000:,.0f}k")


if __name__ == "__main__":
    run()
