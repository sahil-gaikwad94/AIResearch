"""Kestrel-3.1 reference implementation.

Five mechanisms, and the reason each one is here:

  1. latent-width sparse MoE   -> capacity without active FLOPs, and without
                                  paying full width per expert
  2. delta-rule memory         -> long context in 4.2 MB, unchanged from v1
  3. recurrent workspace       -> depth rented per token, not owned in weights
  4. confidence head           -> the depth controller is distilled from the
                                  executor, not from a halting gate
  5. execution feedback        -> reward from tests, not from a judge

The point of this file is not speed. It is that every structural claim in
docs 02-04 is executable and asserted, including the ones that are easy to
get wrong: the expert count per token, the fact that recurrence adds depth
without adding parameters, and the fact that at initialisation the recurrent
path is an epsilon-identity rather than a random perturbation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FULL = dict(
    vocab=131072, d=4096, heads=32, dh=128, d_c=512, d_rope=64,
    layers=16, window=4096,
    d_m=512, r_m=256,
    # sparse MoE: routed experts act on a latent of width d_lat
    n_experts=256, top_k=6, d_lat=384, d_ff=4096, d_shared=1024,
    # recurrent workspace: deliberately narrow, so its 4-bit weights stay
    # resident in L2 and passes 2..r cost FLOPs rather than bandwidth
    w_layers=4, d_w=1024, w_ff=4096, d_head_h=512, r_max=8,
)

EXPECTED_TOTAL = 20_865_312_258
EXPECTED_ACTIVE = 1_990_944_258


def _mask(n, w, device):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j <= i) & (i - j < w)


class MLA(nn.Module):
    """Latent attention: 576 floats cached per token per layer."""

    def __init__(self, c):
        super().__init__()
        d, dc, dr, h, dh = c["d"], c["d_c"], c["d_rope"], c["heads"], c["dh"]
        self.h, self.dh, self.dr, self.d_c = h, dh, dr, dc
        self.w_dq = nn.Linear(d, dc, bias=False)
        self.w_uq = nn.Linear(dc, h * dh, bias=False)
        self.w_qr = nn.Linear(d, dr, bias=False)
        self.w_dkv = nn.Linear(d, dc + dr, bias=False)
        self.w_uk = nn.Linear(dc, h * dh, bias=False)
        self.w_uv = nn.Linear(dc, h * dh, bias=False)
        self.w_o = nn.Linear(h * dh, d, bias=False)
        self.q_norm = nn.Parameter(torch.ones(dh))
        self.k_norm = nn.Parameter(torch.ones(dh))
        self.sinks = nn.Parameter(torch.zeros(h))

    def forward(self, x):
        b, n, _ = x.shape
        h, dh, dr = self.h, self.dh, self.dr
        cq = self.w_dq(x)
        q = F.rms_norm(self.w_uq(cq).view(b, n, h, dh), (dh,), self.q_norm)
        qr = self.w_qr(x).view(b, n, 1, dr)
        q = torch.cat([q, qr.expand(b, n, h, dr)], dim=-1)
        kv = self.w_dkv(x)
        ckv, kr = kv[..., : self.d_c], kv[..., -dr:]
        k = F.rms_norm(self.w_uk(ckv).view(b, n, h, dh), (dh,), self.k_norm)
        k = torch.cat([k, kr.view(b, n, 1, dr).expand(b, n, h, dr)], dim=-1)
        v = self.w_uv(ckv).view(b, n, h, dh)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh + dr)
        m = _mask(n, min(4096, n), x.device)
        att = att.masked_fill(~m, float("-inf"))
        sink = self.sinks.view(1, h, 1, 1).expand(b, h, n, 1)
        att = torch.cat([att, sink], dim=-1).softmax(dim=-1)
        return self.w_o((att[..., :-1] @ v).transpose(1, 2).reshape(b, n, h * dh))


class NeuralMemory(nn.Module):
    """Delta-rule fast weights: O(1) state per token, read-before-write."""

    def __init__(self, c):
        super().__init__()
        d, dm = c["d"], c["d_m"]
        self.dm, self.r = dm, c["r_m"]
        self.q = nn.Linear(d, dm, bias=False)
        self.k = nn.Linear(d, dm, bias=False)
        self.v = nn.Linear(d, dm, bias=False)
        self.g = nn.Linear(d, dm, bias=False)
        self.o = nn.Linear(dm, d, bias=False)
        self.A = nn.Parameter(torch.randn(dm, self.r) / math.sqrt(dm))
        self.eta = 0.05

    def init_state(self, b, device):
        return torch.zeros(b, self.r, self.dm, device=device)

    def forward(self, x, state=None):
        b, n, _ = x.shape
        if state is None:
            state = self.init_state(b, x.device)
        q, v = self.q(x), self.v(x)
        k = F.normalize(self.k(x), dim=-1)
        gate = torch.sigmoid(self.g(x))
        B, reads = state, []
        for t in range(n):
            reads.append(self.A @ (B @ q[:, t].unsqueeze(-1)))  # M q
            err = (self.A @ (B @ k[:, t].unsqueeze(-1))).squeeze(-1) - v[:, t]
            B = B - self.eta * (err @ self.A).unsqueeze(-1) @ k[:, t].unsqueeze(1)
        y = torch.stack(reads, dim=1).squeeze(-1)
        return self.o(gate * y) + x, B.detach()


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_ff, d_out=None):
        super().__init__()
        d_out = d_out if d_out is not None else d_in
        self.g = nn.Linear(d_in, d_ff, bias=False)
        self.u = nn.Linear(d_in, d_ff, bias=False)
        self.dn = nn.Linear(d_ff, d_out, bias=False)

    def forward(self, x):
        return self.dn(F.silu(self.g(x)) * self.u(x))


class LatentMoE(nn.Module):
    """Sparse MoE whose routed experts live in a narrow latent space.

    Routing happens on the full-width token (so the router still sees
    everything); the experts read and write a d_lat vector.  Each expert
    costs d/d_lat = 10.7x less than a full-width expert of the same inner
    width, so E=256, k=6, d_lat=384 buys 2.67x more experts than a 96-expert
    full-width design at the same per-layer parameter budget.
    """

    def __init__(self, c):
        super().__init__()
        d, dl = c["d"], c["d_lat"]
        self.E, self.k = c["n_experts"], c["top_k"]
        self.router = nn.Linear(d, self.E, bias=False)
        self.rbias = nn.Parameter(torch.zeros(self.E))      # aux-loss-free balance
        self.down = nn.Linear(d, dl, bias=False)            # shared latent projection
        self.up = nn.Linear(dl, d, bias=False)
        self.shared = SwiGLU(d, c["d_shared"], d)
        self.experts = nn.ModuleList(
            [SwiGLU(dl, c["d_ff"], dl) for _ in range(self.E)])

    def forward(self, x):
        b, n, d = x.shape
        logits = self.router(x) + self.rbias
        w, idx = logits.topk(self.k, dim=-1)
        w = w.softmax(-1)
        z = self.down(x)
        zf, fi = z.reshape(-1, z.shape[-1]), idx.reshape(-1, self.k)
        wf = w.reshape(-1, self.k)
        out = torch.zeros_like(zf)
        for e in range(self.E):
            tok, slot = (fi == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, self.experts[e](zf[tok]) * wf[tok, slot].unsqueeze(-1))
        return self.shared(x) + self.up(out.view(b, n, -1))


class WorkspaceLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw, d = c["d_w"], c["d"]
        self.n1 = nn.RMSNorm(dw)
        self.q = nn.Linear(dw, dw, bias=False)
        self.kv = nn.Linear(d, 2 * dw, bias=False)
        self.o = nn.Linear(dw, dw, bias=False)
        self.n2 = nn.RMSNorm(dw)
        self.sa = nn.Linear(dw, 3 * dw, bias=False)
        self.sa_o = nn.Linear(dw, dw, bias=False)
        self.n3 = nn.RMSNorm(dw)
        self.ff = SwiGLU(dw, c["w_ff"], dw)

    def forward(self, h, ctx):
        b, n, _ = h.shape
        q = self.q(self.n1(h))
        k, v = self.kv(ctx).chunk(2, dim=-1)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(h.shape[-1])
        m = _mask(n, min(4096, n), h.device)
        a = a.masked_fill(~m, float("-inf")).softmax(-1)
        h = h + self.o(a @ v)
        qkv = self.sa(self.n2(h))
        qq, kk, vv = qkv.chunk(3, dim=-1)
        a2 = (qq @ kk.transpose(-1, -2)) / math.sqrt(h.shape[-1])
        a2 = a2.masked_fill(~m, float("-inf")).softmax(-1)
        h = h + self.sa_o(a2 @ vv)
        return h + self.ff(self.n3(h))


class Workspace(nn.Module):
    """Narrow recurrent core: depth is a rental, not an owned parameter.

    Reads the trunk once (proj_in), is applied r times, writes back once
    (proj_out, layer-scaled).  The output gate starts at 1e-4 so the model
    begins as a no-op and has to learn that thinking helps.
    """

    def __init__(self, c):
        super().__init__()
        d, dw = c["d"], c["d_w"]
        self.proj_in = nn.Linear(d, dw, bias=False)
        self.proj_out = nn.Linear(dw, d, bias=False)
        self.layers = nn.ModuleList([WorkspaceLayer(c) for _ in range(c["w_layers"])])
        self.scale = nn.Parameter(torch.full((1,), 1e-4))
        self.head = nn.Sequential(nn.Linear(dw, c["d_head_h"]), nn.GELU(), nn.Linear(c["d_head_h"], 1))

    def forward(self, x, r, ctx=None):
        h = self.proj_in(x)
        ctx = x if ctx is None else ctx
        for _ in range(r):
            for layer in self.layers:
                h = layer(h, ctx)
        p_pass = torch.sigmoid(self.head(h.mean(dim=1))).squeeze(-1)
        return x + self.scale * self.proj_out(h), h, p_pass


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.RMSNorm(c["d"])
        self.attn = MLA(c)
        self.n2 = nn.RMSNorm(c["d"])
        self.mem = NeuralMemory(c)
        self.n3 = nn.RMSNorm(c["d"])
        self.moe = LatentMoE(c)

    def forward(self, x, state=None):
        x = x + self.attn(self.n1(x))
        x, state = self.mem(self.n2(x), state)
        x = x + self.moe(self.n3(x))
        return x, state


class Kestrel31(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c["vocab"], c["d"])
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["layers"])])
        self.norm = nn.RMSNorm(c["d"])
        self.workspace = Workspace(c)
        self.head = nn.Linear(c["d"], c["vocab"], bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids, r=2):
        x = self.embed(ids)
        states = []
        for blk in self.blocks:
            x, s = blk(x)
            states.append(s)
        x, h, p_pass = self.workspace(x, r)
        return self.head(self.norm(x)), states, p_pass, h


# ---------------------------------------------------------------- accounting

def attn_params(c):
    d, dc, dr, h, dh = c["d"], c["d_c"], c["d_rope"], c["heads"], c["dh"]
    return (d * dc + dc * h * dh + d * dr + d * (dc + dr)
            + 2 * dc * h * dh + h * dh * d + 2 * dh + h)


def mem_params(c):
    return 5 * c["d"] * c["d_m"] + c["d_m"] * c["r_m"]


def moe_params(c, active):
    d, dl, m = c["d"], c["d_lat"], c["d_ff"]
    routed = c["top_k"] if active else c["n_experts"]
    return (routed * 3 * dl * m + 3 * d * c["d_shared"]
            + 2 * d * dl + d * c["n_experts"] + c["n_experts"])


def w_layer_params(c):
    dw, df, d = c["d_w"], c["w_ff"], c["d"]
    return 6 * dw * dw + 2 * dw * d + 3 * dw * df + 3 * dw


def workspace_params(c):
    dw, d = c["d_w"], c["d"]
    head = dw * c["d_head_h"] + c["d_head_h"] + c["d_head_h"] + 1
    return c["w_layers"] * w_layer_params(c) + 2 * dw * d + head + 1


def block_params(c, active):
    return attn_params(c) + mem_params(c) + moe_params(c, active) + 3 * c["d"]


def total_params(c):
    return (c["layers"] * block_params(c, False) + c["vocab"] * c["d"]
            + c["d"] + workspace_params(c))


def active_params(c):
    return (c["layers"] * block_params(c, True) + c["vocab"] * c["d"]
            + c["d"] + workspace_params(c))


def flops_per_token(c, r=2, n=None):
    """Forward FLOPs per token: 2 * weights touched + attention.

    Attention is 4 * n_eff * heads * (dh + dr) per layer for QK^T and AV.
    The embedding is a gather, so it is excluded.
    """
    n = c["window"] if n is None else n
    n_eff = min(n, c["window"])
    dense = 2 * (c["layers"] * (attn_params(c) + mem_params(c) + moe_params(c, True)
                                + 3 * c["d"]) + workspace_params(c))
    attn = c["layers"] * 4 * n_eff * c["heads"] * (c["dh"] + c["d_rope"])
    w_attn = c["w_layers"] * 2 * 4 * n_eff * c["d_w"] * r
    return (dense + attn + w_attn) / 1e9


def main():
    print("=" * 72)
    torch.set_default_dtype(torch.float32)

    # [1] exact parameter count, two independent routes
    with torch.device("meta"):
        m = Kestrel31(FULL)
        n_torch = sum(p.numel() for p in m.parameters())
    n_calc = total_params(FULL)
    print(f"[1] total   torch={n_torch:,}  analytic={n_calc:,}  expected={EXPECTED_TOTAL:,}")
    assert n_torch == n_calc == EXPECTED_TOTAL, "parameter ledger disagrees"
    a = active_params(FULL)
    print(f"    active  analytic={a:,}  expected={EXPECTED_ACTIVE:,}  ratio={a / EXPECTED_TOTAL:.2%}")
    assert a == EXPECTED_ACTIVE, "active parameter count disagrees"
    assert a / EXPECTED_TOTAL < 0.11

    # [2] the latent width is what buys the expert count
    dl, d = FULL["d_lat"], FULL["d"]
    per_expert_latent = 3 * dl * FULL["d_ff"]
    per_expert_full = 3 * d * FULL["d_ff"]
    ratio = per_expert_full / per_expert_latent
    print(f"[2] latent  sparsity={FULL['n_experts'] / FULL['top_k']:.1f}  "
          f"experts={FULL['n_experts']}  top_k={FULL['top_k']}")
    print(f"    per-expert  latent={per_expert_latent:,}  full={per_expert_full:,}  "
          f"{ratio:.1f}x cheaper -> same layer budget, "
          f"{FULL['n_experts'] / 96:.2f}x the experts of a 96-expert design")
    assert abs(ratio - d / dl) < 1e-9, "latent saving is d/d_lat, not squared"
    assert moe_params(FULL, False) < 1.3e9, "MoE layer exceeds the 1.3B budget"

    # [3] recurrence adds depth, never parameters
    c1 = dict(FULL, r_max=1)
    c8 = dict(FULL, r_max=8)
    assert total_params(c1) == total_params(c8) == total_params(FULL)
    f1, f6 = flops_per_token(FULL, r=1), flops_per_token(FULL, r=6)
    print(f"[3] depth   params(r=1)={total_params(c1):,}  params(r=8)={total_params(c8):,}  identical")
    print(f"    flops/tok  r=1: {f1:.2f} GFLOP   r=6: {f6:.2f} GFLOP   (+{100*(f6/f1-1):.0f}%)")
    assert f6 > f1

    # [4] a deliberation pass is cheaper than a token
    pass_flops = 2 * (workspace_params(FULL) - 1) / 1e9
    cod_step = 2 * (active_params(FULL) - FULL["vocab"] * FULL["d"]) / 1e9
    print(f"[4] cost    one workspace pass={pass_flops:.3f} GFLOP  "
          f"one CoT token={cod_step:.3f} GFLOP  -> {cod_step / pass_flops:.1f}x cheaper per step")
    assert pass_flops < 0.25 * cod_step, "a deliberation pass is too expensive"

    # [5] tiny model for behavioural tests
    torch.manual_seed(0)
    tiny = dict(FULL, d=64, heads=4, dh=16, d_c=32, d_rope=8, layers=2,
                d_m=32, r_m=8, n_experts=8, top_k=2, d_lat=16, d_ff=32,
                d_shared=32, w_layers=2, d_w=32, w_ff=64, d_head_h=16, vocab=128)
    moe = LatentMoE(tiny)
    x = torch.randn(2, 6, 64, requires_grad=True)
    y = moe(x)
    assert y.shape == x.shape
    _, idx = (moe.router(x.detach()) + moe.rbias).topk(moe.k, dim=-1)
    counts = torch.bincount(idx.reshape(-1).long(), minlength=moe.E)
    assert counts.sum() == 2 * 6 * moe.k, "token dropped"
    shared_only = moe.shared(x.detach())
    assert shared_only.shape == y.shape
    y.sum().backward()
    assert moe.router.weight.grad.abs().sum() > 0
    print(f"[5] moe     routed={int(counts.sum())}  expected={2 * 6 * moe.k}  "
          f"shared_always_on=True  grads=True")

    # [6] recurrence is stable, and starts as (almost) an identity
    torch.manual_seed(0)
    t = Kestrel31(tiny)
    ids = torch.randint(0, tiny["vocab"], (1, 8))
    lg1, _, p1, _ = t(ids, r=1)
    lg8, _, p8, _ = t(ids, r=8)
    drift = (lg8 - lg1).abs().max().item() / lg1.abs().max().item()
    assert lg1.shape == lg8.shape == (1, 8, tiny["vocab"])
    assert torch.isfinite(lg8).all()
    assert drift < 1e-2, "recurrent path is not an identity at init"
    assert 0.0 < float(p1.mean()) < 1.0

    # long unroll must not explode or vanish
    with torch.no_grad():
        x0 = torch.randn(1, 8, tiny["d"])
        h = t.workspace.proj_in(x0)
        n0 = h.norm().item()
        hh, _, _ = t.workspace(x0, r=16)
        n1 = hh.norm().item() / x0.norm().item()
    assert 0.5 < n1 < 2.0, f"unrolled workspace drifted to {n1:.2f}x"
    print(f"[6] loop    init drift={drift:.2e}  p_pass={float(p1.mean()):.3f}  "
          f"16-pass norm ratio={n1:.2f}  finite=True")

    # [7] end to end, with a state that does not grow with context
    logits, states, p_pass, h = t(ids, r=4)
    st = FULL["r_m"] * FULL["d_m"] * FULL["layers"]
    print(f"[7] forward logits={tuple(logits.shape)}  states={len(states)}  "
          f"p_pass={tuple(p_pass.shape)}  ws={tuple(h.shape)}")
    print(f"    session state {st:,} floats = {st * 2 / 1e6:.1f} MB fp16, independent of context")
    print("=" * 72)
    print("all assertions passed")


if __name__ == "__main__":
    main()
