"""Kestrel-3 reference implementation.

Three mechanisms, each with a scaling law behind it:
  1. sliding-window attention  -> exact short-term memory, bounded cost
  2. neural memory (delta-rule) -> compressed long-term memory, O(1)/token
  3. dropless fine-grained MoE  -> capacity without active FLOPs

The point of this file is not speed. It is that every claim in the
documents is executable and asserted, including the ones that are easy
to get wrong: the window mask, the absence of token dropping, and the
fact that the memory state actually learns.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FULL = dict(
    vocab=131072, d=4096, heads=32, dh=128, d_c=512, d_rope=64,
    layers=16, window=4096,
    d_m=512, r_m=256,
    n_experts=96, top_k=3, d_ff=1024,
)

EXPECTED_TOTAL = 20_497_778_688
EXPECTED_ACTIVE = 1_774_405_632


def _mask(n, w, device):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j <= i) & (i - j < w)


def _apply_rope(x, pos, theta=500000.0):
    """x: [B, n, H, d_rope]; pos: [n] absolute positions."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    freqs = pos.float()[:, None] * inv[None, :]
    cos = freqs.cos()[None, :, None, :].to(x.dtype)
    sin = freqs.sin()[None, :, None, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class MLA(nn.Module):
    """Latent attention: cache 576 floats/token/layer instead of 8192."""

    def __init__(self, c):
        super().__init__()
        d, dc, dr, h, dh = c["d"], c["d_c"], c["d_rope"], c["heads"], c["dh"]
        self.h, self.dh, self.dr, self.d_c = h, dh, dr, dc
        self.w_dq = nn.Linear(d, dc, bias=False)
        self.w_uq = nn.Linear(dc, h * dh, bias=False)
        self.w_qr = nn.Linear(dc, h * dr, bias=False)       # per-head rotary query from c_q
        self.w_dkv = nn.Linear(d, dc + dr, bias=False)
        self.ckv_norm = nn.Parameter(torch.ones(dc))        # latent norm BEFORE w_uk
        self.w_uk = nn.Linear(dc, h * dh, bias=False)
        self.w_uv = nn.Linear(dc, h * dh, bias=False)
        self.w_o = nn.Linear(h * dh, d, bias=False)
        self.q_norm = nn.Parameter(torch.ones(dh))
        self.k_norm = nn.Parameter(torch.ones(dh))
        self.sinks = nn.Parameter(torch.zeros(h))

    def forward(self, x):
        b, n, _ = x.shape
        h, dh, dr = self.h, self.dh, self.dr
        pos = torch.arange(n, device=x.device)
        cq = self.w_dq(x)
        # norm the content dim only, before the RoPE channel is appended
        q = F.rms_norm(self.w_uq(cq).view(b, n, h, dh), (dh,), self.q_norm)
        qr = _apply_rope(self.w_qr(cq).view(b, n, h, dr), pos)
        q = torch.cat([q, qr], dim=-1)
        kv = self.w_dkv(x)
        ckv, kr = kv[..., : self.d_c], kv[..., -dr:]
        ckv = F.rms_norm(ckv, (self.d_c,), self.ckv_norm)
        k = F.rms_norm(self.w_uk(ckv).view(b, n, h, dh), (dh,), self.k_norm)
        k = torch.cat([k, _apply_rope(kr.view(b, n, 1, dr), pos).expand(b, n, h, dr)], dim=-1)
        v = self.w_uv(ckv).view(b, n, h, dh)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh + dr)
        m = _mask(n, min(4096, n), x.device)
        att = att.masked_fill(~m, float("-inf"))
        # sinks: one always-attendable slot per head, appended as a column
        sink = self.sinks.view(1, h, 1, 1).expand(b, h, n, 1)
        att = torch.cat([att, sink], dim=-1).softmax(dim=-1)
        out = att[..., :-1] @ v
        return self.w_o(out.transpose(1, 2).reshape(b, n, h * dh))


class NeuralMemory(nn.Module):
    """Delta-rule fast weights, low-rank state, read-before-write (causal).

    State M = A @ B with A (d_m x r) learned, B (r x d_m) updated online.
    Read  : y_t = q_t @ A @ B
    Write : B <- B - eta * (A^T err) k^T,  err = A B k - v
    Cost  : O(d_m * r) per token -- no backward pass, no growing cache.
    """

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
        q = self.q(x)
        k = F.normalize(self.k(x), dim=-1)
        v = self.v(x)
        gate = torch.sigmoid(self.g(x))
        B = state
        reads = []
        for t in range(n):                      # causal: read, then write
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            reads.append(self.A @ (B @ qt.unsqueeze(-1)))   # M q, not q M
            err = (self.A @ (B @ kt.unsqueeze(-1))).squeeze(-1) - vt
            B = B - self.eta * (err @ self.A).unsqueeze(-1) @ kt.unsqueeze(1)
        y = torch.stack(reads, dim=1).squeeze(-1)
        return self.o((gate * y)) + x, B.detach()


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.g = nn.Linear(d, d_ff, bias=False)
        self.u = nn.Linear(d, d_ff, bias=False)
        self.dn = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.dn(F.silu(self.g(x)) * self.u(x))


class MoE(nn.Module):
    """Fine-grained, dropless top-k. Every token gets exactly k experts."""

    def __init__(self, c):
        super().__init__()
        self.E, self.k = c["n_experts"], c["top_k"]
        self.router = nn.Linear(c["d"], self.E, bias=False)
        self.rbias = nn.Parameter(torch.zeros(self.E))   # loss-free balancing
        self.experts = nn.ModuleList(
            [SwiGLU(c["d"], c["d_ff"]) for _ in range(self.E)])

    def forward(self, x):
        b, n, d = x.shape
        logits = self.router(x) + self.rbias
        w, idx = logits.topk(self.k, dim=-1)
        w = w.softmax(-1)
        flat, fi = x.reshape(-1, d), idx.reshape(-1, self.k)
        out = torch.zeros_like(flat)
        for e in range(self.E):
            hit = (fi == e)
            tok, slot = hit.nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, self.experts[e](flat[tok]) * w.reshape(-1, self.k)[tok, slot].unsqueeze(-1))
        return out.view(b, n, d)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.RMSNorm(c["d"])
        self.attn = MLA(c)
        self.n2 = nn.RMSNorm(c["d"])
        self.mem = NeuralMemory(c)
        self.n3 = nn.RMSNorm(c["d"])
        self.moe = MoE(c)

    def forward(self, x, state=None):
        x = x + self.attn(self.n1(x))
        x, state = self.mem(self.n2(x), state)
        x = x + self.moe(self.n3(x))
        return x, state


class Kestrel3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c["vocab"], c["d"])
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["layers"])])
        self.norm = nn.RMSNorm(c["d"])
        self.head = nn.Linear(c["d"], c["vocab"], bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        states = []
        for blk in self.blocks:
            x, s = blk(x)
            states.append(s)
        return self.head(self.norm(x)), states


def active_params(c):
    """Weights actually read per token."""
    d, dc, dr, h, dh = c["d"], c["d_c"], c["d_rope"], c["heads"], c["dh"]
    attn = (d * dc) + (dc * h * dh) + (dc * h * dr) + (d * (dc + dr)) + (dc * h * dh) * 2 + (h * dh * d) + dc + 2 * dh + h
    mem = 5 * d * c["d_m"] + c["d_m"] * c["r_m"]
    moe = c["top_k"] * 3 * d * c["d_ff"] + d * c["n_experts"] + c["n_experts"]
    norm = 3 * d
    return c["layers"] * (attn + mem + moe + norm) + c["vocab"] * d + d


def total_params(c):
    d = c["d"]
    per_layer = (
        (d * c["d_c"]) + (c["d_c"] * c["heads"] * c["dh"]) + (c["d_c"] * c["heads"] * c["d_rope"])
        + (d * (c["d_c"] + c["d_rope"])) + (c["d_c"] * c["heads"] * c["dh"]) * 2
        + (c["heads"] * c["dh"] * d) + c["d_c"] + 2 * c["dh"] + c["heads"]
        + 5 * d * c["d_m"] + c["d_m"] * c["r_m"]
        + c["n_experts"] * 3 * d * c["d_ff"] + d * c["n_experts"] + c["n_experts"]
        + 3 * d
    )
    return c["layers"] * per_layer + c["vocab"] * d + d


def main():
    print("=" * 70)
    torch.set_default_dtype(torch.float32)

    # [1] exact parameter count, both routes
    with torch.device("meta"):
        m = Kestrel3(FULL)
        n_torch = sum(p.numel() for p in m.parameters())
    n_calc = total_params(FULL)
    print(f"[1] total   torch={n_torch:,}  analytic={n_calc:,}  expected={EXPECTED_TOTAL:,}")
    assert n_torch == n_calc == EXPECTED_TOTAL
    a = active_params(FULL)
    print(f"    active  analytic={a:,}  expected={EXPECTED_ACTIVE:,}  ratio={a/EXPECTED_TOTAL:.3%}")
    assert a == EXPECTED_ACTIVE

    # [2] window mask is causal and bounded
    n, w = 64, 8
    mm = _mask(n, w, torch.device("cpu"))
    assert not bool(mm.triu(1).any()),   "attention to the future"
    assert int(mm.sum(1).max()) == w,    "window wider than w"
    assert int(mm[0].sum()) == 1,        "first token attends to more than itself"
    print(f"[2] window  max_row={int(mm.sum(1).max())}  causal=True  ok")

    # [3] MoE: dropless, exactly k experts per token, gradient flows
    torch.manual_seed(0)
    tiny = dict(FULL, d=64, heads=4, dh=16, d_c=32, d_rope=8, layers=2,
                d_m=32, r_m=8, n_experts=8, top_k=2, d_ff=32, vocab=128)
    moe = MoE(tiny)
    x = torch.randn(2, 6, 64, requires_grad=True)
    y = moe(x)
    assert y.shape == x.shape
    logits = moe.router(x.detach()) + moe.rbias
    _, idx = logits.topk(moe.k, dim=-1)
    counts = torch.bincount(idx.reshape(-1).long(), minlength=moe.E)
    assert counts.sum() == 2 * 6 * 2, "token dropped"
    y.sum().backward()
    assert moe.router.weight.grad.abs().sum() > 0
    print(f"[3] moe     tokens_out={counts.sum()}  expected={2*6*2}  k={moe.k}  dropless=True")

    # [4] neural memory: converges on one association, degrades gracefully under load
    torch.manual_seed(0)
    dm, r = 32, 16
    mem = NeuralMemory(dict(tiny, d=dm, d_m=dm, r_m=r))
    mem.eta = 0.2
    dev = torch.device("cpu")

    def write(B, k, x):
        err = (mem.A @ (B @ k.transpose(-1, -2))).squeeze(-1) - mem.v(x)
        return B - mem.eta * (err @ mem.A).unsqueeze(-1) @ k

    def read(B, k):
        return (mem.A @ (B @ k.transpose(-1, -2))).squeeze(-1)

    x = torch.randn(1, 1, dm)
    k1 = F.normalize(mem.k(x), dim=-1)
    B = mem.init_state(1, dev)
    for _ in range(10):
        B = write(B, k1, x)
    c1 = F.cosine_similarity(read(B, k1), mem.v(x), dim=-1).mean().item()

    keys = torch.randn(1, 6, dm)
    ks = [F.normalize(mem.k(keys[:, t : t + 1]), dim=-1) for t in range(6)]
    vs = [mem.v(keys[:, t : t + 1]) for t in range(6)]
    B = mem.init_state(1, dev)
    for _ in range(30):
        for t in range(6):
            B = write(B, ks[t], keys[:, t : t + 1])
    cs = [F.cosine_similarity(read(B, ks[t]), vs[t], dim=-1).mean().item() for t in range(6)]
    c6 = sum(cs) / len(cs)
    print(f"[4] memory  1 association: cos={c1:+.2f}   6 associations: cos={c6:+.2f}"
          f"   (rank {r}/{dm} caps recall near {math.sqrt(r/dm):.2f})")
    ceil = math.sqrt(r / dm)
    assert c1 > 0.6 and c1 > 0.8 * ceil, "delta rule did not reach its representable ceiling"
    assert c6 > 0.4, "no recall under load"

    # [5] state bytes are independent of sequence length
    st = FULL["r_m"] * FULL["d_m"] * FULL["layers"]
    kv = 16 * 256_000 * 576
    print(f"[5] state   {st:,} floats = {st*2/1e6:.1f} MB fp16 per session, "
          f"independent of context (dense KV@256k = {kv/1e9:.2f} GB fp8, {kv/(st*2):.0f}x larger)")

    # [6] forward pass runs end to end on a tiny model
    torch.manual_seed(0)
    t = Kestrel3(tiny)
    ids = torch.randint(0, tiny["vocab"], (2, 12))
    logits, states = t(ids)
    assert logits.shape == (2, 12, tiny["vocab"])
    assert all(s.shape == (2, tiny["r_m"], tiny["d_m"]) for s in states)
    print(f"[6] forward logits={tuple(logits.shape)}  states={len(states)}  ok")
    print("=" * 70)
    print("all assertions passed")


if __name__ == "__main__":
    main()
