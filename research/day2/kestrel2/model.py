"""
Kestrel-2 — reference implementation (day 2).

Purpose: be the SECOND, independent implementation of the same model, so that a
disagreement with tools/count_params_v2.py localises the error immediately.

Four properties are asserted, and each one is a design claim:

  1. parameter count matches the analytic model exactly (not "within 1%").
  2. sparse attention with nothing dropped is numerically IDENTICAL to dense
     causal attention. If this fails, the mask logic is wrong, not the model.
  3. the KV cache is exact: incremental decode == full prefill, to float noise.
  4. the recurrent block's cache shape does not depend on the step count, which
     is what lets one shared repo prefix be served at any depth setting.

The compression branch of the sparse attention (NSA-style block summaries) is a
latency mechanism; it is modelled in the cost script and deliberately omitted
from this reference kernel, which implements the branches that decide the mask:
window + selected blocks.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 1024
    d: int = 128
    heads: int = 4
    dh: int = 32
    d_c: int = 64
    d_qc: int = 64
    d_rope: int = 16
    d_ff: int = 256
    prelude: int = 1
    recur: int = 2
    coda: int = 1
    window: int = 8
    cmp_stride: int = 4
    sel_block: int = 4
    sel_topk: int = 2
    anchor_budget: int = 2
    idx_dim: int = 16
    mtp: int = 2
    sinks: int = 2
    rope_theta: float = 500000.0
    max_r: int = 8

    @property
    def n_unique(self) -> int:
        return self.prelude + self.recur + self.coda


FULL = Cfg(
    vocab=131072, d=4096, heads=32, dh=128, d_c=512, d_qc=512, d_rope=64,
    d_ff=12288, prelude=2, recur=16, coda=2, window=2048, cmp_stride=64,
    sel_block=64, sel_topk=12, anchor_budget=256, idx_dim=128, mtp=2,
    sinks=4, max_r=8,
)

EXPECTED_INFERENCE = 4_125_828_176
EXPECTED_TRAINING = 4_159_390_800


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope(x, pos, theta):
    """x: [B, n, H, d_rope]; pos: [n] absolute positions."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    f = pos.float()[:, None] * inv[None]                      # [n, d/2]
    cos = f.cos()[None, :, None, :]                            # [1, n, 1, d/2]
    sin = f.sin()[None, :, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class MLA(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        d, h, dh = c.d, c.heads, c.dh
        self.w_dq = nn.Linear(d, c.d_qc, bias=False)
        self.w_uq = nn.Linear(c.d_qc, h * dh, bias=False)
        self.w_qr = nn.Linear(d, c.d_rope, bias=False)
        self.w_dkv = nn.Linear(d, c.d_c + c.d_rope, bias=False)   # FUSED
        self.w_uk = nn.Linear(c.d_c, h * dh, bias=False)
        self.w_uv = nn.Linear(c.d_c, h * dh, bias=False)
        self.w_o = nn.Linear(h * dh, d, bias=False)
        self.q_norm = RMSNorm(dh)
        self.k_norm = RMSNorm(dh)
        self.sink = nn.Parameter(torch.zeros(c.sinks))
        self.w_iq = nn.Linear(d, c.idx_dim, bias=False)
        self.w_ik = nn.Linear(c.d_c, c.idx_dim, bias=False)
        self.w_ip = nn.Linear(c.d_c, c.idx_dim, bias=False)

    def init_kv(self, x, offset: int = 0):
        """Project the cache ONCE; the recurrent block reuses it at every step."""
        c_kv, k_pe = self.w_dkv(x).split([self.c.d_c, self.c.d_rope], dim=-1)
        pos = torch.arange(x.shape[1], device=x.device) + offset
        return c_kv, rope(k_pe[:, :, None], pos, self.c.rope_theta)[:, :, 0]

    def build_mask(self, x, c_kv, anchors=None, dense=False):
        B, n, _ = x.shape
        m = c_kv.shape[1]
        pos = torch.arange(n, device=x.device)
        causal = pos[:, None] >= pos[None, :]
        if m != n:            # decode: every cached key is visible
            return None
        if dense:             # exact causal reference path
            return causal[None, None]
        bs = self.c.sel_block
        nb = (n + bs - 1) // bs
        ki = self.w_ik(c_kv)
        if nb * bs > n:
            ki = F.pad(ki, (0, 0, 0, nb * bs - n))
        pooled = ki.view(B, nb, bs, -1).mean(2)
        scores = torch.einsum("bnd,bmd->bnm", self.w_iq(x), pooled)
        top = scores.topk(min(self.c.sel_topk, nb), dim=-1).indices
        sel = torch.zeros(B, n, nb, dtype=torch.bool, device=x.device)
        sel.scatter_(2, top, True)
        sel = sel.repeat_interleave(bs, dim=2)[:, :, :n]
        win = (pos[:, None] - pos[None, :]) < self.c.window
        mm = win[None] | sel
        if anchors is not None:
            mm = mm.clone()
            mm[:, :, anchors] = True
        return (mm & causal[None]).unsqueeze(1)

    def forward(self, x, kv=None, anchors=None, dense=False):
        c = self.c
        B, n, _ = x.shape

        # the cache owns appending: positions are derived, never passed in
        if kv is None:
            c_kv, k_pe = self.init_kv(x, offset=0)
        else:
            c_prev, k_prev = kv
            c_new, k_new = self.init_kv(x, offset=c_prev.shape[1])
            c_kv = torch.cat([c_prev, c_new], dim=1)
            k_pe = torch.cat([k_prev, k_new], dim=1)
        m = c_kv.shape[1]
        pos = torch.arange(n, device=x.device) + (m - n)

        q = self.q_norm(self.w_uq(self.w_dq(x)).view(B, n, c.heads, c.dh))
        q_pe = rope(self.w_qr(x)[:, :, None], pos, c.rope_theta)
        q = torch.cat([q, q_pe.expand(B, n, c.heads, c.d_rope)], dim=-1)

        k_nope = self.k_norm(self.w_uk(c_kv).view(B, m, c.heads, c.dh))
        v = self.w_uv(c_kv).view(B, m, c.heads, c.dh)
        k = torch.cat([k_nope, k_pe[:, :, None].expand(B, m, c.heads, c.d_rope)], -1)

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * (c.dh + c.d_rope) ** -0.5
        mask = self.build_mask(x, c_kv, anchors=anchors, dense=dense)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        sink = self.sink.view(1, 1, 1, -1).expand(B, c.heads, n, c.sinks)
        att = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :m]
        out = torch.einsum("bhqk,bkhd->bqhd", att, v)
        return self.w_o(out.reshape(B, n, c.heads * c.dh))


class SwiGLU(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.gate = nn.Linear(c.d, c.d_ff, bias=False)
        self.up = nn.Linear(c.d, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.n1 = RMSNorm(c.d)
        self.attn = MLA(c)
        self.n2 = RMSNorm(c.d)
        self.ffn = SwiGLU(c)

    def forward(self, x, **kw):
        x = x + self.attn(self.n1(x), **kw)
        return x + self.ffn(self.n2(x))


class MTPHead(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.n = RMSNorm(c.d)
        self.proj = nn.Linear(c.d, c.d, bias=False)

    def forward(self, h, unembed):
        return F.linear(self.proj(self.n(h)), unembed)


class Kestrel2(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.d)
        self.prelude = nn.ModuleList(Block(c) for _ in range(c.prelude))
        self.recur = nn.ModuleList(Block(c) for _ in range(c.recur))
        self.coda = nn.ModuleList(Block(c) for _ in range(c.coda))
        self.loop_emb = nn.Parameter(torch.zeros(c.max_r, c.d))
        self.norm_f = RMSNorm(c.d)
        self.mtp = nn.ModuleList(MTPHead(c) for _ in range(c.mtp))

    def forward(self, ids, r=1, anchors=None, dense=False, want_kv=False):
        h = self.embed(ids)
        for b in self.prelude:
            h = b(h, dense=dense)

        kv = [b.attn.init_kv(h) for b in self.recur]

        for i in range(r):
            h = h + self.loop_emb[i]
            for b, k in zip(self.recur, kv):
                h = b(h, kv=k, anchors=anchors, dense=dense)

        for b in self.coda:
            h = b(h, dense=dense)
        h = self.norm_f(h)
        logits = F.linear(h, self.embed.weight)
        return (logits, kv) if want_kv else logits


def count(model, skip_mtp=False):
    return sum(p.numel() for n, p in model.named_parameters()
               if not (skip_mtp and n.startswith("mtp.")))


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("KESTREL-2 reference implementation")
    print("=" * 70)

    with torch.device("meta"):
        full = Kestrel2(FULL)
    n_inf = count(full, skip_mtp=True)
    n_train = count(full)
    print(f"\n[1] parameters (inference) : {n_inf:,}  ({n_inf/1e9:.4f} B)")
    print(f"    analytic expected      : {EXPECTED_INFERENCE:,}")
    assert n_inf == EXPECTED_INFERENCE, f"mismatch {n_inf - EXPECTED_INFERENCE:+,}"
    print(f"    + MTP (training only)  : {n_train - n_inf:,}  -> {n_train:,}")
    assert n_train == EXPECTED_TRAINING, f"mismatch {n_train - EXPECTED_TRAINING:+,}"

    c = Cfg()
    tiny = Kestrel2(c)
    ids = torch.randint(0, c.vocab, (1, 12))

    # 2. dropping nothing must change nothing
    with torch.no_grad():
        lg_dense = tiny(ids, r=1, dense=True)
        wide = Kestrel2(Cfg(sel_topk=10**6, window=10**6))
        wide.load_state_dict(tiny.state_dict())
        lg_all = wide(ids, r=1, anchors=torch.arange(12))
    d = (lg_dense - lg_all).abs().max().item()
    print(f"\n[2] sparse(k=all,w=all) vs dense  : max|diff| = {d:.3e}")
    assert d < 1e-4, "mask logic is wrong"

    # 3. cache exactness
    layer = tiny.recur[0].attn
    x = torch.randn(1, 12, c.d)
    with torch.no_grad():
        full_out = layer(x, dense=True)                 # causal, 12 positions
        kv = layer.init_kv(x[:, :11])                   # prefill
        step = layer(x[:, 11:12], kv=kv, dense=True)    # decode appends itself
    d = (full_out[:, 11:12] - step).abs().max().item()
    print(f"[3] incremental decode vs prefill  : max|diff| = {d:.3e}")
    assert d < 1e-4, "KV cache is not exact"

    # 4. cache shape independent of step count
    with torch.no_grad():
        _, kv1 = tiny(ids, r=1, want_kv=True)
        _, kv4 = tiny(ids, r=4, want_kv=True)
    b1 = sum(t.numel() * t.element_size() for p in kv1 for t in p)
    b4 = sum(t.numel() * t.element_size() for p in kv4 for t in p)
    print(f"[4] cache bytes r=1 {b1:,} / r=4 {b4:,}   identical={b1 == b4}")
    assert b1 == b4

    with torch.no_grad():
        a = tiny(ids, r=1)
        b = tiny(ids, r=4)
    dl = (a - b).abs().max().item()
    print(f"[5] r=1 vs r=4 differ             : max|diff| = {dl:.3e}"
          f"   finite={bool(torch.isfinite(b).all())}")
    assert dl > 1e-6 and torch.isfinite(b).all()

    per_tok = FULL.n_unique * (FULL.d_c + FULL.d_rope)
    print(f"\n    cache/token (FP8)             : {per_tok:,} B ({per_tok/1024:.0f} KiB)")
    for ctx in (65536, 262144, 1048576):
        print(f"      ctx {ctx:>9,}              : {per_tok*ctx/2**30:6.2f} GiB")
    assert per_tok == 11520

    print("\nall assertions passed")


if __name__ == "__main__":
    main()
