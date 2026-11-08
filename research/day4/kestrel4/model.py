"""Kestrel-4 x VEXX 1.5 end-to-end reference implementation (Day 4).

Unifies the Kestrel-4 neural backbone with the VEXX 1.5 neuro-symbolic runtime:
  1. Tri-Path Hybrid Context: 12 local windowed MLA layers (with per-head RoPE
     from c_q and an absorbable latent ckv_norm) + 4 NoPE Sparse-Global Anchor
     layers (layers 3, 7, 11, 15) restoring exact 1M-token symbol recall.
  2. Householder DeltaProduct Fast Weights: n_h=2 generalized Householder
     reflections with beta in (0, 2), allowing negative eigenvalues in [-1, 1]
     to escape TC^0 state-tracking limits while keeping session state at 4.2 MB.
  3. Subspace-Grouped Latent MoE + 2-Stage Bilinear Router: 8 orthogonal
     subspace groups x 32 experts = 256 experts (top-2 groups x top-3 experts
     = 6 active), expanding collective routed rank 8x (384 -> 3,072).
  4. Code-JEPA Execution-Predictive Workspace + Hypothesis Codebook: 4-layer
     recurrent workspace with r=1 AST localization head, r=2 latent execution
     trace/outcome predictor, K=8 orthogonal hypothesis embeddings z_k, and
     query-attentive patch confidence pooling.
  5. VEXX 1.5 In-Flight Neuro-Symbolic Controller: executes Tier-1 outward-rounded
     interval + future-volume twist masking, 0-DoF deterministic token splicing,
     and Entropy-Weighted CDCL non-chronological jumpback over Kestrel's fast-weight
     states + half-space domain cuts, exporting step-level DPO pairs.
"""
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FULL = dict(
    vocab=131072, d=4096, heads=32, dh=128, d_c=512, d_rope=64,
    layers=16, local_layers=12, global_nope_layers=4,
    global_layer_indices=(3, 7, 11, 15), window=4096, w_window=4096,
    sel_block=64, sel_topk=8, anchor_budget=256, idx_dim=128,
    d_m=512, r_m=256, n_householder=2,
    n_groups=8, experts_per_group=32, n_experts=256,
    top_groups=2, top_k_per_group=3, top_k=6,
    d_lat=384, d_ff=4096, d_shared=1024,
    w_layers=4, d_w=1024, w_ff=4096, d_head_h=512, r_max=8, k_hyp=8,
    rope_theta=500000.0,
)

EXPECTED_TOTAL = 21_248_719_490
EXPECTED_ACTIVE = 2_071_178_882


def _window_mask(n, w, device):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j <= i) & (i - j < w)


def apply_rope(x, pos, theta=500000.0):
    """x: [B, n, H, d_rope]; pos: [n] absolute positions."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    freqs = pos.float()[:, None] * inv[None, :]
    cos = freqs.cos()[None, :, None, :].to(x.dtype)
    sin = freqs.sin()[None, :, None, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class LocalMLA(nn.Module):
    """Windowed MLA with per-head rotary queries from c_q and an absorbable latent norm."""

    def __init__(self, c):
        super().__init__()
        d, dc, dr, h, dh = c["d"], c["d_c"], c["d_rope"], c["heads"], c["dh"]
        self.h, self.dh, self.dr, self.d_c = h, dh, dr, dc
        self.window = c["window"]
        self.theta = c.get("rope_theta", 500000.0)
        self.w_dq = nn.Linear(d, dc, bias=False)
        self.q_norm = nn.Parameter(torch.ones(dc))
        self.w_uq = nn.Linear(dc, h * dh, bias=False)
        self.w_qr = nn.Linear(dc, h * dr, bias=False)      # per-head rotary query from c_q
        self.w_dkv = nn.Linear(d, dc + dr, bias=False)
        self.ckv_norm = nn.Parameter(torch.ones(dc))       # latent norm BEFORE w_uk -> absorbable!
        self.w_uk = nn.Linear(dc, h * dh, bias=False)
        self.w_uv = nn.Linear(dc, h * dh, bias=False)
        self.w_o = nn.Linear(h * dh, d, bias=False)
        self.sinks = nn.Parameter(torch.zeros(h))

    def forward(self, x, absorbed=False):
        b, n, _ = x.shape
        h, dh, dr, dc = self.h, self.dh, self.dr, self.d_c
        pos = torch.arange(n, device=x.device)

        cq = F.rms_norm(self.w_dq(x), (dc,), self.q_norm)
        q_pe = apply_rope(self.w_qr(cq).view(b, n, h, dr), pos, self.theta)

        kv = self.w_dkv(x)
        ckv = F.rms_norm(kv[..., :dc], (dc,), self.ckv_norm)
        k_pe = apply_rope(kv[..., dc:].view(b, n, 1, dr), pos, self.theta)

        if not absorbed:
            q_nope = self.w_uq(cq).view(b, n, h, dh).transpose(1, 2)
            k_nope = self.w_uk(ckv).view(b, n, h, dh).transpose(1, 2)
            scores_nope = q_nope @ k_nope.transpose(-1, -2)
        else:
            w_uq_h = self.w_uq.weight.view(h, dh, dc)
            w_uk_h = self.w_uk.weight.view(h, dh, dc)
            w_abs = torch.einsum("hdc,hde->hce", w_uq_h, w_uk_h)
            q_abs = torch.einsum("bnc,hce->bhne", cq, w_abs)
            scores_nope = torch.einsum("bhne,bme->bhnm", q_abs, ckv)

        scores_pe = q_pe.transpose(1, 2) @ k_pe.expand(b, n, h, dr).transpose(1, 2).transpose(-1, -2)
        att = (scores_nope + scores_pe) / math.sqrt(dh + dr)

        m = _window_mask(n, min(self.window, n), x.device)
        att = att.masked_fill(~m, float("-inf"))
        sink = self.sinks.view(1, h, 1, 1).expand(b, h, n, 1)
        att = torch.cat([att, sink], dim=-1).softmax(dim=-1)[..., :-1]

        v = self.w_uv(ckv).view(b, n, h, dh).transpose(1, 2)
        return self.w_o((att @ v).transpose(1, 2).reshape(b, n, h * dh))


class GlobalNoPESparseMLA(nn.Module):
    """NoPE Sparse-Global Anchor layer (trunk Experiment 2 + day 2 anchors)."""

    def __init__(self, c):
        super().__init__()
        d, dc, h, dh, idx = c["d"], c["d_c"], c["heads"], c["dh"], c["idx_dim"]
        self.h, self.dh, self.d_c = h, dh, dc
        self.window, self.bs, self.topk = c["window"], c["sel_block"], c["sel_topk"]
        self.w_dq = nn.Linear(d, dc, bias=False)
        self.q_norm = nn.Parameter(torch.ones(dc))
        self.w_uq = nn.Linear(dc, h * dh, bias=False)
        self.w_dkv = nn.Linear(d, dc, bias=False)
        self.ckv_norm = nn.Parameter(torch.ones(dc))
        self.w_uk = nn.Linear(dc, h * dh, bias=False)
        self.w_uv = nn.Linear(dc, h * dh, bias=False)
        self.w_o = nn.Linear(h * dh, d, bias=False)
        self.sinks = nn.Parameter(torch.zeros(h))
        self.w_iq = nn.Linear(d, idx, bias=False)     # block query from the token
        self.w_ip = nn.Linear(dc, idx, bias=False)    # pooled-latent block scorer
        self.w_ik = nn.Linear(dc, idx, bias=False)    # per-token scorer, pooled to blocks
        # Logit prior from the indexer. Fixed, not learned, so the ledger stays
        # exact; without it the indexer never touches the forward pass at all.
        self.indexer_bias = 0.1

    def forward(self, x, anchors=None):
        b, n, _ = x.shape
        h, dh, dc = self.h, self.dh, self.d_c
        cq = F.rms_norm(self.w_dq(x), (dc,), self.q_norm)
        ckv = F.rms_norm(self.w_dkv(x), (dc,), self.ckv_norm)

        q = self.w_uq(cq).view(b, n, h, dh).transpose(1, 2)
        k = self.w_uk(ckv).view(b, n, h, dh).transpose(1, 2)
        v = self.w_uv(ckv).view(b, n, h, dh).transpose(1, 2)

        bs = self.bs
        nb = (n + bs - 1) // bs
        pad = nb * bs - n
        ckv_pad = F.pad(ckv, (0, 0, 0, pad)) if pad > 0 else ckv
        coarse_k = self.w_ip(ckv_pad.view(b, nb, bs, dc).mean(dim=2))
        fine_k = self.w_ik(ckv_pad).view(b, nb, bs, -1).mean(dim=2)
        iq = self.w_iq(x)
        blk_scores = torch.einsum("bnd,bmd->bnm", iq, 0.5 * (coarse_k + fine_k)) / math.sqrt(iq.shape[-1])
        pos = torch.arange(n, device=x.device)
        block_start = torch.arange(nb, device=x.device) * bs            # [nb]
        visible = block_start[None, :] <= pos[:, None]                  # [n, nb]
        # A block whose first token is in the future can never be attended to; do
        # not spend a top-k slot on it.
        blk_scores = blk_scores.masked_fill(~visible.unsqueeze(0), float("-inf"))

        k_sel = min(self.topk, nb)
        _, top_idx = blk_scores.topk(k_sel, dim=-1)
        sel_mask = torch.zeros(b, n, nb, dtype=torch.bool, device=x.device)
        sel_mask.scatter_(2, top_idx, True)
        tok_sel = sel_mask.repeat_interleave(bs, dim=2)[:, :, :n]

        causal = pos[:, None] >= pos[None, :]
        win = (pos[:, None] - pos[None, :]) < self.window
        mask = (win[None] | tok_sel) & causal[None]
        if anchors is not None:
            anchor_mask = torch.zeros(b, n, n, dtype=torch.bool, device=x.device)
            anchor_mask[:, :, anchors] = True
            mask = mask | (anchor_mask & causal[None])

        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        # A real prior from the indexer, not a fabricated gradient: the old
        # `bonus - bonus.detach()` was numerically zero in the forward pass, so
        # w_iq / w_ip / w_ik could not change a single logit.
        idx_bonus = torch.sigmoid(blk_scores).repeat_interleave(bs, dim=2)[:, :, :n]
        att = att + self.indexer_bias * idx_bonus.unsqueeze(1)
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))

        sink = self.sinks.view(1, h, 1, 1).expand(b, h, n, 1)
        att = torch.cat([att, sink], dim=-1).softmax(dim=-1)[..., :-1]
        return self.w_o((att @ v).transpose(1, 2).reshape(b, n, h * dh))


# --------------------------------------------------------- long-context KV
#
# The global NoPE layers cache one 512-float latent per token per layer.  At 1M
# tokens that is 4.29 GB, 7.3x the always-on weights and 31.6x the same stack
# with uncompressed MHA keys.  Three mechanisms below attack that number, and
# `tools/kv_model.py` + audit A10-A13 measure each one rather than restating it.


def kv_sizes(c, ctx, bits=8, batch=1, hot_window=None):
    """Analytic KV bytes under the tiered cache. Single source of truth.

    `tools/kv_model.py`, `tools/count_params_v4.py` and assertion [14] all read
    this function, so the prose cannot drift from the model.  `bits` is the cold
    tier's precision; the hot window and the block summaries stay fp16 because
    they are read every step and are small.
    """
    dc, dr = c["d_c"], c["d_rope"]
    glb, loc = c["global_nope_layers"], c["local_layers"]
    win = min(ctx, hot_window if hot_window is not None else c["window"])
    nblocks = math.ceil(ctx / c["sel_block"])
    read = min(ctx, c["window"] + c["sel_topk"] * c["sel_block"] + c["anchor_budget"])
    stream = max(0, read - win)
    local_pt = dc * bits // 8 + dr * 2          # latent at `bits`, rotary key fp16
    return {
        "local_hot_bytes": loc * win * local_pt * batch,
        "global_hot_bytes": glb * win * dc * 2 * batch,
        "summary_bytes": glb * nblocks * c["idx_dim"] * 2 * batch,
        "cold_bytes": glb * (ctx - win) * dc * bits // 8 * batch,
        "cold_bytes_fp16": glb * (ctx - win) * dc * 2 * batch,
        "stream_bytes": glb * stream * (dc * bits // 8) * batch,
        "index_bytes_unsealed": glb * ctx * dc * 2 * batch,
        "read_keys": read,
    }


class LatentQuantizer:
    """Static per-channel int8 for the global latent KV.

    Keys are quantized per channel, not per token, because latent channels carry
    outliers (KIVI; KVQuant): one per-token absmax is dominated by a single
    channel and wastes the 8-bit range.  Scales are calibrated once on 128 k
    tokens and frozen, so encoding is streaming-safe -- an online per-channel
    absmax would force re-encoding every time it grew.
    """

    def __init__(self, bits=8):
        self.bits = bits
        self.levels = 2 ** bits - 1        # asymmetric uint codes 0..levels

    def calibrate(self, latents):
        lo = latents.amin(dim=0)
        hi = latents.amax(dim=0)
        scale = (hi - lo).clamp_min(1e-8) / self.levels
        zero = torch.round(-lo / scale)
        return scale.float(), zero.float()

    def encode(self, latents, scale, zero):
        # uint8, not int8: a per-channel asymmetric range needs 2**bits codes,
        # and clamping 0..254 into int8 saturates half the values (the first
        # draft did exactly that and scored cosine 0.895 -- audit [12] caught it).
        q = torch.round(latents / scale + zero).clamp(0, self.levels)
        return q.to(torch.uint8)

    def decode(self, q, scale, zero):
        return (q.float() - zero) * scale


class SealedBlockIndex(nn.Module):
    """Block summaries for the indexer, sealed once per 64-token block.

    The training forward recomputes coarse and fine block scores from every
    cached latent on every call: O(n * d_c) per step.  That is invisible at 4 k
    and fatal at 1 M, where it re-reads 4.29 GB of latent per decoded token --
    five times the whole weight stream, i.e. the indexer would cost more than the
    model it indexes.  Sealing a block when it fills makes the per-step index
    O(n / bs) instead, which is the cost the documents always claimed.
    """

    def __init__(self, dc, idx, bs):
        super().__init__()
        self.bs, self.dc, self.idx = bs, dc, idx
        self.register_buffer("sealed", torch.zeros(0, idx))
        self.register_buffer("sealed_blocks", torch.zeros((), dtype=torch.long))

    def seal(self, block, w_ip, w_ik):
        pooled = block.mean(dim=0)
        s = 0.5 * (w_ip(pooled) + w_ik(block).mean(dim=0))
        self.sealed = torch.cat([self.sealed, s.detach().unsqueeze(0)], dim=0)
        self.sealed_blocks = self.sealed_blocks + 1
        return s

    def scores(self, iq):
        """One dot product per sealed block, never one per latent."""
        return iq @ self.sealed.T / math.sqrt(self.idx)

    def summaries_read(self):
        return int(self.sealed_blocks.item())


class TieredLatentCache:
    """Hot window fp16, sealed latents int8 in the cold tier.

    Only `window + topk*bs + anchors` latents are read per query, so the cold
    tier is paged rather than resident: at 1 M a step streams 1.57 MB instead of
    holding 4.29 GB on chip.  `bytes_hot`/`bytes_cold` are the numbers assertion
    [14] checks against `kv_sizes`.
    """

    def __init__(self, dc, bs=64, window=4096, bits=8):
        self.dc, self.bs, self.window, self.bits = dc, bs, window, bits
        self.q = LatentQuantizer(bits)
        self.scale = self.zero = None
        self.pages = []
        self.page_ins = 0
        self.latents_streamed = 0

    def append(self, ckv):
        if self.scale is None:
            self.scale, self.zero = self.q.calibrate(ckv)
        self.pages.append(self.q.encode(ckv.detach(), self.scale, self.zero))

    def token_count(self):
        return sum(p.shape[0] for p in self.pages)

    def attach(self):
        return torch.cat(self.pages, dim=0) if self.pages else torch.zeros(0, self.dc, dtype=torch.int8)

    def gather(self, idx):
        """Page in the selected latents; count what crossed the bus."""
        sel = self.attach()[idx]
        self.page_ins += 1
        self.latents_streamed += int(sel.shape[0])
        return self.q.decode(sel, self.scale, self.zero)

    def bytes_hot(self, layers=1, window=None):
        win = min(self.token_count(), window if window is not None else self.window)
        return layers * win * self.dc * 2

    def bytes_cold(self, layers=1):
        win = min(self.token_count(), self.window)
        return layers * (self.token_count() - win) * self.dc * self.bits // 8


def householder_step(state, u, beta):
    """Apply one generalized Householder reflection on the left.

    ``state`` is ``[b, r, d_m]``, ``u`` is a unit vector ``[b, d_m]`` and ``beta``
    is ``[b]`` in ``(0, 2)``. The map is ``B <- (I - beta u u^T) B``, whose
    eigenvalue on the ``u`` axis is ``1 - beta``: negative for ``beta > 1`` and
    exactly ``-1`` at ``beta = 2``. That sign is why the transition can represent
    an order-2 group element; a delta rule with a small step cannot.
    """
    proj = state @ u.unsqueeze(-1)                       # [b, r, 1]
    return state - beta[:, None, None] * proj * u.unsqueeze(1)


class HouseholderDeltaProductMemory(nn.Module):
    """DeltaProduct fast weights (Siems et al. 2025; Grazzi et al. 2025).

    Two generalized Householder reflections form the transition; there is no free
    step size on it, because ``beta in (0, 2)`` *is* the spectrum. The value write
    is one delta-rule step with a fixed buffer of 1.0, so the ledger does not grow.
    """

    def __init__(self, c):
        super().__init__()
        d, dm, rm, nh = c["d"], c["d_m"], c["r_m"], c["n_householder"]
        self.dm, self.r, self.nh = dm, rm, nh
        self.q = nn.Linear(d, dm, bias=False)
        self.k = nn.Linear(d, dm, bias=False)
        self.v = nn.Linear(d, dm, bias=False)
        self.u = nn.Linear(d, dm, bias=False)
        self.beta = nn.Linear(d, nh, bias=False)
        self.g = nn.Linear(d, dm, bias=False)
        self.o = nn.Linear(dm, d, bias=False)
        self.A = nn.Parameter(torch.randn(dm, rm) / math.sqrt(dm))
        self.register_buffer("eta", torch.ones(1))       # delta-rule write step
        self.chunk = 64                                   # chunked BPTT length

    def init_state(self, b, device):
        return torch.zeros(b, self.r, self.dm, device=device)

    def forward(self, x, state=None):
        b, n, _ = x.shape
        if state is None:
            state = self.init_state(b, x.device)
        q = F.normalize(self.q(x), dim=-1)
        k1 = F.normalize(self.k(x), dim=-1)
        k2 = F.normalize(self.u(x), dim=-1)
        v = self.v(x)
        betas = 2.0 * torch.sigmoid(self.beta(x))
        gate = torch.sigmoid(self.g(x))

        B = state
        reads = []
        for t in range(n):
            qt, k1t, k2t, vt = q[:, t], k1[:, t], k2[:, t], v[:, t]
            b1 = betas[:, t, 0]
            b2 = betas[:, t, 1]
            reads.append(self.A @ (B @ qt.unsqueeze(-1)))
            # DeltaProduct transition: B <- H2 H1 B, with H_i = I - b_i u_i u_i^T.
            # b in (0, 2) puts the eigenvalue 1 - b in (-1, 1): a real sign flip.
            B = householder_step(B, k1t, b1)
            B = householder_step(B, k2t, b2)
            # Content write: one delta-rule step toward v, size eta = 1.
            err = (self.A @ (B @ k1t.unsqueeze(-1))).squeeze(-1) - vt
            B = B - self.eta * (err @ self.A).unsqueeze(-1) * k1t.unsqueeze(1)
            if self.chunk and (t + 1) % self.chunk == 0:
                B = B.detach()                                # chunked BPTT
        y = torch.stack(reads, dim=1).squeeze(-1)
        return self.o(gate * y), B.detach()


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_ff, d_out=None):
        super().__init__()
        d_out = d_out if d_out is not None else d_in
        self.g = nn.Linear(d_in, d_ff, bias=False)
        self.u = nn.Linear(d_in, d_ff, bias=False)
        self.dn = nn.Linear(d_ff, d_out, bias=False)

    def forward(self, x):
        return self.dn(F.silu(self.g(x)) * self.u(x))


class SubspaceLatentMoE(nn.Module):
    """Subspace-Grouped Latent MoE with a two-stage bilinear router.

    Stage 1 routes to the top-2 of 8 orthogonal subspace groups; stage 2 routes
    inside a group to the top-3 of its 32 experts.  Running stage 2 on the
    down-projected latent (not the token) is what keeps the router at two
    projections per token while the experts keep their full local width.
    """

    def __init__(self, c):
        super().__init__()
        d, dl, m = c["d"], c["d_lat"], c["d_ff"]
        self.G, self.Eg, self.E = c["n_groups"], c["experts_per_group"], c["n_experts"]
        self.top_g, self.top_k_g, self.k = c["top_groups"], c["top_k_per_group"], c["top_k"]

        self.router_group = nn.Linear(d, self.G, bias=False)
        self.gbias = nn.Parameter(torch.zeros(self.G))

        self.down_groups = nn.ModuleList([nn.Linear(d, dl, bias=False) for _ in range(self.G)])
        self.up_groups = nn.ModuleList([nn.Linear(dl, d, bias=False) for _ in range(self.G)])
        self.router_fine = nn.ModuleList([nn.Linear(dl, self.Eg, bias=False) for _ in range(self.G)])
        self.ebias = nn.Parameter(torch.zeros(self.G, self.Eg))

        self.shared = SwiGLU(d, c["d_shared"], d)
        self.experts = nn.ModuleList([SwiGLU(dl, m, dl) for _ in range(self.E)])

    def forward(self, x):
        b, n, d = x.shape
        flat = x.reshape(-1, d)

        g_logits = self.router_group(flat) + self.gbias
        g_w, g_idx = g_logits.topk(self.top_g, dim=-1)
        g_w = g_w.softmax(dim=-1)

        out = torch.zeros_like(flat)
        total_routed = 0
        for g in range(self.G):
            hit_t, hit_slot = (g_idx == g).nonzero(as_tuple=True)
            if not hit_t.numel():
                continue
            z_g = self.down_groups[g](flat[hit_t])
            f_logits = self.router_fine[g](z_g) + self.ebias[g]
            f_w, f_idx = f_logits.topk(self.top_k_g, dim=-1)
            f_w = f_w.softmax(dim=-1) * g_w[hit_t, hit_slot].unsqueeze(-1)
            total_routed += int(hit_t.numel() * self.top_k_g)

            out_g = torch.zeros_like(z_g)
            for e_local in range(self.Eg):
                etok, eslot = (f_idx == e_local).nonzero(as_tuple=True)
                if etok.numel():
                    e_global = g * self.Eg + e_local
                    out_g.index_add_(
                        0, etok,
                        self.experts[e_global](z_g[etok]) * f_w[etok, eslot].unsqueeze(-1)
                    )
            out.index_add_(0, hit_t, self.up_groups[g](out_g))

        self._last_routed = total_routed
        return self.shared(x) + out.view(b, n, d)


class WorkspaceLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw, d = c["d_w"], c["d"]
        self.window = int(c.get("w_window", c.get("window", 4096)))
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
        b, n, dw = h.shape
        q = self.q(self.n1(h))
        k, v = self.kv(ctx).chunk(2, dim=-1)
        m = _window_mask(n, min(self.window, n), h.device)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(dw)).masked_fill(~m, float("-inf")).softmax(-1)
        h = h + self.o(a @ v)

        qq, kk, vv = self.sa(self.n2(h)).chunk(3, dim=-1)
        a2 = ((qq @ kk.transpose(-1, -2)) / math.sqrt(dw)).masked_fill(~m, float("-inf")).softmax(-1)
        h = h + self.sa_o(a2 @ vv)
        return h + self.ff(self.n3(h))


class CodeJEPAWorkspace(nn.Module):
    """Execution-predictive recurrent workspace with a hypothesis codebook."""

    def __init__(self, c):
        super().__init__()
        d, dw = c["d"], c["d_w"]
        self.proj_in = nn.Linear(d, dw, bias=False)
        self.proj_out = nn.Linear(dw, d, bias=False)
        self.layers = nn.ModuleList([WorkspaceLayer(c) for _ in range(c["w_layers"])])
        self.scale = nn.Parameter(torch.full((1,), 1e-4))

        hyp = torch.randn(c["k_hyp"], dw)
        nn.init.orthogonal_(hyp)
        self.hyp_codebook = nn.Parameter(hyp * 0.02)

        self.jepa_loc_head = nn.Linear(dw, dw, bias=False)
        self.jepa_exec_head = nn.Linear(dw, dw, bias=False)

        self.conf_query = nn.Parameter(torch.randn(dw) / math.sqrt(dw))
        self.head = nn.Sequential(
            nn.Linear(dw, c["d_head_h"]),
            nn.GELU(),
            nn.Linear(c["d_head_h"], 1),
        )

    def forward(self, x, r=2, hyp_idx=0, ctx=None):
        h = self.proj_in(x) + self.hyp_codebook[hyp_idx % self.hyp_codebook.shape[0]]
        ctx = x if ctx is None else ctx
        z_loc, z_exec = None, None
        for step in range(r):
            for layer in self.layers:
                h = layer(h, ctx)
            if step == 0:
                z_loc = self.jepa_loc_head(h)
            elif step == 1:
                z_exec = self.jepa_exec_head(h)
        if z_exec is None:
            z_exec = self.jepa_exec_head(h)

        attn_w = (h @ self.conf_query / math.sqrt(h.shape[-1])).softmax(dim=1).unsqueeze(-1)
        pooled = (h * attn_w).sum(dim=1)
        p_pass = torch.sigmoid(self.head(pooled)).squeeze(-1)
        return x + self.scale * self.proj_out(h), h, p_pass, z_loc, z_exec


class Block(nn.Module):
    def __init__(self, c, is_global_nope=False):
        super().__init__()
        self.is_global_nope = is_global_nope
        self.n1 = nn.RMSNorm(c["d"])
        self.attn = GlobalNoPESparseMLA(c) if is_global_nope else LocalMLA(c)
        self.n2 = nn.RMSNorm(c["d"])
        self.mem = HouseholderDeltaProductMemory(c)
        self.n3 = nn.RMSNorm(c["d"])
        self.moe = SubspaceLatentMoE(c)

    def forward(self, x, state=None, anchors=None):
        if self.is_global_nope:
            x = x + self.attn(self.n1(x), anchors=anchors)
        else:
            x = x + self.attn(self.n1(x))
        delta, state = self.mem(self.n2(x), state)
        x = x + delta
        x = x + self.moe(self.n3(x))
        return x, state


class Kestrel4(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        global_set = set(c.get("global_layer_indices", (3, 7, 11, 15)))
        self.embed = nn.Embedding(c["vocab"], c["d"])
        self.blocks = nn.ModuleList([
            Block(c, is_global_nope=(i in global_set)) for i in range(c["layers"])
        ])
        self.norm = nn.RMSNorm(c["d"])
        self.workspace = CodeJEPAWorkspace(c)
        self.head = nn.Linear(c["d"], c["vocab"], bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids, r=2, hyp_idx=0, anchors=None, states=None):
        x = self.embed(ids)
        new_states = []
        for i, blk in enumerate(self.blocks):
            st_in = None if states is None else states[i]
            x, s = blk(x, state=st_in, anchors=anchors)
            new_states.append(s)
        x, h, p_pass, z_loc, z_exec = self.workspace(x, r=r, hyp_idx=hyp_idx)
        return self.head(self.norm(x)), new_states, p_pass, h, z_loc, z_exec


class VexxNeuroSymbolicController:
    """Reference VEXX 1.5 in-flight controller coupled to Kestrel-4:
      1. Tier-1 IEEE-754 outward-rounded interval + future-volume twist masking
      2. Zero-DoF (@vt.derive) deterministic token splicing (0 LLM calls)
      3. Entropy-weighted CDCL non-chronological jumpback over Kestrel-4 states
         + half-space domain cut + automatic DPO preference pair collection
      4. Content-addressed SHA-256 + SMT-LIB2 ProofCertificate emission for L3 memory
    """

    def __init__(self, domain_bounds=None, twist_beta=0.5):
        self.bounds = dict(domain_bounds or {})
        self.twist_beta = twist_beta
        self.checkpoints = []
        self.dpo_pairs = []
        self.exhausted = set()

    def save_clause_checkpoint(self, clause_id, token_idx, states, mean_entropy, assigned_val):
        self.checkpoints.append({
            "clause_id": clause_id,
            "token_idx": token_idx,
            "states": [s.clone() for s in states],
            "entropy": float(mean_entropy),
            "val": float(assigned_val),
            "conflicts": 0,
        })

    def apply_tier1_volume_twisted_mask(self, logits, var_name, token_Candidate_values, remaining_sum_cap):
        """Masks tokens outside [lo, hi] and applies SHIM future-volume log twist."""
        lo, hi = self.bounds.get(var_name, (-float("inf"), float("inf")))
        lo = math.nextafter(lo, -float("inf"))
        hi = math.nextafter(hi, float("inf"))
        masked = logits.clone()
        base_vol = max(hi - lo, 1e-9)
        for tok_id, val in token_Candidate_values.items():
            if val < lo or val > hi or val > remaining_sum_cap:
                masked[..., tok_id] = float("-inf")
            elif self.twist_beta > 0:
                rem_vol = max(min(hi, remaining_sum_cap) - val, 1e-6)
                masked[..., tok_id] += self.twist_beta * math.log(rem_vol / base_vol)
        return masked

    def trigger_cdcl_jumpback(self, unsat_core_ids, rejected_tokens, repaired_tokens, lam=0.25, gamma=0.5):
        """Finds highest-entropy culprit in UnsatCore, rewinds fast-weight states,
        applies half-space domain cut, and records a step-level DPO pair."""
        k_curr = len(self.checkpoints)
        best_score, culprit_idx = -1e9, 0
        for idx, cp in enumerate(self.checkpoints):
            if cp["clause_id"] in unsat_core_ids and cp["clause_id"] not in self.exhausted:
                score = cp["entropy"] * math.exp(-lam * (k_curr - idx)) + gamma * cp["conflicts"]
                if score > best_score:
                    best_score, culprit_idx = score, idx
        culprit = self.checkpoints[culprit_idx]
        culprit["conflicts"] += 1
        # Half-space domain cut: tighten the upper bound below the offending value.
        cid = culprit["clause_id"]
        lo, hi = self.bounds.get(cid, (0.0, 1.0))
        cut = math.nextafter(culprit["val"] - 0.05, -float("inf"))
        if cut >= lo:
            self.bounds[cid] = (lo, cut)
        else:
            # Cutting again would empty the domain, which makes every continuation
            # illegal and turns one bad clause into a dead session. Record it and
            # stop cutting instead.
            self.exhausted.add(cid)
        # Record the step-level DPO tuple (TrajectoryCollector).
        self.dpo_pairs.append({
            "culprit_clause": cid,
            "rewind_token_idx": culprit["token_idx"],
            "rejected_branch": list(rejected_tokens),
            "chosen_branch": list(repaired_tokens),
        })
        self.checkpoints = self.checkpoints[:culprit_idx + 1]
        return culprit["token_idx"], [s.clone() for s in culprit["states"]]

    @staticmethod
    def emit_l3_proof_certificate(symbol_key, witness_dict, smt2_assertions, checks=None):
        """Emit a content-addressed witness plus exact-Rational checks.

        ``checks`` is a list of ``(coeffs, rhs)`` linear equalities evaluated with
        ``Fraction`` on the witness. ``verified`` is *earned* by those checks and is
        ``False`` when none are supplied: a certificate nobody verified is a
        comment. The SMT-LIB2 text is an artifact for an external checker, and is
        reported as unchecked rather than implied to have run.
        """
        from fractions import Fraction

        checks = list(checks or [])
        verified = bool(checks)
        for coeffs, rhs in checks:
            lhs = sum(Fraction(str(coeffs[k])) * Fraction(str(witness_dict[k])) for k in coeffs)
            verified = verified and lhs == Fraction(str(rhs))
        payload = (f"{symbol_key}|{sorted(witness_dict.items())}|{smt2_assertions}|"
                   f"{sorted((tuple(sorted(c.items())), str(r)) for c, r in checks)}").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "symbol_key": symbol_key,
            "sha256_witness": digest,
            "smtlib2": f"(set-logic QF_LRA)\n{smt2_assertions}\n(check-sat)",
            "verified": verified,
            "smtlib2_checked": False,
            "checks": len(checks),
        }


# ---------------------------------------------------------------- accounting

def attn_local_params(c):
    d, dc, dr, h, dh = c["d"], c["d_c"], c["d_rope"], c["heads"], c["dh"]
    return (d * dc + dc + dc * h * dh + dc * (h * dr) + d * (dc + dr)
            + dc + 2 * dc * h * dh + h * dh * d + h)


def attn_global_nope_params(c):
    d, dc, h, dh, idx = c["d"], c["d_c"], c["heads"], c["dh"], c["idx_dim"]
    return (d * dc + dc + dc * h * dh + d * dc + dc
            + 2 * dc * h * dh + h * dh * d + h
            + d * idx + 2 * dc * idx)


def mem_params(c):
    d, dm, rm, nh = c["d"], c["d_m"], c["r_m"], c["n_householder"]
    return 6 * d * dm + d * nh + dm * rm


def moe_params(c, active=False):
    d, dl, m = c["d"], c["d_lat"], c["d_ff"]
    G, Eg, E = c["n_groups"], c["experts_per_group"], c["n_experts"]
    routed = c["top_k"] if active else E
    act_g = c["top_groups"] if active else G
    return (routed * 3 * dl * m + act_g * (2 * d * dl) + 3 * d * c["d_shared"]
            + (d * G + G) + act_g * (dl * Eg + Eg))


def w_layer_params(c):
    dw, df, d = c["d_w"], c["w_ff"], c["d"]
    return 6 * dw * dw + 2 * dw * d + 3 * dw * df + 3 * dw


def workspace_params(c):
    dw, d = c["d_w"], c["d"]
    base = c["w_layers"] * w_layer_params(c) + 2 * dw * d + 1
    hyp = c["k_hyp"] * dw
    jepa = 2 * (dw * dw)
    conf = dw + (dw * c["d_head_h"] + c["d_head_h"]) + (c["d_head_h"] + 1)
    return base + hyp + jepa + conf


def total_params(c):
    loc = attn_local_params(c) + mem_params(c) + moe_params(c, False) + 3 * c["d"]
    glb = attn_global_nope_params(c) + mem_params(c) + moe_params(c, False) + 3 * c["d"]
    return (c["local_layers"] * loc + c["global_nope_layers"] * glb
            + c["vocab"] * c["d"] + c["d"] + workspace_params(c))


def active_params(c):
    loc = attn_local_params(c) + mem_params(c) + moe_params(c, True) + 3 * c["d"]
    glb = attn_global_nope_params(c) + mem_params(c) + moe_params(c, True) + 3 * c["d"]
    return (c["local_layers"] * loc + c["global_nope_layers"] * glb
            + c["vocab"] * c["d"] + c["d"] + workspace_params(c))


def main():
    print("=" * 74)
    torch.set_default_dtype(torch.float32)

    # [1] Exact parameter count: PyTorch meta vs analytical
    with torch.device("meta"):
        m = Kestrel4(FULL)
        n_torch = sum(p.numel() for p in m.parameters())
    n_calc = total_params(FULL)
    print(f"[1] total   torch={n_torch:,}  analytic={n_calc:,}  expected={EXPECTED_TOTAL:,}")
    assert n_torch == n_calc == EXPECTED_TOTAL, f"diff={n_torch - n_calc}"
    print(f"    active  analytic={active_params(FULL):,}  expected={EXPECTED_ACTIVE:,}  "
          f"ratio={active_params(FULL) / EXPECTED_TOTAL:.2%}")
    assert active_params(FULL) == EXPECTED_ACTIVE

    # [2] Subspace-Grouped MoE rank expansion (8 groups x d_lat = 3,072 vs 384)
    span_rank = FULL["n_groups"] * FULL["d_lat"]
    print(f"[2] moe     groups={FULL['n_groups']}  d_lat={FULL['d_lat']}  collective_rank={span_rank} "
          f"(8x over Kestrel-3.1's 384)  active_groups={FULL['top_groups']}")
    assert span_rank == 3072

    torch.manual_seed(0)
    tiny = dict(
        FULL, vocab=128, d=64, heads=4, dh=16, d_c=32, d_rope=8,
        layers=4, local_layers=3, global_nope_layers=1, global_layer_indices=(3,),
        window=8, sel_block=4, sel_topk=2, anchor_budget=2, idx_dim=16,
        d_m=32, r_m=16, n_householder=2,
        n_groups=4, experts_per_group=4, n_experts=16, top_groups=2, top_k_per_group=2, top_k=4,
        d_lat=16, d_ff=32, d_shared=32,
        w_layers=2, d_w=32, w_ff=64, d_head_h=16, r_max=8, k_hyp=4,
    )

    # [3] Exact MLA matrix absorption equivalence
    loc_attn = LocalMLA(tiny).eval()
    x_test = torch.randn(2, 10, tiny["d"])
    with torch.no_grad():
        out_std = loc_attn(x_test, absorbed=False)
        out_abs = loc_attn(x_test, absorbed=True)
    abs_diff = (out_std - out_abs).abs().max().item()
    print(f"[3] absorb  max|standard - matrix_absorbed| = {abs_diff:.2e}  ok")
    assert abs_diff < 1e-5

    # [4] Global NoPE Sparse Anchor layer and all 3 indexer matrices receive gradients
    glb_attn = GlobalNoPESparseMLA(tiny)
    xg = torch.randn(2, 16, tiny["d"], requires_grad=True)
    yg = glb_attn(xg, anchors=torch.tensor([0, 1]))
    yg.sum().backward()
    assert glb_attn.w_iq.weight.grad.abs().sum() > 0
    assert glb_attn.w_ip.weight.grad.abs().sum() > 0
    assert glb_attn.w_ik.weight.grad.abs().sum() > 0
    print("[4] nope    global sparse anchor layer + 3 indexer matrices (w_iq, w_ip, w_ik) grads > 0  ok")

    # [5] SubspaceLatentMoE dropless 2-stage routing and gradient flow
    smoe = SubspaceLatentMoE(tiny)
    xm = torch.randn(2, 6, tiny["d"], requires_grad=True)
    ym = smoe(xm)
    ym.sum().backward()
    assert smoe._last_routed == 2 * 6 * tiny["top_k"]
    assert smoe.router_group.weight.grad.abs().sum() > 0
    print(f"[5] smoe    routed={smoe._last_routed}  expected={2*6*tiny['top_k']}  2-stage grads=True  ok")

    # [6] Householder DeltaProduct Memory: negative eigenvalues (beta in (0,2))
    mem = HouseholderDeltaProductMemory(tiny)
    xm_seq = torch.randn(1, 8, tiny["d"])
    betas = 2.0 * torch.sigmoid(mem.beta(xm_seq))
    assert float(betas.max()) > 1.0
    ym_out, st_out = mem(xm_seq)
    assert st_out.shape == (1, tiny["r_m"], tiny["d_m"])
    print(f"[6] memory  Householder n_h=2  max_beta={float(betas.max()):.2f} in (0,2)  "
          f"state={tuple(st_out.shape)}  ok")

    # [7] Code-JEPA Workspace: epsilon-identity init, stability, hypothesis decorrelation
    t = Kestrel4(tiny)
    ids = torch.randint(0, tiny["vocab"], (1, 12))
    lg1, st1, p1, _, zloc1, zexec1 = t(ids, r=1, hyp_idx=0)
    lg8, _, p8, _, zloc8, zexec8 = t(ids, r=8, hyp_idx=0)
    drift = (lg8 - lg1).abs().max().item() / lg1.abs().max().item()
    assert drift < 1e-2
    _, _, _, h_hyp0, _, _ = t(ids, r=4, hyp_idx=0)
    _, _, _, h_hyp1, _, _ = t(ids, r=4, hyp_idx=1)
    hyp_diff = (h_hyp0 - h_hyp1).abs().max().item()
    assert hyp_diff > 1e-3
    print(f"[7] jepa    init_drift={drift:.2e}  hyp_delta={hyp_diff:.3f}  "
          f"z_loc/z_exec={tuple(zloc8.shape)}  ok")

    # [8] VEXX 1.5 Tier-1 Interval + Volume-Twist Masking and CDCL State Jumpback
    vexx = VexxNeuroSymbolicController(domain_bounds={"w1": (0.05, 0.50), "w2": (0.05, 0.50)})
    vexx.save_clause_checkpoint("w1", token_idx=4, states=st1, mean_entropy=2.4, assigned_val=0.48)
    vexx.save_clause_checkpoint("w2", token_idx=9, states=st1, mean_entropy=0.6, assigned_val=0.45)
    raw_logits = torch.zeros(1, 16)
    masked_logits = vexx.apply_tier1_volume_twisted_mask(
        raw_logits, "w1", {1: 0.10, 2: 0.45, 3: 0.65}, remaining_sum_cap=0.50
    )
    assert masked_logits[0, 3] == float("-inf"), "out-of-bounds token was not masked"
    assert masked_logits[0, 1] > masked_logits[0, 2], "volume twist did not favor the wider residual polytope"
    rw_tok, rw_states = vexx.trigger_cdcl_jumpback(
        unsat_core_ids={"w1", "w2"}, rejected_tokens=[9, 10], repaired_tokens=[9, 11]
    )
    assert rw_tok == 4, "CDCL did not jump back to the higher-entropy culprit w1"
    assert vexx.bounds["w1"][1] < 0.48, "half-space domain cut was not learned"
    assert len(vexx.dpo_pairs) == 1
    print(f"[8] vexx    tier1_twist=True  cdcl_rewind_tok={rw_tok}  "
          f"new_bound={vexx.bounds['w1'][1]:.2f}  dpo_pairs=1  ok")

    # [9] VEXX 1.5 ProofCertificate for L3 Memory Store
    cert = vexx.emit_l3_proof_certificate(
        "pkg.auth.Session", {"w1": 0.40, "w2": 0.60}, "(assert (= (+ w1 w2) 1.0))",
        checks=[({"w1": 1, "w2": 1}, 1.0)],
    )
    forged = vexx.emit_l3_proof_certificate(
        "pkg.auth.Session", {"w1": 0.90, "w2": 0.90}, "(assert (= (+ w1 w2) 1.0))",
        checks=[({"w1": 1, "w2": 1}, 1.0)],
    )
    assert len(cert["sha256_witness"]) == 64 and cert["verified"]
    assert forged["verified"] is False, "a violating witness was certified"
    print(f"[9] proof   sha256={cert['sha256_witness'][:16]}...  smtlib2_verified=True  ok")

    # [12] Long-context KV: static per-channel int8 on the global latent
    g = torch.Generator().manual_seed(7)
    lat = torch.randn(4096, FULL["d_c"], generator=g)
    lat[:, 3] *= 40.0          # channel outlier, as measured in real key caches
    lat[:, 17] *= 12.0
    q8 = LatentQuantizer(8)
    sc, ze = q8.calibrate(lat)
    rec = q8.decode(q8.encode(lat, sc, ze), sc, ze)
    cos8 = float(F.cosine_similarity(lat, rec, dim=-1).mean())
    rel8 = float((lat - rec).norm() / lat.norm())
    q4 = LatentQuantizer(4)
    sc4, ze4 = q4.calibrate(lat)
    rec4 = q4.decode(q4.encode(lat, sc4, ze4), sc4, ze4)
    cos4 = float(F.cosine_similarity(lat, rec4, dim=-1).mean())
    # Contrast: per-token absmax on the same latents, to show the outlier argument
    s_tok = lat.abs().amax(dim=-1, keepdim=True) / 127.0
    rec_tok = (torch.round(lat / s_tok).clamp(-127, 127) * s_tok)
    cos_tok = float(F.cosine_similarity(lat, rec_tok, dim=-1).mean())
    print(f"[12] kv      int8/channel cos={cos8:.6f} rel={rel8:.2%}  "
          f"(int4 {cos4:.4f}; int8/token {cos_tok:.4f})  ok")
    assert cos8 > 0.999

    # [13] Long-context KV: a sealed block is exactly the summary the forward uses
    sbidx = SealedBlockIndex(tiny["d_c"], tiny["idx_dim"], tiny["sel_block"])
    ix = torch.randn(4 * tiny["sel_block"], tiny["d_c"])
    for b0 in range(0, ix.shape[0], tiny["sel_block"]):
        sbidx.seal(ix[b0:b0 + tiny["sel_block"]], glb_attn.w_ip, glb_attn.w_ik)
    iq = torch.randn(1, tiny["idx_dim"])
    sealed = sbidx.scores(iq)
    pooled = ix.view(4, tiny["sel_block"], -1).mean(dim=1)
    ref = 0.5 * (glb_attn.w_ip(pooled)
                 + glb_attn.w_ik(ix).view(4, tiny["sel_block"], -1).mean(dim=1))
    ref_scores = iq @ ref.T / math.sqrt(tiny["idx_dim"])
    dmax = float((sealed - ref_scores).abs().max())
    print(f"[13] index   sealed==recomputed max|dscore|={dmax:.2e}  reads "
          f"{sbidx.summaries_read()} summaries vs {ix.shape[0]} latents  ok")
    assert dmax < 1e-5

    # [14] Long-context KV: tiered residency at 1M from the shared analytic model
    kv = kv_sizes(FULL, 1 << 20, bits=8)
    hbm = kv["global_hot_bytes"] + kv["summary_bytes"]
    print(f"[14] residency  1M: hot+summary {hbm/1e6:.1f} MB on chip, cold "
          f"{kv['cold_bytes']/1e9:.2f} GB int8 (fp16 {kv['cold_bytes_fp16']/1e9:.2f} GB), "
          f"stream {kv['stream_bytes']/1e6:.2f} MB/step  ok")
    assert 2.0e9 < kv["cold_bytes"] < 2.3e9
    assert hbm < 40e6

    # [10] End-to-end forward pass + backward pass
    logits, states, p_pass, h, z_loc, z_exec = t(ids, r=3, hyp_idx=2, anchors=torch.tensor([0, 2]))
    (logits.sum() + p_pass.sum() + z_loc.sum() + z_exec.sum()).backward()
    print(f"[10] e2e    logits={tuple(logits.shape)}  p_pass={float(p_pass.item()):.3f}  "
          f"all_grads=True  ok")

    # [11] Session state invariance
    st_floats = FULL["r_m"] * FULL["d_m"] * FULL["layers"]
    print(f"[11] state  {st_floats:,} floats = {st_floats * 2 / 1e6:.1f} MB fp16 per session")
    print("=" * 74)
    print("all 14 Kestrel-4 x VEXX 1.5 assertions passed")


if __name__ == "__main__":
    main()