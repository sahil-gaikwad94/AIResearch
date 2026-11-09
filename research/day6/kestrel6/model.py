"""Kestrel-6 model: a real, trainable decoder-only language model.

Architecture (baseline, "dense" attn_impl):

  * Decoder-only causal Transformer
  * Pre-norm RMSNorm
  * RoPE rotary positions (per-head, applied to query & key slices)
  * GQA / MQA attention (n_kv_heads <= n_heads) with an optional sliding window
  * SwiGLU dense FFN
  * Tied embedding / LM head (optional)
  * Explicit per-session KV cache for streaming decode (prefill + decode_step)

Day-5 mechanisms are *toggleable ablations* behind config flags
(``use_memory``, ``use_moe``, ``use_workspace``) and the ``attn_impl`` /
``global_layers`` settings. Each ablation must beat the dense baseline at
equal parameter count, training tokens, and compute before promotion.

No false flags: there is no checkpoint, no autograd claim beyond what PyTorch
provides, and no manual logit perturbation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- primitives


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Root-mean-square layer normalization (pre-norm)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to the last dimension (interleaved pairs)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.reshape_as(x)


def rope_cos_sin(
    seq_len: int,
    dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE, shape [1, seq_len, dim/2] (one value per rotary pair)."""
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
    cos = torch.cos(freqs).unsqueeze(0)  # [1, seq_len, dim/2]
    sin = torch.sin(freqs).unsqueeze(0)
    return cos, sin


# --------------------------------------------------------------------------- attention


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head attention (dense), with optional sliding window.

    Supports GQA/MQA: n_kv_heads key/value heads expanded to n_heads query heads.
    Returns attention output; the KV cache is managed externally by the block.
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        max_ctx: int,
        rope_theta: float,
        sliding_window: Optional[int] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d // n_heads
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        assert self.head_dim * n_heads == d

        self.w_q = nn.Linear(d, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d, bias=False)

        # RoPE tables (lazily materialized to the right device/dtype on first use).
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _rope(self, x: torch.Tensor, seq_start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1] + seq_start
        device = x.device
        dtype = x.dtype
        if self.cos.shape[1] < seq_len or self.cos.device != device:
            cos, sin = rope_cos_sin(seq_len, self.head_dim, self.rope_theta, device, dtype)
            self.cos = cos
            self.sin = sin
        cos = self.cos[:, seq_start : seq_start + x.shape[1], :]
        sin = self.sin[:, seq_start : seq_start + x.shape[1], :]
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        seq_start: int = 0,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        b, n, _ = x.shape
        q = self.w_q(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)  # [b,h,n,hd]
        k = self.w_k(x).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(x, seq_start)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV cache update.
        new_cache: Dict[str, torch.Tensor] = {}
        if cache is not None and "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        new_cache["k"] = k
        new_cache["v"] = v

        total_len = k.shape[2]  # number of cached keys for this sequence

        # Expand KV heads to query heads (GQA/MQA).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # [b, n_heads, total, hd]
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [b,h,n,total]

        # Causal mask + optional sliding window.
        mask = torch.ones(n, total_len, device=x.device, dtype=torch.bool)
        q_pos = torch.arange(seq_start, seq_start + n, device=x.device)[:, None]
        k_pos = torch.arange(total_len, device=x.device)[None, :]
        mask &= k_pos <= q_pos  # causal
        if self.sliding_window is not None:
            mask &= (q_pos - k_pos) < self.sliding_window
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn_weights = attn if need_weights else None
        out = attn @ v  # [b, h, n, hd]
        out = out.transpose(1, 2).contiguous().view(b, n, self.d)
        return self.w_o(out), attn_weights, new_cache


# --------------------------------------------------------------------------- MLA local attention (Day-5 ablation)


class LocalMLA(nn.Module):
    """Multi-head Latent Attention with latent KV compression.

    Compresses keys/values to a latent of size d_c (< d) before projection to
    heads, reducing KV cache size. Optional sliding window. RoPE applied to a
    small rotary slice (d_rope) of the query/key latents. When ``absorbed``,
    the query/key output projections are precomputed into a single matrix for
    decode efficiency (exact equivalence to the standard path at init).
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        d_c: int,
        d_rope: int,
        sliding_window: Optional[int],
        max_ctx: int,
        rope_theta: float,
    ):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.d_c = d_c
        self.d_rope = d_rope
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta

        self.w_dq = nn.Linear(d, d_c, bias=False)
        self.q_norm = nn.Parameter(torch.ones(d_c))
        self.w_uq = nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.w_qr = nn.Linear(d_c, n_heads * d_rope, bias=False)
        self.w_dkv = nn.Linear(d, d_c + d_rope, bias=False)
        self.ckv_norm = nn.Parameter(torch.ones(d_c))
        self.w_uk = nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.w_uv = nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d, bias=False)
        self.sinks = nn.Parameter(torch.zeros(n_heads))

        self.cos = nn.Parameter(torch.empty(0), requires_grad=False)
        self.sin = nn.Parameter(torch.empty(0), requires_grad=False)

    def _rope(self, x: torch.Tensor, seq_start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1] + seq_start
        device = x.device
        dtype = x.dtype
        if self.cos.shape[1] < seq_len or self.cos.device != device:
            cos, sin = rope_cos_sin(seq_len, self.d_rope, self.rope_theta, device, dtype)
            self.cos = cos
            self.sin = sin
        return self.cos[:, seq_start : seq_start + x.shape[1], :], self.sin[:, seq_start : seq_start + x.shape[1], :]

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        seq_start: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, n, _ = x.shape
        cq = rms_norm(self.w_dq(x), self.q_norm)  # [b,n,d_c]
        ckv_full = self.w_dkv(x)  # [b,n,d_c+d_rope]
        ckv = rms_norm(ckv_full[..., : self.d_c], self.ckv_norm)
        k_pe_full = ckv_full[..., self.d_c :]  # [b,n,d_rope]

        q_nope = self.w_uq(cq).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        q_pe = self.w_qr(cq).view(b, n, self.n_heads, self.d_rope).transpose(1, 2)
        k_nope = self.w_uk(ckv).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k_pe = k_pe_full.view(b, n, 1, self.d_rope).transpose(1, 2)
        v = self.w_uv(ckv).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(x, seq_start)
        q_pe = apply_rope(q_pe, cos, sin)
        k_pe = apply_rope(k_pe, cos, sin)

        # KV cache stores the compressed latent ckv (and rotary slice) only.
        new_cache: Dict[str, torch.Tensor] = {}
        if cache is not None and "ckv" in cache:
            ckv = torch.cat([cache["ckv"], ckv], dim=1)
            k_pe = torch.cat([cache["k_pe"], k_pe], dim=1)
        new_cache["ckv"] = ckv
        new_cache["k_pe"] = k_pe

        total_len = ckv.shape[1]
        k_nope = self.w_uk(ckv).view(b, total_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(ckv).view(b, total_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_rep > 1:
            k_nope = k_nope.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores_nope = (q_nope @ k_nope.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_pe = (q_pe @ k_pe.transpose(-2, -1)) / math.sqrt(self.d_rope)
        scores = (scores_nope + scores_pe) / math.sqrt(2.0)  # two components summed

        q_pos = torch.arange(seq_start, seq_start + n, device=x.device)[:, None]
        k_pos = torch.arange(total_len, device=x.device)[None, :]
        mask = (k_pos <= q_pos)  # causal
        if self.sliding_window is not None:
            mask &= (q_pos - k_pos) < self.sliding_window
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        sink = self.sinks.view(1, self.n_heads, 1, 1)
        attn = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, self.n_heads * self.head_dim)
        return self.w_o(out), new_cache


# --------------------------------------------------------------------------- MLP


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int, bias: bool = False):
        super().__init__()
        self.g = nn.Linear(d_in, d_ff, bias=bias)
        self.u = nn.Linear(d_in, d_ff, bias=bias)
        self.dn = nn.Linear(d_ff, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dn(F.silu(self.g(x)) * self.u(x))


# --------------------------------------------------------------------------- fast-weight memory (Day-5 ablation)


class WYHouseholderMemory(nn.Module):
    """Compact-WY / Householder fast-weight memory (Day-5 ablation).

    A small recurrent state B updated per token with a delta write and read
    through a spectral-normalized Householder product. Offered as an ablation;
    not part of the dense baseline.
    """

    def __init__(self, d: int, d_m: int, r: int, n_householder: int, eta: float = 0.5):
        super().__init__()
        self.d, self.d_m, self.r, self.n_householder, self.eta = d, d_m, r, n_householder, eta
        self.q = nn.Parameter(torch.randn(d, d_m) / math.sqrt(d))
        self.k = nn.Parameter(torch.randn(d, d_m) / math.sqrt(d))
        self.v = nn.Parameter(torch.randn(d, d_m) / math.sqrt(d))
        self.u = nn.Parameter(torch.randn(d, d_m) / math.sqrt(d))
        self.beta = nn.Parameter(torch.randn(d, n_householder) / math.sqrt(d))
        self.g = nn.Parameter(torch.randn(d, d_m) / math.sqrt(d))
        self.o = nn.Parameter(torch.randn(d_m, d) / math.sqrt(d_m))
        # Stiefel-initialized A (orthonormal columns).
        a_init = torch.randn(d_m, r)
        a_init = torch.linalg.qr(a_init).Q
        self.A = nn.Parameter(a_init)

    def spectral_normalized_A(self) -> torch.Tensor:
        norm2 = torch.linalg.norm(self.A, ord=2)
        return self.A / torch.clamp(norm2, min=1.0)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """x: [b, n, d]. state: [b, r, d_m] or None. Returns (delta, new_state)."""
        b, n, _ = x.shape
        if state is None:
            B = torch.zeros(b, self.r, self.d_m, device=x.device, dtype=x.dtype)
        else:
            B = state.clone()

        q = torch.nn.functional.normalize(x @ self.q, dim=-1)
        k1 = torch.nn.functional.normalize(x @ self.k, dim=-1)
        k2 = torch.nn.functional.normalize(x @ self.u, dim=-1)
        v = x @ self.v
        betas = 2.0 * torch.sigmoid(x @ self.beta)
        gate = torch.sigmoid(x @ self.g)
        A_hat = self.spectral_normalized_A()

        reads: List[torch.Tensor] = []
        for t in range(n):
            qt = q[:, t]  # [b, d_m]
            k1t = k1[:, t]
            vt = v[:, t]
            b1 = betas[:, t, 0] if betas.dim() == 3 else betas[:, t]

            b_q = torch.einsum("brd,bd->br", B, qt)
            y_t = b_q @ A_hat.T
            reads.append(y_t)

            # Compact-WY reflection step on row 0 (Day-5 regime).
            u1 = k1t
            u2 = k2
            b1v = torch.clamp(b1, 1.6, 1.9)
            b2v = torch.clamp(betas[:, t, 1] if betas.dim() == 3 else betas[:, t], 1.6, 1.9)
            dot12 = (u1 * u2).sum(dim=-1, keepdim=True)
            bu1 = torch.einsum("brd,bd->br", B, u1)
            bu2 = torch.einsum("brd,bd->br", B, u2)
            t1 = b1v.unsqueeze(-1) * bu1
            t2 = (-b1v * b2v * dot12.squeeze(-1)).unsqueeze(-1) * bu1 + b2v.unsqueeze(-1) * bu2
            B = B - t1.unsqueeze(-1) * u1.unsqueeze(1) - t2.unsqueeze(-1) * u2.unsqueeze(1)

            b_k = torch.einsum("brd,bd->br", B, k1t)
            pred = b_k @ A_hat.T
            err = pred - vt
            grad_r = err @ A_hat
            B = B - self.eta * grad_r.unsqueeze(-1) * k1t.unsqueeze(1)

        y = torch.stack(reads, dim=1)  # [b, n, d_m]
        delta = (gate * y) @ self.o  # [b, n, d]
        return delta, B


# --------------------------------------------------------------------------- subspace MoE (Day-5 ablation)


class SubspaceLatentMoE(nn.Module):
    """Stiefel-orthogonal subspace MoE (Day-5 ablation), simplified top-1 router.

    Not part of the dense baseline. Routes each token to a small number of
    experts within selected groups, with shared SwiGLU expert and a Stiefel
    constraint on group projections.
    """

    def __init__(
        self,
        d: int,
        d_lat: int,
        d_ff: int,
        n_groups: int,
        experts_per_group: int,
        top_groups: int,
        top_k_per_group: int,
    ):
        super().__init__()
        self.d, self.d_lat, self.d_ff = d, d_lat, d_ff
        self.G, self.Eg = n_groups, experts_per_group
        self.top_g, self.top_k_g = top_groups, top_k_per_group
        assert top_groups * top_k_per_group <= experts_per_group * n_groups

        self.router_group = nn.Linear(d, n_groups, bias=False)
        self.gbias = nn.Parameter(torch.zeros(n_groups))
        self.down_groups = nn.ParameterList(
            [nn.Parameter(torch.randn(d, d_lat) / math.sqrt(d)) for _ in range(n_groups)]
        )
        self.up_groups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_lat, d) / math.sqrt(d_lat)) for _ in range(n_groups)]
        )
        self.router_fine = nn.ParameterList(
            [nn.Parameter(torch.randn(d_lat, experts_per_group) / math.sqrt(d_lat)) for _ in range(n_groups)]
        )
        self.ebias = nn.Parameter(torch.zeros(n_groups, experts_per_group))

        self.shared = SwiGLU(d, d, d)
        self.experts = nn.ModuleList([SwiGLU(d_lat, d_ff, d_lat) for _ in range(n_groups * experts_per_group)])

    def retract_stiefel(self):
        with torch.no_grad():
            U = torch.cat([g.t() for g in self.up_groups], dim=0)  # [G*d_lat, d]
            U = torch.linalg.qr(U).Q
            dl = self.d_lat
            for g in range(self.G):
                block = U[g * dl : (g + 1) * dl, :]
                self.up_groups[g].copy_(block.t())
                self.down_groups[g].copy_(block)

    def orthogonality_error(self) -> torch.Tensor:
        U = torch.cat([g.t() for g in self.up_groups], dim=0)
        gram = U @ U.t()
        off = gram - torch.eye(gram.shape[0], device=U.device, dtype=U.dtype)
        return (off * off).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        flat = x.reshape(-1, d)
        g_logits = self.router_group(flat) + self.gbias  # [tokens, G]
        g_idx = torch.topk(g_logits, self.top_g, dim=-1).indices  # [tokens, top_g]
        g_w = F.softmax(torch.gather(g_logits, -1, g_idx), dim=-1)

        out = torch.zeros_like(flat)
        for g in range(self.G):
            mask_g = g_idx == g  # [tokens, top_g] bool
            hit_t, hit_slot = mask_g.nonzero(as_tuple=True)
            if hit_t.numel() == 0:
                continue
            z_g = flat[hit_t] @ self.down_groups[g]  # [hit, d_lat]
            f_logits = z_g @ self.router_fine[g] + self.ebias[g]  # [hit, Eg]
            f_idx = torch.topk(f_logits, self.top_k_g, dim=-1).indices
            f_w = F.softmax(torch.gather(f_logits, -1, f_idx), dim=-1) * g_w[hit_t, hit_slot].unsqueeze(-1)

            out_g = torch.zeros_like(z_g)
            for e_local in range(self.Eg):
                etok, eslot = (f_idx == e_local).nonzero(as_tuple=True)
                if etok.numel() == 0:
                    continue
                exp = self.experts[g * self.Eg + e_local]
                act = exp.forward(z_g[etok])
                out_g[etok] += act * f_w[etok, eslot].unsqueeze(-1)
            out[hit_t] += out_g @ self.up_groups[g]

        return self.shared(x) + out.reshape(b, n, d)


# --------------------------------------------------------------------------- recurrent workspace (Day-5 ablation)


class WorkspaceLayer(nn.Module):
    def __init__(self, dw: int, d: int, window: int):
        super().__init__()
        self.dw, self.d = dw, d
        self.q = nn.Parameter(torch.randn(dw, dw) / math.sqrt(dw))
        self.kv = nn.Parameter(torch.randn(d, 2 * dw) / math.sqrt(d))
        self.o = nn.Parameter(torch.randn(dw, dw) / math.sqrt(dw))
        self.sa = nn.Parameter(torch.randn(dw, 3 * dw) / math.sqrt(dw))
        self.sa_o = nn.Parameter(torch.randn(dw, dw) / math.sqrt(dw))
        self.ff = SwiGLU(dw, dw, dw)
        self.window = window
        self.n1 = nn.Parameter(torch.ones(dw))
        self.n2 = nn.Parameter(torch.ones(dw))
        self.n3 = nn.Parameter(torch.ones(dw))

    def forward(self, h: torch.Tensor, k_ctx: torch.Tensor, v_ctx: torch.Tensor) -> torch.Tensor:
        q = rms_norm(h, self.n1) @ self.q  # [b,n,dw]
        b, n, _ = h.shape
        mask = torch.tril(torch.ones(n, n, device=h.device, dtype=torch.bool))
        scores = (q @ k_ctx.transpose(-2, -1)) / math.sqrt(self.dw)
        a = F.softmax(scores.masked_fill(~mask[None, :, :], float("-inf")), dim=-1)
        h = h + (a @ v_ctx) @ self.o

        qkv = rms_norm(h, self.n2) @ self.sa
        qq, kk, vv = qkv.chunk(3, dim=-1)
        scores2 = (qq @ kk.transpose(-2, -1)) / math.sqrt(self.dw)
        a2 = F.softmax(scores2.masked_fill(~mask[None, :, :], float("-inf")), dim=-1)
        h = h + (a2 @ vv) @ self.sa_o
        return h + self.ff(rms_norm(h, self.n3))


class CodeJEPAWorkspace(nn.Module):
    """Recurrent refinement workspace (Day-5 ablation).

    Runs r refinement passes over a projected hidden state attending to a
    cached context K/V. Returns (delta_hidden, h, p_pass, z_loc, z_exec).
    Offered as an ablation; no JEPA objective is implemented here (per the Day-6
    directive: the workspace stays until a real training objective is formalized).
    """

    def __init__(self, d: int, d_w: int, w_layers: int, w_ff: int, d_head_h: int, k_hyp: int, window: int, r_max: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(d, d_w, bias=False)
        self.proj_out = nn.Linear(d_w, d, bias=False)
        self.layers = nn.ModuleList([WorkspaceLayer(d_w, d, window) for _ in range(w_layers)])
        self.scale = nn.Parameter(torch.tensor([9.0e-4]))

        raw_hyp = torch.randn(d_w, k_hyp)
        raw_hyp = torch.linalg.qr(raw_hyp).Q
        self.hyp_codebook = nn.Parameter(raw_hyp.t() * 25.0)

        self.jepa_loc_head = nn.Parameter(torch.randn(d_w, d_w) / math.sqrt(d_w))
        self.jepa_exec_head = nn.Parameter(torch.randn(d_w, d_w) / math.sqrt(d_w))

        self.conf_query = nn.Parameter(torch.randn(d_w) / math.sqrt(d_w))
        self.head_w1 = nn.Parameter(torch.randn(d_w, d_head_h) / math.sqrt(d_w))
        self.head_b1 = nn.Parameter(torch.zeros(d_head_h))
        self.head_w2 = nn.Parameter(torch.randn(d_head_h, 1) / math.sqrt(d_head_h))
        self.head_b2 = nn.Parameter(torch.zeros(1))
        self.k_hyp = k_hyp
        self.r_max = r_max

    def forward(
        self, x: torch.Tensor, r: int = 2, ctx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x @ self.proj_in + self.hyp_codebook[0]
        ctx = x if ctx is None else ctx
        cached_kv = [(layer, (ctx @ self.kv).chunk(2, dim=-1)) for layer in self.layers]
        z_exec = torch.zeros_like(h)
        for step in range(min(r, self.r_max)):
            for layer, (k_ctx, v_ctx) in cached_kv:
                h = layer.forward(h, k_ctx, v_ctx)
            z_exec = 0.5 * z_exec + 0.5 * (h @ self.jepa_exec_head)
        hidden_tok = F.gelu(h @ self.head_w1 + self.head_b1)
        p_pass_tok = torch.sigmoid(hidden_tok @ self.head_w2 + self.head_b2).squeeze(-1)
        ws_delta = h @ self.proj_out
        eff_scale = self.scale[0]
        return x + eff_scale * ws_delta, h, p_pass_tok, h @ self.jepa_loc_head, z_exec


# --------------------------------------------------------------------------- block


class Block(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        d_ff: int,
        norm_eps: float,
        rope_theta: float,
        max_ctx: int,
        sliding_window: Optional[int],
        attn_impl: str = "dense",
        d_c: Optional[int] = None,
        d_rope: Optional[int] = None,
        use_memory: bool = False,
        d_m: int = 512,
        r_m: int = 256,
        n_householder: int = 2,
        use_moe: bool = False,
        d_lat: int = 384,
        d_moe_ff: int = 4096,
        n_groups: int = 8,
        experts_per_group: int = 8,
        top_groups: int = 6,
        top_k_per_group: int = 1,
        use_workspace: bool = False,
        d_w: int = 1024,
        w_layers: int = 4,
        w_ff: int = 4096,
        d_head_h: int = 512,
        k_hyp: int = 8,
        window: int = 4096,
    ):
        super().__init__()
        self.attn_impl = attn_impl
        self.n1 = nn.Parameter(torch.ones(d))
        if attn_impl == "mla":
            self.attn = LocalMLA(
                d, n_heads, n_kv_heads, head_dim,
                d_c or d // 2, d_rope or 64, sliding_window, max_ctx, rope_theta,
            )
        else:
            self.attn = CausalSelfAttention(d, n_heads, n_kv_heads, max_ctx, rope_theta, sliding_window, norm_eps)
        self.n2 = nn.Parameter(torch.ones(d))
        self.use_memory = use_memory
        if use_memory:
            self.mem = WYHouseholderMemory(d, d_m, r_m, n_householder)
        self.n3 = nn.Parameter(torch.ones(d))
        self.mlp = SwiGLU(d, d_ff, d)
        self.use_moe = use_moe
        if use_moe:
            self.moe = SubspaceLatentMoE(d, d_lat, d_moe_ff, n_groups, experts_per_group, top_groups, top_k_per_group)
        self.use_workspace = use_workspace
        if use_workspace:
            self.workspace = CodeJEPAWorkspace(d, d_w, w_layers, w_ff, d_head_h, k_hyp, window)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        seq_start: int = 0,
        memory_state: Optional[torch.Tensor] = None,
        workspace_r: int = 2,
    ):
        attn_cache = cache.get("attn") if cache else None
        if self.attn_impl == "mla":
            attn_out, new_attn_cache = self.attn.forward(rms_norm(x, self.n1), cache=attn_cache, seq_start=seq_start)
        else:
            attn_out, _, new_attn_cache = self.attn.forward(rms_norm(x, self.n1), cache=attn_cache, seq_start=seq_start)
        x = x + attn_out

        new_cache = {"attn": new_attn_cache}
        mem_state_out = memory_state
        if self.use_memory:
            delta, mem_state_out = self.mem.forward(rms_norm(x, self.n2), state=memory_state)
            x = x + delta
        x = x + self.mlp(rms_norm(x, self.n3))

        if self.use_moe:
            x = x + self.moe(rms_norm(x, self.n3))
        if self.use_workspace:
            ws_delta, _, p_pass, z_loc, z_exec = self.workspace.forward(x, r=workspace_r)
            x = x + ws_delta

        return x, new_cache, mem_state_out


# --------------------------------------------------------------------------- model


class Kestrel6(nn.Module):
    """Kestrel-6 trainable language model.

    Args:
        config: model config dict (see config.json presets).
        workspace_r: number of recurrent workspace passes (only if use_workspace).

    Truthful metadata attributes:
        model_status        = "randomly_initialized" until a checkpoint is loaded
        training_backend    = "pytorch"
        checkpoint_loaded   = False
        supports_autograd   = True
    """

    def __init__(self, config: dict, workspace_r: int = 2):
        super().__init__()
        self.config = config
        self.workspace_r = workspace_r
        self.vocab = config["vocab"]
        self.d = config["d"]
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.head_dim = self.d // self.n_heads
        self.d_ff = config["d_ff"]
        self.layers_n = config["layers"]
        self.max_ctx = config["max_ctx"]
        self.norm_eps = config.get("norm_eps", 1e-5)
        self.rope_theta = config.get("rope_theta", 500000.0)
        self.tie_embeddings = config.get("tie_embeddings", True)

        self.embed = nn.Embedding(self.vocab, self.d)
        self.blocks = nn.ModuleList()
        global_set = set(config.get("global_layers", []))
        for i in range(self.layers_n):
            # Global layers (NoPE landmark) when attn_impl=="mla" and i in global_set.
            is_global = i in global_set
            self.blocks.append(
                Block(
                    d=self.d,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    head_dim=self.head_dim,
                    d_ff=self.d_ff,
                    norm_eps=self.norm_eps,
                    rope_theta=self.rope_theta,
                    max_ctx=self.max_ctx,
                    sliding_window=config.get("sliding_window"),
                    attn_impl=config["attn_impl"],
                    d_c=config.get("d_c"),
                    d_rope=config.get("d_rope"),
                    use_memory=config.get("use_memory", False),
                    d_m=config.get("d_m", 512),
                    r_m=config.get("r_m", 256),
                    n_householder=config.get("n_householder", 2),
                    use_moe=config.get("use_moe", False),
                    d_lat=config.get("d_lat", 384),
                    d_moe_ff=config.get("d_moe_ff", self.d_ff),
                    n_groups=config.get("n_groups", 8),
                    experts_per_group=config.get("experts_per_group", 8),
                    top_groups=config.get("top_groups", 6),
                    top_k_per_group=config.get("top_k_per_group", 1),
                    use_workspace=config.get("use_workspace", False),
                    d_w=config.get("d_w", 1024),
                    w_layers=config.get("w_layers", 4),
                    w_ff=config.get("w_ff", 4096),
                    d_head_h=config.get("d_head_h", 512),
                    k_hyp=config.get("k_hyp", 8),
                    window=config.get("w_window", config.get("max_ctx", 4096)),
                )
            )
        self.norm = nn.Parameter(torch.ones(self.d))
        if self.tie_embeddings:
            self.lm_head = None  # tied to embed
        else:
            self.lm_head = nn.Linear(self.d, self.vocab, bias=False)

        # Truthful metadata.
        self.model_status = "randomly_initialized"
        self.training_backend = "pytorch"
        self.checkpoint_loaded = False
        self.supports_autograd = True

    def config_hash(self) -> str:
        """SHA-256 over the sorted config JSON; used to validate checkpoints."""
        return hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        seq_start: int = 0,
        workspace_r: Optional[int] = None,
    ):
        b, n = input_ids.shape
        x = self.embed(input_ids)  # [b, n, d]
        r = self.workspace_r if workspace_r is None else workspace_r

        new_caches: List[Dict[str, torch.Tensor]] = []
        mem_states: List[Optional[torch.Tensor]] = [None] * self.layers_n
        if cache is not None:
            for i, blk in enumerate(self.blocks):
                if blk.use_memory and cache[i] is not None and "mem" in cache[i]:
                    mem_states[i] = cache[i]["mem"]

        for i, blk in enumerate(self.blocks):
            blk_cache = cache[i] if cache is not None else None
            x, new_cache, mem_state = blk.forward(
                x,
                cache=blk_cache,
                seq_start=seq_start,
                memory_state=mem_states[i],
                workspace_r=r,
            )
            new_caches.append(new_cache)
            if blk.use_memory:
                new_caches[i]["mem"] = mem_state

        # Optional recurrent workspace applied once after the trunk (Day-5 ablation).
        if self.blocks[-1].use_workspace:
            pass  # workspace already applied inside the last block

        x = rms_norm(x, self.norm)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed.weight)
        return logits, new_caches

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, workspace_r: Optional[int] = None):
        logits, _ = self._forward_with_cache(input_ids, cache=None, seq_start=0, workspace_r=workspace_r)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        """Run the prompt through the model, building the KV cache.

        Returns (logits for the final position, cache). Call decode_step() to
        continue autoregressively.
        """
        logits, caches = self._forward_with_cache(input_ids, cache=None, seq_start=0)
        return logits[:, -1, :], caches

    @torch.no_grad()
    def decode_step(self, token_id: torch.Tensor, cache: List[Dict[str, torch.Tensor]], seq_start: int):
        """Decode a single token given the KV cache. Returns (logits, new_cache)."""
        x = token_id.view(1, 1)
        logits, caches = self._forward_with_cache(x, cache=cache, seq_start=seq_start)
        return logits[:, -1, :], caches


def build_model(config_name_or_dict, workspace_r: int = 2) -> Kestrel6:
    """Build a Kestrel-6 model from a config preset name or a config dict."""
    if isinstance(config_name_or_dict, str):
        cfg = get_config(config_name_or_dict)
    else:
        cfg = dict(config_name_or_dict)
    return Kestrel6(cfg, workspace_r=workspace_r)


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
