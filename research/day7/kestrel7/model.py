"""Kestrel-7: a 3:1 Gated DeltaNet / gated-attention hybrid language model.

Architecture, and the alternative each choice beat:

*   **3:1 hybrid layout** (``[gdn, gdn, gdn, full]``). Three quarters of the
    layers keep an O(1) recurrent state (Gated DeltaNet) and every fourth layer
    is full/sliding causal attention, the layout used by Qwen3.5/Qwen3-Next and
    explained by Zeng et al. (Nov 2026) as an Encode-Address-Reader circuit.
    Beat: an all-full-attention stack at equal width/depth. Falsifier: hybrid
    perplexity worse than dense at equal tokens *and* no KV-cache reduction, or
    the streaming cache disagreeing with a full forward pass (>1e-4).
*   **Zero-centered RMSNorm** ``y = x * rsqrt(mean(x^2)+eps) * (1 + gamma)``
    with ``gamma=0`` at init, computed in fp32. Beat: standard ``* gamma`` with
    ``gamma=1`` init, which drifts the residual scale during long runs.
    Falsifier: activation RMS diverging from 1 across depth at init.
*   **Per-head QK-RMSNorm + sigmoid output gate.** Beat: unnormalized QK, whose
    logits grow with head_dim and collapse attention entropy. Falsifier: max
    attention logit growing without bound over training.
*   **Neural MTP head, depth 1.** Predicts token t+2 from h_t and embed(x_{t+1}),
    trained jointly with weight 0.2. Beat: a pure next-token objective, which
    gives no planning signal for the next token. Falsifier: MTP head receiving
    zero gradient, or MTP loss not decreasing.
*   **Single label shift.** Labels are stored unshifted and shifted exactly once
    here. Beat: Day 6's double shift (labels shifted in the dataset *and* in
    forward), which trained the model to predict t+2 and capped exact match at 0.
    Falsifier: loss computed against ``labels[..., 1:]`` not matching a manual
    cross-entropy over the same tensors.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import get_config, validate_config
from .tokenizer import BPETokenizer


# --------------------------------------------------------------------------- primitives


def rms_norm_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    zero_centered: bool = True,
) -> torch.Tensor:
    """FP32-accumulated RMSNorm with optional zero-centered ``(1 + weight)``."""
    x_fp32 = x.to(torch.float32)
    inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    scale = (1.0 + weight) if zero_centered else weight
    return (x_fp32 * inv_rms).to(x.dtype) * scale


def rope_cos_sin(
    seq_len: int,
    dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE tables of shape ``[1, seq_len, dim // 2]`` (one angle per pair)."""
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(dtype).unsqueeze(0)
    sin = torch.sin(freqs).to(dtype).unsqueeze(0)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.reshape_as(x)


# --------------------------------------------------------------------------- attention


class GatedCausalSelfAttention(nn.Module):
    """Gated GQA with QK-RMSNorm, RoPE, sliding-window cache, and document mask."""

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        max_ctx: int,
        rope_theta: float,
        sliding_window: Optional[int] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        output_gate: bool = True,
        zero_centered_norm: bool = True,
    ):
        super().__init__()
        assert d % n_heads == 0 and n_heads % n_kv_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d // n_heads
        self.max_ctx = max_ctx
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.output_gate = output_gate
        self.zero_centered_norm = zero_centered_norm

        self.w_q = nn.Linear(d, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d, bias=False)

        if qk_norm:
            init_fn = torch.zeros if zero_centered_norm else torch.ones
            self.q_norm_w = nn.Parameter(init_fn(self.head_dim))
            self.k_norm_w = nn.Parameter(init_fn(self.head_dim))
        if output_gate:
            self.w_gate = nn.Linear(d, n_heads * self.head_dim, bias=False)

        # Non-persistent 3D buffers: shape[1] is a valid sequence length even
        # before the first forward pass (Day-6 P0-01 was a 1D empty buffer).
        self.register_buffer("cos", torch.empty(1, 0, self.head_dim // 2), persistent=False)
        self.register_buffer("sin", torch.empty(1, 0, self.head_dim // 2), persistent=False)

    def _rope(self, seq_len: int, device, dtype):
        if (
            self.cos.ndim != 3
            or self.cos.shape[1] < seq_len
            or self.cos.device != device
            or self.cos.dtype != dtype
        ):
            self.cos, self.sin = rope_cos_sin(
                max(seq_len, 64), self.head_dim, self.rope_theta, device, dtype
            )
        return self.cos, self.sin

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        seq_start: int = 0,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, n, _ = x.shape
        q = self.w_q(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = rms_norm_fp32(q, self.q_norm_w, self.norm_eps, self.zero_centered_norm)
            k = rms_norm_fp32(k, self.k_norm_w, self.norm_eps, self.zero_centered_norm)

        cos_all, sin_all = self._rope(seq_start + n, x.device, x.dtype)
        cos = cos_all[:, seq_start : seq_start + n, :]
        sin = sin_all[:, seq_start : seq_start + n, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None and "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cap = self.sliding_window or self.max_ctx
        using_cache = cache is not None and "k" in cache
        # Eviction is only sound for a single incremental query: that query's
        # legal keys all lie in the last ``cap`` positions. During prefill (or a
        # multi-token cached chunk) early queries still need the oldest keys, so
        # trimming here would leave their softmax rows empty and produce NaNs.
        if cap is not None and k.shape[2] > cap and using_cache and n == 1:
            k = k[:, :, -cap:, :]
            v = v[:, :, -cap:, :]
        new_cache = {"k": k, "v": v}

        total_len = k.shape[2]
        if self.n_rep > 1:
            k_exp = k.repeat_interleave(self.n_rep, dim=1)
            v_exp = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_exp, v_exp = k, v

        scores = (q @ k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Absolute positions make causal/sliding masks correct even after the
        # cache has evicted its oldest entries.
        q_pos = torch.arange(seq_start, seq_start + n, device=x.device)[:, None]
        k_pos = torch.arange(seq_start + n - total_len, seq_start + n, device=x.device)[None, :]
        mask = k_pos <= q_pos
        if self.sliding_window is not None:
            mask = mask & ((q_pos - k_pos) < self.sliding_window)
        mask_4d = mask[None, None, :, :]

        if doc_ids is not None and total_len == n:
            # Block-diagonal intra-document mask. The diagonal is always kept so
            # padding queries (doc_id -1) never produce an all -inf softmax row,
            # which would emit NaNs into the backward pass.
            same_doc = (doc_ids[:, None, :, None] == doc_ids[:, None, None, :]) & (
                doc_ids[:, None, :, None] >= 0
            )
            diag = torch.eye(n, device=x.device, dtype=torch.bool)[None, None, :, :]
            mask_4d = mask_4d & (same_doc | diag)

        # A legal mask can still leave a query row with no key (padded rows,
        # empty segments, sliding-window edges). Explicitly zero those rows
        # instead of letting an all -inf softmax emit NaN into the backward pass.
        has_any_key = mask_4d.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask_4d, float("-inf"))
        scores = torch.where(has_any_key, scores, torch.zeros_like(scores))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn = attn * has_any_key.to(attn.dtype)
        out = (attn @ v_exp).transpose(1, 2).contiguous().view(b, n, self.d)

        if self.output_gate:
            out = out * torch.sigmoid(self.w_gate(x))
        return self.w_o(out), new_cache


# --------------------------------------------------------------------------- gated delta net


class GatedDeltaNetLayer(nn.Module):
    """Gated DeltaNet linear attention with a depthwise causal conv and O(1) state.

    Update (per head, per token), following Yang et al. (ICLR 2025):

        S_t = g_t * S_{t-1} + k_t (x) [ beta_t * (v_t - (g_t * S_{t-1})^T k_t) ]
        o_t = S_t^T q_t / sqrt(d_h)

    with ``g_t = exp(-exp(A_log) * softplus(a_t + dt_bias))`` in (0, 1] and
    ``beta_t = sigmoid(b_t)``. Beat: a pure SSM (Mamba-2) update, which can only
    add associations and cannot correct them. The delta rule subtracts the
    already-stored value before writing, so repeated keys stay retrievable.
    Falsifier: associative-recall accuracy worse than a plain decayed state.
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        conv_kernel: int = 4,
        norm_eps: float = 1e-5,
        zero_centered_norm: bool = True,
    ):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.conv_kernel = conv_kernel
        self.norm_eps = norm_eps
        self.zero_centered_norm = zero_centered_norm

        self.in_proj_qkvz = nn.Linear(d, 4 * d, bias=False)
        self.in_proj_ba = nn.Linear(d, 2 * n_heads, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(3 * d, conv_kernel) / math.sqrt(conv_kernel))
        # A_log = -3 gives an initial retention of exp(-exp(-3)*softplus(1)) ~ 0.937.
        self.A_log = nn.Parameter(torch.full((n_heads,), -3.0))
        self.dt_bias = nn.Parameter(torch.ones(n_heads))
        init_fn = torch.zeros if zero_centered_norm else torch.ones
        self.out_norm_w = nn.Parameter(init_fn(self.head_dim))
        self.out_proj = nn.Linear(d, d, bias=False)

    def _causal_conv1d(
        self,
        qkv: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        doc_hist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        b, n, c = qkv.shape
        k_size = self.conv_kernel
        if conv_state is None:
            pad = torch.zeros(b, k_size - 1, c, device=qkv.device, dtype=qkv.dtype)
        else:
            pad = conv_state
        padded = torch.cat([pad, qkv], dim=1)  # [b, n + K - 1, 3d]

        docs_padded = None
        if doc_ids is not None:
            if doc_hist is None:
                doc_hist = torch.full(
                    (b, k_size - 1), -1, device=qkv.device, dtype=torch.long
                )
            docs_padded = torch.cat([doc_hist, doc_ids], dim=1)

        out = torch.zeros_like(qkv)
        for ki in range(k_size):
            w_slice = self.conv_weight[:, k_size - 1 - ki]
            start = k_size - 1 - ki
            contrib = padded[:, start : start + n, :] * w_slice
            if docs_padded is not None:
                # A convolution tap may only read tokens from the current
                # document; cross-boundary taps are zeroed, exactly like the
                # recurrent state reset below.
                prev_docs = docs_padded[:, start : start + n]
                valid = (prev_docs == doc_ids) | (prev_docs < 0) | (doc_ids < 0)
                contrib = contrib * valid.unsqueeze(-1).to(contrib.dtype)
            out = out + contrib

        new_conv_state = padded[:, -(k_size - 1) :, :].detach()
        new_doc_hist = docs_padded[:, -(k_size - 1) :].detach() if docs_padded is not None else None
        return F.silu(out), new_conv_state, new_doc_hist

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, n, _ = x.shape
        h, hd = self.n_heads, self.head_dim

        qkvz = self.in_proj_qkvz(x)
        qkv, z = qkvz[..., : 3 * self.d], qkvz[..., 3 * self.d :]
        conv_state_in = cache.get("conv") if cache else None
        doc_hist_in = cache.get("doc_hist") if cache else None
        qkv_conv, conv_state_out, doc_hist_out = self._causal_conv1d(
            qkv, conv_state_in, doc_ids=doc_ids, doc_hist=doc_hist_in
        )

        q, k, v = qkv_conv.chunk(3, dim=-1)
        q = F.normalize(q.view(b, n, h, hd), dim=-1)
        k = F.normalize(k.view(b, n, h, hd), dim=-1)
        v = v.view(b, n, h, hd)
        z = z.view(b, n, h, hd)

        ba = self.in_proj_ba(x)
        b_raw, a_raw = ba.chunk(2, dim=-1)
        beta = torch.sigmoid(b_raw)
        g = torch.exp(-torch.exp(self.A_log) * F.softplus(a_raw + self.dt_bias))
        g = g.clamp(max=1.0, min=1e-6)

        if cache is not None and cache.get("state") is not None:
            S = cache["state"].clone()
        else:
            S = torch.zeros(b, h, hd, hd, device=x.device, dtype=x.dtype)

        prev_doc = cache.get("last_doc") if cache is not None else None

        outputs: List[torch.Tensor] = []
        scale = 1.0 / math.sqrt(hd)
        for t in range(n):
            # Document-boundary reset: zero the recurrent state whenever the
            # packed stream crosses into a new document.
            if doc_ids is not None:
                if t == 0 and prev_doc is not None:
                    same_doc = (doc_ids[:, 0] == prev_doc) & (doc_ids[:, 0] >= 0)
                    S = S * same_doc[:, None, None, None].to(S.dtype)
                elif t > 0:
                    same_doc = (doc_ids[:, t] == doc_ids[:, t - 1]) & (doc_ids[:, t] >= 0)
                    S = S * same_doc[:, None, None, None].to(S.dtype)

            gt = g[:, t, :, None, None]
            bt = beta[:, t, :, None]
            kt = k[:, t]
            vt = v[:, t]
            qt = q[:, t]

            S_decayed = S * gt
            retrieved = torch.einsum("bhkv,bhk->bhv", S_decayed, kt)
            delta = bt * (vt - retrieved)
            S = S_decayed + kt.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append(torch.einsum("bhkv,bhk->bhv", S, qt) * scale)

        out = torch.stack(outputs, dim=1)
        out = rms_norm_fp32(out, self.out_norm_w, self.norm_eps, self.zero_centered_norm) * F.silu(z)
        out = out.reshape(b, n, self.d)
        new_cache = {"state": S, "conv": conv_state_out}
        if doc_ids is not None:
            new_cache["last_doc"] = doc_ids[:, -1].clone()
            new_cache["doc_hist"] = doc_hist_out
        return self.out_proj(out), new_cache


# --------------------------------------------------------------------------- feed-forward / MTP


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int):
        super().__init__()
        self.g = nn.Linear(d_in, d_ff, bias=False)
        self.u = nn.Linear(d_in, d_ff, bias=False)
        self.dn = nn.Linear(d_ff, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dn(F.silu(self.g(x)) * self.u(x))


class MTPDraftHead(nn.Module):
    """Depth-1 multi-token-prediction head: plan token t+2 from h_t and e_{t+1}."""

    def __init__(self, d: int, norm_eps: float = 1e-5, zero_centered: bool = True):
        super().__init__()
        self.norm_eps = norm_eps
        self.zc = zero_centered
        init_fn = torch.zeros if zero_centered else torch.ones
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.norm = nn.Parameter(init_fn(d))
        self.ffn = SwiGLU(d, 2 * d, d)

    def forward(self, h_trunk: torch.Tensor, next_token_embeds: torch.Tensor) -> torch.Tensor:
        combined = self.proj(torch.cat([h_trunk, next_token_embeds], dim=-1))
        out = combined + self.ffn(rms_norm_fp32(combined, self.norm, self.norm_eps, self.zc))
        return rms_norm_fp32(out, self.norm, self.norm_eps, self.zc)


# --------------------------------------------------------------------------- block


class Kestrel7Block(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        d = config["d"]
        self.norm_eps = config["norm_eps"]
        self.zc = config.get("zero_centered_norm", True)
        init_fn = torch.zeros if self.zc else torch.ones

        schedule = config.get("hybrid_schedule")
        self.mixer_type = schedule[layer_idx] if schedule is not None else "full"

        self.n1 = nn.Parameter(init_fn(d))
        if self.mixer_type == "gdn":
            self.mixer = GatedDeltaNetLayer(
                d=d,
                n_heads=config["n_heads"],
                conv_kernel=config.get("conv_kernel", 4),
                norm_eps=self.norm_eps,
                zero_centered_norm=self.zc,
            )
        else:
            self.mixer = GatedCausalSelfAttention(
                d=d,
                n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"],
                max_ctx=config["max_ctx"],
                rope_theta=config["rope_theta"],
                sliding_window=config.get("sliding_window"),
                norm_eps=self.norm_eps,
                qk_norm=config.get("qk_norm", True),
                output_gate=config.get("attn_output_gate", True),
                zero_centered_norm=self.zc,
            )
        self.n2 = nn.Parameter(init_fn(d))
        self.mlp = SwiGLU(d, config["d_ff"], d)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        seq_start: int = 0,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = rms_norm_fp32(x, self.n1, self.norm_eps, self.zc)
        if self.mixer_type == "gdn":
            mix_out, new_cache = self.mixer(h, cache=cache, doc_ids=doc_ids)
        else:
            mix_out, new_cache = self.mixer(h, cache=cache, seq_start=seq_start, doc_ids=doc_ids)
        x = x + mix_out
        x = x + self.mlp(rms_norm_fp32(x, self.n2, self.norm_eps, self.zc))
        return x, new_cache


# --------------------------------------------------------------------------- model


class Kestrel7(nn.Module):
    """Kestrel-7 hybrid LM with a strict tokenizer/vocab contract.

    Truthful metadata:
        model_status      = "randomly_initialized" until a real checkpoint is loaded
        checkpoint_loaded = False
        supports_autograd = True
    """

    def __init__(self, config: dict, tokenizer_hash: Optional[str] = None):
        super().__init__()
        self.config = copy.deepcopy(dict(config))
        validate_config(self.config, "Kestrel7")
        self.vocab = int(config["vocab"])
        self.d = int(config["d"])
        self.layers_n = int(config["layers"])
        self.max_ctx = int(config["max_ctx"])
        self.norm_eps = float(config.get("norm_eps", 1e-5))
        self.zc = bool(config.get("zero_centered_norm", True))
        self.tie_embeddings = bool(config.get("tie_embeddings", True))
        self.mtp_depth = int(config.get("mtp_depth", 0))

        self.logit_scale = 1.0 / math.sqrt(self.d)
        self.embed = nn.Embedding(self.vocab, self.d)
        self.blocks = nn.ModuleList([Kestrel7Block(self.config, i) for i in range(self.layers_n)])
        init_fn = torch.zeros if self.zc else torch.ones
        self.norm = nn.Parameter(init_fn(self.d))
        self.lm_head = None if self.tie_embeddings else nn.Linear(self.d, self.vocab, bias=False)
        self.mtp_head = MTPDraftHead(self.d, self.norm_eps, self.zc) if self.mtp_depth > 0 else None

        self._init_weights()

        self.model_status = "randomly_initialized"
        self.training_backend = "pytorch"
        self.checkpoint_loaded = False
        self.supports_autograd = True
        self.tokenizer_hash = tokenizer_hash

    def _init_weights(self) -> None:
        """Explicit initialization policy (PyTorch defaults are not a policy).

        A tied embedding at the default ``N(0, 1)`` produces output logits with
        standard deviation ``~sqrt(d)`` (21.55 measured at ``d=256``), so the
        initial CE is ~252 against a uniform baseline of 5.61 -- gradients start
        badly scaled and the first optimizer steps chase enormous logits. The
        embedding is therefore scaled ``1/sqrt(d)``; every other weight matrix
        starts at ``N(0, 0.02)``. Zero-centered norms already start at the
        identity. Beat: default initialization, which made the 50-step demo
        report a 108 loss at step 10.
        """
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.d))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for blk in self.blocks:
            mixer = blk.mixer
            if isinstance(mixer, GatedDeltaNetLayer):
                nn.init.normal_(mixer.conv_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    mixer.A_log.fill_(-3.0)
                    mixer.dt_bias.fill_(1.0)

    def config_hash(self) -> str:
        return hash_config(self.config)

    def _project_logits(self, h: torch.Tensor) -> torch.Tensor:
        # Fixed output scale. A tied embedding plus RMSNorm makes the final
        # hidden state a ~sqrt(d)-scaled copy of the input embedding, so the
        # position-0 logit for the input token reaches ~d*std(embed) (measured
        # 11.1 at d=256) and the initial CE is 11.7 against a 5.61 uniform
        # baseline. Dividing by sqrt(d) restores a near-uniform start; the
        # learnable final-norm scale can undo it during training.
        if self.lm_head is not None:
            return self.lm_head(h) * self.logit_scale
        return F.linear(h, self.embed.weight) * self.logit_scale

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        seq_start: int = 0,
        doc_ids: Optional[torch.Tensor] = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, time], got shape {tuple(input_ids.shape)}")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if doc_ids is not None and doc_ids.shape != input_ids.shape:
            raise ValueError(
                f"doc_ids shape {tuple(doc_ids.shape)} != input_ids shape {tuple(input_ids.shape)}"
            )
        x = self.embed(input_ids)
        new_caches: List[Dict[str, torch.Tensor]] = []
        for i, blk in enumerate(self.blocks):
            blk_cache = cache[i] if cache is not None else None
            x, new_c = blk(x, cache=blk_cache, seq_start=seq_start, doc_ids=doc_ids)
            new_caches.append(new_c)
        h = rms_norm_fp32(x, self.norm, self.norm_eps, self.zc)
        return self._project_logits(h), h, new_caches

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        mtp_weight: float = 0.2,
    ):
        logits, h, _ = self._forward_with_cache(input_ids, cache=None, seq_start=0, doc_ids=doc_ids)
        loss = None
        if labels is not None:
            # Exactly one shift: position t predicts token t+1.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_labels = shift_labels.view(-1)
            n_lm_targets = int((flat_labels != -100).sum().item())
            if n_lm_targets == 0:
                raise ValueError(
                    "Batch contains no valid language-model targets (all labels are -100)"
                )
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab), flat_labels, ignore_index=-100
            )
            loss = ce_loss
            if self.mtp_head is not None and input_ids.shape[1] >= 3 and mtp_weight > 0:
                # Depth-1: h_t plus embed(x_{t+1}) predicts token t+2.
                h_mtp = self.mtp_head(h[:, :-2, :], self.embed(input_ids[:, 1:-1]))
                mtp_logits = self._project_logits(h_mtp)
                mtp_labels = labels[:, 2:].contiguous().view(-1)
                # The MTP mask is not assumed identical to the LM mask; a batch
                # with no t+2 target simply contributes no auxiliary term.
                if int((mtp_labels != -100).sum().item()) > 0:
                    mtp_loss = F.cross_entropy(
                        mtp_logits.reshape(-1, self.vocab),
                        mtp_labels,
                        ignore_index=-100,
                    )
                    loss = ce_loss + mtp_weight * mtp_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {float(loss.detach())}")
        return logits, loss

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, doc_ids: Optional[torch.Tensor] = None):
        if input_ids.shape[1] == 0:
            raise ValueError("prefill requires at least one token")
        if input_ids.shape[1] > self.max_ctx:
            input_ids = input_ids[:, -self.max_ctx :]
            doc_ids = doc_ids[:, -self.max_ctx :] if doc_ids is not None else None
        logits, _, caches = self._forward_with_cache(
            input_ids, cache=None, seq_start=0, doc_ids=doc_ids
        )
        return logits[:, -1, :], caches

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, cache: List[Dict[str, torch.Tensor]], seq_start: int):
        b = token_ids.shape[0]
        x = token_ids.view(b, 1)
        logits, _, caches = self._forward_with_cache(x, cache=cache, seq_start=seq_start)
        return logits[:, -1, :], caches


# --------------------------------------------------------------------------- builders & ledgers


def hash_config(config: dict) -> str:
    """Stable SHA-256 over the sorted config JSON (shared by save/load checks)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_model(config_name_or_dict, tokenizer: Optional[BPETokenizer] = None) -> Kestrel7:
    """Build a Kestrel-7 model, enforcing ``model.vocab == tokenizer.vocab_size``."""
    if isinstance(config_name_or_dict, str):
        cfg = get_config(config_name_or_dict)
    else:
        cfg = copy.deepcopy(dict(config_name_or_dict))
    tok_hash = None
    if tokenizer is not None:
        cfg["vocab"] = int(tokenizer.vocab_size)
        tok_hash = tokenizer.tokenizer_hash()
    return Kestrel7(cfg, tokenizer_hash=tok_hash)


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def clone_hybrid_cache(cache: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Deep-copy KV caches *and* Gated DeltaNet recurrent states for CDCL rewinds."""
    copied: List[Dict[str, torch.Tensor]] = []
    for layer_c in cache:
        c_new = {}
        for key, value in layer_c.items():
            c_new[key] = value.clone() if isinstance(value, torch.Tensor) else value
        copied.append(c_new)
    return copied


def kv_cache_bytes(model: Kestrel7, seq_len: Optional[int] = None, dtype_bytes: int = 2) -> int:
    """KV bytes for one sequence, counting full-attention layers only.

    Gated DeltaNet layers store a fixed ``[h, d_h, d_h]`` state per layer, not a
    per-token KV cache, so they contribute O(1) rather than O(T) memory.
    """
    t = int(seq_len if seq_len is not None else model.max_ctx)
    total = 0
    for blk in model.blocks:
        if isinstance(blk.mixer, GatedCausalSelfAttention):
            total += 2 * blk.mixer.n_kv_heads * blk.mixer.head_dim * t * dtype_bytes
    return total


def gdn_state_bytes(model: Kestrel7, batch: int = 1, dtype_bytes: int = 4) -> int:
    """Fixed recurrent + convolution state bytes for Gated DeltaNet layers."""
    total = 0
    for blk in model.blocks:
        if isinstance(blk.mixer, GatedDeltaNetLayer):
            h = blk.mixer.n_heads
            hd = blk.mixer.head_dim
            state = batch * h * hd * hd * dtype_bytes
            conv = batch * (blk.mixer.conv_kernel - 1) * 3 * model.d * dtype_bytes
            total += state + conv
    return total
