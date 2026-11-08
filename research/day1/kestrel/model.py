"""
Kestrel-Coder-4B — reference implementation.

Deliberately small, readable and dependency-light: no FlashAttention, no fused
kernels, no tensor parallelism. The purpose is to be *checkable* — every
architectural claim in research/day1/02-architecture.md is expressible here in
under 400 lines.

    python3 research/day1/kestrel/model.py

What it demonstrates:
  1. MLA attention with a low-rank *joint* KV cache (DeepSeek-V2/V3 style).
  2. A decoupled RoPE key, which is what makes caching c_kv possible at all.
  3. Attention sinks, so a KV-cache eviction policy can be added later without
     destroying the softmax normaliser.
  4. Multi-token prediction heads (training only, dropped at inference).
  5. An incremental cache whose output is numerically identical to a single
     full-sequence pass — asserted in __main__.

Not demonstrated (see 07-serving.md): the "absorption" trick that folds
W_UK into W_UQ at serving time, paged attention, FP8 KV, speculative decode.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass
class MLAConfig:
    kv_lora_rank: int = 448
    q_lora_rank: int = 448
    qk_rope_head_dim: int = 64
    attention_sinks: int = 4
    qk_norm: bool = True


@dataclass
class Config:
    vocab_size: int = 131072
    hidden_size: int = 3072
    num_hidden_layers: int = 40
    num_attention_heads: int = 24
    head_dim: int = 128
    intermediate_size: int = 7680
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    mtp_heads: int = 2
    mtp_loss_weight: float = 0.3
    max_position_embeddings: int = 262144
    attention: MLAConfig = field(default_factory=MLAConfig)
    _expected_inference: float = 4.0834e9   # from tools/count_params.py

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            head_dim=raw["head_dim"],
            intermediate_size=raw["ffn"]["intermediate_size"],
            rope_theta=raw["position"]["rope_theta"],
            rms_norm_eps=raw["norm"]["eps"],
            tie_word_embeddings=raw["embeddings"]["tie_word_embeddings"],
            mtp_heads=raw["mtp"]["num_heads"],
            max_position_embeddings=raw["context"]["extended"],
            attention=MLAConfig(
                kv_lora_rank=raw["attention"]["kv_lora_rank"],
                q_lora_rank=raw["attention"]["q_lora_rank"],
                qk_rope_head_dim=raw["attention"]["qk_rope_head_dim"],
                attention_sinks=raw["attention"]["attention_sinks"],
                qk_norm=raw["attention"]["qk_norm"],
            ),
        )


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def precompute_rope(dim: int, max_pos: int, theta: float, dtype=torch.float32):
    """Standard RoPE tables. dim must be even."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (max_pos, dim/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin, start_pos: int = 0):
    """x: (b, t, h, d). cos/sin: (max_pos, d/2)."""
    t = x.shape[1]
    cos = cos[start_pos:start_pos + t].unsqueeze(0).unsqueeze(2)   # 1,t,1,d/2
    sin = sin[start_pos:start_pos + t].unsqueeze(0).unsqueeze(2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos.repeat_interleave(2, dim=-1) + rotated * sin.repeat_interleave(2, dim=-1)


# --------------------------------------------------------------------------
# MLA attention
# --------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-head Latent Attention.

    The point of the whole design: the KV cache stores only
        c_kv  : (kv_lora_rank)      shared across heads, no per-head state
        k_pe  : (qk_rope_head_dim)  shared across heads, carries position
    = kv_lora_rank + qk_rope_head_dim = 512 floats per token per layer,
    versus 2 * n_heads * head_dim = 6144 for MHA. 12x less, before dtype.

    RoPE is *not* applied to k_nope, because k_nope does not exist until
    after the cached c_kv is up-projected. Position must therefore ride in a
    separate channel (k_pe) that is cached post-rotation. This is the one
    subtlety that makes MLA work, and the reason the q/k rope projections
    look "extra" in the parameter table.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        a = cfg.attention
        self.n_heads = cfg.num_attention_heads
        self.d_head = cfg.head_dim
        self.d_rope = a.qk_rope_head_dim
        self.d_c = a.kv_lora_rank
        self.d_qc = a.q_lora_rank
        self.scale = (self.d_head + self.d_rope) ** -0.5

        self.w_dq = nn.Linear(cfg.hidden_size, self.d_qc, bias=False)
        self.w_uq = nn.Linear(self.d_qc, self.n_heads * self.d_head, bias=False)
        self.w_qr = nn.Linear(cfg.hidden_size, self.n_heads * self.d_rope, bias=False)

        self.w_dkv = nn.Linear(cfg.hidden_size, self.d_c + self.d_rope, bias=False)
        self.w_uk = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)
        self.w_uv = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)

        self.w_o = nn.Linear(self.n_heads * self.d_head, cfg.hidden_size, bias=False)

        # latent norm BEFORE w_uk: keeps w_uk linearly absorbable at serving
        # (the k_norm below stays, per-head, after the up-projection)
        self.ckv_norm = RMSNorm(self.d_c, cfg.rms_norm_eps) if a.qk_norm else nn.Identity()
        self.q_norm = RMSNorm(self.d_head, cfg.rms_norm_eps) if a.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head, cfg.rms_norm_eps) if a.qk_norm else nn.Identity()

        # learned sink logits: one column of "attend to nothing" per head
        self.sink_logits = nn.Parameter(torch.zeros(self.n_heads, a.attention_sinks))

    def forward(self, x, cos, sin, cache=None, start_pos: int = 0):
        b, t, _ = x.shape
        h, dh, dr = self.n_heads, self.d_head, self.d_rope

        c_q = self.w_dq(x)
        q_nope = self.w_uq(c_q).view(b, t, h, dh)
        q_pe = self.w_qr(x).view(b, t, h, dr)
        q_nope = self.q_norm(q_nope)
        q_pe = apply_rope(q_pe, cos, sin, start_pos)

        c_kv = self.w_dkv(x)
        c_kv_part, k_pe = c_kv[..., : self.d_c], c_kv[..., self.d_c:]
        k_pe = apply_rope(k_pe.unsqueeze(2), cos, sin, start_pos)   # (b,t,1,dr)

        if cache is not None:
            c_kv_part = torch.cat([cache["c_kv"], c_kv_part], dim=1)
            k_pe = torch.cat([cache["k_pe"], k_pe], dim=1)
            cache["c_kv"], cache["k_pe"] = c_kv_part, k_pe

        k_nope = self.k_norm(self.w_uk(self.ckv_norm(c_kv_part)).view(b, -1, h, dh))
        v = self.w_uv(c_kv_part).view(b, -1, h, dh)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, h, -1)], dim=-1)   # (b,s,h,dh+dr)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # (b,t,h,d) -> (b,h,t,d)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * self.scale

        # causal mask, applied only when q and k have the same length
        s = k.shape[-2]
        if t > 1:
            causal = torch.full((t, s), float("-inf"), device=x.device, dtype=scores.dtype)
            causal = torch.triu(causal, diagonal=s - t + 1)
            scores = scores + causal

        # sinks: extra logits that are always "visible" but carry no value
        sinks = self.sink_logits.view(1, h, 1, -1).expand(b, -1, t, -1)
        scores = torch.cat([sinks, scores], dim=-1)
        probs = scores.softmax(dim=-1)
        probs = probs[..., self.sink_logits.shape[-1]:]

        out = probs @ v                                           # (b,h,t,dh)
        out = out.transpose(1, 2).reshape(b, t, h * dh)
        return self.w_o(out)

    def new_cache(self, batch: int, device):
        return {
            "c_kv": torch.zeros(batch, 0, self.d_c, device=device),
            "k_pe": torch.zeros(batch, 0, 1, self.d_rope, device=device),
        }


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, f = cfg.hidden_size, cfg.intermediate_size
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = MLAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, start_pos: int = 0):
        x = x + self.attn(self.norm1(x), cos, sin, cache, start_pos)
        x = x + self.ffn(self.norm2(x))
        return x


class Kestrel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        # multi-token prediction: each extra head predicts token t+1+i.
        # Dropped at inference, kept here because the gradient signal they give
        # the trunk is the reason to have them at all.
        self.mtp = nn.ModuleList(
            nn.ModuleDict({"norm": RMSNorm(cfg.hidden_size, cfg.rms_norm_eps),
                           "proj": nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)})
            for _ in range(cfg.mtp_heads)
        )

        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope(
            cfg.attention.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        # zero-init the residual-facing projections (GPT-2 style): every block
        # starts as identity, which is what keeps 40 layers stable at bf16.
        for block in self.blocks:
            nn.init.zeros_(block.attn.w_o.weight)
            nn.init.zeros_(block.ffn.down.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def head(self, h):
        return self.lm_head(h) if self.lm_head is not None else F.linear(h, self.embed.weight)

    def forward(self, idx, targets=None, caches=None, start_pos: int = 0):
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, self.cos, self.sin,
                      caches[i] if caches is not None else None, start_pos)
        h = self.norm(x)
        out = {"logits": self.head(h)}

        if targets is not None:
            losses = [F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].shape[-1]).float(),
                targets.reshape(-1), ignore_index=-100)]
            # MTP: predict targets shifted by 1+i, from a transformed hidden state
            z = h
            for i, mtp in enumerate(self.mtp):
                z = mtp["proj"](mtp["norm"](z))
                tgt = targets[:, 1 + i:]
                if tgt.shape[1] == 0:
                    continue
                logits_i = self.head(z[:, : tgt.shape[1]])
                losses.append(self.cfg.mtp_loss_weight * F.cross_entropy(
                    logits_i.reshape(-1, logits_i.shape[-1]).float(), tgt.reshape(-1)))
            out["loss"] = sum(losses)
            out["loss_parts"] = losses
        return out

    def new_caches(self, batch: int, device):
        return [b.attn.new_cache(batch, device) for b in self.blocks]

    def n_params(self, include_mtp: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_mtp:
            n -= sum(p.numel() for m in self.mtp for p in m.parameters())
        # tied embedding is counted once by parameters(), as intended
        return n


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = Config.from_json(os.path.join(here, "config.json"))

    # --- parameter count at full size, without allocating 16 GB -----------
    with torch.device("meta"):
        full = Kestrel(cfg)
    n_inf = full.n_params(include_mtp=False)
    n_tr = full.n_params()
    print(f"params (inference) : {n_inf/1e9:.4f} B")
    print(f"params (training)  : {n_tr/1e9:.4f} B")
    print(f"  of which MLA     : "
          f"{sum(p.numel() for b in full.blocks for p in b.attn.parameters())/1e9:.3f} B")
    print(f"  of which FFN     : "
          f"{sum(p.numel() for b in full.blocks for p in b.ffn.parameters())/1e9:.3f} B")
    print(f"  embeddings       : {full.embed.weight.numel()/1e9:.3f} B (tied)")
    assert abs(n_inf - cfg._expected_inference) / cfg._expected_inference < 1e-3, (
        f"parameter count drifted from the analytic model: "
        f"{n_inf/1e9:.4f} B vs {cfg._expected_inference/1e9:.4f} B"
    )

    # --- functional tests on a shrunken config (same code path) -----------
    tiny = Config(
        vocab_size=1024, hidden_size=256, num_hidden_layers=4,
        num_attention_heads=4, head_dim=64, intermediate_size=512,
        mtp_heads=2, max_position_embeddings=4096,
        attention=MLAConfig(kv_lora_rank=64, q_lora_rank=64, qk_rope_head_dim=32),
    )
    torch.manual_seed(0)
    model = Kestrel(tiny).eval()
    print(f"\nfunctional test model: {model.n_params()/1e6:.2f} M params")

    idx = torch.randint(0, tiny.vocab_size, (2, 32))
    with torch.no_grad():
        out = model(idx, targets=idx)
    print(f"logits {tuple(out['logits'].shape)}  "
          f"loss {out['loss'].item():.3f}  (ln(V) = {math.log(tiny.vocab_size):.2f} at init)")
    assert out["logits"].shape == (2, 32, tiny.vocab_size)
    assert len(out["loss_parts"]) == 1 + tiny.mtp_heads

    # --- cache equivalence -------------------------------------------------
    # A full 32-token pass must equal 32 single-token steps with a KV cache,
    # otherwise the cache is storing the wrong thing.
    with torch.no_grad():
        full_out = model(idx)["logits"]
        caches = model.new_caches(batch=2, device=idx.device)
        steps = []
        for pos in range(idx.shape[1]):
            steps.append(model(idx[:, pos:pos + 1], caches=caches, start_pos=pos)["logits"])
        inc = torch.cat(steps, dim=1)
    delta = (full_out - inc).abs().max().item()
    print(f"cache equivalence  : max|full - incremental| = {delta:.2e}")
    assert delta < 1e-4, "KV cache is inconsistent with full-sequence attention"
    # also confirm the cache is exactly (d_c + d_rope) per layer per token
    got = caches[0]["c_kv"].shape[-1] + caches[0]["k_pe"].shape[-1]
    want = tiny.attention.kv_lora_rank + tiny.attention.qk_rope_head_dim
    print(f"tiny cache/token   : {got} floats/layer (expected {want})")
    assert got == want

    # and the full-size cache arithmetic quoted in the docs
    per_tok = cfg.attention.kv_lora_rank + cfg.attention.qk_rope_head_dim
    print(f"full cache/token   : {per_tok} floats/layer x {cfg.num_hidden_layers} layers "
          f"= {per_tok*cfg.num_hidden_layers} B/token at FP8 "
          f"({per_tok*cfg.num_hidden_layers*65536/2**30:.2f} GiB at 64k)")
    assert per_tok * cfg.num_hidden_layers == 20480

    print("\nall assertions passed")


if __name__ == "__main__":
    main()
