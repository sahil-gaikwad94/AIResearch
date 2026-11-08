# 02 — Llama 3 8B (2024)

> **The modern dense layer.** Between 2019 and 2024, every component of the GPT-2 block was replaced. Llama 3 is the cleanest record of the result: RMSNorm instead of LayerNorm, RoPE instead of learned positions, SwiGLU instead of GELU, GQA instead of full multi-head attention. If you memorise one architecture, memorise this one — it is still the base of most open models in 2026, and every chapter after this is a modification of it.

| Property | Value |
|---|---|
| Released | April 2024, Meta (Llama 3); 3.1 update July 2024 |
| Parameters | 8,030,261,248 (8.03B), all active |
| Layers | 32 |
| Width | 4,096 |
| Attention | 32 query heads, 8 key/value heads (GQA, 4:1), head dim 128 |
| Feed-forward | SwiGLU, width 14,336 |
| Vocabulary | 128,256 |
| Context | 8,192 (Llama 3); 131,072 (Llama 3.1, same architecture + RoPE scaling) |
| Training data | ~15 trillion tokens |
| License | Meta community licence (gated download) |

The key structural fact: **nothing here is a mixture of experts.** Every token uses the same 8.03B parameters. That is why it is the right second model — you can hold the entire computation in your head before sparsity makes it conditional.

---

## 1. The config, annotated

```json
{
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 128256,
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "hidden_act": "silu",
  "attention_bias": false,
  "mlp_bias": false,
  "tie_word_embeddings": false
}
```

Compare to GPT-2's ten lines and notice what is *absent*: there is no `n_positions` table (positions are computed, not stored), no `scale_attn_weights` flag (always on), no bias flags set true anywhere. Two fields do most of the work of this chapter: `num_key_value_heads: 8` and `rope_theta: 500000`.

---

## 2. The five replacements

| GPT-2 | Llama 3 | What changes |
|---|---|---|
| LayerNorm (mean + var, γ and β) | **RMSNorm** | Delete the mean subtraction and the β shift |
| Learned position table | **RoPE** | No table; rotate q and k by an angle proportional to position |
| GELU MLP, 4× width | **SwiGLU**, 3.5× width | Three matrices with a multiplicative gate |
| 12 heads, each with own K and V | **GQA**, 32 q / 8 kv | Four query heads share one K/V pair |
| 50,257 vocab, tied embeddings | 128,256 vocab, untied | 2.5× more tokens, separate output matrix |

Everything else — attention as a dot-product lookup, the residual stream, pre-norm, the causal mask, next-token prediction — is unchanged from GPT-2.

---

## 3. RMSNorm

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ        (no mean subtraction, no β)
```

Divide each token's 4,096-vector by its root-mean-square, then scale by a learned vector. That is it. No centring, no shift.

Why it works: LayerNorm's mean subtraction and β shift turn out to be unnecessary — the network can represent any shift it needs inside the following linear layer anyway. Removing them means one fewer reduction over the feature dimension and one fewer parameter vector per norm, which at 32 layers × 2 norms is real time on a GPU. The paper that introduced it (Zhang & Sennrich, 2019) reported ~7% faster training with equal or better quality, and empirically it also slightly improves stability because the norm cannot blow up through the re-centring path.

The residual stream in an RMSNorm model is *not* zero-mean. Interpretability work that assumes centred activations has to account for this.

---

## 4. RoPE: positions as rotation

This is the most important idea in the file, and the one most often memorised without understanding.

**The problem.** Attention is permutation-invariant: `q·k` has no idea where either token sits. GPT-2 fixed that by adding a learned vector to each position's embedding — cheap, but the table has a fixed number of rows and, worse, the model has to *learn* what "position 700" means.

**The trick.** Instead of adding something, **rotate** the query and key vectors by an angle proportional to their position. Take the 128-dimensional head vector and view it as 64 two-dimensional pairs. For pair `i`, rotate by angle `m · θᵢ`, where `m` is the token's position and `θᵢ = base^(-2i/128)`.

```
q' = R(mθ) q          k' = R(nθ) k

q'·k' = qᵀ R(mθ)ᵀ R(nθ) k = qᵀ R((n−m)θ) k
```

The rotation matrices compose, so the absolute positions cancel and only the **difference** `n − m` survives. That is the whole point: relative position falls out of the algebra for free, and the model never has to learn absolute offsets.

In code it is applied without building matrices:

```python
q_rot = q * cos(m * theta) + rotate_half(q) * sin(m * theta)
# rotate_half([x1, x2, x3, x4]) = [-x3, -x4, x1, x2]
```

**Why `rope_theta = 500000` instead of the usual 10,000.** `θᵢ` sets the wavelength of each pair. With base 10,000, the fastest pair completes a full turn roughly every **6 tokens** and the slowest every **~54,000 tokens**. With base 500,000, the slowest pair's wavelength stretches to roughly **2.5 million tokens**. Longer wavelengths mean distant positions stay distinguishable instead of wrapping around into ambiguity, which is what long-context extension needs. The cost: those slow dimensions rotate so little during an 8k-token training run that they are barely trained — which is exactly why extending Llama 3 from 8k to 128k (Llama 3.1) needed a *scaled* RoPE, not just a bigger base.

**Consequence for you.** Context length is no longer a table size. It is a *training* decision plus a *scaling* trick (`rope_scaling` in the 3.1 config: factor 8.0, `rope_type: llama3`). The same weights can be stretched to a longer window if the position frequencies are remapped; quality degrades gracefully rather than crashing at a hard wall.

---

## 5. SwiGLU

```
SwiGLU(x) = ( SiLU(x W_gate) ⊙ (x W_up) ) W_down
SiLU(z)   = z · sigmoid(z)
```

Three matrices instead of two, and the gate multiplies the value elementwise. The intuition: a plain MLP computes `f(xW₁)W₂`, so every hidden unit's contribution is additive. A gate lets the network *decide how much of each feature to let through* as a function of the token itself — a multiplicative interaction, which is strictly more expressive per unit.

To keep the parameter count fair, you shrink the hidden width. A plain MLP with 4× width costs `2·d·4d = 8d²`. A SwiGLU with width `w` costs `3·d·w`. Setting them equal gives `w = 8d/3 ≈ 2.67d`.

Meta did not do that. Llama 2 used `11008 ≈ 2.69d` (matched). **Llama 3 uses `14336 = 3.5d`** — 31% *more* FFN parameters than the matched size. This is a deliberate, checkable design choice: Llama 3 moved parameters out of attention and into the feed-forward, and the MLP is now 176.2M of the 218.1M parameters in a block (81%).

---

## 6. GQA: the KV cache is the real constraint

Here is the fact that reorganises how you think about inference.

When generating token `t`, the model needs the keys and values of all previous tokens. Recomputing them every step would be quadratic waste, so they are cached — the **KV cache**. Its size, per token, per layer, in BF16:

```
bytes = 2 (K and V) × n_kv_heads × head_dim × 2 bytes
```

With plain multi-head attention (32 KV heads): `2 × 32 × 128 × 2 = 16,384 bytes = 16 KiB` per layer, **512 KiB per token** across 32 layers. An 8k-token conversation would need 4 GiB of cache — more than the weights of a 1B model, for one user.

**Grouped-query attention** makes 8 KV heads serve 32 query heads: each K/V pair is shared by 4 queries. The cache drops 4×, to **4 KiB per layer = 128 KiB per token**. At 8k context that is 1 GiB, which fits on a consumer GPU. Quality loss is small — much smaller than multi-query attention (1 KV head), which was the earlier aggressive version.

| Scheme | KV heads | Cache/token (32 layers, dim 128) | At 8k |
|---|---|---|---|
| MHA | 32 | 512 KiB | 4.0 GiB |
| **GQA (Llama 3)** | 8 | **128 KiB** | **1.0 GiB** |
| MQA | 1 | 16 KiB | 128 MiB |

Note what did *not* change: the parameter count barely moves (k_proj and v_proj shrink from 16.8M to 4.2M each), and the FLOPs barely move. GQA is almost purely a **memory-bandwidth** optimisation. This distinction — parameters vs FLOPs vs bytes moved — is the lens for the rest of the folder, because every later model trades among these three.

---

## 7. Count the parameters yourself

| Component | Shape | Parameters |
|---|---|---|
| `q_proj` | 4096 × 4096 | 16.78M |
| `k_proj` | 4096 × 1024 | 4.19M |
| `v_proj` | 4096 × 1024 | 4.19M |
| `o_proj` | 4096 × 4096 | 16.78M |
| `gate_proj` | 4096 × 14336 | 58.72M |
| `up_proj` | 4096 × 14336 | 58.72M |
| `down_proj` | 14336 × 4096 | 58.72M |
| 2 × RMSNorm | 4096 each | 8,192 |
| **one block** | | **218.1M** |

- 32 blocks × 218.1M = 6.98B
- embeddings 128,256 × 4096 = 525.3M, **counted twice** (untied input and output) = 1.05B
- **total = 8.03B** ✓

Two things to carry forward:

1. **The MLP is 81% of each block.** Attention is 41.9M. In every dense model, the FFN dominates.
2. **The embedding is now 13% of the model** (1.05B of 8.03B), up from 31% in GPT-2 — not because it grew relative to the model, but because the model grew faster than the vocabulary. Note also that Llama 3 *unties* embeddings, paying 525M extra. At 8B scale that is affordable; the usual rationale is that input and output representations want different geometry, and tying costs quality at scale.

**Training data.** 15T tokens ÷ 8.03B parameters ≈ **1,870 tokens per parameter**. Chinchilla-optimal (2022) says 20. That is a ~90× overrun, and it is not a mistake — it is the whole post-Chinchilla shift: when inference cost dominates lifetime cost, you train smaller models much longer (chapter 4 of the main book has the arithmetic).

---

## 8. The forward pass, with shapes

Batch 1, sequence length T = 8192, BF16.

```
input_ids                       (1, 8192)              int64
embed_tokens                    (1, 8192, 4096)
  ── per block ×32 ──
  h = rmsnorm(x)                (1, 8192, 4096)
  q = h @ Wq                    (1, 8192, 4096)  → (1, 32, 8192, 128)
  k = h @ Wk                    (1, 8192, 1024)  → (1, 8, 8192, 128)  → repeat 4× → 32 heads
  v = h @ Wv                    (1, 8192, 1024)  → (1, 8, 8192, 128)  → repeat 4× → 32 heads
  scores = q @ kᵀ / √128        (1, 32, 8192, 8192)   ← 2.1B numbers, 4.3 GB in BF16
  ... mask, softmax, @ v        (1, 32, 8192, 128)
  o = merge heads @ Wo          (1, 8192, 4096)
  x = x + o
  h = rmsnorm(x)
  g = silu(h @ Wgate) * (h @ Wup)   (1, 8192, 14336)
  x = x + g @ Wdown             (1, 8192, 4096)
final rmsnorm                   (1, 8192, 4096)
lm_head @ embedᵀ                (1, 8192, 128256)      ← 1.05B numbers
```

That `(1, 32, 8192, 8192)` scores tensor is why FlashAttention exists: materialising it costs 4.3 GB, so real kernels never write it to memory — they compute the softmax in tiles and accumulate. The maths is identical; the memory traffic is not. (Chapter 2 of the main book covers this; all you need here is that the *shape* is quadratic and the *implementation* avoids paying for it.)

---

## 9. Reverse-engineering lab

```python
from transformers import AutoModelForCausalLM, AutoConfig
import torch

# meta-llama/Meta-Llama-3-8B is gated; this mirror is identical and open
mid = "NousResearch/Meta-Llama-3-8B"
cfg = AutoConfig.from_pretrained(mid)
print(cfg.hidden_size, cfg.num_hidden_layers, cfg.num_key_value_heads,
      cfg.num_attention_heads, cfg.rope_theta, cfg.intermediate_size)

# 1. Verify 8.03B from the config alone
d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
ff, kv = cfg.intermediate_size, cfg.num_key_value_heads
hd = d // cfg.num_attention_heads
attn = d*d + 2*d*kv*hd + d*d
mlp  = 3*d*ff
total = L*(attn+mlp) + 2*V*d
print(f"{total/1e9:.2f}B")                     # 8.03B  ✓

# 2. KV cache, the number that decides your GPU
per_tok = 2 * L * kv * hd * 2                  # bytes, BF16
print(f"{per_tok/1024:.0f} KiB/token -> 8k: {per_tok*8192/2**30:.2f} GiB")
                                               # 128 KiB/token -> 1.00 GiB

# 3. Load and inspect one block's tensors
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16)
blk = model.model.layers[0]
for n, p in blk.named_parameters():
    print(f"{n:28s} {tuple(p.shape)}")
# self_attn.q_proj.weight  (4096, 4096)
# self_attn.k_proj.weight  (1024, 4096)     <- 8 heads, not 32
# self_attn.v_proj.weight  (1024, 4096)
# self_attn.o_proj.weight  (4096, 4096)
# mlp.gate_proj.weight     (14336, 4096)
# mlp.up_proj.weight       (14336, 4096)
# mlp.down_proj.weight     (4096, 14336)
# input_layernorm.weight   (4096,)          <- no bias anywhere

# 4. Watch the residual stream grow (pre-norm signature)
x = model.model.embed_tokens(torch.tensor([[1, 100, 200, 300]]))
norms = []
for layer in model.model.layers:
    x = layer(x, use_cache=False)[0] if False else layer(x)[0]
    norms.append(round(x.norm(dim=-1).mean().item(), 1))
print(norms)   # monotonically increasing — each block ADDS to the stream
```

Step 4 is worth doing once. In a pre-norm model the residual stream's norm climbs steadily with depth, because every sublayer adds its output. It is visual proof that the model is not rewriting a representation but accumulating one.

---

## 10. What Llama 3 still does the old way

The 8B is dense: every token multiplies all 8.03B weights. At batch 1 that is fine, but it means **compute per token is fixed and large**. You cannot get a 400B model's knowledge with 4B model's speed.

The remaining constraints, and who removes them:

| Constraint | Cost | Removed by |
|---|---|---|
| Dense MLP | every token pays for all knowledge | MoE — Mixtral (03), Flint (04), DeepSeek (05), Qwen3-Next (06) |
| Full attention at every layer | quadratic compute; 128 KiB/token cache | sliding windows (03, 04), latent compression (05), linear attention (06) |
| 128 KiB/token KV cache | 16 GiB at 128k context for one sequence | MLA → 68.6 KiB (05); hybrid linear → 12 KiB (06) |
| Fixed 4:1 GQA ratio | a blunt instrument: 4 or 8 share, no adaptivity | MLA learns the compression (05) |
| One token predicted per forward pass | decode is memory-bound and slow | MTP heads — DeepSeek (05), Qwen3-Next (06) |

---

## 11. Self-check

1. Why does RMSNorm not need a bias term, when LayerNorm has one?
2. Derive from `qᵀR(mθ)ᵀR(nθ)k` why RoPE gives relative position. What would break if you rotated only `q`?
3. A SwiGLU MLP at width `w = 8d/3` matches a plain MLP at `4d` in parameters. Llama 3 uses `3.5d`. How many extra parameters does that add at d = 4096, L = 32, and what else in the block shrank to pay for it?
4. Your GPU has 24 GB. Weights take 16 GB. How many 8k-token sequences can you serve concurrently with Llama 3 8B, ignoring activation memory? (Answer: ~7 — and that number, not the parameter count, is what caps your throughput.)
5. Llama 3 8B and Llama 3.1 8B have the same config except `max_position_embeddings` and `rope_scaling`. Why can the same weights handle 16× more context?
6. Why is the FFN 81% of the block rather than the 50% you might guess from "attention is all you need"?

*(Answers: 2 — only the difference survives because rotations compose about the same axis; rotating only q leaves an absolute reference. 3 — `3·d·(3.5d − 2.667d) = 3·4096·3413 ≈ 42M` per layer, ~1.3B total, paid for by shrinking k/v projections and by accepting a larger model. 5 — RoPE frequencies are remapped by a scaling factor, so positions beyond 8k fall into the trained frequency range instead of out of distribution. 6 — the hidden width is 3.5× the model width, and three matrices instead of two, so `3·3.5 = 10.5d²` of FFN against `4d²` of attention.)*

---

**Next:** [03 — Mixtral 8x7B](03-mixtral-8x7b.md), where the MLP stops being one MLP and the parameter count stops meaning what you think it means.
