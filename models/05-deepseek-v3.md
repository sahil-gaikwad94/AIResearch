# 05 — DeepSeek-V3 (671B total / 37B active, 2024)

> **The architecture that defined 2025–2026.** Almost every open frontier model since has copied its parts: Kimi K2 is explicitly "DeepSeek-V3 plus more experts", Flint borrows sigmoid routing and the shared expert, Qwen3-Next borrows the fine-grained expert layout and MTP. DeepSeek-V3 is also the model that proved frontier-scale training could be done for single-digit millions of dollars.
>
> Its three inventions are independent and each is worth understanding on its own: **MLA** (compress the KV cache into a latent), **fine-grained MoE with aux-loss-free balancing** (256 experts, balanced without a penalty term), and **MTP** (predict two tokens per forward pass).

| Property | Value |
|---|---|
| Released | December 2024, DeepSeek (MIT licence for code; open weights) |
| Parameters | 671B total / **37B active per token** |
| Layers | 61: 3 dense + 58 MoE |
| Width | 7,168 |
| Attention | **MLA**, 128 heads, head dim 192 (128 no-position + 64 rotary) |
| KV cache | 576 values per token per layer → **68.6 KiB/token** total |
| Experts | 256 routed (width 2,048) + 1 shared, top-8, group-limited routing (8 groups, 4 allowed) |
| Extra head | 1 MTP module (predicts the token after next) |
| Numerics | FP8 training with 128×128 block scaling |
| Vocabulary | 129,280 |
| Context | 128,000 trained, 163,840 max positions via YaRN |
| Training | 14.8T tokens, 2.788M H800 GPU-hours |

---

## 1. The config, annotated

```json
{
  "hidden_size": 7168,
  "intermediate_size": 18432,
  "moe_intermediate_size": 2048,
  "num_hidden_layers": 61,
  "first_k_dense_replace": 3,
  "num_attention_heads": 128,
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 8,
  "n_group": 8,
  "topk_group": 4,
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "topk_method": "noaux_tc",
  "num_nextn_predict_layers": 1,
  "vocab_size": 129280,
  "rope_theta": 10000,
  "rope_scaling": {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096}
}
```

Everything here is either familiar (RMSNorm, SwiGLU, RoPE, GQA-style head sharing) or one of the three inventions. The unfamiliar fields:

- `q_lora_rank` / `kv_lora_rank` — MLA's compression dimensions: 1,536 for queries, 512 for the key/value latent.
- `qk_nope_head_dim: 128` / `qk_rope_head_dim: 64` — each head's query and key are split into a 128-dim **content** part and a 64-dim **position** part. This "decoupled RoPE" is essential to MLA (§3).
- `first_k_dense_replace: 3` — the first three layers use a normal dense MLP of width 18,432; routing starts at layer 3.
- `n_group: 8`, `topk_group: 4` — group-limited routing (§4).
- `routed_scaling_factor: 2.5` — multiplies the routed experts' output (§4b).
- `topk_method: "noaux_tc"` — "no auxiliary loss, top-k choice": the aux-loss-free balancer.
- `num_nextn_predict_layers: 1` — one extra prediction head, i.e. MTP with depth 1.
- `rope_scaling` — YaRN with factor 40 extends a 4,096-token training window to 163,840 positions.

---

## 2. One block, drawn

```
x (1, T, 7168)
│
├─ rmsnorm
├─ MLA attention:
│     q: x → q_a_proj (7168→1536) → norm → q_b_proj → 128 heads × 192
│     kv: x → kv_a_proj_with_mqa (7168→576) ─┬─ 512 latent c_kv → kv_b_proj → K_nope, V
│                                            └─ 64 k_rope (shared by all heads)
│     CACHE ONLY c_kv (512) + k_rope (64) = 576 numbers per layer per token
│  x = x + attn_out
│
├─ rmsnorm
├─ layer 0..2:  dense SwiGLU (7168 → 18432 → 7168)
│  layer 3..60:
│     scores s = sigmoid(x @ W_r)        (T, 256)
│     select: s + b   (b = learned bias, NOT a gradient)  → groups → top-8
│     weights: s[top8] renormalised × 2.5
│     out = Σ wᵢ·Expertᵢ(x)  +  SharedExpert(x)
│  x = x + out
▼
```

---

## 3. Invention one: MLA, or how to cut the KV cache 57×

### The problem, in numbers

With plain multi-head attention at DeepSeek's width, each of 128 heads stores a 192-dim key and a 128-dim value, per token, per layer:

```
61 layers × 128 heads × (192 + 128) dims × 2 bytes ≈ 4.0 MB per token
```

At 128k context that is **512 GiB of KV cache for a single sequence.** Not slow — impossible. This single number is why long context was expensive for everyone in 2024, and why GQA (file 02) was the first response: cut the number of KV heads. But GQA is blunt; it forces groups of query heads to share identical K/V, whether or not they want to.

### The MLA idea

Don't store K and V. Store a **compressed latent** that both can be reconstructed from, and reconstruct on the fly.

```
c_kv   = x @ W_DKV              (7168 → 512)      ← this is what you cache
k_nope, v = c_kv @ W_UK, W_UV   (512 → per-head)  ← reconstructed at attention time
k_rope = x @ W_KR               (7168 → 64)       ← shared across heads, cached too
```

So the cache holds **512 + 64 = 576 numbers** per token per layer instead of 40,960:

```
61 × 576 × 2 bytes = 70,272 bytes ≈ 68.6 KiB per token
```

At 128k: **8.6 GiB instead of 512 GiB — a 57× reduction.** That is the whole point of MLA, and it is the largest single memory win in this folder.

### The trick that makes it free at inference: absorption

Reconstructing per-head K and V at every step would be expensive. But matrix multiplication is associative: instead of `(c_kv @ W_UK)` then `q · k`, you can precompute `q @ W_UKᵀ` and multiply the latent directly. The up-projection is **absorbed** into the query and output projections, so at inference the model attends *in the 512-dim latent space* and never materialises the per-head keys and values. Training keeps the explicit form (it is better conditioned); inference uses the absorbed form. Same maths, different cost profile.

This is why MLA is elegant: the compression is not an approximation, it is an exact re-parameterisation. Nothing is discarded — the 671B model's attention is mathematically identical to a wide MHA model, just organised so the cache is small.

### Why RoPE had to be decoupled

Here is the subtle part, and the reason for `qk_rope_head_dim`.

If you rotate the *latent* with RoPE, then the absorbed form breaks: the rotation depends on position, so it cannot be folded into a position-independent matrix. The fix is to split each query/key into two parts:

- a **content** part (128 dims) carried through the latent — no rotation, absorbable;
- a **position** part (64 dims) rotated with RoPE, shared across all heads, cached separately.

The head's score is the sum of the two dot products. Content matching is compressed and cheap; position is exact and small. This "decoupled RoPE" is the piece that made latent attention compatible with rotary positions, and it appears in every MLA model since (Kimi K2 included).

**Cost of the design:** MLA is genuinely harder to implement than GQA, needs custom kernels, and the absorption has to be done carefully to avoid numerical drift. Kimi K2 adopted it wholesale; Flint did not, choosing GQA + windows + sinks instead (file 04). Both are defensible; MLA wins on cache size, GQA wins on simplicity.

---

## 4. Invention two: fine-grained experts, balanced without a loss

### 4a. 256 narrow experts instead of 8 wide ones

Mixtral: 8 experts × 14,336 wide (176M each). DeepSeek: 256 experts × 2,048 wide (44M each), top-8, **plus one shared expert of width 2,048**.

Why finer is better at equal active compute: a token that needs "Python syntax + a bit of maths + some JSON formatting" can select three narrow experts that each do one thing, rather than two wide generalists that must each do everything. The combinatorial number of reachable expert *sets* is enormous — C(256, 8) ≈ 4×10¹⁴ — so capacity grows without any single expert needing to be broad.

The arithmetic is striking: 8 active experts × 44M = 352M of routed FFN per token, almost exactly the 2 × 176M = 352M Mixtral activates. **Same compute per token, 16× more stored capacity.**

### Group-limited routing

256 experts cannot all live on one GPU. With expert parallelism they are spread across nodes, and a token that selects experts on four different nodes forces four all-to-all round trips. DeepSeek bounds this: the 256 experts are divided into **8 groups of 32**; the router first picks the **top 4 groups** (by the sum of their two best expert scores), then chooses 8 experts from within those. Worst case: at most 4 nodes are involved.

This is a pure systems constraint expressed in the architecture — the kind of thing you only learn by reading the config rather than the abstract.

### 4b. Aux-loss-free balancing

Mixtral's aux loss (file 03) balances the router by *penalising* imbalance in the training objective, which fights the router's actual job. DeepSeek's fix:

```
score for selection:   sᵢ + bᵢ        bᵢ is a per-expert bias
score for weighting:   sᵢ             bᵢ is NOT used here
after each step:       bᵢ ← bᵢ + γ · sign(target − loadᵢ)
```

`b` is updated by a rule, not by a gradient. It never touches the loss, so it cannot distort the language-model objective; it only nudges *which* experts get chosen. If an expert is overloaded, its bias decreases; if starved, it increases. Over training, the load equalises while the router's preference signal (`s`) stays pure.

`routed_scaling_factor: 2.5` compensates for a second-order effect: sigmoid scores renormalised over top-8 produce a different typical magnitude than softmax weights did, so the output is multiplied by 2.5 to land back in the range the rest of the network expects. A one-number reminder that every routing scheme needs its output scale calibrated.

The result: DeepSeek-V3 trained 14.8T tokens with no auxiliary-loss gradient, no token dropping, and balanced load. By 2026 this is the default; Mixtral-style aux losses look crude in comparison.

---

## 5. Invention three: MTP (multi-token prediction)

Every model so far predicts one token per forward pass: `h_t → token t+1`. MTP adds a small module that predicts `token t+2` from `h_t` plus the embedding of the *actual* `t+1`:

```
h'_t   = MTP_block( concat(rmsnorm(h_t), rmsnorm(embed(token_{t+1}))) )
logits2 = h'_t @ embedᵀ        → predicts token t+2
loss    = CE(token_{t+1}) + λ · CE(token_{t+2})
```

Two distinct benefits, and they are often conflated:

1. **Training signal.** Forcing the model to plan two tokens ahead gives a denser gradient — the representation at position `t` must encode enough to constrain `t+2`, not just `t+1`. DeepSeek reports this improves the main model, not just the extra head.
2. **Inference speed.** The MTP head is a free draft model. Generate a candidate `t+2` cheaply, then verify both tokens in one forward pass — classic speculative decoding, but with the draft model built into the checkpoint. DeepSeek reports >1.8× generation throughput.

The cost is one extra transformer block plus projections (a small fraction of 671B), and a slightly more complex training loop. It is now standard: Qwen3-Next and Kimi K2 both ship MTP heads.

---

## 6. Count the parameters yourself

| Component | Shape | Parameters |
|---|---|---|
| MLA attention (q_a, q_b, kv_a, kv_b, o) | see §3 | **187.1M** |
| dense MLP (layers 0–2) | 3 × 7168 × 18432 | 396.4M each |
| one routed expert | 3 × 7168 × 2048 | 44.04M |
| 256 experts | 256 × 44.04M | 11,274M |
| shared expert | 3 × 7168 × 2048 | 44.04M |
| **one MoE layer (total)** | | **11,505M** |
| **one MoE layer (active)** | attn + shared + 8 experts | **583.4M** |

- 3 dense layers × (187.1 + 396.4) = 1.75B
- 58 MoE layers × 11.505B = 667.3B
- embeddings + head (untied): 2 × 129,280 × 7,168 = 1.853B
- **total = 670.9B** ✓ (671B)

Active: 58 × 583.4M + 1.75B + output head 927M = **36.5B** ✓ (37B, with the MTP module and norms rounding it up)

Notice what the ratios say:

- **Attention is 1.7% of the model** (187M × 61 = 11.4B of 671B) but 32% of the *active* count (187M of 583M). Compression made it small in memory and relatively large in compute.
- **The shared expert is 0.4% of total, 7.5% of active.** Cheap insurance against routing gaps.
- **Sparsity is 18×** (671/37), 5× more than Mixtral.

---

## 7. What it cost

DeepSeek's own accounting: 2.788M H800 GPU-hours for the final pretraining run, on 14.8T tokens. At ~$2 per H800-hour that is **≈$5.6M**.

Read that number carefully. It is:
- **the compute for one successful run**, not the R&D, the data pipeline, the ablations, or the failed attempts;
- at H800 prices available to a Chinese lab in 2024;
- for a model that was, at release, competitive with models costing an order of magnitude more to train.

The lesson people draw ("frontier models are cheap") is wrong; the lesson that is right: **the *algorithmic* efficiency gains — FP8 training, MLA, aux-loss-free MoE, MTP, better data — were worth more than the compute gap.** That is the actual content of the paper.

---

## 8. Reverse-engineering lab

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
print(cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_experts_per_tok,
      cfg.kv_lora_rank, cfg.qk_rope_head_dim)     # 61 256 8 512 64

# 1. Total vs active
d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
h, hd = cfg.num_attention_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
qk_n, qk_r, vh = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
kvl, qvl = cfg.kv_lora_rank, cfg.q_lora_rank
attn = (d*qvl) + qvl*(h*(qk_n+qk_r)) + d*(kvl+qk_r) + kvl*(h*(qk_n+vh)) + (h*vh)*d
expert = 3*d*cfg.moe_intermediate_size
dense  = 3*d*cfg.intermediate_size
total  = 3*(attn+dense) + 58*(attn + 256*expert + expert) + 2*V*d
active = 3*(attn+dense) + 58*(attn + 9*expert) + V*d
print(f"{total/1e9:.0f}B total, {active/1e9:.0f}B active")     # 671B, 37B

# 2. MLA cache vs the MHA it replaces  <- the headline number
mla = L*(kvl+qk_r)*2
mha = L*h*(qk_n+vh)*2
print(f"{mla/1024:.1f} KiB/token vs {mha/2**20:.1f} MiB/token  ({mha/mla:.0f}x smaller)")

# 3. The tensor names that give the design away
#   model.layers.3.self_attn.kv_a_proj_with_mqa   (576, 7168)   <- 512 latent + 64 rope
#   model.layers.3.self_attn.kv_b_proj            (32768, 512)  <- reconstructs K_nope and V
#   model.layers.3.self_attn.q_a_proj / q_b_proj  (1536,) (24576,) <- query compression
#   model.layers.3.mlp.gate.e_score_correction_bias (256,)      <- the aux-loss-free bias
#   model.layers.3.mlp.shared_experts.*                          <- always-on expert
```

`kv_a_proj_with_mqa` is the single most recognisable tensor name in modern open models: 576 = 512 + 64 tells you MLA with decoupled RoPE, before you read a line of the paper. And `e_score_correction_bias` with no corresponding gradient is the signature of aux-loss-free routing.

---

## 9. What DeepSeek-V3 still gets wrong

| Limitation | Detail | Fixed by |
|---|---|---|
| **Full attention at all 61 layers** | 68.6 KiB/token is 57× better than MHA but still 8.6 GiB at 128k. | V3.2-Exp: **DSA** (sparse attention, top-2,048 tokens) — Sept 2025, with a >50% API price cut |
| **8 of 256 is coarse** | Sparsity 18×; Qwen3-Next pushes to 51×. | Qwen3-Next (06) |
| **MLA complexity** | Custom kernels, absorption subtleties, harder to quantise. | Adopted anyway by Kimi K2; avoided by Flint |
| **MTP depth 1** | Only one extra token predicted; deeper MTP is mostly unexplored at scale. | Open question |
| **FP8 fragility** | Block-scaled FP8 needs careful accumulation; naive ports degrade quality. | Still an engineering hazard |

**The V3.2-Exp sequel is worth knowing because it shows the pattern:** same 671B skeleton, same experts, but attention replaced with DeepSeek Sparse Attention — a tiny "lightning indexer" scores every token and the layer attends to only the top 2,048. Cost per 1M tokens dropped by more than half at release. The architecture did not get bigger; it got *sparser in attention*, which is exactly the trend file 07 documents.

---

## 10. Self-check

1. Explain the 57× KV-cache reduction without using the word "compress". Where exactly are the bytes saved?
2. Why can't you just apply RoPE to the 512-dim latent and be done?
3. What does "absorption" let you avoid computing at inference, and why does the maths allow it?
4. 256 experts × 8 active = 352M of FFN per token, same as Mixtral's 2 × 176M. So what did DeepSeek actually gain?
5. The aux-loss-free bias is updated with `sign()`, not a gradient. Why does that matter for the language-modelling loss?
6. Group-limited routing (8 groups, top-4) has nothing to do with model quality. What is it for?
7. MTP helps training *and* inference. Give the mechanism for each.

*(Answers: 1 — you cache one 512-vector plus a 64-vector per layer instead of 128 separate key and value vectors; the per-head K/V are rebuilt from the latent when needed. 2 — rotation depends on position, so it cannot be folded into a position-independent projection, and absorption breaks; hence a small separate rotary part. 3 — materialising per-head K and V: you fold the up-projection into the query and output matrices and attend in latent space. 4 — 16× more stored capacity at equal compute, i.e. more knowledge per FLOP, plus finer specialisation. 5 — a bias updated outside the gradient cannot distort the LM objective; it only changes selection. 6 — it bounds the number of all-to-all network hops to at most 4 nodes per token. 7 — training: denser gradient, the state must constrain t+2; inference: the MTP head is a built-in draft model for speculative decoding.)*

---

**Next:** [06 — Qwen3-Next 80B-A3B](06-qwen3-next.md), which removes most of the attention entirely.
