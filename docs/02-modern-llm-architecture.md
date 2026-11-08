# 02 — The Modern LLM: Anatomy of a 2026 Model

*Read time: 75–90 minutes. Prerequisites: `01`.*

---

## 2.1 Why this chapter matters

If you read a 2023 blog post about Transformers and then look at a 2026 model card, you will recognise maybe half of it. The backbone is the same; nearly every *component* has been replaced by something cheaper. This chapter is the parts list, in the order information flows, with the reason each part exists and the numbers from real models.

The canonical modern stack (say DeepSeek-V3, Llama 4, Qwen3.5, GLM-5 class):

```
tokens
  → embedding
  → [ for each of L layers:
        RMSNorm → Attention(MoE-independent, GQA or MLA or hybrid) → residual
        RMSNorm → MoE(shared expert + routed experts) → residual
    ]
  → final RMSNorm → LM head (often + MTP head)
```

Below, each choice, why it exists, and what it costs.

---

## 2.2 Pre-norm and RMSNorm

**Problem:** deep networks with post-LayerNorm blow up. Gradients through 100+ layers of normalisation are unstable, and the original Transformer needed learning-rate warmup hacks that never fully fixed it.

**Solution:** move normalisation *before* each sublayer and use a simpler norm.

```
pre-norm:  x ← x + Sublayer(Norm(x))
RMSNorm(x) = γ ⊙ x / sqrt(mean(x²) + ε)
```

**Why RMSNorm instead of LayerNorm?** LayerNorm computes mean and variance; RMSNorm only variance (the root-mean-square). Empirically it works as well or better, and it's cheaper: fewer reductions, fewer parameters, better kernel fusion. Introduced by Zhang & Sennrich (2019), adopted by T5, Llama, and essentially everything since.

**Detail that matters:** modern models often apply **QK-norm** (normalising queries and keys per head) for training stability at long context, and some 2025–26 models apply RMSNorm to the *output* of attention in addition to the input ("sandwich norm"). DeepSeek-V3 and Qwen3 are examples.

---

## 2.3 SwiGLU: the gated MLP

**Original:** `FFN(x) = W₂ · ReLU(W₁x)`.

**Modern (from PaLM/LLaMA onward):**

```
SwiGLU(x) = W₂ · ( Swish(W_gate x) ⊙ (W_up x) )
Swish(z) = z · sigmoid(z)
```

Two parallel projections instead of one, multiplied elementwise (a **gate**). With three matrices instead of two, the hidden dimension is reduced to `≈ 8/3 · d_model` to keep parameter count equal, and the block still wins — typically 0.5–1% perplexity at equal parameters, consistently reproduced (Shazeer 2020, "GLU Variants Improve Transformer").

**Why gating helps:** multiplicative interactions give the network a cheap way to do conditional computation — the gate decides *whether* a feature passes, not just how much. It effectively doubles the expressiveness per parameter.

**Variants worth knowing:** GeGLU (GELU gate), ReGLU, and 2025–26 "fine-grained" MoE designs that shrink the expert hidden size and increase expert count instead (§2.7).

---

## 2.4 Positional information: RoPE and its scaling

Attention as defined in `01` is **permutation-invariant** — it has no idea about word order. So we must inject position.

### History in three steps

1. **Sinusoidal absolute encodings** (Vaswani 2017): add `sin/cos` of position at fixed frequencies to embeddings. Extrapolates poorly.
2. **Learned absolute embeddings** (GPT-2/BERT): a lookup table of length = max context. Cannot exceed the table.
3. **RoPE — Rotary Position Embedding** (Su et al. 2021, "RoFormer"): **rotate** the query and key vectors by an angle proportional to position, *inside* attention.

### How RoPE works

For a 2-D slice of the head dimension at position `m`, rotate by angle `mθ`:

```
[q'_1, q'_2] = [q_1 cos(mθ) − q_2 sin(mθ),  q_1 sin(mθ) + q_2 cos(mθ)]
```

The magic property: for two positions `m` and `n`, the dot product `q'_m · k'_n` depends only on the **relative** offset `m − n`. So attention becomes naturally relative, which is what you want ("the word two before me"), while still being computed with absolute rotations — a neat algebraic trick with no extra parameters.

Frequencies `θ_i = 10000^{−2i/d}` give a geometric spread: fast rotations for local order, slow rotations for global position.

### Extending context: the RoPE-scaling family

The problem: train at 4k, deploy at 128k, and performance collapses, because rotations at unseen positions are out-of-distribution.

| Method | Idea | Result |
|---|---|---|
| **Position Interpolation (PI)** | Compress positions: `m → m·(L_train/L_target)` | Simple, needs fine-tuning |
| **NTK-aware scaling** | Scale the *base* of the frequencies, not positions, changing the spread non-uniformly | Better zero-shot extension |
| **YaRN** (Peng et al. 2023) | Combine NTK-by-parts interpolation + attention temperature scaling; only interpolate the high-frequency dims, extrapolate low ones | ~10× extension with ~400 steps of fine-tuning |
| **LongRoPE** (Ding et al. 2024) | Non-uniform per-dimension rescaling found by search; progressive 256k → 2M extension | 2M tokens demonstrated |
| **LongRoPE2** (2025) | Diagnoses the residual OOD as *insufficient training* of high RoPE dimensions; evolutionary search + mixed-context training | 128k on Llama-3-8B with 10B tokens (80× less than Meta's approach), retaining 98.5% short-context performance |

**The honest summary (2026):** extending context is a *two-part* problem — (a) fix the position encoding, (b) keep training on long documents. Papers that only do (a) produce models that pass needle-in-a-haystack and fail real long-context reasoning. The trend in 2025–26 is **NoPE** (no positional embedding at all) in some layers of sparse/hybrid models, relying on attention's implicit order sensitivity plus data, which extrapolates unusually well.

---

## 2.5 The KV cache: the real cost of generation

During autoregressive decoding, at step `t` the model needs keys and values for all positions `1..t`. Recomputing them each step would be O(t²) per token. So we **cache** them: the **KV cache**.

Size:

```
bytes = 2 (K and V) × L (layers) × h_kv (kv heads) × d_head × seq_len × bytes_per_value
```

Worked example — Llama 3 70B, 80 layers, 8 KV heads (GQA), d_head 128, 8192 context, FP16:

```
2 × 80 × 8 × 128 × 8192 × 2 bytes = 2.1 GB per sequence
```

At 128k context that's ~34 GB *per user* — larger than the model's own weights in 4-bit. This single arithmetic fact drove the entire architectural agenda of 2023–2026:

### The four solutions

**(1) MQA — Multi-Query Attention** (Shazeer 2019). All query heads share *one* key/value head. KV cache shrinks by factor `h` (e.g. 32×). Quality drops slightly; training can be unstable.

**(2) GQA — Grouped-Query Attention** (Ainslie et al. 2023). Compromise: queries are grouped, each group shares a KV head. With `h=32` queries and `h_kv=8`, cache shrinks 4×. **This is the default in Llama 2/3/4, Mistral, Qwen, and most 2024–25 models.** It's the "boring correct answer": minimal quality loss, big memory win, and it speeds decode because you read fewer bytes from HBM.

**(3) MLA — Multi-head Latent Attention** (DeepSeek-V2, 2024). Instead of caching K and V per head, project the input down to a **low-rank latent** `c_t` (dimension 512 in DeepSeek-V2/V3), cache *only that*, and reconstruct K and V on the fly via up-projections. Because the up-projection matrices are shared across positions, they can be folded into the subsequent matmuls.

Result: DeepSeek reported a **93% reduction in KV cache** vs. their previous dense model, with *better* quality than MHA. It is strictly better than GQA on the memory/quality frontier, at the cost of extra compute to decompress (and a more complex kernel — `FlashMLA` exists for this reason). By 2026, MLA is the most-copied architectural idea in the field: Kimi K2.5 and GLM-5 use it (`02` §2.10).

**(4) Sparse attention (DSA)** — see §2.9. Reduces the *compute* term, which MLA does not.

**Bonus (5): KV cache compression/eviction** — quantize the cache to FP8/INT4, or drop low-attention tokens (H2O, SnapKV), or share across layers (cross-layer attention). All trade accuracy for memory; all are used in serving stacks (`09`).

---

## 2.6 FlashAttention: why attention is fast despite being quadratic

Naive attention materialises the `n×n` score matrix in GPU memory (HBM). At n=128k that's 16 billion entries — utterly infeasible, and even at n=8k it's the memory-bandwidth bottleneck.

**FlashAttention** (Dao et al., 2022; v2 2023; v3 2024) computes exact attention **without ever materialising the full matrix**, by *tiling*: process the sequence in blocks small enough to fit in on-chip SRAM, accumulate the softmax incrementally (using the online-softmax trick with running max/sum), and never write intermediates to HBM.

Consequences:
- **Memory** drops from O(n²) to O(n).
- **Speed** rises 2–4× because the bottleneck was memory traffic, not FLOPs.
- Enables 128k+ context training and inference.
- FlashAttention-2 improved parallelism/occupancy; **FlashAttention-3** (2024) exploits Hopper's asynchronous TMA + FP8 to hit ~75% of peak on H100.

**Why this belongs in an architecture chapter:** it's not an optimisation detail, it's the reason the quadratic term was survivable long enough for the field to build sparse/hybrid alternatives.

---

## 2.7 Mixture-of-Experts (MoE): more parameters, same compute

### The idea

Replace the single FFN in each layer with `E` expert FFNs plus a **router** that, for each token, selects the top-`k` experts:

```
y = Σ_{i ∈ TopK(router(x))} g_i · Expert_i(x)
```

where `g_i` are the (softmax or sigmoid) routing weights.

### Why it works

Parameters model *capacity*; FLOPs model *cost*. MoE decouples them. A 671B-parameter MoE with 37B active gives you the storage of a huge model and the per-token compute of a small one. Empirically, MoE models at fixed *active* parameters match or beat dense models at fixed *total* parameters trained on the same tokens — you get more quality per FLOP, at the price of more memory and more complex infrastructure.

### The hard parts (this is where most of the engineering is)

**(a) Load balancing.** If the router sends most tokens to a few experts, those experts saturate and the rest are dead weight. Solutions:
- **Auxiliary load-balancing loss** (Switch Transformer): penalise the deviation of expert load from uniform. Works, but the aux loss fights the language-model loss.
- **Capacity factor / token dropping**: cap tokens per expert; drop the overflow. Simple, lossy.
- **Auxiliary-loss-free balancing** (DeepSeek-V3, 2024): add a per-expert *bias* to the routing scores, updated by a rule (bias up if expert underloaded, down if overloaded), with the bias not affecting the gradient. This removed the quality/balance tradeoff and is now standard. DeepSeek-V3 also uses **sigmoid** routing scores with normalisation over selected experts, and a **shared expert** (below).

**(b) Shared / always-on experts.** DeepSeekMoE introduced **fine-grained expert segmentation** (more, smaller experts) plus **shared experts** that process every token and absorb common knowledge, while routed experts specialise. This improves the parameter-efficiency of MoE substantially.

**(c) Communication.** With **expert parallelism**, experts live on different GPUs, so each MoE layer requires an **all-to-all** exchange: send each token to the GPU hosting its expert, get the result back. This is the dominant cost of training large MoEs and the reason NVLink/NVSwitch topology matters so much (`09`).

**(d) Memory.** All experts must be resident (or offloaded). A 671B FP8 model is ~671 GB before KV cache — multi-node territory. At 4-bit, ~340 GB, which fits on 4–8 H100/H200s, which is why "run DeepSeek locally" became plausible in 2025.

### Real configurations (a sense of scale)

| Model (date) | Total / active | Experts | Notes |
|---|---|---|---|
| Switch Transformer (2021) | 1.6T / ~ | 2048 | First at trillion scale; top-1 routing |
| Mixtral 8×7B (Dec 2023) | 47B / 13B | 8 | The model that made MoE mainstream in open weights |
| DeepSeek-V2 (May 2024) | 236B / 21B | 160 routed + 2 shared | Introduced **MLA** + fine-grained MoE |
| DeepSeek-V3 (Dec 2024) | 671B / 37B | 256 routed + 1 shared, top-8 | FP8 training, aux-loss-free balancing, MTP; trained on 14.8T tokens |
| Qwen3-Next-80B (Sept 2025) | 80B / 3B | very sparse | Hybrid attention + extreme sparsity |
| Qwen3.5 (2026) | 397B / 17B | — | Gated DeltaNet hybrid, MLA-class attention |
| Kimi K2 (2025) | ~1T / ~32B | — | Trillion-scale open MoE |

Notice the trend: **total/active ratio has grown from ~4× to ~20–30×**. Sparsity is the main lever for capacity per FLOP.

---

## 2.8 Multi-Token Prediction (MTP)

Standard training predicts token `t+1`. **MTP** adds extra heads predicting `t+2`, `t+3`, … (DeepSeek-V3 uses one extra head at depth 1; some models use more).

Two benefits:
1. **Denser training signal** — each position teaches multiple predictions, improving sample efficiency (reported gains in code and reasoning benchmarks).
2. **Free speculative decoding** — the MTP head is a built-in drafter: propose `k` tokens cheaply, verify in one forward pass (`09`). Qwen3-Next ships MTP layers explicitly for "NEXTN speculative decoding".

Cost: extra parameters and a more complex loss. Net win at scale — now near-universal in 2026 flagships.

---

## 2.9 Sparse attention: DSA and friends

MLA/GQA shrink the *cache*; they don't reduce the `O(n²)` *compute* of attention over a long context. At 128k tokens, dense attention dominates the FLOP budget.

**DeepSeek Sparse Attention (DSA)**, introduced in DeepSeek-V3.2-Exp (Sept 2025):

1. A tiny **"lightning indexer"** scores every past token against the current query (cheap: few heads, low dimension, FP8).
2. Take the **top-k** tokens (e.g. k = 2048).
3. Run full MLA attention **only over those k**.

```
scores_i = TopK( q_i · Kᵀ )          # k ≪ L
Attn_i    = Σ_{j ∈ scores_i} α_ij V_j
```

Complexity per query drops from **O(L)** to **O(k)**, i.e. total from O(L²) to O(L·k). DeepSeek deliberately matched all other training settings to V3.1-Terminus and reported **essentially unchanged benchmark quality** with large long-context efficiency gains — and cut API prices by >50%. That "we changed one thing and measured it honestly" methodology is a model of good practice.

**Follow-ons (2026):** GLM-5 integrates DSA on top of MLA; systems work (HISA, 2026) attacks the indexer bottleneck itself, which still scans the full prefix, by adding a block-level coarse filter. This is the current frontier of attention engineering: the *selection mechanism* is now the cost.

**Other sparse families:** sliding-window attention (Mistral 2023, Longformer), dilated/block-sparse patterns, native sparse attention (DeepSeek's earlier NSA), and learned sparsity. The unifying idea: most tokens are irrelevant to most other tokens, so don't pay for all pairs.

---

## 2.10 Hybrid linear attention in production

The fastest-moving architectural story of 2025–26. Full detail in `03`; the production summary:

**Qwen3-Next (Sept 2025):** 75% of layers use **Gated DeltaNet** (a linear-attention/RNN-like layer with constant-size state), 25% use gated full attention, in a 3:1 repeating pattern. 80B total / **3B active** parameters, 262k native context. Qwen reports this beats pure linear or pure full attention at equal budget, and that linear attention is fast but weak at *recall* — hence keeping some full-attention layers.

**Qwen3.5 (2026)** scales the same idea to 397B-A17B. **Kimi Linear / GLM-5** combine KDA (Kimi Delta Attention) layers with sparse MLA layers in a 3:1 pattern; one 2026 analysis of a GLM-5.3-class model describes 11 sparse MLA layers (NoPE, 512-dim latent cache, 64 heads, DSA indexer selecting 2048 positions) interleaved with 33 linear-attention layers, 8-of-288 sigmoid-routed experts, aux-loss-free balancing, 154,880-token vocab, and one MTP layer.

That single sentence is basically the state of the art of LLM architecture in 2026. Read it twice.

---

## 2.11 Putting it together: three reference designs

**A. Llama-3-class dense (2024)**
GQA (8 KV heads), SwiGLU, RoPE (θ=500k), RMSNorm pre-norm, 128k vocab, 8k→128k context, no MoE. Simple, easy to serve, the workhorse of the open ecosystem.

**B. DeepSeek-V3-class MoE (2024–25)**
MLA (512-dim latent), 1 shared + 256 routed experts (top-8, sigmoid, aux-loss-free), 61 layers, MTP depth-1, FP8 training, 128k context, 671B/37B.

**C. 2026 hybrid flagship**
Gated DeltaNet : sparse-MLA at 3:1, NoPE in sparse layers, 512-dim latent cache, DSA top-2048, 8-of-288 experts + 1 shared, MTP, 1M context, FP4/FP8 mixed precision, 150k+ vocab.

---

## 2.12 Common misconceptions

- **"MoE is cheaper."** Cheaper *per token of compute*. More expensive in memory, more fragile in training, harder to serve (all-to-all, expert imbalance, tail latency).
- **"MLA is just GQA with extra steps."** MLA caches a low-rank latent and reconstructs; it dominates GQA on the memory/quality frontier. The "extra steps" are decompression matmuls that can be fused.
- **"Sparse attention loses information."** DSA-style selection keeps top-k *per query*, adaptively — very different from a fixed sliding window, and empirically near-lossless in the reported regime.
- **"Bigger vocab is free."** Embedding + unembedding parameters scale with V; at V=250k and d=8192 that's 4B parameters of embeddings alone.
- **"These architectural details are minor."** At frontier scale, a 10% KV-cache reduction or a 1.4× training speedup changes the economics of the whole product. Architecture is where the money is.

---

## 2.13 If you remember five things

1. The block is: pre-RMSNorm → attention → residual → pre-RMSNorm → SwiGLU/MoE → residual.
2. RoPE encodes relative position by rotation; extending context is a position-encoding *and* a data problem.
3. KV cache size, not FLOPs, is what limits long-context serving; GQA/MLA are the two answers.
4. MoE decouples capacity from compute via top-k routing; the hard parts are balancing (solved-ish by aux-loss-free biases), communication (all-to-all), and memory.
5. 2026's frontier is hybrid: linear-attention layers for cheap long context + a few sparse-attention layers for recall + MoE for capacity + MTP for decoding.

---

## 2.14 Self-check

1. Compute the KV cache for a 32-layer, 32-head, d_head=128, FP16 model at 32k context with (a) MHA, (b) GQA with 8 KV heads, (c) MLA with a 512-dim latent. What's the ratio?
2. Why does MLA beat GQA on quality at equal cache size?
3. Explain the RoPE relative-position property in one sentence, then explain why naive extrapolation fails.
4. Why did DeepSeek's aux-loss-free balancing matter? What tradeoff did it remove?
5. A model has 288 experts, top-8 routing, 1 shared expert, 4096 hidden. How many parameters per MoE layer (roughly)? Why might that be better than 8 large experts?
6. What does DSA reduce that MLA doesn't?

**Next:** [`03-beyond-transformers.md`](03-beyond-transformers.md)
