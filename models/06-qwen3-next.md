# 06 — Qwen3-Next 80B-A3B (2025)

> **The model that stopped using attention for most of its layers.** 36 of Qwen3-Next's 48 layers have no attention at all — they use a linear-time recurrence with a fixed-size state, so their cost per token does not grow with context. The other 12 keep ordinary attention, because exact recall still needs it.
>
> This is the hardest file because it requires abandoning the mental model of "every token keeps a key and a value". If you follow the four steps in §3, you will have it.

| Property | Value |
|---|---|
| Released | 12 September 2025, Alibaba Qwen (Apache 2.0) |
| Parameters | 80B total / 3B active (3.9B by NVIDIA's convention) |
| Layers | 48, arranged as **12 × (3 Gated DeltaNet + 1 Gated Attention)** |
| Width | 2,048 |
| Linear layers (36) | Gated DeltaNet: 32 value heads, 16 QK heads, head dim 128 |
| Attention layers (12) | 16 query heads, 2 KV heads, head dim 256, RoPE on 64 dims |
| Experts | 512 routed, top-10, + 1 shared, expert width 512 |
| Vocabulary | 151,936 |
| Context | 262,144 native, ~1,010,000 with YaRN |
| Training | 15T tokens |
| Extras | MTP head, zero-centred RMSNorm with weight decay |

---

## 1. Why this is the last file

Everything up to here kept one assumption: **each token stores its own key and value, and attention compares the query against all of them.** That is what makes the KV cache grow (files 02–05) and what makes cost quadratic in context length. Every previous fix — GQA, windows, MLA — reduced the *constant factor*. Qwen3-Next removes the *scaling law* for three quarters of its layers.

The price is the thing attention is uniquely good at: exact recall of an arbitrary earlier token. The design is a bet that you can have both, by keeping a quarter of the layers exact.

---

## 2. The problem, quantified

Take a hypothetical 48-layer model at Qwen3-Next's width with pure attention and 2 KV heads × 256 dims:

```
2 × 48 layers × 2 heads × 256 dims × 2 bytes = 49,152 bytes = 48 KiB per token
at 262,144 tokens: 12 GiB of KV cache, for one sequence
```

And that is with aggressive GQA. The cache must also be *read* at every decode step, so long-context decoding becomes memory-bandwidth-bound on the cache rather than the weights.

Now the structural point: after MoE, attention is no longer a small part of the per-token cost. In Qwen3-Next, roughly **40–45% of active compute per layer is the mixing operator** (attention or the recurrent layer), because the experts are so cheap. Fixing attention stopped being an optimisation and became the main event.

---

## 3. Four steps from attention to Gated DeltaNet

### Step 1 — Rewrite attention without the softmax

Ordinary attention, for one head, at position `t`:

```
o_t = Σ_{j≤t} softmax(q_t · k_j) · v_j
```

Every previous token's `k` and `v` must still exist. Now delete the softmax and use a plain sum:

```
o_t = Σ_{j≤t} (q_t · k_j) · v_j  =  q_tᵀ · ( Σ_{j≤t} k_j v_jᵀ )
```

That refactoring is the whole trick. The inner sum is a **matrix** — call it the state `S` — and it does not depend on the query at all:

```
S_t = S_{t-1} + k_t v_tᵀ            (outer product: d_k × d_v)
o_t = q_tᵀ S_t                      (a single matrix-vector product)
```

Cost per token: O(d_k · d_v), **constant**, no matter how long the context. Memory: one matrix per head — for 128×128 that is 16,384 numbers, fixed, instead of 262,144 key/value pairs. This is "linear attention", and the reason it fell out of fashion after 2020 is step 2.

### Step 2 — See why it is bad, and fix it with the delta rule

The additive state has two pathologies. It **never forgets** (the state keeps accumulating forever, so old content is buried under noise), and writes **interfere** (two different keys writing to the same directions of `S` collide, and retrieval becomes mush). Empirically, such models are poor at in-context recall: ask them to reproduce a fact from 50k tokens back and they fail.

**DeltaNet** replaces "add" with "correct":

```
predicted value for key k_t:   ŷ = S_{t-1}ᵀ k_t
error:                         e = v_t − ŷ
state update:                  S_t = S_{t-1} + β_t · k_t eᵀ
```

Instead of writing `k vᵀ` blindly, the layer first asks *what does my state currently predict for this key?*, computes the error, and writes only the correction. `β_t ∈ (0,1)` controls how hard to correct.

Why this is a big deal: it is exactly the **delta rule** from 1960s learning theory (Widrow–Hoff), and it is equivalent to **one step of gradient descent** on `‖Sᵀk − v‖²`. The state is a fast-weight memory that is being trained *during the forward pass*. Two consequences follow immediately:

- writing a new association with key `k` **overwrites** the old association for `k` rather than adding to it, so interference is bounded;
- the state is a much better associative memory, which is why DeltaNet variants beat Mamba-2 and sliding-window attention on in-context learning tasks.

In matrix form the update is `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ` — a rank-one projection that removes the old association, then writes the new one.

### Step 3 — Add a gate, so it can forget

The delta rule fixes interference but still never decays. Add a per-channel decay:

```
S_t = α_t ⊙ S_{t-1}  +  β_t k_t (v_t − α_t ⊙ S_{t-1}ᵀ k_t)ᵀ
```

`α_t ∈ (0,1)` is computed from the current token (a small linear projection followed by a sigmoid). Near 1: remember everything, good for long documents. Near 0: wipe the state, good for a new topic or a hard reset at a delimiter. The gate is what makes the state *selectively* forgetful rather than uniformly decaying, and it is the difference between DeltaNet and **Gated** DeltaNet.

That is the whole operator. Three learned projections produce `q`, `k`, `v`; two more produce the gate `α` and the write strength `β`; the state update is two matrix products per token.

### Step 4 — Make it fast on a GPU (the chunked form)

The recurrence above is sequential: token `t` needs `S_{t-1}`. GPUs hate that. The fix is to process the sequence in **chunks of 64 tokens**:

```
within a chunk:  all 64 updates computed as dense matmuls (the "WY representation")
across chunks:   the state is carried forward once per chunk, sequentially
```

So you do 64 tokens' worth of work per sequential step instead of 1, and 99% of the FLOPs land in matmuls the tensor cores like. This chunkwise algorithm — not the maths — is what made linear attention competitive in production in 2025. It also explains the practical reality: **linear-attention layers need custom kernels**, and their speed advantage only appears once those kernels exist for your hardware.

---

## 4. The hybrid: why keep 12 attention layers

If Gated DeltaNet were strictly better, you would use 48 of them. It is not. Its weakness is exactly the one you would predict: the state is a **fixed-size bottleneck**, so exact retrieval of an arbitrary earlier token is lossy. A 128×128 state cannot losslessly store 262,144 tokens of key/value pairs — that is 16,384 numbers of memory trying to hold millions.

So Qwen uses a 3:1 hybrid:

```
layer 0: GDN   layer 1: GDN   layer 2: GDN   layer 3: attention
layer 4: GDN   layer 5: GDN   layer 6: GDN   layer 7: attention
...                                           (×12 blocks = 48 layers)
```

The division of labour mirrors Flint's (file 04) but with a different mechanism:

| | Flint (04) | Qwen3-Next (06) |
|---|---|---|
| Local layer type | sliding-window attention (513) | Gated DeltaNet (fixed state) |
| Local cost per token | O(513), constant | O(1), constant |
| Global layer type | full attention, **no RoPE** | full attention, RoPE |
| Global share | 14/49 = 29% | 12/48 = 25% |
| Local memory | cache capped at 513 | state, ~1 MB per layer |

Qwen's own claim is that Gated DeltaNet gave better in-context learning than either sliding-window attention or Mamba-2 at the same budget, and that the 3:1 mix beat any single-operator model. Treat the comparison as vendor evidence; the *structural* claim — that a recurrent local layer plus a sparse exact layer beats either alone — now has several independent instances (Jamba, Samba, GPT-OSS, Flint, Qwen3-Next) and is the closest thing to consensus in the field.

---

## 5. The 12 attention layers are not ordinary attention

Given only 12 full layers instead of 48, each has to do more work. Three changes:

**Head dim 256 instead of 128.** Each head holds twice as much per token. With 16 query heads × 256 = 4,096 (2× the 2,048 residual width), the attention layers are wide even though they are few.

**2 KV heads for 16 query heads** — an 8:1 GQA ratio, more aggressive than Llama's 4:1, which is affordable precisely because there are only 12 of them.

**An output gate.** After attention produces its output, it is multiplied elementwise by `sigmoid(x W_g)`:

```
o = attention(x) ⊙ sigmoid(x W_gate)
```

Qwen describes this as fixing "low-rank issues in attention" — the attention output lives in a much smaller subspace than the residual stream, and a multiplicative gate lets the layer decide per-channel how much of it to admit, instead of always writing the full projection into the residual stream. It also acts as a soft attention-sink mechanism. GPT-OSS and several 2025–26 models use the same trick.

**RoPE on 64 of the 256 head dimensions**, not all of them — the same decoupling idea as MLA (file 05): rotate a subset for position, leave the rest for content.

---

## 6. Extreme sparsity

512 experts, top-10, one shared, expert width **512** — the same 3.1M-per-expert size as Flint.

| Per layer | Parameters |
|---|---|
| one expert (3 × 2048 × 512) | 3.15M |
| 10 active experts | 31.5M |
| shared expert | 3.15M |
| **active FFN per token** | **34.6M** |
| 512× active FFN per layer, in weights | 1,613M |

Model-level sparsity: 80B / 3.9B ≈ **20×**. Expert-level sparsity: 512 / 10 = **51×** — the highest in this folder. 1.6B parameters of feed-forward per layer, of which a token touches 2%.

Compare with Mixtral (file 03), which activated 25% of its FFN per token. Qwen3-Next activates **2%**. That is the two-year arc of the field in one number, and it is why the FFN stopped being the bottleneck and attention became one again.

The shared expert is worth noting: at width 512 it is deliberately *small* — one thirtieth of the layer's FFN. Its job is not capacity but continuity: it guarantees every token gets some always-on processing even if routing is uncertain. Flint's shared expert is 2× a routed expert; Qwen's is 1×. There is no consensus on the size, only on having one.

---

## 7. Stability tricks at 15T tokens

Two changes from the default recipe, both aimed at a specific failure:

**Zero-centred RMSNorm.** Parameterise the scale as `(1 + γ)` with `γ` initialised to zero, so the layer starts as exact identity normalisation. With the usual parameterisation, every norm starts at a random scale and the model must learn to undo it.

**Weight decay applied to norm parameters.** Normally normalisation weights are excluded from weight decay. Qwen applies it, which keeps the norms from drifting upward over a very long run.

Underlying problem: at this scale, "loss spikes" — sudden divergences mid-training that can cost weeks of compute — are common, and they are usually traced to a norm or an attention logit growing unboundedly. Every 2025–26 model has a story about this: DeepSeek used FP8 with careful per-block scaling, Kimi K2 invented QK-clip, Qwen used these norm changes. **Stability engineering is now a first-class architecture topic**, and it is almost invisible in the config.

---

## 8. Count the parameters yourself

| Component | Parameters |
|---|---|
| Gated attention layer (16q × 256, 2kv × 256, output gate, 2048→4096→2048) | ≈ 27M |
| Gated DeltaNet layer (q,k: 16×128; v: 32×128; output; gate and beta projections; short conv) | ≈ 29–35M |
| one routed expert | 3.15M |
| shared expert | 3.15M |
| **per MoE layer, total** | ≈ 1.65B |
| **per MoE layer, active** | ≈ 62–67M |

- 12 attention layers × 1.65B + 36 GDN layers × 1.65B = 48 × 1.65B ≈ **79.3B**
- embeddings + head: 2 × 151,936 × 2,048 = 0.62B
- **total ≈ 79.9B** ✓ (vendor: 80B)

Active per token:

- 12 × (27 + 3.1 + 31.5) + 36 × (29 + 3.1 + 31.5) + output head
- ≈ 0.74B + 2.29B + 0.31B ≈ **3.2B**

**Here my count and the vendor's disagree**: Qwen says 3B, NVIDIA's model card says 3.9B, and I get 3.2B. The gap is entirely counting conventions — whether the embedding matrix is included, whether the DeltaNet gate/beta/convolution parameters are counted as "activated", and whether the MTP head is included. This is not a mistake in either direction; it is a demonstration that **"active parameters" is not a defined standard.** When you compare two MoE models, compare their FLOPs or their measured tokens/second, not their marketing active-parameter counts.

---

## 9. The cache: 12 KiB/token plus a fixed state

```
attention layers:  12 × 2 KV heads × 256 dims × 2 bytes = 12,288 bytes = 12 KiB per token
DeltaNet layers:   36 × 32 heads × 128 × 128 × 2 bytes  = 37.7 MB, FIXED per sequence
```

| Context | Attention KV cache | DeltaNet state | Total |
|---|---|---|---|
| 8k | 96 MiB | 37.7 MB | ~0.13 GiB |
| 32k | 384 MiB | 37.7 MB | ~0.42 GiB |
| 262k | **3.0 GiB** | 37.7 MB | ~3.04 GiB |

Compare a 48-layer pure-attention model at the same width: 12.0 GiB at 262k. So the hybrid is a **4× reduction in cache** — but the interesting number is not the 4×, it is that *three quarters of the model's memory no longer grows at all*.

That is what "efficient context modelling for ultra-long context" means in Qwen's model card, and it is why the million-token extension is plausible: the growing part is now one quarter of the model instead of all of it.

---

## 10. Reverse-engineering lab

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

# The fields that tell you this is a hybrid, before reading any paper
print(cfg.num_hidden_layers)              # 48
print(cfg.linear_num_value_heads)         # 32   <- DeltaNet value heads
print(cfg.linear_num_key_heads)           # 16   <- DeltaNet QK heads
print(cfg.linear_key_head_dim)            # 128
print(cfg.num_attention_heads, cfg.num_key_value_heads)   # 16, 2  <- the 12 real attention layers
print(cfg.head_dim)                       # 256  <- attention heads are twice as wide
print(cfg.num_experts, cfg.num_experts_per_tok)           # 512, 10
print(cfg.shared_expert_intermediate_size)                # 512
print(cfg.layer_types[:8])
# ['linear_attention','linear_attention','linear_attention','full_attention', ...]

# Where the recurrence becomes concrete: these four tensors
#   linear_attn.A_log        (32,)     <- per-head decay, log-space
#   linear_attn.dt_bias      (32,)     <- per-head gate bias
#   linear_attn.conv1d.weight          <- short causal convolution before the recurrence
#   linear_attn.in_proj_qkv  / in_proj_z / in_proj_a / in_proj_b
```

`A_log` and `dt_bias` are the universal signature of a state-space or gated-linear layer — the same names appear in Mamba. If you see them plus `full_attention` layers in `layer_types`, you are looking at a hybrid, and you now know exactly what that means: a fixed-size state for most layers, exact attention for a quarter.

---

## 11. What Qwen3-Next gets wrong

| Limitation | Detail |
|---|---|
| **State bottleneck** | 36 layers compress the whole prefix into a fixed matrix; exact recall of arbitrary distant tokens is still lossy, which is why 25% of layers are exact. |
| **Two operators, two kernels** | A serving stack must schedule two different attention implementations in one forward pass; naive implementations lose the speed advantage entirely. |
| **80B weights for 3.9B compute** | Same workstation problem as Flint: ~160 GB in BF16. |
| **Extreme sparsity risks** | 10 of 512 experts is a hard routing problem; MTP and the tiny shared expert are partly there to stabilise it. |
| **Benchmarks** | Vendor-reported; the claim "beats Qwen3-235B-A22B within 256k" is a claim about a specific evaluation setup. |

---

## 12. Self-check

1. Derive `o_t = q_tᵀ S_t` from the softmax-free attention sum. Which assumption did you drop relative to ordinary attention?
2. What exactly does the delta rule overwrite, and what is the equivalent optimisation statement?
3. Why does a fixed-size state hurt needle-in-a-haystack retrieval, in one sentence with a number in it?
4. The chunked algorithm changes no maths. Why does it change the speed by an order of magnitude?
5. Qwen3-Next activates 2% of its FFN per token, versus Mixtral's 25%. What does that do to attention's share of per-token compute, and what design consequence follows?
6. 262k context costs ~3 GiB of cache here versus ~12 GiB for pure attention at the same width. How much of that comes from smaller per-token cache, and how much from the cache not growing?
7. Why would you gate the *output* of an attention layer rather than its input?

*(Answers: 1 — you drop the softmax normalisation, which is what allowed the key/value sum to be precomputed independently of the query; without it, retrieval is additive rather than competitive. 2 — the association for the current key, replaced by the residual error `v − Sᵀk`; it is one gradient step on `‖Sᵀk−v‖²`. 3 — 16,384 numbers of state cannot losslessly hold 262,144 tokens' key/value pairs, so recall degrades as the prefix grows. 4 — it converts 64 sequential rank-one updates into dense matmuls the tensor cores can execute in parallel, removing the dependency chain. 5 — attention's share rises to roughly 40–45%, so the mixing operator becomes the bottleneck and must itself be made sub-quadratic. 6 — 12 KiB vs 48 KiB is a 4× per-token reduction, and all of the growth-free benefit comes from the 36 recurrent layers holding a constant state. 7 — the output lives in a lower-dimensional subspace than the residual stream; gating the output controls how much of that subspace is written per channel, which also damps attention-sink behaviour.)*

---

**Next:** [07 — Insights and Patterns](07-insights-and-patterns.md), the audit: what the six models say together that none of them says alone.
