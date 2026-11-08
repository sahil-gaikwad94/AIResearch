# 01 — GPT-2 Small (124M, 2019)

> **The model that is still the best teaching artifact in AI.** Every architecture in the rest of this folder is GPT-2 with specific parts swapped out. Learn this one properly and the other five become "ah, they changed the norm" instead of "this is incomprehensible."

| Property | Value |
|---|---|
| Released | February 2019, OpenAI |
| Parameters | 123.5M (commonly quoted as 124M) |
| Layers | 12 |
| Width (d_model) | 768 |
| Attention heads | 12 (so 64 dimensions per head) |
| Feed-forward width | 3072 |
| Vocabulary | 50,257 tokens |
| Context | 1,024 tokens (a hard ceiling, not a preference) |
| Weights | ~500 MB in float32, ~250 MB in BF16 |
| Runs on | A laptop CPU, slowly; any 4 GB GPU, comfortably |

---

## 1. What a language model actually is

Before any architecture: a language model is a function that takes a sequence of tokens and returns a probability for every possible next token.

"Token" means a chunk of text — roughly a word-piece. GPT-2's vocabulary has 50,257 of them; `"unbelievable"` might be `un` + `believ` + `able`, three tokens. The model never sees letters, only integer IDs.

"Probability for every possible next token" means the output is 50,257 numbers. Feed them through a softmax (explained in §8) and they become probabilities that sum to 1. Pick one — the highest, or sample randomly — append it to the sequence, and run the whole thing again. That loop is text generation. Nothing else is happening. Every model in this folder does exactly this; they differ only in how they compute those 50,257 numbers.

So the real question of architecture is: **what happens between the token IDs and those numbers?** For GPT-2, the answer is 12 identical blocks, and each block does two things — mix information between tokens (attention), then think about each token individually (the MLP).

---

## 2. The config, annotated

This is the actual `config.json` shipped with `gpt2` on Hugging Face:

```json
{
  "vocab_size": 50257,
  "n_positions": 1024,
  "n_embd": 768,
  "n_layer": 12,
  "n_head": 12,
  "n_inner": 3072,
  "activation_function": "gelu_new",
  "layer_norm_epsilon": 1e-05,
  "tie_word_embeddings": true,
  "scale_attn_weights": true
}
```

Ten lines. That is the entire model. Read it as a sentence:

- `vocab_size: 50257` — the alphabet. 50,000 merged subwords from byte-pair encoding (BPE), plus 257 special slots.
- `n_positions: 1024` — the model has a **learned lookup table** of 1024 position vectors. That is where the context limit comes from: there is no row 1025. Context length here is a property of a table, not of a mechanism, which is why later models moved to rotary positions that can be computed for any index.
- `n_embd: 768` — the width. Every token is represented by 768 numbers as it flows through the network. Everything else is sized relative to this.
- `n_layer: 12` — how many times the block repeats. Depth.
- `n_head: 12` — attention is split into 12 parallel "heads", each 64-dimensional (768 ÷ 12). Each head can learn a different relationship.
- `n_inner: 3072` — the MLP's hidden width, exactly 4× the model width. This 4× rule was folk wisdom in 2019 and is now known to be roughly where parameter-efficiency peaks for a dense model.
- `activation_function: "gelu_new"` — the nonlinearity inside the MLP.
- `layer_norm_epsilon: 1e-05` — a tiny number added before dividing, to avoid dividing by zero in the normalisation.
- `tie_word_embeddings: true` — the input embedding matrix and the output projection are **the same 50257×768 matrix**, used twice. This saves 38.6M parameters (31% of the model!) and is the reason GPT-2 is 124M rather than 162M.

---

## 3. One block, drawn

```
token IDs  (1, T)                       T = sequence length
    │
    ├─► embedding table  (50257, 768)  ──► (1, T, 768)
    ├─► position table   (1024, 768)   ──► (1, T, 768)
    └─► add                            ──► x  (1, T, 768)

  ┌─────────────── × 12 blocks ───────────────┐
  │                                            │
  │  a = LayerNorm(x)                 (1,T,768)│
  │  q,k,v = a @ W_qkv                (1,T,2304)│
  │  split into 12 heads of 64 dims            │
  │  attn  = softmax(q kᵀ/√64 + mask) v         │
  │  x = x + attn @ W_o               (1,T,768)│   ← residual
  │                                            │
  │  b = LayerNorm(x)                 (1,T,768)│
  │  h = GELU(b @ W_fc)               (1,T,3072)│
  │  x = x + h @ W_proj               (1,T,768)│   ← residual
  └────────────────────────────────────────────┘

final LayerNorm  ──► logits = x @ embeddingᵀ  ──► (1, T, 50257)
```

Two details in that diagram are load-bearing and easy to miss:

**The residual stream.** `x = x + something` appears twice per block. This means the 768-wide vector is not overwritten; it is a shared bus that each sublayer *adds to*. Information has a direct path from layer 0 to layer 12 that no sublayer can destroy. Almost everything modern interpretability relies on this: you can read the state of the model at any depth by looking at `x`. It is also why deep models train at all — gradients flow straight back along the additions.

**Pre-norm.** `LayerNorm` is applied to the *input* of each sublayer (`a = LayerNorm(x)`), not to its output. This is called pre-norm and it was not obvious in 2019. Post-norm (normalising after the residual add, as in the original 2017 Transformer) makes deep stacks unstable; pre-norm is one of the main reasons 100+ layer models became trainable.

---

## 4. Attention, exactly

Attention is a lookup. Each token emits three vectors:

- a **query** `q` — "what am I looking for?"
- a **key** `k` — "what do I contain?"
- a **value** `v` — "what do I hand over if you pick me?"

For token `i`, score it against every earlier token `j` with a dot product `qᵢ · kⱼ`. High score = relevant. Convert scores to weights with softmax so they are positive and sum to 1. The output for token `i` is the weighted average of the `v` vectors.

```
scores  = q @ kᵀ / √d_head          (T, T) per head
scores  = scores + mask              mask = 0 on/below diagonal, -∞ above
weights = softmax(scores, dim=-1)    each row sums to 1
out     = weights @ v                (T, 64) per head
```

Three things a beginner should actually understand rather than memorise:

**Why the causal mask.** A language model predicting token `t` must not see token `t+1`, or it would just copy the answer. The mask sets all future scores to −∞ before the softmax, which makes them exactly 0 weight. This is what "causal" in "causal language model" means.

**Why divide by √d_head.** If `q` and `k` entries are roughly unit-variance, their dot product over 64 dimensions has standard deviation ≈ √64 = 8. Softmax of numbers with spread 8 is nearly one-hot — it saturates, gradients vanish. Dividing by 8 brings the spread back to ~1. This is not a detail: it is the difference between a model that learns and one that does not.

**Why multiple heads.** One attention pattern can only express one kind of relationship ("attend to the previous noun"). Twelve heads give twelve independent patterns, and their outputs are concatenated back to 768. Empirically, different heads specialise: some track syntax, some copy the previous token, some attend to a delimiter far back.

Attention is the only operation in the model that moves information **between** positions. Everything else is per-token.

---

## 5. The MLP (feed-forward network)

```
h = GELU(x @ W_fc)     768 → 3072
y = h @ W_proj         3072 → 768
x = x + y
```

No mixing between tokens. Each position is processed independently — it is a two-layer perceptron applied to every token identically. It expands to 4× width, applies a nonlinearity, and projects back.

This is where most of the parameters live (4.7M of the 7.1M per block, 66%), and modern interpretability suggests it is where most factual knowledge is stored — the widely cited picture is that the MLP acts as a key-value memory: the first matrix detects a pattern, the second writes associated content into the residual stream.

`GELU` (Gaussian Error Linear Unit) is a smooth version of ReLU: it multiplies the input by the probability that a standard normal is below it. In practice `GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`. It is smooth, so gradients are well-behaved, and it slightly allows small negative values through instead of clamping them to zero.

---

## 6. LayerNorm

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) · γ + β      (normalise over the 768 features)
```

For each token separately: subtract the mean of its 768 numbers, divide by their standard deviation, then scale by a learned `γ` and shift by a learned `β`. Output has mean 0, variance 1 per token before the learned affine.

Why it exists: without it, activations drift in scale as they pass through many layers, and training becomes hypersensitive to learning rate. LayerNorm re-centres every token at every step, which makes the optimisation landscape much flatter.

Keep this formula in mind. Model 2 replaces it with RMSNorm — same idea, one term deleted.

---

## 7. Count the parameters yourself

Do this with a calculator once. It removes all mystery from model sizes.

**Per block:**

| Matrix | Shape | Parameters |
|---|---|---|
| `c_attn` (q, k, v fused) | 768 × 2304 | 1,769,472 |
| `c_proj` (attention output) | 768 × 768 | 589,824 |
| `c_fc` (MLP up) | 768 × 3072 | 2,359,296 |
| `c_proj` (MLP down) | 3072 × 768 | 2,359,296 |
| biases + 2 LayerNorms | — | ~8,448 |
| **one block** | | **≈ 7.08M** |

**Whole model:**

- 12 blocks × 7.08M = 85.0M
- embedding table 50,257 × 768 = 38.6M (counted **once**, because it is tied)
- **total ≈ 123.5M** ✓

Note the ratio: 69% of the model is in the 12 blocks, and two-thirds of *that* is MLP. The attention machinery — the famous part — is 2.4M per block, about a third of the block. This ratio recurs in every model in this folder, and it is the single most useful fact for predicting where parameters go.

---

## 8. From logits to text

The final `x` (1, T, 768) is multiplied by the transposed embedding matrix to produce (1, T, 50257) **logits** — unbounded real numbers, one per vocabulary entry. Then:

```
p = softmax(logits / temperature)
```

`temperature = 1` is the model's honest distribution. `< 1` sharpens it (more deterministic, more repetitive). `> 1` flattens it (more surprising, more incoherent). Real sampling also truncates: `top_k` keeps only the k most likely tokens, `top_p` keeps the smallest set whose probabilities sum to p, then renormalises. All of that happens outside the model. The weights never change at inference.

At this point you know the entire forward pass. Everything that follows in this folder is a variation on §§3–7.

---

## 9. Reverse-engineering lab

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = model.state_dict()

# 1. Every tensor in the model, with its shape
for name, w in list(sd.items())[:14]:
    print(f"{name:28s} {tuple(w.shape)}")

# Expected:
#   wte.weight                 (50257, 768)      <- token embedding
#   wpe.weight                 (1024, 768)       <- position table (the context limit)
#   h.0.ln_1.weight            (768,)
#   h.0.attn.c_attn.weight     (768, 2304)       <- fused q|k|v   (Conv1D: in, out)
#   h.0.attn.c_attn.bias       (2304,)
#   h.0.attn.c_proj.weight     (768, 768)
#   h.0.ln_2.weight            (768,)
#   h.0.mlp.c_fc.weight        (768, 3072)
#   h.0.mlp.c_proj.weight      (3072, 768)
#   ...
#   ln_f.weight                (768,)
#   lm_head.weight             (50257, 768)       <- SAME tensor as wte.weight

print(torch.equal(sd["wte.weight"], sd["lm_head.weight"]))   # True  <- weight tying, proven

# 2. Total parameters, and how much is embedding
total = sum(p.numel() for p in model.parameters())
embed = sd["wte.weight"].numel()
print(f"total {total/1e6:.1f}M, embedding {embed/1e6:.1f}M ({100*embed/total:.0f}%)")

# 3. Look inside attention: one head's query slice
qkv = sd["h.0.attn.c_attn.weight"]            # (768, 2304)
q, k, v = qkv.split(768, dim=1)
head0_q = q[:, :64]                            # first head, 64 dims
print(head0_q.shape, head0_q.std().item())
```

That last line is the whole trick of reverse engineering: **the weight shapes tell you the architecture before you read any paper.** A (768, 2304) fused matrix says "three projections of a 768-wide model"; the fact that it is 2304 and not 2560 says the heads are 64-dimensional and there are 12 of them; the existence of `wpe` says positions are learned, not rotary.

**Tensor-name heuristic.** Once you have seen a few models, the *names* give away the design: `c_attn` = 2019-era fused QKV; `q_proj`/`k_proj`/`v_proj` = the 2023+ style; `kv_a_proj_with_mqa` = compressed (MLA) attention; `A_log`/`dt_bias` = a state-space model; `softmax_offset` = attention sinks. File 07 turns this into a checklist.

---

## 10. What GPT-2 does the old way

Each of these is a constraint that a later model in this folder removes. Read this table as a map of the rest of the course.

| GPT-2 (2019) | Problem it causes | Fixed by |
|---|---|---|
| **Learned absolute positions** (`wpe`, 1024 rows) | Hard 1024-token cap; cannot extrapolate; positions are memorised data, not a mechanism | RoPE — model 2 |
| **Multi-head attention** (all 12 heads have own K, V) | KV cache is 36 KiB per token *at 124M scale*; at 8B it would be gigabytes | GQA — model 2; MLA — model 5 |
| **Quadratic attention** | Cost grows with T²; long context is unaffordable | Sliding window — models 3, 4; linear attention — model 6 |
| **LayerNorm** (mean + variance) | Extra reduction and two parameter vectors per norm; slight instability | RMSNorm — model 2 |
| **GELU** | MLP has 2 matrices; a gated version is more expressive per parameter | SwiGLU — model 2 |
| **Dense MLP** | Every token pays for all knowledge in the model | MoE — models 3–6 |
| **50,257-token vocab** | Poor code and non-English tokenisation; 31% of the model is the embedding table | 128k–160k vocabs — models 2–6 |
| **Tied embeddings** | Saves 38.6M here; at scale input and output embeddings appear to want different geometry | Untied — models 2–6 |
| **Biases everywhere** | ~100k parameters that empirically do nothing; extra kernel complexity | Removed — models 2–6 |
| **`n_inner` = 4× d_model** | Folk rule; modern models use ~3.5× with SwiGLU | Model 2 |

Two of these deserve a warning, because they are the ones people over-claim:

**"Attention is all you need" is misleading at this scale.** In GPT-2, attention is a third of the block parameters and a small fraction of the FLOPs at 1k context. The MLP is the bulk. Attention matters enormously for *capability at long range*, but it is not where the weights are.

**Emergence.** GPT-2 small cannot do arithmetic, follow instructions, or write working code — those abilities appear only at 10–100× this scale, and (see file 07) part of that apparent jump is a measurement artifact, not a phase change.

---

## 11. What you should now be able to say without looking

- A transformer block = mix across tokens (attention) + think per token (MLP), each wrapped in a residual add, each preceded by a normalisation.
- Parameter count is dominated by MLP matrices: `2 × d × 4d` per block for GPT-2.
- Context length in GPT-2 is literally the number of rows in a lookup table.
- Text generation is the same forward pass repeated, with sampling on top.

Everything else in this folder is one of: a better norm, a better position scheme, a better MLP, a cheaper attention, or not using all the weights at once.

---

## 12. Self-check

1. Why is GPT-2 124M parameters and not 162M? What would you change to make it 162M?
2. A block has 7.08M parameters. How much of that is attention, and how much is MLP? Which would you shrink first to make the model 20% smaller, and what would you expect to lose?
3. The mask sets future scores to −∞. Why −∞ and not 0?
4. If you doubled `n_head` to 24 while keeping `n_embd` at 768, what happens to (a) parameter count, (b) head dimension, (c) KV cache size?
5. GPT-2 cannot accept a 2,000-token prompt. Give two distinct reasons from the config.
6. Why is the same matrix used as both `wte` and `lm_head`, and what does that imply about the geometry of the embedding space?

*(Answers: 1 — weight tying saves the 38.6M embedding; untie by setting `tie_word_embeddings=false`. 2 — attention 2.36M, MLP 4.72M; shrink the MLP, expect loss of knowledge recall more than syntax. 3 — softmax(−∞)=0 exactly; 0 would still give a nonzero share to the future. 4 — params unchanged if head dim halves to 32, but head dim 32 is too small to be useful; KV cache unchanged in total size. 5 — `n_positions=1024` has no row 1025, and attention cost grows as T². 6 — the logit for token w is the dot product of the final state with w's embedding, so tokens that are "used similarly" end up near each other.)*

---

**Next:** [02 — Llama 3 8B](02-llama3-8b.md), where the block gets rebuilt with the five components that every 2024–2026 model still uses.
