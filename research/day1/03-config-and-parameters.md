# 3 — Config and Parameters

> The full count, derived by hand, then checked against the instantiated model. If the two disagree, one of them is wrong and it is usually the hand.

---

## 3.1 One layer, by hand

With `d = 3072`, `d_c = 448`, `d_rope = 64`, `h = 24`, `d_h = 128`, `d_ff = 7680`:

**Attention (MLA):**

| Matrix | Shape | Params |
|---|---|---|
| `W_DQ` | 3072 × 448 | 1,376,256 |
| `W_UQ` | 448 × 3072 | 1,376,256 |
| `W_DKV` | 3072 × 512 | 1,572,864 |
| `W_UK` | 448 × 3072 | 1,376,256 |
| `W_UV` | 448 × 3072 | 1,376,256 |
| `W_O` | 3072 × 3072 | 9,437,184 |
| `W_QR` | 3072 × 1536 | 4,718,592 |
| `q_norm`, `k_norm` | 128 each | 256 |
| **subtotal** | | **21,233,920** |

**FFN (SwiGLU):**

| Matrix | Shape | Params |
|---|---|---|
| `gate` | 3072 × 7680 | 23,592,960 |
| `up` | 3072 × 7680 | 23,592,960 |
| `down` | 7680 × 3072 | 23,592,960 |
| **subtotal** | | **70,778,880** |

**Norms:** two RMSNorm of 3072 = 6,144.

**Per layer:** 21,233,920 + 70,778,880 + 6,144 = **92,018,944** (92.019 M).

**Body:** 92,018,944 × 40 = **3,680,757,760** (3.681 B).

**Embeddings:** 131,072 × 3,072 = 402,653,184, tied — counted once.

**MTP:** 2 × (3072 × 3072 + 3072) = **18,882,048**.

| | Params |
|---|---|
| Body | 3,680,757,760 |
| Embeddings (tied) | 402,653,184 |
| **Inference total** | **4,083,410,944** → 4.083 B |
| MTP (training only) | 18,882,048 |
| **Training total** | **4,102,292,992** → 4.102 B |

Verified against the model: `params (inference) : 4.0834 B` — the assertion in `model.py` fails the build if this drifts by more than 0.1%.

---

## 3.2 Derived ratios worth checking

These are the numbers that tell you whether a shape is sane before you spend $100k on it.

| Ratio | Kestrel | Why it matters |
|---|---|---|
| tokens per parameter | **294** | 1.2 T / 4.08 B. Well above Chinchilla's 20; modern small models are deliberately overtrained because inference cost dominates training cost |
| `d_ff / d` | **2.5** | Llama uses 3.5. Lower because the 131 k vocabulary and MLA take budget |
| attention : FFN | **1 : 3.33** | Typical is 1:4 to 1:5. Attention is comparatively expensive because MLA stores a low-rank latent rather than reusing head dims |
| embedding share | **9.9 %** | Above 12% is a warning sign at this scale — the vocabulary is eating the model |
| FLOPs per token (train) | **2.45 × 10¹⁰** | `6N` |
| params per head | 128 M | 3072 / 24 heads = 128 dims each — no head padding, no wasted parameters |
| KV per token per layer | 512 floats | `d_c + d_rope` — the design's defining constant |

Two of these deserve comment.

**Tokens per parameter = 294** is the most consequential ratio in the table and it is a *choice about the business*, not about learning. A Chinchilla-optimal 4 B model would train on ~80 B tokens, cost $7k, and be mediocre. At 1.2 T it costs 15× more and is worth deploying, because you pay training once and inference millions of times. Every serious small model of the last two years has made the same bet.

**Embedding share = 9.9%** is why ablation A1 matters. If the code-aware tokeniser delivers what §4 predicts, a 65 k vocabulary would free 200 M parameters — 5% of the model — for the FFN. If it does not, the larger vocabulary is 200 M wasted.

---

## 3.3 The config, annotated

[`kestrel/config.json`](kestrel/config.json) is the canonical source of truth; `model.py` reads it. Field by field, with the reasoning:

```jsonc
"vocab_size": 131072        // 2^17. Large enough for code-aware tokens,
                            // small enough that embeddings cost 9.9% not 15%.
"hidden_size": 3072,
"num_hidden_layers": 40,    // depth-heavy, see §2.7
"num_attention_heads": 24,  // 24 x 128 = 3072 exactly; no padding
"head_dim": 128,
```

```jsonc
"attention": {
  "type": "mla",
  "kv_lora_rank": 448,      // 0.146 x d_model. DeepSeek uses 0.071 at 7168;
                            // smaller models need a relatively larger latent.
  "q_lora_rank": 448,       // symmetric, and it is what makes MLA cheaper
                            // in parameters than GQA-4.
  "qk_rope_head_dim": 64,   // must be even; 64 carries position for all 24 heads
  "qk_nope_head_dim": 128,  // = head_dim
  "attention_sinks": 4,     // eviction + long-context safety, costs 96 params
  "qk_norm": true,          // per-head RMSNorm; the cheapest stability win
  "softcap": null           // Gemma-style logit capping not used; QK-norm
                            // already bounds the pre-softmax dynamic range
}
```

```jsonc
"ffn": { "type": "swiglu", "intermediate_size": 7680 },   // 2.5 x d
"norm": { "type": "rmsnorm", "eps": 1e-6 },
"position": { "rope_theta": 500000.0, "no_pe_layers": [] },  // see A7
"embeddings": { "tie_word_embeddings": true },
"mtp": { "num_heads": 2, "drop_at_inference": true },
"context": { "native": 65536, "staged_training": [4096, 16384, 65536] }
```

```jsonc
"init": {
  "residual_output_proj_zero": true,   // every block starts as identity
  "depth_scaled_init": true            // residual stream does not grow with depth
},
"precision": { "train": "bf16-mixed", "optimizer_states": "fp32", "kv_cache": "fp8" }
```

Nothing here is a novelty. That is intentional: the design's bet is in the *combination* (full attention + MLA + read-heavy serving), not in a new block. Every individual component has been validated by someone else's ablation, which is the only way a 45,000-GPU-hour budget produces a model rather than a bug report.

---

## 3.4 Reproducing

```bash
python3 research/day1/tools/count_params.py    # all params, KV, FLOPs, budget
python3 research/day1/kestrel/model.py         # instantiate + assert + cache test
```

The second one requires torch. It builds the 4.08 B model on a `meta` device (no memory allocated), asserts the count, then runs the functional tests on a 2.72 M shrunken config that exercises the identical code path — including the KV-cache equivalence check that catches the classic MLA bug of rotating a cached vector.

**Next:** [4 — Tokeniser](04-tokenizer.md).
