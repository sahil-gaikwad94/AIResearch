# 5 — Parameters and Config

> One layer by hand, then the totals, then the ratios that tell you whether the shape is sane. The reference implementation reproduces the inference count **exactly** — not within 1%, not within 0.1%.

---

## 5.1 One layer, by hand

`d = 4096`, `d_c = 512`, `d_qc = 512`, `d_rope = 64`, `h = 32`, `d_h = 128`, `d_ff = 12288`.

**Attention (MLA):**

| Matrix | Shape | Params |
|---|---|---|
| `W_DQ` | 4096 × 512 | 2,097,152 |
| `W_UQ` | 512 × 4096 | 2,097,152 |
| `W_QR` | 4096 × 64 | 262,144 |
| `W_DKV` | 4096 × 576 | 2,359,296 |
| `W_UK` | 512 × 4096 | 2,097,152 |
| `W_UV` | 512 × 4096 | 2,097,152 |
| `W_O` | 4096 × 4096 | 16,777,216 |
| `q_norm`, `k_norm` | 128 each | 256 |
| attention sinks | 4 per head | 4 |
| **subtotal** | | **27,787,524** |

**Indexer (hierarchical):**

| Matrix | Shape | Params |
|---|---|---|
| `W_IQ` (query) | 4096 × 128 | 524,288 |
| `W_IK` (fine key) | 512 × 128 | 65,536 |
| `W_IP` (coarse pool) | 512 × 128 | 65,536 |
| **subtotal** | | **655,360** |

**FFN (SwiGLU) and norms:**

| Matrix | Shape | Params |
|---|---|---|
| `gate` | 4096 × 12288 | 50,331,648 |
| `up` | 4096 × 12288 | 50,331,648 |
| `down` | 12288 × 4096 | 50,331,648 |
| `ffn_norm`, `attn_norm` | 4096 each | 8,192 |
| **subtotal** | | **151,003,136** |

**Per layer:** 27,787,524 + 655,360 + 151,003,136 = **179,446,020**

**Body:** 179,446,020 × 20 = **3,588,920,400**

| | Params |
|---|---|
| Body (20 unique layers) | 3,588,920,400 |
| Final RMSNorm | 4,096 |
| Loop-index embeddings (8 × 4096) | 32,768 |
| Embeddings (131,072 × 4096), tied | 536,870,912 |
| **Inference total** | **4,125,828,176** |
| MTP heads, 2 × (4096² + 4096) | 33,562,624 |
| **Training total** | **4,159,390,800** |

Verified by two independent implementations:

```
analytic  : 4,125,828,176    torch (meta) : 4,125,828,176
```

The assertion is `==`, not a tolerance. Day 1's "within 0.1%" allowed a 4 M-parameter error to pass silently; Kestrel-2's exactness is why the two real bugs found during implementation (a missing positional offset and a miscounted MTP norm) were caught in minutes rather than at 300 B tokens.

---

## 5.2 The ratios

The numbers that say whether a 4 B model is shaped like a 4 B model.

| Ratio | Kestrel-2 | Day 1 | Reference point | Verdict |
|---|---|---|---|---|
| tokens / parameter | 309 | 294 | Chinchilla 20; modern small models 200–500 | deliberately overtrained |
| `d_ff / d_model` | 3.0 | 2.5 | Llama-3-8B: 3.5 | width restored (day 1 was narrow) |
| attention : FFN | 1 : 5.4 | 1 : 3.3 | Llama-3-8B: 1 : 2.6 | attention is cheap when sparse |
| embeddings / total | 13.0 % | 9.9 % | 4 B models: 8–15 % | acceptable |
| **params per layer of effective depth** | **79.3 M** | **102 M** | — | **22 % better** |
| KV / token (FP8) | 11,520 B | 20,480 B | GQA-4: 40,960 B | 3.6× better than GQA-4 |
| KV / token / layer | 576 B | 512 B | — | MLA latent grew with width |

That "params per layer of effective depth" row is the design's central accounting. At `R = 3` the model has depth 52; dividing 4.126 B by 52 gives 79.3 M per layer, versus day 1's 102 M for depth 40. **The same parameter budget buys 22% more layers of computation**, and pays for it in FLOPs rather than memory.

The counter-column, stated honestly:

| Cost of the loop | Value |
|---|---|
| Training FLOPs/token at 4 k, R=2 vs R=1 | 1.61× |
| Inference FLOPs/token at 4 k, R=2 vs R=1 | 1.64× |
| Cache penalty | **zero** (fixed-memory recurrence) |

The cache penalty being zero is the design's justification for accepting the FLOP penalty.

---

## 5.3 Why 20 unique layers and not 40

Because 40 unique layers at `d = 4096` is 8 B parameters, and the loop is a cheaper way to buy depth than width is to buy capability.

The alternative — 40 unique layers at `d = 3072`, which is roughly day 1's shape — was rejected for a specific reason: **sparse attention benefits from a wider head dimension and from having fewer layers to traverse**, because each sparse layer's fixed cost (window 2,048 + anchors 256 + selection 768) is paid per layer. At 40 layers, that fixed cost is 3,072 × 40 = 123 k tokens of reads per query even at short context. At 20 layers it is 61 k. The shallow-wide shape is *cheaper in sparse attention*, which is not true of dense attention.

That is a genuine interaction between two decisions, and it is the kind of thing that only shows up when the cost model is written before the architecture is frozen.

---

## 5.4 Why `d_c = 512` and not 448

Day 1 used 448 because it was the largest latent that kept MLA's parameter count below GQA-4's at `d = 3072`.

At `d = 4096` that constraint is looser, and 512 has two practical advantages: it is a multiple of 64 (kernel alignment for FP8 scaled matmuls), and `d_c / d_model = 0.125` exactly, which makes the up/down projections a clean 8× expansion in each direction.

The cost is 64 more floats per layer per token: 20 × 64 = 1,280 B/token, or 11% of the cache. Bought deliberately, for kernel simplicity, in a design that reads 96× less of the cache than day 1 does anyway.

---

## 5.5 Why six of the twenty layers are unshared

Four of the twenty (2 prelude + 2 coda) are unshared, following the `prelude-recur-coda` template that both the recurrent-depth paper and the iso-depth scaling study use. The reasons are not decorative:

- **Prelude:** the first two layers do local feature extraction and establish the representation the recurrent block iterates on. Sharing them would force one block to do both jobs.
- **Coda:** the last two layers must turn a *converged* state into a distribution. Sharing them means every intermediate iteration also writes to the output — which the entropy-exit training uses as a signal, but which destabilises the final head.
- **Parameter accounting:** 4 of 20 layers unshared means 20% of parameters are outside the loop, which bounds how much of the model's behaviour is a fixed-point iteration. A fully shared model is more parameter-efficient and less stable; 4/20 is the conservative version, and B4 tests the extreme.

---

## 5.6 Config

[`kestrel2/config.json`](kestrel2/config.json) is the canonical artefact. The fields that carry a decision, annotated:

```jsonc
"depth": {
  "prelude_layers": 2, "recurrent_layers": 16, "coda_layers": 2,
  "recurrent_steps_train": [1, 2, 3],       // sampled per batch, mean 2.0
  "recurrent_steps_default": 3,             // serving default (depth 52)
  "recurrent_steps_max": 8,                 // depth 68
  "kv_policy": "fixed-memory"               // makes R free in cache terms
},
"attention": {
  "sparsity": {
    "window": 2048,
    "compression_stride": 64,               // n/64 global summaries
    "selection_block": 64, "selection_topk": 12,
    "anchor_budget": 256,                   // ALWAYS attended
    "anchor_source": "deterministic: def/class/import/signature lines",
    "anchor_learned": false                 // cannot be trained away
  }
},
"optimizer": {
  "pretrain": "muon",
  "post_train": "adamw"                     // the advantage reverses; see 06 §6.6
}
```

---

## 5.7 What the accounting cannot tell you

Parameters are necessary and not sufficient. The count says the shape is plausible; it says nothing about whether:

- the indexer learns to select rather than to attend to position (B1);
- the loop improves reasoning at 4 B rather than only at the scales in the literature (B3);
- the fixed-memory recurrence loses enough expressivity to matter (B3, arm 2);
- sparse attention at 4 k is worth its 1.6× cost (B1, and it may not be — which is why pretraining runs at R=1 for 63% of tokens).

Those are four separate experiments, budgeted in [09](09-evals-and-ablations.md).

**Next:** [06 — Training](06-training.md).
