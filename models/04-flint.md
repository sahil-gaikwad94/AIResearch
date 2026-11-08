# 04 — Flint (39.8B total / 2.5B active, 2026)

> **The most modern design in this folder, at a size you can actually run.** Flint is a coding-agent model from Recursive (October 2026) that combines every trend of the last two years at once: extreme sparsity, sigmoid routing with a shared expert, mostly-windowed attention, attention sinks, and position encoding applied only where it is needed. It also shows what happens when architecture search is automated — the team started from a 35B/3B baseline and let a system explore expert configurations.
>
> Its distinguishing property: it stores **16× more parameters than it uses per token**, at a total size small enough for a workstation.

| Property | Value |
|---|---|
| Released | 18 October 2026, Recursive (Apache 2.0) |
| Parameters | 39.8B total / **≈2.5B active** (body alone ≈2.2B) |
| Layers | 49: 2 dense + 47 MoE |
| Width | 2,048 |
| Attention | 32 query heads, 4 KV heads, head dim 128 → q width 4,096 (wider than the residual stream) |
| Attention pattern | 35 sliding-window layers (window 513) + 14 full-attention layers |
| Position encoding | RoPE **only on sliding layers**; none on full-attention layers |
| Experts | 256 per MoE layer, top-6, plus 1 always-on shared expert |
| Expert width | 512 (shared expert 1,024) |
| Router | sigmoid scores, renormalised, aux-loss coefficient 0.0005 |
| Sink | one learned logit per attention head (`softmax_type: "learnable"`) |
| Vocabulary | 150,144 |
| Context | 262,144 (vendor recommends capping at 256k) |
| Vendor benchmarks | 54.9% Terminal-Bench 2.1, 89.7% SWE-bench Pro v2, 74.2% SWE-bench Verified |

---

## 1. Why include a model this new

Three reasons.

First, it is the clearest example of the **2026 default stack**: a small-active, high-sparsity MoE with hybrid attention. Qwen3-Next (file 06) does the same at 80B, DeepSeek-V3.2-Exp does it with sparse attention at 671B — Flint does it at 40B, which means the config is small enough to reason about completely.

Second, it is **honest about its target**: a coding agent on local hardware. Every design decision follows from "we want few active parameters per token so it decodes fast on an M4 Max, but enough total capacity to be good at code."

Third, it contains two components nothing earlier in this folder has — attention sinks and no-RoPE full-attention layers — and both are worth understanding because they are appearing everywhere.

---

## 2. The config, annotated

This is the real `config.json`. It is long because the layer pattern is written out explicitly.

```json
{
  "hidden_size": 2048,
  "num_hidden_layers": 49,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "intermediate_size": 6144,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 1024,
  "num_experts": 256,
  "num_experts_per_tok": 6,
  "mlp_only_layers": [0, 1],
  "sliding_window": 513,
  "layer_types": ["full_attention","full_attention","sliding_attention", ... 49 entries ...],
  "no_rope_layer_indices": [0,1,5,9,...,45,48],
  "softmax_type": "learnable",
  "router_score_function": "sigmoid",
  "norm_topk_prob": true,
  "router_aux_loss_coef": 0.0005,
  "rope_theta": 10000.0,
  "rms_norm_eps": 1e-06,
  "vocab_size": 150144,
  "max_position_embeddings": 262144,
  "tie_word_embeddings": false
}
```

Field by field, because each one is a decision:

- `hidden_size: 2048` — a *narrow* model. Compare Llama 3's 4,096 at 8B. Width sets the cost of every dense operation; Flint keeps it small and buys capacity in the experts instead.
- `num_attention_heads: 32, head_dim: 128` — the query projection outputs 32 × 128 = 4,096, **twice the residual width**. Modern models routinely make attention wider than the stream: more room for heads to specialise, while the expensive per-token MLP stays narrow.
- `num_key_value_heads: 4` — aggressive GQA. 8 query heads share each K/V pair.
- `intermediate_size: 6144` — the dense MLP width, used only in layers 0 and 1 (see `mlp_only_layers`).
- `moe_intermediate_size: 512` — each of the 256 experts is a *tiny* MLP: 3 × 2048 × 512 ≈ 3.1M parameters. Compare Mixtral's 176M per expert. This is the "fine-grained" idea taken to its limit.
- `shared_expert_intermediate_size: 1024` — one always-on expert, twice as wide as a routed one, that every token uses.
- `num_experts_per_tok: 6` — of 256. Sparsity ratio 42.7×.
- `sliding_window: 513` — attend to the current token plus the 512 before it. (513 and not 512 because the window includes the query position. Off-by-one like this is extremely common in configs; always ask whether a window counts the current token.)
- `layer_types` — the explicit per-layer schedule: full attention at indices 0, 1, 5, 9, 13, …, 45, 48 (14 layers), sliding attention everywhere else (35 layers). The pattern is one full layer every four, with two at the start and one at the end.
- `no_rope_layer_indices` — **exactly the full-attention layers.** Position encoding is switched off precisely where the model looks at the whole context, and on everywhere else. §5 explains why this is not a bug.
- `softmax_type: "learnable"` — attention sinks. One learned scalar per head (§6).
- `router_score_function: "sigmoid"` — routing scores are independent per expert, not a normalised distribution (§4).
- `norm_topk_prob: true` — renormalise the selected six weights to sum to 1.
- `router_aux_loss_coef: 0.0005` — 40× weaker than Mixtral's 0.02, because sigmoid routing plus renormalisation needs far less balancing pressure.

---

## 3. One block, drawn

```
x (1, T, 2048)
│
├─ rmsnorm
├─ attention:  sliding (window 513)  OR  full  ──────  depends on layer index
│              └─ QK-RMSNorm per head, learned sink logit per head
│              └─ RoPE applied only on sliding layers
│  x = x + attn_out
│
├─ rmsnorm
├─ if layer ∈ {0,1}:  dense SwiGLU, width 6144
│  else:             router (sigmoid over 256) → top-6
│                    out = 1.0 · shared(x)  +  Σ wᵢ · expertᵢ(x)
│  x = x + out
▼
```

The layer schedule across all 49:

```
0  full, no rope, dense
1  full, no rope, dense
2  slide, rope, MoE
3  slide, rope, MoE
4  slide, rope, MoE
5  full, no rope, MoE
6..8   slide
9  full        ... repeating 4-layer period ...
...
45 full
46 slide
47 slide
48 full, no rope
```

Two dense layers at the bottom is a recurring pattern: cheap, always-on computation to get basic features (token identity, local syntax) before routing begins. DeepSeek uses three. The reason is practical — routers work badly on raw embeddings, so you give them one or two processed layers to look at.

---

## 4. Idea one: extreme sparsity at small size

Do the arithmetic on what a token actually costs.

| Per MoE layer | Parameters activated |
|---|---|
| attention (q, k, v, o) | 18.9M |
| shared expert | 6.3M |
| 6 routed experts × 3.1M | 18.9M |
| **total active** | **≈ 44.1M** |

The three contributions are almost exactly **1 : ⅓ : 1**. Attention and routed experts are balanced; the shared expert is a third of either. That is a deliberately tuned ratio, and it is the opposite of Mixtral, where attention was about 11% of the active block (41.9M of 394.3M).

**Why 6 of 256 instead of 2 of 8.** With 256 tiny experts, the router composes a token's feed-forward from six small bases. The 256 experts together hold 805M parameters per layer — 16× more than the 50M a token actually uses. Capacity is decoupled from compute by a factor of 16.

The cost is routing difficulty. Choosing 6 of 256 is a much harder learning problem than 2 of 8, and the router must be stable or you get pathological specialisation. That is why the routing design (§4b) is more careful than Mixtral's.

### 4b. Sigmoid routing

Mixtral: `p = softmax(W_r x)`, then take top-2. Softmax couples all eight scores — raising one lowers the others.

Flint: `sᵢ = sigmoid(wᵢ · x)`, a score in (0,1) per expert, **independent**. Take the top six, then renormalise:

```
s      = sigmoid(x @ W_r)        (T, 256), each entry independent
top6   = argtopk(s, 6)
w      = s[top6] / Σ s[top6]     renormalise to sum to 1  (norm_topk_prob)
```

Why this matters at 256 experts: with softmax over 256, every probability is ~1/256 and the top-k boundary is razor-thin — a tiny change in one logit reshuffles the selection. With independent sigmoids, an expert's score says "how relevant am I", full stop, and the selection is far more stable across similar tokens.

The renormalisation matters too: sigmoid scores do not sum to 1, so without it the layer's output scale would drift with however many experts happened to score high.

**The shared expert is the other half of this.** One expert (width 1,024, always on, ungated) handles everything common: syntax, frequent words, formatting. The 256 routed experts are then free to specialise narrowly without any of them having to be a generalist. DeepSeek introduced this idea; by 2026 it is standard (Flint, DeepSeek-V3, Qwen3-Next all have exactly one shared expert).

---

## 5. Idea two: most attention is local, and global attention has no positions

This is the most interesting thing in the config, and it is easy to skim past.

**Only 14 of 49 layers see the full context.** The other 35 see 513 tokens. Cost implications:

- Sliding attention is O(T · 513) instead of O(T²). At 256k tokens, a sliding layer does ~1/500th the attention work of a full layer.
- Its KV cache can be evicted: a sliding layer only ever needs the last 513 keys and values, so a correct serving stack keeps 513 per layer instead of 262,144.

**And the 14 global layers have no positional encoding at all.** `no_rope_layer_indices` is exactly the set of full-attention layers. So the model is:

- local layers: *where* something is (RoPE, 513-token window)
- global layers: *what* is in the context, with position deliberately destroyed

Think about why that is defensible. A head doing global retrieval — "find the function definition mentioned earlier", "find the error message" — wants content matching, not proximity. Applying RoPE to a 200k-token context forces the model to reason about enormous distances that it saw rarely in training, and creates the long-range "attention dilution" problem where far tokens get tiny but nonzero weight. Removing position from exactly those layers sidesteps it: the layer sees a bag of tokens and can match on content alone.

The division of labour is: sliding layers maintain order and locality; full layers maintain identity and recall. Qwen3-Next reaches a similar conclusion by a different route (file 06: linear recurrent layers for local state, 12 full layers for global) — and Flint's earlier sibling design, sliding-window + a few global layers, is the pattern GPT-OSS used too.

**Caveat, stated plainly:** position-free global attention should hurt tasks that need precise long-range ordering ("the third item in the list 100k tokens ago"). Recursive's benchmarks are coding-agent tasks where retrieval dominates. This is a design bet, not a free lunch, and it is one of the things to watch as independent evaluations arrive.

---

## 6. Idea three: attention sinks

A softmax must put all of its probability mass somewhere. Every row sums to 1. So a head is *forced* to attend to something even when nothing is relevant — and when it has nothing useful, the least-bad option is usually to dump mass on the first token (this is the well-documented "attention sink" behaviour in unmodified transformers).

Flint makes that explicit and learnable: **each head gets one extra logit that participates in the softmax denominator but has no value vector attached.**

```
scores  = q·kᵀ / √d                     (T, T)
scores  = concat(scores, sink_logit)    (T, T+1)     sink_logit is a learned scalar
weights = softmax(scores)               (T, T+1)
out     = weights[:, :T] @ v            the sink's weight is simply discarded
```

The sink is an escape hatch: the head can attend to "nothing", contributing zero to the output while still satisfying the normalisation. Empirically this stabilises training (no head is forced to invent an attention target) and makes streaming inference possible, because evicting a token from the cache no longer distorts the softmax — the mass can move to the sink instead of being redistributed over unrelated tokens.

**The engineering cost, and why it is instructive.** The transformers documentation says it directly: SDPA cannot apply attention sinks, so Flint falls back to the **eager** attention implementation, computing scores in float32. To get speed you need a sink-aware FlashAttention kernel. This is a general law of architecture: *every compression or extra term buys a new class of kernel incompatibility.* DeepSeek's MLA had the same problem in 2024; so did every linear-attention model.

**QK-RMSNorm** is the companion trick. Before the dot product, normalise each head's query and key vectors. Without it, the norm of `q` and `k` grows during training, logits grow, softmax saturates into a near one-hot pattern, and the head stops learning (a phenomenon often called entropy collapse). Normalising q and k bounds the logits by construction. Qwen3 and GPT-OSS do the same thing.

---

## 7. Count the parameters yourself

| Component | Shape | Parameters |
|---|---|---|
| `q_proj` | 2048 × 4096 | 8.39M |
| `k_proj` | 2048 × 512 | 1.05M |
| `v_proj` | 2048 × 512 | 1.05M |
| `o_proj` | 4096 × 2048 | 8.39M |
| **attention** | | **18.88M** |
| one routed expert | 3 × 2048 × 512 | 3.15M |
| shared expert | 3 × 2048 × 1024 | 6.29M |
| dense MLP (layers 0–1) | 3 × 2048 × 6144 | 37.75M |

- layers 0–1: 2 × (18.88 + 37.75) = 113.3M
- layers 2–48 (47 MoE layers): 47 × (18.88 + 256 × 3.15 + 6.29) = 47 × 830.5M = 39.03B
- embeddings + output head (untied): 150,144 × 2048 × 2 = 615.0M
- **total = 39.76B** ✓ (vendor: 39.8B)

Active per token:

- layers 0–1: 113.3M
- 47 × (18.88 + 6.29 + 6 × 3.15) = 47 × 44.07M = 2.071B
- output head only (the vendor convention excludes the input embedding): 307.5M
- **active ≈ 2.49B** ✓ (vendor: ≈2.5B including the output head, ≈2.2B body only)

Both vendor numbers make sense now, and the discrepancy between "2.5B" and "2.2B" is just the embedding-head convention. That is a useful lesson: **active-parameter counts are not standardised.** When comparing two models, check whether embeddings and the output head are included.

---

## 8. The KV cache, honestly

Per-token cache for Flint at BF16: `2 × 4 heads × 128 dims × 2 bytes = 2 KiB` per layer, over 49 layers ≈ **98.0 KiB per token**. That is *higher* than DeepSeek's 68.6 KiB despite Flint being 17× smaller — because Flint's cache is dominated by 49 layers of full-precision K/V at a 128-dim head, while DeepSeek compresses into a 576-dim latent.

But the sliding layers can be evicted:

| | If nothing is evicted | If windows are enforced |
|---|---|---|
| 35 sliding layers | 35 × 262,144 × 2 KiB = 17.5 GiB | 35 × 513 × 2 KiB = **35 MiB** |
| 14 full layers | 7.0 GiB | 7.0 GiB |
| **at 256k context** | **24.5 GiB** | **≈7.0 GiB** |

The lesson generalises: **a sliding-window architecture is only as cheap as its serving stack.** The weights describe an opportunity; the kernel and the cache manager decide whether you get it. When someone benchmarks a hybrid model's long-context memory and gets a bad number, ask whether the runtime was actually evicting.

---

## 9. How it was built (and why that matters)

Recursive's account: Flint was produced largely through an internal automated research system ("Eureka Machine") that proposed architecture changes, ran experiments and planned follow-ups. The starting point was a 35B-total / 3B-active baseline; the search varied expert configurations and widths and landed on 39.8B / 2.5B. Total build time: four months, from scratch through pretraining, mid-training, SFT and RL.

Treat the marketing framing with normal scepticism, but the *artefact* is checkable: 40B of open weights with a specific, unusual config. And the direction of the search result is itself informative — given freedom, the search moved **fewer active parameters and more total parameters**, i.e. it bought speed by buying sparsity. If architecture search is doing its job, that is exactly what it should discover for a latency-bound local target, and it matches what human teams chose independently (Qwen3-Next: 3.9B active, 80B total).

---

## 10. Reverse-engineering lab

```python
import json, urllib.request
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("RecursiveSI/flint")
print(cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok,
      cfg.sliding_window, cfg.softmax_type)      # 49 256 6 513 learnable

# 1. Recover the layer schedule from the config itself
full  = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
slide = [i for i, t in enumerate(cfg.layer_types) if t == "sliding_attention"]
print(len(full), len(slide), full[:6])           # 14 35 [0, 1, 5, 9, 13, 17]
print(set(full) == set(cfg.no_rope_layer_indices))   # True — global layers have no RoPE

# 2. Verify 39.8B / 2.5B
d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
hq, hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
E, k = cfg.num_experts, cfg.num_experts_per_tok
attn = d*hq*hd + 2*d*hkv*hd + hq*hd*d
expert = 3*d*cfg.moe_intermediate_size
shared = 3*d*cfg.shared_expert_intermediate_size
dense  = 3*d*cfg.intermediate_size
total  = 2*(attn+dense) + 47*(attn + E*expert + shared) + 2*V*d
active = 2*(attn+dense) + 47*(attn + shared + k*expert) + V*d
print(f"{total/1e9:.2f}B total, {active/1e9:.2f}B active")   # 39.76B, 2.49B

# 3. Find the sink tensor — the giveaway is the name
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("RecursiveSI/flint", torch_dtype=torch.bfloat16)
print(m.model.layers[5].self_attn.softmax_offset.shape)   # one scalar per head
print(m.model.layers[5].self_attn.q_norm.weight.shape)    # QK-RMSNorm, per head
```

The tensor names here are the whole reverse-engineering story: `softmax_offset` = sinks, `q_norm`/`k_norm` = QK-RMSNorm, 256 `experts.<i>.gate_proj` = fine-grained MoE, and a `shared_expert` that has no router entry.

---

## 11. What Flint gets wrong

| Choice | The cost |
|---|---|
| **Sinks + no-RoPE global layers** | SDPA cannot run sinks; you fall back to eager or need a custom kernel. No-RoPE may hurt precise long-range ordering. |
| **40B of weights for 2.5B of compute** | Excellent on a machine with unified memory and good bandwidth (M4/M5 Max, a single B200); awkward on many small GPUs, where the weights dominate. |
| **Very narrow experts (512)** | Each expert is small enough that specialisation is fine-grained but shallow; more experts must agree to compute anything complex. |
| **Coding-agent specialisation** | Benchmarks are code-agent benchmarks. Chat, multilingual and long-document behaviour are under-tested. |
| **Vendor-reported evals** | Terminal-Bench and SWE-bench scores are from the publisher, with different harnesses from the comparison numbers. |

---

## 12. Self-check

1. Flint activates 2.5B of 39.8B. What is the sparsity ratio, and how does it compare with Mixtral's 3.6×? What does the model buy with the difference?
2. Why does `sliding_window` equal 513 rather than 512?
3. Explain in one sentence each: why full-attention layers have no RoPE, and why sliding layers do.
4. A sink logit has no value vector. What exactly happens to its softmax weight, and why does that help a streaming server?
5. `norm_topk_prob: true` renormalises the six selected sigmoid scores. What breaks if you set it to false?
6. The KV cache at 256k is 24.5 GiB without eviction and ~7 GiB with it. Which number would a naive benchmark report, and why?
7. Flint's router aux-loss coefficient is 0.0005 — 40× smaller than Mixtral's. Give two reasons sigmoid routing needs less balancing pressure.

*(Answers: 1 — 39.8/2.5 = 15.9×; Mixtral buys 3.6×, so Flint holds ~4.4× more unused capacity per active parameter, i.e. more knowledge per FLOP. 2 — the window includes the current query position, so 512 past tokens + 1 = 513. 3 — global layers do content matching where absolute distance is noise; local layers need order to encode syntax and recency. 4 — its weight stays in the denominator and is simply not multiplied by any value, so it contributes zero; evicted tokens' mass can transfer to the sink instead of distorting real tokens. 5 — output magnitude would fluctuate with how many experts scored high, changing the residual stream's scale layer by layer. 6 — the un-evicted 24.5 GiB, because most naive stacks allocate max-context buffers. 7 — sigmoid scores are independent so a high score for one expert does not suppress others; and the shared expert absorbs common traffic, so routed experts are not all competing for the same tokens.)*

---

**Next:** [05 — DeepSeek-V3](05-deepseek-v3.md), where the same ideas appear at 671B — plus the attention compression that made long context affordable at all.
