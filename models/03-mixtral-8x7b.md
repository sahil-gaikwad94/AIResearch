# 03 — Mixtral 8x7B (2023)

> **The model that made "active parameters" a phrase you have to understand.** Mixtral has 46.7B parameters in its weights file and uses 12.9B of them per token. It is the first open mixture-of-experts model that people actually deployed, and its design is simple enough to hold in your head: eight feed-forward networks per layer, pick two.
>
> If you take one thing from this file: **MoE is a memory-for-compute trade, not a magic parameter multiplier.**

| Property | Value |
|---|---|
| Released | December 2023, Mistral AI |
| Parameters | 46.7B total / **12.9B active per token** |
| Layers | 32 |
| Width | 4,096 |
| Attention | 32 query heads, 8 KV heads (GQA) — a 4:1 ratio, same as Llama 3 |
| Experts | 8 per layer, top-2 routing, **no shared expert** |
| Expert width | 14,336 (same as a Llama 3 MLP) |
| Vocabulary | 32,000 |
| Context | 32,768 (`sliding_window: null` in the config — see §7) |
| License | Apache 2.0 |

---

## 1. The problem MoE solves

A dense model has one embarrassing property: **capacity and cost are welded together.** Llama 3 8B knows what 8B parameters can know, and every token costs 8B parameters' worth of multiply-adds. Want a model that knows twice as much? Pay twice as much per token, forever, on every request.

But knowledge is not uniformly needed. The weights that answer a question about Kubernetes manifests are irrelevant to a question about Renaissance painting. A dense model cannot exploit that; it runs everything through every token anyway.

Mixture-of-experts exploits it: keep many specialised feed-forward networks in memory, and for each token run only a couple. Capacity scales with the number of experts, compute scales with how many you select.

The trade is explicit and unavoidable: **you must store all of it, and you must move the selected parts fast enough.** MoE converts a compute problem into a memory-capacity and memory-bandwidth problem.

---

## 2. The config, annotated

```json
{
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "num_local_experts": 8,
  "num_experts_per_tok": 2,
  "vocab_size": 32000,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-05,
  "max_position_embeddings": 32768,
  "sliding_window": null,
  "router_aux_loss_coef": 0.02,
  "router_jitter_noise": 0.0
}
```

New fields, compared to Llama 3:

- `num_local_experts: 8` — eight independent MLPs in every layer. ("local" means per-device; on one GPU it is just "8".)
- `num_experts_per_tok: 2` — the router picks the best two for each token. This is the **top-k**.
- `router_aux_loss_coef: 0.02` — the strength of the load-balancing penalty (§5). Note it is 20× the library default of 0.001; Mistral needed stronger pressure than usual.
- `router_jitter_noise: 0.0` — noise injection during training, disabled here.
- `sliding_window: null` — no window; full causal attention over 32k. This contradicts the marketing around Mistral 7B (which used a 4,096-token window) and is a good first lesson: **read the config, not the blog post.**

Everything else is Llama: RMSNorm, RoPE, GQA, SwiGLU, no biases.

---

## 3. One block, drawn

```
x  (1, T, 4096)
│
├─ rmsnorm → attention (GQA, unchanged, dense)         ← shared by ALL tokens
│  x = x + attn(x)
│
├─ rmsnorm → router: logits = x @ W_router   (8 numbers per token)
│            probs  = softmax(logits)
│            pick top-2 experts, keep their 2 probabilities
│
│   ┌── token 1 → experts 3, 7 ──┐
│   │   token 2 → experts 1, 3   │    different tokens, different experts
│   │   token 3 → experts 3, 6   │    in the SAME layer
│   └── token 4 → experts 0, 3 ──┘
│
│  out = Σ_{i ∈ top2}  p_i · SwiGLU_i(x)      ← only 2 of 8 MLPs run
│  x = x + out
│
▼  (next block)
```

Three things to notice, because they are where beginners misread MoE:

1. **Only the MLP is replicated.** Attention is one module, shared by everyone. This is why "8×7B" is not 56B: each expert is a 176M-parameter MLP, not a 7B model.
2. **Routing is per token, per layer.** Token 5 can use experts {3,7} in layer 0 and {1,4} in layer 1. There is no per-sequence routing.
3. **The expert output is weighted, not summed equally.** The two probabilities are renormalised to sum to 1 and used as the mixing weights, so a confident router can lean almost entirely on one expert.

---

## 4. The routing maths

```
h      = rmsnorm(x)                     (T, 4096)
logits = h @ W_r                        (T, 8)          W_r is 4096×8 — tiny
p      = softmax(logits / τ)            (T, 8)          τ = 1 at inference
top2   = argtopk(p, 2)                  (T, 2)
w      = p[top2] / Σ p[top2]            (T, 2)          renormalised weights
y      = Σ_i w_i · Expert_i(h)          (T, 4096)
```

The router is a single 4096×8 matrix — **32,768 parameters per layer**, 1M in the whole model, about 0.002% of the weights. The entire behaviour of a 46.7B-parameter model is gated by 1M parameters of routing. That asymmetry is the central engineering risk of MoE and the reason half of DeepSeek's architecture paper is about routers.

Softmax over 8 then top-2 is *token-choice* routing: each token independently picks. The alternative, *expert-choice* (each expert picks its tokens), guarantees balanced load but breaks causality — an expert cannot know which future tokens will want it.

---

## 5. Why routers collapse, and what the auxiliary loss does

Left alone, routing degenerates. Early in training a few experts get slightly better, get picked more, get more gradient, get better still. Within a few thousand steps a handful of experts handle everything and the rest are dead weight — you paid for 46.7B parameters and are effectively running a 13B model.

The fix in Mixtral is an **auxiliary load-balancing loss**, added to the language-modelling loss:

```
L_aux = α · N · Σ_i  f_i · P_i

f_i = fraction of tokens in the batch routed to expert i     (hard count)
P_i = mean router probability for expert i over the batch    (soft)
N   = number of experts
α   = router_aux_loss_coef = 0.02
```

It is minimised when every expert receives the same fraction *and* the router's mean probability is uniform. `f` is not differentiable, so it is treated as a constant; the gradient flows through `P`. With `α = 0.02`, this is a real term in the objective, not a rounding error.

**Why this is a compromise rather than a solution.** Minimising `L_aux` means pushing the router toward uniform routing — but uniform routing is not what makes a good model. The router's *job* is to specialise; the loss tells it not to. Every MoE before 2024 lived with this tension. DeepSeek-V3's contribution (file 05) was to replace the loss with a bias term updated outside the gradient, so the router is balanced without being lobotomised. Keep that in mind: it is one of the more elegant fixes in the field.

---

## 6. Count the parameters yourself — and the two numbers that matter

**Per block:**

| Component | Shape | Parameters |
|---|---|---|
| attention (q, k, v, o) | as Llama 3 | 41.94M |
| one expert (gate, up, down) | 3 × 4096 × 14336 | 176.16M |
| 8 experts | 8 × 176.16M | **1,409M** |
| router | 4096 × 8 | 0.03M |
| **one block (total)** | | **1,451M** |
| **one block (active)** | attention + 2 experts | **394.3M** |

**Whole model:**

- total: 32 × 1,451M + 2 × 32,000 × 4096 = 46.4B + 0.26B = **46.7B**
- active: 32 × 394.3M + 0.26B = **12.9B**

So Mixtral stores 46.7B and computes like a 13B dense model. **Sparsity ratio = 46.7 / 12.9 = 3.6×.** Only 25% of the FFN runs per token.

Sanity check against the name: "8×7B" invites you to compute 56B. The real number is 46.7B because attention (1.34B) and embeddings (0.26B) are not replicated eight times, and each expert is 176M, not 7B.

---

## 7. What it actually costs to serve

This is where MoE stops being free.

**Memory.** 46.7B × 2 bytes = 93.4 GB in BF16. Mixtral needs two 80 GB GPUs or four 48 GB GPUs to hold, where a 13B dense model fits on one 24 GB card. You pay for capacity whether or not you use it.

**Bandwidth.** Decoding is memory-bound: each generated token must read every weight it touches. Per token, Mixtral reads ~12.9B parameters ≈ 25.8 GB at BF16 — comparable to a 13B dense model, which is the point.

**But the arithmetic intensity depends on batch size.** At batch 1, reading an expert's 176M parameters serves exactly one token. At batch 64, the same read can serve all the tokens routed to that expert. MoE throughput therefore *improves* with batching, and gets *worse* at batch 1 relative to its FLOP count. This is why MoE is a datacenter technology and why a 46.7B MoE can feel slower than a 13B dense model on a laptop despite identical FLOPs.

**Expert parallelism.** In multi-GPU serving, experts are spread across devices and every token's hidden state must be **all-to-all**-shipped to whichever devices hold its chosen experts, then shipped back. That interconnect traffic is the MoE tax, and it is why NVLink-class bandwidth matters for these models.

**Dropped tokens.** Capacity limits per expert (in training) can overflow, dropping tokens entirely; Mixtral-era implementations avoided this by not enforcing a capacity factor, at the cost of variable compute per batch.

---

## 8. Reverse-engineering lab

```python
from transformers import AutoConfig
import torch

cfg = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
print(cfg.num_local_experts, cfg.num_experts_per_tok,
      cfg.hidden_size, cfg.intermediate_size)          # 8 2 4096 14336

# 1. Total vs active, from first principles
d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
ff, E, k = cfg.intermediate_size, cfg.num_local_experts, cfg.num_experts_per_tok
hd, kv = d // cfg.num_attention_heads, cfg.num_key_value_heads
attn = d*d + 2*d*kv*hd + d*d
expert = 3*d*ff
total  = L*(attn + E*expert) + 2*V*d
active = L*(attn + k*expert) + 2*V*d
print(f"total {total/1e9:.1f}B  active {active/1e9:.1f}B  "
      f"sparsity {total/active:.1f}x")                 # 46.7B 12.9B 3.6x

# 2. How much of the model is a single router?
print(f"router per layer: {d*E} params = {d*E/total*100:.4f}% of the model")

# 3. See the experts in the tensor list
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16)
blk = m.model.layers[0]
print([n for n, _ in blk.named_parameters() if "experts" in n][:4])
# block_sparse_moe.experts.0.w1.weight  (14336, 4096)
# block_sparse_moe.experts.0.w2.weight  (4096, 14336)
# block_sparse_moe.experts.0.w3.weight  (14336, 4096)
# ...and the router, which is easy to miss:
print(blk.block_sparse_moe.gate.weight.shape)          # torch.Size([8, 4096])
```

Note the naming convention change: experts use `w1`/`w2`/`w3` (gate, down, up in that library's order) instead of `gate_proj`/`up_proj`/`down_proj`, and the router is called `gate` — *not* to be confused with the SwiGLU `gate_proj`. Misreading which "gate" is which is the most common mistake when reading MoE code.

---

## 9. What Mixtral gets wrong

Its design is deliberately crude, and each crudeness motivated a specific fix in the models that follow.

| Mixtral choice | Why it is limiting | Fixed by |
|---|---|---|
| **8 experts, each 176M wide** | Coarse granularity: a token either gets a whole 176M MLP or nothing. Specialisation is blunt. | Fine-grained experts — 256 experts of 2048 wide (05), 512 of 512 (06) |
| **Top-2 of 8 = 25% active FFN** | Weak sparsity (3.6×). Most of the capacity is still paid for per token. | Sparsity ratios of 8–50× (04, 05, 06) |
| **Aux loss (α = 0.02)** | Balancing fights specialisation; the loss perturbs the LM objective. | Aux-loss-free bias update (05) |
| **No shared expert** | Common knowledge (syntax, frequent words) must be duplicated in every expert or learned eight times. | Shared expert (04, 05, 06) |
| **Full attention at every layer** | 128 KiB/token cache; quadratic cost at 32k. | Sliding windows (04), MLA (05), linear layers (06) |
| **One token per forward pass** | No speculative decoding built into the model. | MTP heads (05, 06) |

Point three is worth dwelling on, because it is counter-intuitive: **fine-grained experts are better than coarse ones at equal active parameters.** Splitting 8×176M into 64×22M and picking more, narrower experts gives the router a much finer-grained basis to compose from — like choosing 8 of 64 small tiles instead of 2 of 8 big ones. DeepSeek's 2024 MoE paper is essentially the empirical case for this, and by 2026 it is the universal default.

---

## 10. Self-check

1. Mixtral is 46.7B total, 12.9B active. Which parts are *not* active per token, and why can the model not simply skip them at load time?
2. Why does MoE help at batch 256 and hurt at batch 1, when the FLOP count per token is identical?
3. What does `router_aux_loss_coef = 0.02` do, and why can't you just set it to 0?
4. If you doubled the experts to 16 while keeping top-2, what happens to total parameters, active parameters, and quality-per-FLOP — and what does the sparsity ratio become?
5. Explain, without using the word "mixture", why "8×7B = 56B" is wrong.
6. A 46.7B MoE and a 13B dense model have nearly identical active compute. Name two reasons the MoE can still be strictly better, and two reasons it can be worse.

*(Answers: 1 — six of eight experts per layer are never read for a given token, but you cannot know which before the token arrives, and the router's choice changes per layer and per token. 2 — bandwidth: at batch 1 an expert read serves one token; at batch 256 it amortises across the tokens routed to it. 3 — it penalises unbalanced routing so experts do not die; setting it to 0 lets early random advantages snowball into a collapsed router. 4 — total goes to ~93B, active stays 12.9B, sparsity 7.2×; quality per FLOP typically improves until routing becomes hard to learn. 5 — attention and embeddings are shared, and an "expert" is a 176M MLP, not a 7B model. 6 — better: more knowledge per FLOP, and better specialisation of rare domains; worse: 93 GB of weights, and interconnect all-to-all plus batch sensitivity.)*

---

**Next:** [04 — Flint](04-flint.md), where routing gets *finer* (6 of 256), the expert gets *narrower*, and most attention layers stop looking at the whole context.
