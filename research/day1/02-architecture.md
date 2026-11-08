# 2 — Architecture

> The spec, the math, and the alternatives that lost. Every number here is produced by `tools/count_params.py` and asserted against the instantiated model in `kestrel/model.py`.

---

## 2.1 The block

```
 token ids ──► Embedding (131072 × 3072, tied with lm_head)
                 │
                 ▼
      ┌──────────────────────────────────────────────────┐
      │  x ──► RMSNorm ──► MLA ──────────► + ──┐         │
      │  │                                     │         │  × 40
      │  └──► RMSNorm ──► SwiGLU ─────────────► + ──┐    │
      └────────────────────────────────────────────┼────┘
                                                   ▼
                                              RMSNorm
                                        ┌──────────┴──────────┐
                                        ▼                     ▼
                                  lm_head (tied)        MTP head 1 ──► t+2
                                        │               MTP head 2 ──► t+3
                                        ▼               (training only)
                                    logits (131072)
```

Two deviations from a standard 2026 block, both deliberate:

- **Every layer is full attention.** Not 75% local — zero local. See §2.3.
- **MLA, not GQA.** Not for the usual reason (KV savings) but because at this size it is *also* cheaper in parameters. See §2.4.

---

## 2.2 Dimensions

| Symbol | Value | Note |
|---|---|---|
| `vocab` | 131,072 | tied embeddings, see [04](04-tokenizer.md) |
| `d_model` | 3,072 | |
| `n_layers` | 40 | |
| `n_heads` | 24 | `24 × 128 = 3072` exactly — no head padding |
| `d_head` | 128 | |
| `d_c` (kv latent) | 448 | `d_c / d_model = 0.146` |
| `d_qc` (q latent) | 448 | query low-rank bottleneck |
| `d_rope` | 64 | decoupled positional channels |
| `d_ff` | 7,680 | `2.5 × d_model` |
| `mtp_heads` | 2 | training only, dropped at inference |
| `rope_theta` | 500,000 | |
| context | 64 k native, 256 k extended | |

| Parameter group | Count | Share |
|---|---|---|
| Per-layer attention (MLA) | 21.234 M | 23.1 % of body |
| Per-layer FFN (SwiGLU) | 70.779 M | 76.9 % of body |
| Per layer | 92.019 M | |
| Body (40 layers) | 3.681 B | |
| Embeddings (tied) | 0.403 B | |
| MTP modules (train only) | 0.019 B | |
| **Total, inference** | **4.083 B** | |
| **Total, training** | **4.102 B** | |

---

## 2.3 The central bet: full attention at every layer

### The case for hybrid, stated fairly

At 64 k context, attention arithmetic is not a rounding error:

| Context | Attention FLOPs/token (fwd+bwd) | As a share of `6N` |
|---|---|---|
| 4,096 | 6.0 × 10⁹ | 24.7 % |
| 16,384 | 2.4 × 10¹⁰ | 98.6 % |
| 65,536 | 9.7 × 10¹⁰ | **394 %** |
| 262,144 | 3.9 × 10¹¹ | 1,578 % |

At 64 k, full attention costs ~5× the parameter FLOPs. That is exactly why DeepSeek, Qwen, Kimi and Flint all went hybrid: at 4× the cost of the FFN, attention stops being affordable, so you replace 75–82% of it with a linear-time recurrence and keep a few full layers for global mixing.

### Why that reasoning does not transfer here

Three separate arguments, in increasing order of strength.

**(a) Training pays the quadratic once; inference pays it once per unique token.** The 394% figure is a *training* number. At inference, prefix caching means a 64 k repo is prefilled once and then read from cache for the rest of the session. The 20-turn session in §1.1 prefills 190 k unique tokens, not 1.04 M.

**(b) Cache, not compute, is the residency constraint.** Concurrent sessions are limited by KV bytes, and hybrid attention's advantage there is real (a fixed 37.7 MB state instead of 1.25 GiB of cache at 64 k) but is exactly what MLA already buys at 4× (1.25 GiB vs 5 GiB for GQA-4). Using both is belt and braces; using the cheaper one alone keeps exact recall.

**(c) Recall is the requirement, not a nice-to-have.** The companion experiment on state capacity argues that hybrid models are limited by a *capacity* law — the number of distinct associations the fixed state can hold scales with state dimension, roughly `m* ∝ d_v` — and not by context length. For prose that is a graceful degradation. For code it is fatal: the model must reproduce `_build_attention_mask` exactly, and a state that stores an association approximately will occasionally produce `_build_attn_mask`, which does not exist.

The failure mode is not "slightly worse summaries". It is "the patch does not compile", which the verifier scores as zero. Under a verifiable reward, approximate recall is not a partial credit — it is a hard failure.

### The trade we are making, explicitly

| | Hybrid (75% linear) | Kestrel (full MLA) |
|---|---|---|
| Training FLOPs @64k | ~1.4× 6N | 4.9× 6N |
| Prefill FLOPs @64k, uncached | ~1.4× 6N | 4.9× 6N |
| Prefill FLOPs, cached session | ~1.4× 6N (still O(n) per token) | amortised to 190 k unique tokens |
| KV per token | 37.7 MB fixed state | 20 KiB/token (1.25 GiB @64k) |
| Exact recall over 40 k spans | degraded by design | **exact** |
| Kernel complexity | high (two attention paths) | medium (one path, MLA) |

We pay ~3.5× the training compute of the 64 k stage — which is only 8% of the total budget ([05 §5.3](05-training.md)) — to keep a property that a verifier punishes us for losing.

That is the bet. Ablation **A2** in [08](08-evals-and-ablations.md) tests it directly: full-MLA vs 3:1 hybrid at equal KV budget, judged on repo-level task success, with a pre-registered rule to switch if hybrid wins by more than 3 points.

---

## 2.4 MLA, and the surprising parameter result

### The mechanism

```
x ─┬─► W_DQ (3072→448) ─► RMSNorm ─► W_UQ (448→3072) ─► q_nope  (24 × 128)
   └─► W_QR (3072→1536) ─────────────────────────────► q_pe    (24 × 64) ─► RoPE
x ──► W_DKV (3072→512) ─┬─ c_kv  (448)  ── CACHED
                        └─ k_pe  (64)   ── CACHED (post-RoPE)

c_kv ─┬─► W_UK (448→3072) ─► k_nope (24 × 128)
      └─► W_UV (448→3072) ─► v      (24 × 128)

scores = [q_nope ‖ q_pe] · [k_nope ‖ k_pe]ᵀ / √192
```

**Why the rope key must be decoupled.** The whole point of MLA is that the cache holds `c_kv` — a single 448-dim vector shared by all 24 heads — instead of 24 separate key and value vectors. But RoPE must be applied *after* the per-head up-projection, and RoPE is position-dependent while `c_kv` is cached once. You cannot rotate a cached vector later, because the rotation depends on absolute position and the cache is reused at every position.

The fix is to carry position in a channel that is cheap to cache and cheap to rotate: a single shared `k_pe` of 64 dims, rotated *before* caching. Then the cache is `c_kv ‖ k_pe` = 512 floats, and `k_nope`/`v` are reconstructed on the fly from `c_kv` by two small matmuls.

This is the one subtlety that makes MLA work, and it is why the parameter table has a 4.7 M `W_QR` that looks redundant.

### The comparison that decided it

Same model, three attention designs, identical `d_model`, `n_heads`, `d_head`:

| Design | Params / layer | KV per token (FP16) | KV @64 k | Note |
|---|---|---|---|---|
| MHA | 37.75 M | 480 KiB | 30 GiB | 2 × 24 × 128 floats |
| GQA-4 | 22.02 M | 80 KiB | 5.0 GiB | 4 kv heads |
| **MLA (448 + 64)** | **21.23 M** | **40 KiB** | **2.5 GiB** | **12× compression vs MHA** |

Read the first column. **MLA uses fewer parameters than GQA-4 and a third fewer than MHA, while using 4× less cache than GQA-4.** The reason is that the query path is bottlenecked through 448 dims (`W_DQ` + `W_UQ` = 2.75 M instead of 9.44 M for a full `W_Q`), which pays for the 4.72 M rope projection and still comes out ahead.

This is not a trade-off. It is a domination, and it is the single most important empirical fact in this document. It is also why the design could afford to refuse hybrid attention: the cache problem had already been solved three times over.

At FP8, the number that matters:

| Context | Cache per session |
|---|---|
| 8 k | 0.16 GiB |
| 64 k | **1.25 GiB** |
| 256 k | 5.00 GiB |

---

## 2.5 Attention sinks

Four learned sink logits per head, prepended to the softmax denominator and carrying no value.

They cost 96 parameters. They matter for three reasons:

1. **Eviction.** A cache-eviction policy that drops old tokens shifts the softmax normaliser and produces a distribution the model never saw in training. With sinks, "attend to nothing" is representable, so evicting is safe in the way it is for StreamingLLM.
2. **Long-context drift.** Heads that have nothing useful to attend to currently spread probability over the whole prefix; sinks give them a legitimate null option.
3. **Compatibility.** Every serious long-context model shipping in 2026 has them, so the failure mode of *not* having them is well documented.

The reference implementation applies sinks by concatenating them into the score matrix before softmax and dropping their weights afterwards — the exact semantics of the production kernel.

---

## 2.6 The 77/23 split

FFN is 70.78 M per layer, attention 21.23 M. Attention is 23% of the body, versus ~19% for Llama-3-8B. The difference is entirely MLA's `W_QR` and the up/down projections — the price of the small cache, and worth it.

`d_ff / d_model = 2.5`, below the 3.5 of Llama. This is deliberate: at 4 B parameters with a 131 k vocabulary taking 403 M, the FFN has to give up width to keep the total. The alternative — shrink the vocabulary to 65 k and widen the FFN — is tested as part of ablation **A1**, since vocabulary size and tokeniser efficiency are the same decision.

---

## 2.7 Depth over width

Ablation **A3** compares, at exactly 4.08 B parameters:

| | Depth-heavy (chosen) | Width-heavy |
|---|---|---|
| Layers | 40 | 28 |
| `d_model` | 3,072 | 4,096 |
| `d_ff` | 7,680 | 9,728 |
| KV per token (FP8) | 20,480 B | 14,336 B |
| KV @64 k | 1.25 GiB | 0.87 GiB |

The cache is **not** identical between the two, and it is worth being precise about why: MLA costs 512 floats per layer per token regardless of width, so cache scales with *layers*. The width-heavy model is 30% cheaper in cache and faster to decode (one large matmul beats many small ones at small batch). The depth argument therefore has to overcome a real cost, not win for free.

The case for depth is that small-model reasoning tracks sequential composition steps, and 40 layers give 12 more of them than 28. The decision rule in A3 is a ≥1.5-point win on repo-level task success at 1 B parameters with μP transfer; below that, the width-heavy variant wins on cache and serving cost. This is a genuinely open question at this scale, which is why it is an ablation rather than a claim.

---

## 2.8 Multi-token prediction

Two MTP heads. Each is `RMSNorm → Linear(3072→3072)` applied to the trunk's final hidden state, followed by the tied output head, predicting `t+2` and `t+3` with loss weight 0.3.

MTP is included for two reasons, and the honest ranking is that the second matters more:

1. **Gradient density.** Each position contributes three prediction targets instead of one, which measurably improves the trunk at small scale — the DeepSeek-V3 ablation is the canonical evidence.
2. **Speculative decoding for free.** The heads provide draft tokens at zero extra memory and zero extra forward passes. For a coding model whose output is long and structurally predictable (indentation, closing braces, common identifiers), draft acceptance rates are high, and the measured speedup is ~2×.

Note that (2) improves *decode* latency, which is 1–11% of compute. That sounds negligible and is not: the 800 generated tokens are the ones a human waits for. Prefill is throughput; decode is the user's experience.

Both heads are dropped at inference, which is why the parameter count is quoted twice (4.083 B inference / 4.102 B training).

---

## 2.9 Norm, init, stability

Nothing exotic, everything load-bearing:

- **RMSNorm, ε = 1e-6**, pre-norm only.
- **QK-norm**: per-head RMSNorm on queries and keys before the dot product. Costs 256 parameters per layer; prevents attention-logit blowup, which is the most common cause of loss spikes above 1 B parameters.
- **Zero-init on residual-facing projections** (`w_o` and `down_proj`). Every block starts as the identity function, so a 40-layer model at initialisation is a 40-layer residual stream with no signal amplification. This is GPT-2's trick and it is still the cheapest stability guarantee available.
- **No biases anywhere.** They cost parameters, add a synchronisation point, and buy nothing measurable at this scale.
- **No dropout.** 1.2 T tokens on 4 B parameters; overfitting is not the failure mode.
- **bf16 mixed precision, fp32 master weights and optimizer states.** FP8 matmuls are revisited in [05 §5.5](05-training.md) as a *stage-2* decision with a loss-matching gate, not as a default.

---

## 2.10 Tied embeddings

Saves 403 M parameters — 10% of the model — which is real money at this size. The known cost is that the input embedding and output classifier are forced to share a geometry, which hurts when `d_model` is small relative to vocabulary. At `d_model = 3072` and 131 k tokens the ratio is 42.6, comfortably in the regime where tying is neutral.

If ablation **A1** shrinks the vocabulary to 65 k, tying becomes even safer and the savings halve.

---

## 2.11 Rejected, with reasons

| Alternative | Why not | Revisit when |
|---|---|---|
| **Hybrid linear attention** (3:1) | Loses exact recall (C1); cache problem already solved by MLA (§2.4) | A2 shows hybrid wins >3 points at equal KV |
| **Sliding-window + global interleave** (Flint-style) | Same objection; window 513 would be catastrophic for repo recall | Never at this scale |
| **MoE** (≈12 B total / 1.9 B active) | 3× rollout cost for RL (C3); routing instability under 20 different reward shapes; expert-parallel serving complexity for a single-GPU deployment | A8 shows >8% gain at equal active FLOPs |
| **NoPE on some layers** | Plausible per the companion experiment, but unmeasurable at this budget and RoPE-on-all already gives position | A7, as a 4-layer variant |
| **Looped / recurrent depth** | Attractive for reasoning-per-parameter, but no evidence at 4 B and it breaks the fixed-depth kernel path | After v1, if depth-dependence is confirmed |
| **Diffusion LM** | 10× latency win in theory; quality and tool-calling reliability unproven | Never for this product |
| **Mamba/DeltaNet as the trunk** | Recall (C1); the companion experiment argues the limit is capacity, not length | If a state-capacity law shows `m*` is large enough |
| **Per-layer attention variants** | No ablation budget to justify; complexity without a thesis | — |
| **Learning-rate-free / Muon optimizer** | Promising, but 1.2 T tokens is not the place to discover optimizer bugs | A9 side-experiment at 1 B |

---

## 2.12 What I would change at 10× the budget

Not part of the spec, but the honest answer to "what would you do with more money":

1. **Hybrid, but only in the bottom third.** Keep full attention in the top 2/3 (where recall matters most for generation) and linear in the bottom (where it is mostly feature extraction). This is a *different* bet from the field's uniform 3:1, and it is only testable with a real budget.
2. **A 1.5 T-token vocabulary-matched tokeniser trained jointly with the model** rather than fixed upfront.
3. **Depth 56 at the same width**, with the extra layers paid for by a 65 k vocabulary.
4. **Process-level RL on plans**, not just outcomes — but only if a verifier for plan quality can be built, which is the hard part.

**Next:** [3 — Config and parameters](03-config-and-parameters.md).
