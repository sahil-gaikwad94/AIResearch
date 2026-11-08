# 07 — Insights and Patterns

> Everything above describes one model at a time. This file is the audit: put all six side by side, compute the same quantities for each, and see which differences are real design choices and which are noise. Several of the patterns below only become visible when you line up the numbers, and a few of them contradict what the field's own marketing says.

**Method.** For every model I read the published `config.json` (or the technical report where a field is absent), recomputed total and active parameters from tensor shapes, recomputed KV-cache size per token, and compared each result to the vendor's stated number. My counts agree with vendor figures to within 0.3% on five of six models; the sixth is discussed in §9. The arithmetic is reproducible with the code in each chapter.

---

## 1. The master table

| | GPT-2 S | Mixtral 8x7B | Llama 3 8B | DeepSeek-V3 | Qwen3-Next | Flint |
|---|---|---|---|---|---|---|
| **Released** | 2019-02 | 2023-12 | 2024-04 | 2024-12 | 2025-09 | 2026-10 |
| **Total params** | 123.5M | 46.7B | 8.03B | 671B | 79.9B | 39.8B |
| **Active params** | 123.5M | 12.9B | 8.03B | 36.5B | 3.2B | 2.5B |
| **Model-level sparsity** | 1× | 3.6× | 1× | 18.4× | 25× | 16× |
| **Expert-level sparsity** | — | 4× | — | 32× | 51× | 43× |
| **Layers** | 12 | 32 | 32 | 61 | 48 | 49 |
| **Width** | 768 | 4,096 | 4,096 | 7,168 | 2,048 | 2,048 |
| **Width ÷ layers** | 64 | 128 | 128 | 117 | 43 | 42 |
| **Mixing operator** | MHA | MHA + top-2 | GQA | MLA | 3:1 GDN : attention | 3:1 window : full |
| **Q heads / KV heads** | 12 / 12 | 32 / 8 | 32 / 8 | 128 latent | 16 / 2 (12 layers) | 32 / 4 |
| **FFN width** | 3,072 | 8 × 14,336 | 14,336 | 256 × 2,048 | 512 × 512 | 256 × 512 |
| **Norm** | LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMS + zero-centre | RMS + QK-norm |
| **Position** | learned | RoPE | RoPE | RoPE (decoupled) | RoPE, 12 layers only | RoPE, 35 layers only |
| **Full-attention share** | 100% | 100% | 100% | 100% | 25% | 29% |
| **KV bytes / token / layer** | 3 KiB | 4 KiB | 4 KiB | 1.1 KiB | 0.25 KiB + state | 2 KiB |
| **Cache at max context** | 36 MiB @1k | 4.0 GiB @32k | 1.0 GiB @8k | 8.6 GiB @128k | 3.0 GiB @262k | 7.0 GiB @262k |
| **Extra heads** | none | none | none | MTP (1) | MTP (1) | none |
| **Vocab** | 50,257 | 32,000 | 128,256 | 129,280 | 151,936 | 150,144 |

Plus one reference point outside the six, because it is the same recipe at maximum scale: **Kimi K2** (Jul 2025) — 1.04T total, 32.6B active, 61 layers, MLA, 384 experts top-8, sparsity 32× (model) / 48× (expert), 15.5T tokens, trained with MuonClip. It is DeepSeek-V3 architecture with 50% more experts, half the attention heads, and a different optimiser.

---

## 2. Seven things that never changed

Before the differences, the constants — because they are what "transformer" actually means, and they survived 650× growth in parameter count:

1. **The block is still `norm → mix-across-tokens → add → norm → think-per-token → add`.** Not one of the six reorders these.
2. **The residual stream is still the backbone.** Every sublayer adds; nothing overwrites.
3. **Pre-norm still wins.** Normalise before the sublayer, not after.
4. **The causal mask is unchanged.** No model has replaced masked next-token prediction for pretraining.
5. **The objective is still cross-entropy on the next token** (+ MoE balancing, + MTP terms).
6. **Attention is still a query–key dot product.** Even the linear layers keep `q`, `k`, `v` as the conceptual interface — Gated DeltaNet is a refactoring of the same three projections, and MLA is a compressed parameterisation of the same operation.
7. **The output is still logits over a fixed vocabulary**, sampled with temperature/top-p.

If you are ever lost in a new architecture paper, find these seven. The paper will be a change to item 1's "mix" or "think" term, or to how the weights behind them are stored.

---

## 3. Ten things that changed, with the numbers

### 3.1 The MLP ate the model

Share of total parameters in the feed-forward (including all experts):

| GPT-2 | Llama 3 | Mixtral | Flint | DeepSeek-V3 | Qwen3-Next |
|---|---|---|---|---|---|
| 67% | 81% | 97% | 96% | 98% | 98% |

Two separate forces, same direction. In dense models, the FFN grew from 4× to 3.5×-of-a-3-matrix-block (Llama 3's `14336 = 3.5d`), pushing the block from two-thirds FFN to more than four-fifths. Then MoE multiplied the FFN by 8, 256, 512 — and attention, which cannot be replicated per expert (it would break the residual stream's shared meaning), became a rounding error in the weight file.

**Practical consequence:** when someone says "the model is 671B parameters", they mean 98% of it is feed-forward weights sitting in memory. Context length, reasoning, and routing all live in the remaining 2%.

### 3.2 Attention's share of *compute* collapsed, then rebounded

Share of **active** parameters per token spent on the mixing operator:

| | GPT-2 | Mixtral | Llama 3 | DeepSeek-V3 | Qwen3-Next | Flint |
|---|---|---|---|---|---|---|
| Share | 33% | **11%** | 19% | 32% | ~45% | 43% |

This is the most useful pattern in the file, and it is the causal chain behind everything in files 04–06.

- Dense era: attention is a third of the block, so it is a third of the cost.
- MoE arrives (Mixtral): the FFN is replicated 8×, so attention's *share* of per-token compute collapses to 11%. MoE made attention cheap — relatively.
- Then sparsity deepens (DeepSeek 8/256, Flint 6/256, Qwen 10/512): the active FFN shrinks back toward the size of the attention stack, so attention's share climbs to ~44%.
- At that point **the mixer is the bottleneck again** — and the field responds by making the mixer sub-quadratic: sliding windows (Flint), sparse top-k attention (DeepSeek-V3.2-Exp), linear recurrence (Qwen3-Next).

Read that way, "linear attention is the future" is not a fashion. It is the predictable consequence of winning the MoE battle: once the FFN is 2% of active compute, the only thing left to optimise is the part that grows with context.

### 3.3 The KV cache fell 16× per layer, while models grew 650×

Bytes of KV cache per token, per layer — the number that decides how many users fit on a GPU:

| GPT-2 | Llama 3 | Mixtral | DeepSeek-V3 | Flint | Qwen3-Next |
|---|---|---|---|---|---|
| 3 KiB | 4 KiB | 4 KiB | **1.1 KiB** | 2 KiB | **0.25 KiB** (+fixed state) |

Four distinct mechanisms, each buying roughly 2–8×:

1. **GQA** (2023–24): fewer KV heads than query heads. 512 → 128 KiB/token at Llama's width.
2. **MLA** (2024): cache a 512-dim latent + 64 rotary dims instead of 128 heads' keys and values. 57× versus MHA at DeepSeek's width.
3. **Sliding windows** (2025–26): cache only the window — *if the server evicts*.
4. **Linear recurrence** (2025–26): no cache at all; a fixed state.

And the absolute totals at each model's advertised context are strikingly flat — 1–8 GiB — which is the real design constraint: **every model is engineered so that one long sequence fits.** Context grew 256× (1k → 262k) while cache stayed within an order of magnitude, which means per-token cache had to fall by ~30×. That is exactly what happened.

### 3.4 Sparsity converged into a narrow band

Expert-level sparsity (how many times more experts exist than are selected):

| Mixtral (2023) | DeepSeek-V3 (2024) | Kimi K2 (2025) | Flint (2026) | Qwen3-Next (2025) |
|---|---|---|---|---|
| 4× | 32× | 48× | 43× | 51× |

Four independent teams, two countries, three years: **32–51×**. That is a convergence, not a coincidence. Below ~20× the compute saving is too small to justify routing complexity; above ~64× the router's job becomes impossible (choosing 10 of 640 experts with a single linear projection) and load balancing destabilises. Between 32 and 51 there is a wide, flat optimum.

The other half of the convergence is **expert width**: 512 (Flint, Qwen), 2,048 (DeepSeek, K2), against Mixtral's 14,336. Fine-grained experts won completely — nobody shipped a coarse-expert model after 2024.

### 3.5 Active parameters converged to 2–4B for the fast tier

| Mixtral | DeepSeek-V3 | Flint | Qwen3-Next | Kimi K2 |
|---|---|---|---|---|
| 12.9B | 36.5B | 2.5B | 3.2B | 32.6B |

Decoding is memory-bandwidth-bound: each token must read every weight it touches. So **active parameters ≈ bytes moved per token ≈ decode latency**, which makes the active count a direct proxy for speed. Interactive products (coding agents, chat, local models) cluster at 2.5–3.9B active in 2025–26; batch/quality products (API flagships) sit at 32–37B.

The absence of anything in between is the interesting part. There is no 8B-active 2026 frontier model — you either optimise for latency or for quality per token, and the middle is dominated by dense models at the same compute.

### 3.6 Depth is stable; width tracks *active* compute

Layers: 12 → 32 → 32 → 49 → 61 → 48. Since 2024, everything is 32–61 layers. Nobody is going deeper for capability at fixed compute; the returns are in width and sparsity.

Width is the tell: 2,048 for models with ~3B active (Flint, Qwen3-Next), 7,168 for 37B active (DeepSeek), 4,096 for 8B dense (Llama 3). Width ÷ layers is ~42 for both narrow 2026 models and ~117–128 for the wide ones. **Width is set by per-token compute; total capacity is set by expert count.** Once you know a model's width and its active count, you can guess its layer count to within 20%.

### 3.7 Positional encoding is disappearing

| | GPT-2 | 2023–24 | Flint (2026) | Qwen3-Next |
|---|---|---|---|---|
| Scheme | learned table | RoPE, every layer | RoPE on 35 of 49 layers | RoPE on 12 of 48 |
| Where absent | — | — | all 14 global layers | all 36 recurrent layers |

Nobody removed RoPE from the layers that need order. They removed it from the layers that need *content matching*: global attention in Flint (which retrieves by identity, not distance) and linear recurrence in Qwen (where position is implicit in the state's decay).

The conceptual shift: **position is a property of the mixing operator, not of the model.** A layer that matches content should not be told how far away the content is. Expect this to spread.

### 3.8 Vocabulary plateaued

50,257 (2019) → 128,256 (Llama 3) → 129,280 (DeepSeek) → 150,144 (Flint) → 151,936 (Qwen) → 160,000 (Kimi K2). The 2024 jump was real (better code and multilingual tokenisation); since then it has flattened at 128k–160k. Above that, the embedding matrix and the softmax cost start to bite, and the compression gains are marginal.

Related and counter-intuitive: **the embedding's share of the model collapsed** — 31% in GPT-2, 13% in Llama 3 8B, under 2% in every MoE. That is not because vocabularies shrank; it is because total parameters grew far faster. It also explains why "untied embeddings cost 525M extra" is a rounding error in a 671B model but a real 6.5% in an 8B one.

### 3.9 Normalisation became a stability tool

| GPT-2 | Llama 3 | Flint | Qwen3-Next |
|---|---|---|---|
| LayerNorm | RMSNorm | RMSNorm + **QK-RMSNorm** + sinks | **zero-centred** RMSNorm + weight decay on norms |

Two additions, both about *bounding* rather than centring:

- **QK-RMSNorm** bounds attention logits, preventing entropy collapse (a head that attends to exactly one token forever).
- **Zero-centred RMSNorm** (`1 + γ`, initialised at identity) plus weight decay on norm weights stops the residual stream's scale from drifting over 15T tokens.

At 671B parameters and ~$5M per training run, a single loss spike is catastrophic, so stability engineering became architecture. Kimi K2's QK-clip is the same instinct; so is DeepSeek's block-scaled FP8. **Anyone reading a 2026 config should read the norm and numerics fields as reliability engineering, not as maths.**

### 3.10 A second prediction head is now standard

None in GPT-2, Mixtral or Llama 3. Present in DeepSeek-V3 (MTP depth 1), Qwen3-Next (MTP depth 1), Kimi K2 (MTP). Three of the three post-2024 frontier open models.

It is the cheapest win in the stack: a small extra block that (a) forces the representation to plan further ahead during training, and (b) doubles as a draft model for speculative decoding at inference. If a 2026 model has no MTP head, that is worth asking about.

---

## 4. The six knobs

Strip the names and every model in this folder is the same skeleton with six dials. This is the design-space map; the value in each cell is what that model chose.

| Knob | Options seen | GPT-2 | Mixtral | Llama 3 | DeepSeek | Qwen3-Next | Flint |
|---|---|---|---|---|---|---|---|
| **1. Mixing operator** | full attn / window / sparse top-k / linear | full | full | full | full (→ sparse in V3.2) | 3:1 linear | 3:1 window |
| **2. KV sharing** | MHA / GQA / MLA / latent state | MHA | GQA 4:1 | GQA 4:1 | MLA latent | 8:1 + state | 8:1 |
| **3. FFN sparsity** | dense / E,k | dense | 8,2 | dense | 256,8 | 512,10 | 256,6 |
| **4. Norm** | LN / RMS / +QK / +zero-centre | LN | RMS | RMS | RMS | RMS+0c | RMS+QK |
| **5. Position** | learned / RoPE / conditional / none | learned | RoPE all | RoPE all | RoPE decoupled | 12 layers | 35 layers |
| **6. Extra head** | none / MTP(depth) | none | none | none | MTP(1) | MTP(1) | none |

Two things follow from the table being this small:

**The search space is exhausted in the obvious directions.** Every combination of these six dials is now explored territory. Genuine novelty in 2027 will have to come from a seventh knob — something like learned routing at the *layer* level, or a mixing operator with sub-linear memory that is not a recurrence (diffusion LMs, for instance, replace the whole left-to-right structure and are not represented here at all).

**Most "new architecture" announcements are a dial setting.** Before reading a launch post, check the six dials against the model it claims to beat. Usually two differ, and both were already published elsewhere.

---

## 5. One trade, in six costumes

Every optimisation in this folder is the same trade — you pay in one resource to save another. Written as a table, it becomes a decision procedure.

| Technique | Saves | Costs | Model |
|---|---|---|---|
| MoE | compute per token (5–25×) | memory capacity (all experts stored), interconnect traffic, routing instability | 03–06 |
| GQA | KV bytes (4–8×) | a little quality; forced sharing between query heads | 02–04 |
| MLA | KV bytes (57× vs MHA) | extra projection, kernel complexity, hard to quantise | 05 |
| Sliding window | attention FLOPs, cache | exact long-range recall | 04 |
| Linear recurrence | attention FLOPs and cache growth | exact recall (state bottleneck), kernel complexity | 06 |
| Attention sinks | stability, streaming | kernel compatibility (loses FlashAttention) | 04 |
| FP8 training | ~2× training throughput, memory | numerical drift, careful scaling | 05 |
| MTP | decode latency (~1.8×) | extra params, training complexity | 05, 06 |
| Distillation/quantisation | memory, latency | accuracy at the tail | (inference-time) |

The roofline framing that makes this concrete: **decoding is bandwidth-bound, not FLOP-bound.** At batch 1, an H100 can do ~1,000 TFLOP/s of BF16 math but only reads ~3 TB/s from memory. For a 3B-active model you need ~6 GB per token, so you get ~500 tokens/s at best — while the GPU's compute could in principle support 100× that. Every technique above is an attempt to move fewer bytes or spend fewer FLOPs *per useful token*, and the ones that fail commercially are the ones that save FLOPs while adding bytes (or vice versa).

---

## 6. Every compression buys a new failure mode

This is the pattern I did not expect to be so consistent: **you can predict a model's bugs from its compressions.**

| Compression | The failure it introduces | Seen in |
|---|---|---|
| MoE routing | Router collapse (a few experts do everything); load imbalance; all-to-all congestion; batch-size sensitivity | 03, 04, 05, 06 |
| Aux-loss balancing | Balancing fights specialisation; the penalty perturbs the LM objective | 03 |
| Fine-grained experts | Harder routing problem; needs a shared expert to cover common traffic | 04, 05, 06 |
| MLA | Custom kernels; absorption numerics; quantisation hostility | 05 (and K2) |
| Attention sinks | SDPA/FlashAttention incompatibility → eager fallback, float32 scores | 04 |
| Sliding window | Recall loss at range; benefit evaporates if the server does not evict | 04 |
| Linear recurrence | State bottleneck; recall degradation with prefix length | 06 |
| Sparse attention (DSA) | Indexer errors are silently unrecoverable — a dropped token is gone | 05 (V3.2-Exp) |
| FP8 | Silent quality loss from bad scaling | 05 |
| Extreme sparsity | Routing becomes a search problem; tiny experts specialise shallowly | 06 |

The practical version: when you evaluate a new architecture, do not only ask "is it better on benchmarks". Ask **what does it make fragile**, then test exactly that. For a hybrid linear model, test recall at 100k. For a sink-based model, test that your serving stack is not silently falling back to eager attention (a 3–4× throughput cliff that looks like "the model is slow" rather than "the model is misconfigured").

---

## 7. What `config.json` cannot tell you

This is the most important insight in the file, and it is the one beginners get wrong most often.

**Architecture sets the price and the ceiling. Data and post-training decide where you land underneath.**

Evidence from inside this folder's own model families:

- **Llama 3 8B → Llama 3.1 8B**: identical layer structure; the 3.1 config differs mainly by `rope_scaling` and context. Performance gains came from data and post-training, not architecture.
- **DeepSeek-V3 → V3.1 → V3.2-Exp**: the same 671B skeleton across all three. V3.2 changed attention to sparse DSA and cut API prices by more than half — a systems change, on unchanged weights structure.
- **Qwen3-Next Instruct vs Thinking**: the *same* architecture, two checkpoints, wildly different benchmark profiles. One config, two products.
- **Kimi K2 vs DeepSeek-V3**: near-identical architecture (same MLA, same 61 layers, same 8-of-N routing). K2's leap came from MuonClip (optimiser), 15.5T tokens, and a large agentic RL stage — none of which appears in a config.

A reasonable quantitative guess, from the public ablations: **architecture explains maybe 20–30% of the gap between two models at similar compute; data quality, data quantity, and post-training explain the rest.** That is why two 8B models with byte-identical configs can differ by 15 points on MMLU, and why "we copied the architecture" has never reproduced a frontier model.

Corollary for reading announcements: an architecture change is only interesting if it changes the **cost curve** (throughput, memory, context) or the **ceiling** (what is representable at all). Changes that only reshuffle parameters at equal cost usually show up as noise.

---

## 8. The reverse-engineering method

Six models in, the procedure is mechanical. It works on any open-weight model, and it takes about twenty minutes.

1. **Read the config, in order of information density.** `num_hidden_layers`, `hidden_size`, expert fields (`num_experts`, `num_experts_per_tok`), attention fields (`num_key_value_heads`, `head_dim`, any `*_lora_rank`), then `layer_types`.
2. **Classify the block.** Dense or sparse? What is the mixing operator? Any non-standard fields (`softmax_type`, `no_rope_layer_indices`, `mla_*`)?
3. **Count parameters by hand.** `L × (attn + ffn)` plus embeddings. If your number disagrees with the vendor's by more than 1%, you have misread a field — find it. (Mine disagreed by 15% on Qwen3-Next, and it taught me that `active` includes gate projections.)
4. **Compute the two cost numbers.** KV bytes per token per layer, and FLOPs per token (`2 × active`). These, not the parameter count, determine whether the model is usable.
5. **Draw one block with shapes.** Every error in your understanding shows up as a shape mismatch. This step catches more mistakes than reading the paper.
6. **List the tensors and look for the odd names.** The dictionary below.
7. **Find what is *missing*.** No `q_norm` → no QK-norm. No `shared_expert` → no shared expert. No `mtp` → no speculative decoding. Absences are design decisions too.
8. **Read one benchmark with the harness in mind.** Vendor numbers use vendor harnesses; the same model can move 5 points between evaluation frameworks.

### Tensor-name dictionary

| You see | It means |
|---|---|
| `c_attn` (fused 768×2304) | 2019-era fused QKV, GPT-2 lineage |
| `q_proj` / `k_proj` / `v_proj` | separate projections, 2023+ style |
| `k_proj` narrower than `q_proj` | GQA — the ratio is the group size |
| `q_norm` / `k_norm` | QK-RMSNorm (logit bounding) |
| `kv_a_proj_with_mqa`, `kv_b_proj`, `q_a_proj`, `q_b_proj` | **MLA** — latent attention with decoupled RoPE |
| `softmax_offset` / `sinks` | **attention sinks** — a learned logit in the denominator |
| `mlp.gate.weight` shaped (E, d) | MoE **router** (not to be confused with SwiGLU's `gate_proj`) |
| `experts.N.gate_proj` / `w1`,`w2`,`w3` | routed experts (naming differs by library) |
| `shared_expert*` | always-on expert, no router entry |
| `e_score_correction_bias` | **aux-loss-free** routing bias (no gradient) |
| `A_log`, `dt_bias` | state-space / gated linear layer (Mamba, GDN) |
| `conv1d` inside `linear_attn` | short causal convolution before a recurrence |
| `nextn` / `mtp` / `num_nextn_predict_layers` | multi-token prediction head |
| `lm_head` tied to `embed_tokens` | weight tying on |
| `q_lora_rank` in config | query compression — MLA or a low-rank attention variant |

---

## 9. Contradictions and loose ends found during the audit

Honest reporting, because these matter if you are going to rely on the numbers:

1. **"Active parameters" has no standard definition.** Qwen states 3B for Qwen3-Next; NVIDIA's model card for the same checkpoint says 3.9B; my hand-count gives 3.2B. The differences are whether the embedding matrix, the output head, the MTP module and the Gated DeltaNet gate projections are included. **Do not compare active-parameter counts across vendors**; compare measured throughput or recompute both from configs the same way.
2. **Mixtral is 46.7B, not 45B, and certainly not 56B.** Some library docs say 45B; the config arithmetic gives 46.7B (45.1B of experts + 1.34B attention + 0.26B embeddings). "8×7B" is a naming convention in which the 7B refers to a *dense* Mistral-7B-shaped model, not to each expert.
3. **Flint's context is 262,144 in the config**, while one press account describes "up to 512,000 with 256k recommended". The model card's number is the one the weights are trained for; treat larger figures as extrapolation.
4. **Mixtral's config sets `sliding_window: null`.** Mistral 7B's launch emphasised a 4,096-token sliding window; Mixtral does full attention over 32k. If you assumed Mixtral inherited the window, its long-context behaviour makes more sense now.
5. **DeepSeek-V3 has one shared expert, not two.** The config says `n_shared_experts: 1`; some secondary write-ups say 2. The config is authoritative.
6. **NVIDIA's model card labels Qwen3-Next "Hybrid Transformer-Mamba".** It is not Mamba — it is Gated DeltaNet, a different linear-attention family (delta rule with gating). The label is a category error that has propagated into several comparison tables.
7. **Llama 3 8B is 8.03B only because the embeddings are untied.** The "8B" is honest, but 1.05B of it is two copies of the embedding matrix, which matters if you are computing compute-per-token from the headline number.
8. **"Sparsity" is used for two different ratios.** Model-level (total ÷ active: 16–32×) and expert-level (experts ÷ selected: 32–51×). Flint is 16× by one measure and 43× by the other. Always ask which.

---

## 10. Predictions, with falsifiable markers

Stated as bets, each with the observation that would prove it wrong. These are my inferences from the six configs, not consensus.

| Prediction | Marker that would falsify it |
|---|---|
| Full-attention share in frontier open models falls to 15–20% by end-2027 | A 2027 flagship shipping ≥50% full layers |
| Attention sinks become default (they solve a real problem and cost only kernel work) | New configs with no `softmax_offset`/sink tensor and no sink paper in the model card |
| Expert-level sparsity goes to 64–128× | The band staying at 32–51 as models get bigger |
| MTP depth increases beyond 1 (depth 2–4) | MTP being dropped entirely for cost reasons |
| Position encoding disappears from all recurrent layers | A hybrid that puts RoPE back on its linear layers |
| The 2–4B active "fast tier" stays where it is | A 10B-active model becoming the latency leader |
| Vocabulary stops growing past ~200k | A 500k-vocab frontier model with a clear win |
| A recall-focused paper forces a partial retreat from pure linear attention | Hybrids moving to 50%+ full-attention layers |

The one I am least confident about is MTP depth: it multiplies training complexity for a speedup that speculative decoding can also get from a separate draft model. The one I am most confident about is sparsity continuing to rise — every mechanism in the stack rewards it, and the router is the only thing in the way.

---

## 11. Cheat sheet

| If you want to… | Do this | Evidence |
|---|---|---|
| Make decode fast | shrink **active** params (target 2–4B) | §3.5 |
| Add knowledge without slowing down | add experts, keep top-k fixed | 03, 05 |
| Serve long context cheaply | compress the cache (MLA) or remove it (linear layers) | 05, 06 |
| Keep exact recall | keep ≥25% full-attention layers | 04, 06 |
| Stabilise a big training run | QK-norm, zero-centred norms, clip attention logits | §3.9 |
| Predict a model's bugs | list its compressions | §6 |
| Estimate inference cost | `2 × active_params` FLOPs, `active_params × 2 bytes` read per token | §5 |
| Read a new architecture fast | six knobs, then the tensor names | §4, §8 |
| Know whether it will actually be good | look at data and post-training, not the config | §7 |

---

## 12. Self-check (the audit-level questions)

1. Llama 3 8B and Qwen3-Next both have 2,048-or-4,096 width and ~48 layers, but one is dense and one is 25× sparse. Which is faster to decode per token, and which needs more GPU memory? Give the number for each.
2. Attention's share of active compute went 33% → 11% → 45%. Explain each step with a mechanism, not a trend.
3. Why did nobody ship a coarse-expert MoE after 2024?
4. Flint and Qwen3-Next both put 25–29% of layers on global attention and remove position from the rest. Give one reason this is the same idea and one reason it is a coincidence.
5. A new model is announced as "1.2T parameters, 4B active, 64 experts of which 8 are used". Compute its model-level and expert-level sparsity, and say which of the two numbers the marketing will quote.
6. Two models have identical configs and differ by 12 points on a coding benchmark. Name three non-architectural explanations, in order of how likely they are.
7. You have to serve 262k-token contexts on 8×80 GB GPUs. Using §1, which of the six is the best fit and which is disqualified, and why?
8. Why is "active parameters" a better proxy for speed than "total parameters", and where does that proxy break down?

*(Answers: 1 — Qwen3-Next decodes far faster per token (3.2B vs 8.03B active ⇒ ~2.5× fewer bytes read) but needs ~160 GB for weights versus 16 GB, so it is worse on memory. 2 — dense era: attention is a third of the block; MoE: the FFN is replicated so attention's share falls to 11%; deeper sparsity: the active FFN shrinks toward the attention stack, so attention's share rises, which forces sub-quadratic mixing. 3 — fine-grained experts strictly dominate at equal active compute: same FLOPs, 16× the stored capacity, and finer specialisation. 4 — same: separate content retrieval from local ordering and stop paying attention cost at range; coincidence: Flint uses windowed attention for locality while Qwen uses a recurrence, so the "local" half is a different mechanism. 5 — 300× model-level, 8× expert-level; marketing quotes 300×. 6 — post-training data quality, then SFT/RL recipe, then evaluation harness; architecture is not in the top three. 7 — Qwen3-Next (3 GiB cache at 262k) is the fit; DeepSeek-V3 (8.6 GiB at 128k, and 671B of weights) and Mixtral (4 GiB at only 32k) are disqualified. 8 — decoding reads every active weight per token so active ≈ bytes moved; it breaks down at large batch (weights are reused, so FLOPs and interconnect dominate) and when the KV cache, not the weights, is the bottleneck.)*

---

**Back to:** [the index](README.md) · **Cross-references:** the six-model view complements the architecture chapter of the main book ([modern LLM architecture](../docs/02-modern-llm-architecture.md)) and the systems chapter ([efficiency and inference](../docs/09-efficiency-and-inference.md)).
