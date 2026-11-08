# 09 — Efficiency and Inference: Where the Money Goes

*Read time: 65 minutes. Prerequisites: `02` §2.5, `01` §1.8.*

---

## 9.1 The asymmetry that drives everything

**Prefill** (reading the prompt): all tokens processed in parallel, one big matmul, **compute-bound**. Fast per token, high FLOPs, cheap relative to output length.

**Decode** (generating output): one token at a time, and *every* step must read the entire model's weights from HBM plus the whole KV cache. **Memory-bandwidth-bound.** A 671B FP8 model means ~671 GB read per token. At 3 TB/s of HBM bandwidth (H100), that's a hard floor of ~4.5 tokens/sec per sequence regardless of how fast the math is.

Everything in this chapter follows from that one sentence. Optimisations either:

- **reduce bytes moved** (quantization, MoE sparsity, MLA, smaller weights), or
- **amortise the read over more tokens** (batching, speculative decoding, MTP), or
- **reduce wasted work** (paged KV cache, prefix caching, continuous batching).

---

## 9.2 Quantization

**Idea:** store weights (and maybe activations/KV) in fewer bits.

- **FP16/BF16** = 16 bits. **FP8** (E4M3) = 8. **INT8/FP4/NF4** = 8/4/4. **Ternary (BitNet b1.58)** ≈ 1.58 bits.

### Post-training quantization (PTQ)

Train in high precision, then compress. The question is how to choose the scale/rounding so accuracy survives.

| Method | Key idea |
|---|---|
| **GPTQ** (2023) | Layer-wise, uses a second-order (Hessian) approximation of the reconstruction error to decide rounding; 3–4 bit weights with small loss |
| **AWQ** (2023) | Observes a few *activation-salient* channels matter disproportionately; scales weights to protect them. 4-bit with strong results; robust and simple |
| **SmoothQuant** | Migrates activation outliers into the weights so both can be quantized to INT8 |
| **GGUF / llama.cpp k-quants** | A family of CPU-friendly formats and mixed-precision schemes; the standard for local inference |
| **bitsandbytes NF4** | The 4-bit format QLoRA uses; normal-float quantile-based bins |

**The empirical rules of thumb:**
- **8-bit weights**: essentially free (no measurable quality loss) for most models.
- **4-bit weights**: small loss, usually acceptable; the standard for local serving.
- **≤3-bit weights**: quality degrades noticeably and non-uniformly — math and code suffer first. Needs distillation or quantization-aware training to recover.
- **KV-cache quantization** is often more valuable than weight quantization for long-context serving, because the cache is what limits batch size. FP8 KV cache is now default in vLLM/SGLang for many models.

### Training-time quantization

- **FP8 training** (DeepSeek-V3, 2024): per-block scaling, selective BF16 promotion for sensitive ops; ~2× throughput, no loss degradation. Now standard.
- **FP4 / NVFP4** (Blackwell): 4-bit floats with per-block microscaling; mainly inference today, with 2026 work on FP4 *training*.
- **BitNet b1.58** (Microsoft): train natively with ternary weights `{−1, 0, +1}` and 8-bit activations. The 2B model (bitnet-b1.58-2B-4T) was trained on 4T tokens from scratch with matmul-free-ish kernels: **1.37–5.07× speedup on ARM, 2.37–6.17× on x86, 55–82% energy reduction**, matching full-precision peers at that scale. The claim is that extreme quantization should be a *training-time design choice*, not a post-hoc squeeze. Whether it scales to frontier quality is still open, but it's the most credible path to cheap local inference.

---

## 9.3 Parameter-efficient fine-tuning: LoRA and friends

**Full fine-tuning** of a 70B model needs ~1 TB of optimizer state. **LoRA** (Hu et al. 2021) instead freezes `W` and learns a low-rank update:

```
W' = W + (α/r) · B A        where A ∈ R^{r×d}, B ∈ R^{d×r}, r ≪ d
```

Trainable parameters drop by 100–10,000×; you can fine-tune a 70B model on a single 80GB GPU. **QLoRA** (Dettmers 2023) additionally quantizes the frozen base to 4-bit NF4, enabling 65B fine-tuning on one 48GB card.

**Why it works:** fine-tuning mostly needs a low-dimensional *direction* of change, not new capacity. Empirically, LoRA matches full fine-tuning on most tasks up to a point; it underperforms on tasks requiring substantial new knowledge or very large distribution shift.

**Operational superpower:** you can serve many LoRAs over one base model, swapping adapters per request (multi-LoRA serving). This is how teams ship per-customer customisation cheaply.

**Variants:** DoRA (decompose magnitude and direction), rsLoRA (better scaling), LoRA+ (different LR for A and B), GaLore (gradient low-rank projection for full-parameter-ish training at low memory).

---

## 9.4 Speculative decoding: buy tokens in bulk

**Problem:** decode is memory-bound, so you're paying to read 671 GB to produce one token. Wasteful.

**Solution:** use a cheap **drafter** to propose `k` tokens, then run the big model **once** to verify all `k` in parallel. Verification is a single forward pass over `k+1` positions (prefill-like, compute-bound, cheap). Accept the longest correct prefix; sample the first rejected token from the corrected distribution.

Crucially, **the output distribution is exactly the target model's** (with the right acceptance rule), so this is a pure speedup, not an approximation.

**Drafter options:**
- A small model of the same family (classic).
- **Medusa** (2024): extra decoding heads predicting multiple future tokens; no separate model.
- **EAGLE / EAGLE-2/3** (2024–25): draft at the *feature* level with a lightweight head and a tree of candidates; ~3–4× speedups in practice, widely deployed (vLLM supports EAGLE).
- **MTP heads** (`02` §2.8): the model's own multi-token-prediction head is a free drafter (Qwen3-Next's "NEXTN").
- **N-gram / suffix / prompt-lookup drafting**: no model at all — match suffixes from the context or a lookup table. Cheap, works well for code and repetitive output. GLM-5.3-Flash reportedly ships 51B-parameter n-gram tables alongside the model for exactly this.

**When it doesn't help:** high batch sizes (the GPU is already saturated, so verification isn't free) and low acceptance rates. 2025–26 work (Nightjar, AgentSpec) makes speculation *adaptive* — disable it when the load is high, or constrain drafting to semantically coherent segments in agent workflows.

---

## 9.5 Serving systems: vLLM, SGLang, TensorRT-LLM

**PagedAttention / vLLM** (Kwon et al. 2023) is the single most important serving innovation. Before it, each request pre-allocated a contiguous KV cache for its maximum length → 60–80% of GPU memory wasted on fragmentation. PagedAttention borrows OS virtual memory: store the KV cache in fixed-size **blocks** with a page table, allocate on demand, share blocks across requests with the same prefix (**prefix caching**).

Consequences: 2–4× higher throughput, continuous batching, and prefix sharing. Every modern serving stack does this.

**Other essentials:**
- **Continuous batching**: add/remove requests every step instead of waiting for a batch to finish.
- **Chunked prefill**: interleave prefill chunks with decode steps so a long prompt doesn't stall everyone.
- **Prefix caching**: reuse the KV of a shared system prompt or document — enormous for RAG/agents (`08`).
- **Disaggregated prefill/decode**: run prefill on one pool of GPUs and decode on another, passing the KV cache between them. Since prefill is compute-bound and decode is memory-bound, separating them lets each pool be provisioned and quantized differently. NVIDIA Dynamo, vLLM, and SGLang all support it; 2026 work even quantizes the two phases differently ("disaggregated quantization": NVFP4 prefill, 2–4-bit decode).
- **Kernel libraries**: FlashAttention/FlashInfer/FlashMLA, fused MoE kernels (CUTLASS), CUDA graphs, `torch.compile`.

**The stack in one line:** vLLM and SGLang are the open-source defaults; TensorRT-LLM is the NVIDIA-optimised path; Dynamo is the orchestration layer.

---

## 9.6 Hardware: what changed 2022–2026

| Generation | Year | Key specs | Relevance |
|---|---|---|---|
| **A100** | 2020 | 80 GB HBM2e, 2 TB/s, BF16 | Trained GPT-3-era and Chinchilla-era models |
| **H100** | 2022 | 80 GB HBM3, 3.35 TB/s, FP8, TMA | The workhorse of 2023–25; FlashAttention-3 exploits its async pipeline |
| **H200** | 2024 | 141 GB HBM3e, 4.8 TB/s | More capacity/bandwidth → better decode and bigger KV caches |
| **B200 / GB200** | 2024–25 | 192 GB HBM3e, FP4 tensor cores, 2-die package | FP4 inference, cheaper per token |
| **GB300 / Blackwell Ultra** | 2025 | 288 GB HBM3e/GPU, 20 TB HBM per 72-GPU rack, 130 TB/s NVLink, ~1.1–1.4 EF FP4 | Rack-scale: a 72-GPU NVLink domain acts like one giant GPU |
| **2026** | | higher-bandwidth HBM, optical interconnect, custom ASICs (TPU v6+, Trainium, MTIA) | Competition on $/token, not peak FLOPS |

**The point of rack-scale (NVLink domain = 72 GPUs):** MoE all-to-all and tensor parallelism are *communication*-bound. Making 72 GPUs share one high-bandwidth fabric changes what's feasible — you can put a 1T-parameter MoE's experts in one domain.

**Power is now the binding constraint.** A 100k-GPU cluster draws ~100+ MW; grid interconnection and cooling are multi-year problems. This is why "tokens per megawatt" is now a first-class metric (NVIDIA claims ~5× improvement GB300 vs H100 per MW).

---

## 9.7 The cost arithmetic you should be able to do

**Cost per 1M output tokens** for a dense model, roughly:

```
time_per_token ≈ (bytes_of_weights_moved) / (bandwidth × efficiency)
cost = time × price_per_gpu_second / batch_size
```

Practical worked example (2026, order of magnitude):
- 70B model, FP8 (70 GB), H100 at 3 TB/s, 70% efficiency → ~30 tokens/s/sequence unbatched.
- With batch 32: ~960 tokens/s aggregate (weights read once for the whole batch).
- At $2/GPU-hour = $0.00056/s → $0.58 per 1M tokens. Add overheads, KV cache, prefill, and a margin → the market price of ~$1–3/M output tokens for mid-tier models is consistent.

**This is why batching is everything in serving economics**, why MoE (fewer active bytes) is cheaper at scale, and why quantization gives an almost linear cost reduction until accuracy breaks.

---

## 9.8 Efficiency techniques ranked by real-world impact

1. **Batching + PagedAttention/continuous batching** — 5–20× throughput. Nothing else is close.
2. **Quantization to FP8/INT4 weights + FP8 KV cache** — 1.5–3× throughput, ~4× memory.
3. **Prefix caching** — 2–10× for RAG/agent workloads with shared context.
4. **Speculative decoding / MTP** — 2–4× latency improvement at low batch.
5. **MoE sparsity** — 3–10× cheaper per token at fixed total parameters (already baked into architectures).
6. **Disaggregated prefill/decode** — 1.3–2× at scale, better tail latency.
7. **Distillation to a smaller model** — 5–20× cost reduction if quality holds.
8. **Pruning / structured sparsity** — modest, often disappointing on modern hardware.
9. **Custom ASICs / better MFU** — large but capital-intensive.

---

## 9.9 Common misconceptions

- **"Quantization degrades quality uniformly."** It degrades unevenly: math, code, and long-context recall suffer first; style suffers last.
- **"Speculative decoding always speeds things up."** It's a latency optimisation for low-to-medium batch; at high load it can hurt, which is why 2026 systems adaptively disable it.
- **"FLOPS are the bottleneck."** For decode, memory bandwidth is. This is why H200 (more bandwidth) can beat H100 by more than its FLOPS advantage.
- **"MoE reduces memory."** It increases total memory and requires all experts resident (or offloaded). It reduces *active* compute.
- **"Pruning gives free speedups."** Unstructured sparsity rarely maps to hardware speedups; structured pruning helps but damages accuracy.

---

## 9.10 If you remember five things

1. Decode is memory-bandwidth-bound; everything follows from reducing bytes moved or amortising them over a batch.
2. PagedAttention + continuous batching is the biggest single serving win ever shipped.
3. Speculative decoding (EAGLE/MTP/Medusa) is exact, not approximate, when done right.
4. Quantization is the cheapest way to cut cost, and 4-bit is the practical floor without distillation.
5. Prefix caching and disaggregated serving are what make agent workloads economically viable.

---

## 9.11 Self-check

1. Compute the theoretical maximum tokens/s for a 70B FP8 model on a 3 TB/s GPU, batch 1, ignoring FLOPs.
2. Why does batching improve throughput so dramatically for decode but not for prefill?
3. Explain why speculative decoding preserves the target distribution.
4. When would FP4 inference hurt more than 2× weight compression suggests?
5. Why does prefix caching matter more for agents than for chatbots?

**Next:** [`10-generative-media.md`](10-generative-media.md)
