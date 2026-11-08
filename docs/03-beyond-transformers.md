# 03 — Beyond the Transformer: Recurrence, Linear Attention, and Diffusion LMs

*Read time: 60 minutes. Prerequisites: `01`, `02` §2.5–2.6.*

---

## 3.1 The problem these architectures attack

Attention does two things at once:

- **Mixing**: at each step, gather information from everywhere earlier in the sequence.
- **Storage**: keep every past token's key/value around, so it *can* be gathered later.

The second is what costs. The KV cache grows linearly with context, and attention compute grows quadratically. For a 1M-token context you are paying for a million cached vectors per layer.

Every architecture in this chapter asks the same question: **can we compress the past into a fixed-size state, and pay O(1) per new token?**

That's a **recurrent** model. And the answer turns out to be "yes, with caveats" — the caveat being *recall*: a fixed-size state cannot perfectly remember an arbitrary earlier token, and Transformers can.

---

## 3.2 The RNN lineage: why they came back

Classic RNNs (pre-2017) had exactly this property: state `h_t = f(h_{t-1}, x_t)`. They were killed by Transformers for three reasons:

1. **They can't parallelise over time.** You must compute step 1 before step 2. GPUs hate this.
2. **Vanishing/exploding gradients** over long sequences (mitigated but not solved by LSTM/GRU gates).
3. **Fixed-size state = information bottleneck.** A 512-dim vector cannot hold a 100k-token document.

The 2023–2026 revival fixes (1) and (3) with two ideas:

- **Linear recurrence with a matrix state**: `S_t = A_t S_{t-1} + B_t`, where `S` is a *matrix* (not a vector) of size `d_head × d_state` per head. This can be computed in **chunks** via associative scans and matrix multiplications — GPU-friendly — and the state is much larger than a classic RNN's.
- **Input-dependent (selective) dynamics**: the transition/gating depends on the input, which recovers content-based addressing (the thing attention is good at).

Everything below is a variation on that.

---

## 3.3 S4: structured state space models (2021)

**State space model** (continuous-time):

```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```

Discretise (via zero-order hold with step Δ):

```
h_t = Ā h_{t-1} + B̄ x_t
y_t = C h_t
```

This is a linear RNN. Its saving grace: because it's *linear* and *time-invariant*, the whole sequence can be computed as a **convolution** with a kernel derived from (A, B, C) — so training is parallel (FFT or a dedicated scan), while inference is a constant-cost recurrence. Best of both worlds.

**S4's key trick** ("Structured State Spaces", Gu et al. 2021): initialise `A` with a specific structure (HiPPO matrices) that provably compresses long histories in an optimal polynomial basis. This made SSMs finally competitive at long-range tasks (e.g. 16k Long Range Arena) where earlier SSMs failed.

**Limitation:** `A, B, C` are fixed after training — *time-invariant*. The model cannot decide "this token matters, remember it". That's a content-based decision, and attention does it naturally.

---

## 3.4 Mamba: selectivity (2023)

**Mamba** (Gu & Dao, Dec 2023) makes `B`, `C`, and `Δ` **functions of the input**:

```
Δ_t = softplus(Linear(x_t))
B_t = Linear(x_t),  C_t = Linear(x_t)
Ā_t = exp(Δ_t A)
h_t = Ā_t h_{t-1} + B_t x_t
y_t = C_t h_t
```

Now the model can *choose* to write a token into state (large `B_t`) or ignore it, and choose how fast to forget (via `Δ_t`: large Δ = fast forgetting, small Δ = long memory). This is **selectivity**, and it was the breakthrough.

Because the recurrence is now input-dependent, the convolution trick no longer applies. Mamba's second contribution is a **hardware-aware parallel scan**: a custom CUDA kernel that keeps the expanded state in SRAM and recomputes it in the backward pass instead of storing it (gradient checkpointing by construction), giving linear-time training with real speedups.

**Mamba's results:** matched or beat Transformers of equal size on language modelling, with **5× higher generation throughput** at long sequences and linear scaling. It spawned a wave of "SSM will replace Transformers" commentary and the first genuinely competitive non-attention language models.

**Mamba's limitation** — and this is the important part — follow-up work found pure Mamba models are notably weaker at **in-context learning** and **exact recall** (copying a specific earlier token, associative recall, "needle in a haystack"). The state is a lossy summary; attention is an exact lookup. This is why the field converged on *hybrids*, not replacements.

---

## 3.5 Mamba-2 and State Space Duality (2024)

**Mamba-2** (Dao & Gu, 2024) is a theoretical consolidation with practical consequences. It shows that a broad class of SSMs is **dual** to a class of attention-like mechanisms: the same computation can be expressed as (a) a linear recurrence with matrix state, or (b) a masked matrix multiplication with a structured (semiseparable) mask.

Why this matters concretely:

- **Speed.** Because it can be expressed as matmuls, Mamba-2 uses tensor cores efficiently — 2–8× faster than Mamba-1 at matched quality, and it can reuse the *same parallelism strategy* as Transformers (tensor parallelism works unchanged because the layer's macro-structure — residual + norm — is identical).
- **Unification.** Attention and SSMs are points on a spectrum of "how structured is the mixing matrix?". This reframed the debate from "which wins" to "what's the right structure per layer".
- **The state size can be much larger** (e.g. 64–256 per head) because the matmul formulation makes it cheap.

**The hybrid finding.** At 2.7B parameters / 300B tokens, a hybrid with **6 attention layers and 58 SSD layers** beat both 64 pure-SSD layers and a pure Transformer++ baseline. Jamba (ICLR 2025) independently found that a 1:7 attention-to-Mamba ratio performed as well as 1:3, and that pure Mamba struggled with in-context learning while hybrids matched Transformers.

**This is the empirical basis for §2.10.** A handful of attention layers appears to supply the exact-recall capability the recurrent layers lack.

---

## 3.6 DeltaNet and Gated DeltaNet: the current best linear layer

**DeltaNet** (Schlag et al. 2021, revived 2024) reformulates linear attention as an *associative memory with error correction*. Instead of blindly adding `k vᵀ` to the state, it applies the **delta rule**:

```
S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

Read it as: "erase the old value stored at key `k_t`, then write the new value." That's the Widrow–Hoff / delta learning rule from the 1960s, applied to a fast-weight memory. It gives much more precise storage than plain accumulation — you can *overwrite* rather than pile up.

**Gated DeltaNet** (Yang, Kautz, Hatamizadeh, ICLR 2025) combines two complementary mechanisms:

- **Gating** (from Mamba-2/GLA): a scalar decay `α_t` that lets the model rapidly *erase* the whole state when context switches.
- **Delta rule**: precise, targeted *writes*.

They're complementary because gating is global and fast, while the delta rule is local and precise. Gated DeltaNet beat Mamba-2, DeltaNet, RetNet, HGRN2, and Mamba at 400M and 1.3B on language modelling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. Hybrid variants (Gated DeltaNet + sliding-window attention, or + Mamba-2) were better still.

**Gated DeltaNet-2** (May 2026) further decouples the *erase* gate from the *write* gate per channel, arguing a single scalar was conflating two different decisions; it reports the best results among Mamba-2/Gated DeltaNet/KDA/Mamba-3 variants at 1.3B/100B tokens, with the biggest gains on multi-key retrieval in RULER.

**KDA — Kimi Delta Attention** (2025) is Moonshot's variant with channel-wise decay, used in Kimi Linear and interleaved with sparse MLA in GLM-5-class models. The family (DeltaNet → Gated DeltaNet → KDA → Gated DeltaNet-2) is the most active architecture research area of 2025–26.

---

## 3.7 RWKV and xLSTM: the other two lineages

**RWKV** ("Receptance, Weighted Key Value", Peng et al. 2023–2026) is a hand-designed linear-attention/RNN hybrid that can be trained in parallel like a Transformer and run as an RNN at inference. Versions: v4 (2023), v5 (2024, matrix-valued state), v6 "Finch" (2024, data-dependent decay), **v7 "Goose"** (2025, delta-rule-like state tracking with a diagonal-plus-low-rank transition). RWKV-7 7.2B is used in production by its community; it remains a smaller ecosystem than Transformer-family models but is the most mature fully-recurrent open line.

**xLSTM** (Beck et al., NeurIPS 2024, Sepp Hochreiter's group) is the modernisation of LSTM in two variants:

- **sLSTM** (scalar memory): exponential gating + normalisation + a memory-mixing trick, restoring the original LSTM's ability to revise stored values.
- **mLSTM** (matrix memory): a *matrix* state with covariance-style update — fully parallelisable, and the workhorse.

xLSTM[7:1] (7 mLSTM blocks : 1 sLSTM) matched or beat Llama, Mamba, RWKV-4, and GLA at 1.3B, and — memorably — showed *better length extrapolation* than Transformers (perplexity stayed flat out to 16k when trained at 2k, while Llama's blew up). It's a reminder that the 1997 architecture was never the problem; the training recipe and state size were.

---

## 3.8 Jamba and the hybrid recipe

**Jamba** (AI21, Mar 2024; ICLR 2025) is the first production-grade hybrid: interleaved Transformer and Mamba blocks, with **MoE in some layers**, giving:

- Jamba-1.5-Large: 94B active parameters, 256k effective context
- Jamba-1.5-mini: 12B active
- **ExpertsInt8** quantization to fit 256k-context inference on a single 8×80GB node

Key empirical findings worth internalising:
1. Attention:Mamba ratio 1:3 vs 1:7 made *virtually no difference* — you need *some* attention, not much.
2. Pure Mamba fails to develop in-context learning; hybrids match Transformers.
3. The memory/throughput advantage grows with context length (that's the whole point).

**Samba** (Microsoft, 2024) is a similar hybrid with sliding-window attention + Mamba, reporting 3.7× throughput and 3.6× memory reduction vs. grouped-query-attention Transformers at 128k, with better quality.

---

## 3.9 Test-time training: learning at inference

A third path between "cache everything" (attention) and "compress into fixed state" (RNN): **let the model update its own weights on the context it's reading.**

**TTT layers** (Sun et al. 2024) replace the recurrent state with a small model whose weights are updated by gradient descent on a self-supervised objective at each chunk — the state *is* the weights.

**TTT-E2E** (Dec 2025) simplifies this radically: standard Transformer with sliding-window attention, but at test time it keeps doing next-token prediction on the given context, compressing that context into its weights, with a meta-learned initialisation. At 3B/164B tokens it scales with context length **like full attention** (which Mamba-2 and Gated DeltaNet do not) while having **constant inference latency** — 2.7× faster at long context.

**LaCT** (ICLR 2026) makes TTT practical by using very large chunk updates (2K–1M tokens), making hardware utilisation orders of magnitude better and allowing nonlinear state up to 40% of model size.

**Why this is philosophically interesting:** it reframes long context as *continual learning* rather than architecture. The model doesn't retrieve the past; it has been *changed* by it. If it works at frontier scale, it blurs the line between "context window" and "training" (`17`).

---

## 3.10 Diffusion language models: a genuinely different generation process

Autoregressive (AR) models generate left-to-right, one token at a time, because the factorisation is `p(x) = Π p(x_t | x_<t)`. **Diffusion language models (dLLMs)** abandon that factorisation.

### Masked diffusion (the dominant variant)

Training: take a clean sequence, randomly **mask** a fraction of tokens (or corrupt them), and train the model to recover the originals given the corrupted sequence. Vary the mask ratio over training. This is a denoising objective, and it generalises BERT's masked LM to a full generative model.

Sampling: start from a fully-masked sequence; iteratively predict all masked positions, commit the most confident ones, re-mask the rest; repeat for `T` steps.

### Why it's attractive

- **Parallel decoding**: generate many tokens per forward pass → 5–10× lower latency at equal quality in the best cases.
- **Bidirectional context**: every token is conditioned on both sides, which helps editing, infilling, and structured output (code, JSON).
- **Error correction**: because tokens can be re-masked and revised, the model can fix early mistakes — AR models cannot.

### Why it hasn't replaced AR

- **Quality per FLOP is worse.** Bidirectional attention means you can't cache as cheaply (KV cache for dLLMs is an active research problem), and you need multiple refinement steps.
- **Variable-length generation is awkward** (you pre-allocate a canvas).
- **Training at scale from scratch is expensive**, so the practical path is *conversion* from a pretrained AR model.

### The state of it in 2026

| Model | Date | Note |
|---|---|---|
| **LLaDA** (8B) | Feb 2025 | First dLLM trained from scratch at 8B, competitive with same-size AR models |
| **Mercury** (Inception Labs) | 2025 | First *commercial* dLLM; >1,000 tok/s/user, quality near speed-optimised frontier models; Mercury 2 ~1,200 tok/s |
| **Gemini Diffusion** | 2025 | Google demo, impressive speed; no public API as of late 2026 |
| **Seed Diffusion** (ByteDance) | 2025 | 1,479 tok/s, HumanEval 89.6% vs 90.2% for its AR counterpart |
| **LLaDA2.0** (16B mini, 100B flash) | Dec 2025 | Scaled to 100B by *converting* AR MoE models with a 3-phase block-size WSD schedule; open weights |
| **LLaDA2.2-mini** | Sept 2026 | 16B/1.4B-active MoE, 128k context, edit tokens (DELETE/INSERT) so it can revise rather than just fill masks |

**Block diffusion** is the pragmatic middle ground (and what most production dLLMs actually do): diffuse over *blocks* of e.g. 256 tokens, left to right, denoising within each block in parallel. This keeps AR-like dependency structure (so you can cache across blocks) while getting intra-block parallelism. BD3-LM (Mar 2025) formalised it; Gemma-diffusion-style models and 2026 work on generation order (PARD, 2026) build on it.

**The honest verdict:** dLLMs are a real, shipping technology for **latency-critical, structured** outputs. They are not yet competitive with frontier AR models on hard reasoning at equal cost. The most likely 2027 outcome is *hybrid*: diffusion draft + AR verify (which preserves the AR distribution exactly, like speculative decoding).

---

## 3.11 Comparison table

| Family | Train parallel? | Inference cost/token | Cache | Exact recall | Maturity (2026) |
|---|---|---|---|---|---|
| Transformer (GQA/MLA) | ✅ | O(L) attention | O(L) | ★★★★★ | Production default |
| Sparse attention (DSA) | ✅ | O(k) | O(L) (compressible) | ★★★★☆ | Production (2025→) |
| Mamba / Mamba-2 | ✅ (scan) | O(1) | O(1) | ★★☆☆☆ | Hybrids only |
| Gated DeltaNet / KDA | ✅ (chunked) | O(1) | O(1) | ★★★☆☆ | Production in hybrids (2025→) |
| RWKV-7 | ✅ | O(1) | O(1) | ★★★☆☆ | Niche/open |
| xLSTM | ✅ | O(1) | O(1) | ★★★☆☆ | Research |
| Test-time training | ✅ | O(1) amortised | weights | ★★★★☆ (reported) | Research frontier |
| Diffusion LM | ✅ (denoise) | O(1) per step, T steps | partial | ★★★☆☆ | Shipping for speed |

---

## 3.12 Common misconceptions

- **"Mamba beat Transformers."** Mamba beat Transformers *on some benchmarks at some scales*, with better throughput. It lost on recall and in-context learning, which is why no frontier model is pure Mamba.
- **"Linear attention is just attention with a kernel trick."** The modern versions (delta rule + gating) are much closer to *fast-weight associative memory with error correction* than to a kernel approximation.
- **"Hybrids are a compromise."** At equal compute they often *win*, and at long context they win decisively on memory. It's not a compromise; it's an allocation of scarce resources (exact recall is expensive, buy a little of it).
- **"Diffusion LMs are automatically faster."** They're faster only if you can afford to commit many tokens per step without losing accuracy. In practice that requires block structure, KV caching, and confidence-aware decoding.

---

## 3.13 If you remember five things

1. The whole family trades *exact recall* for *constant-memory state*.
2. Selectivity (input-dependent dynamics) was Mamba's breakthrough; the delta rule (overwrite, don't accumulate) is the current best write mechanism.
3. 3:1 or 7:1 linear:full attention hybrids are the production answer in 2025–26.
4. Diffusion LMs generate in parallel with bidirectional context; they're a latency play, not a capability play (yet).
5. Test-time training reframes long context as continual learning and is the most interesting open direction here.

---

## 3.14 Self-check

1. Why can't a fixed-size recurrent state implement exact copy of an arbitrary earlier token?
2. What does the `Δ_t` in Mamba control, and what happens if it's near zero / very large?
3. Explain Mamba-2's state-space duality in one sentence and one practical benefit.
4. Why does the delta rule beat plain accumulation for associative recall?
5. In a 3:1 hybrid, what job are the full-attention layers doing?
6. Why is block diffusion easier to serve than pure masked diffusion?

**Next:** [`04-scaling-and-training.md`](04-scaling-and-training.md)
