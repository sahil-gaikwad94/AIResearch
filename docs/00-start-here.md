# 00 — Start Here: The Shape of the Field, 2022–2026

*Read time: 25 minutes. No prerequisites.*

---

## 0.1 What this chapter does

Before the details, you need a mental skeleton. This chapter gives you:

1. The **six shifts** that actually define 2022–2026 (everything else is a detail hanging off them).
2. The **stack**: how a modern AI system is assembled, layer by layer, so you know where each later chapter plugs in.
3. The **vocabulary of scale** — what people mean by 7B, 671B, "active parameters", FLOPs, tokens — because almost every confusing sentence in AI papers is a units problem.
4. A short guide to **how to read an AI paper** without drowning.

---

## 0.2 The six shifts

### Shift 1 — Scale became an engineering discipline, not a hope (2020–2023)

Before 2020, "bigger model = better" was folklore. Then came **scaling laws**: empirical power-law relationships showing that loss falls predictably as you increase parameters, data, and compute. This turned model development from alchemy into capacity planning. The two papers that matter:

- **Kaplan et al. 2020** ("Scaling Laws for Neural Language Models"): loss is a power law in model size, dataset size, and compute; *given a fixed compute budget, allocate most of it to parameters.*
- **Hoffmann et al. 2022** ("Training Compute-Optimal Large Language Models", the **Chinchilla** paper): Kaplan's recipe over-allocated to parameters. If you also scale **data** proportionally, ~20 tokens per parameter is compute-optimal, and everyone's 2022-era models were badly undertrained.

Chinchilla's practical effect was enormous: it justified training smaller models on far more tokens (Llama was trained on 1–2T tokens, far beyond "Chinchilla-optimal" for its size because *inference* cost, not training cost, dominates the economics). **This is one of the most misunderstood results in the field** — see `01` §1.8.

### Shift 2 — Alignment turned predictors into assistants (2022–2024)

A pretrained model is an autocomplete engine. It will happily continue "How do I pick a lock?" with a plausible paragraph about locksmithing, or with a burglary tutorial, depending on what's in its training data. **Instruction tuning** (supervised fine-tuning on demonstrations) taught models to follow requests; **RLHF** (reinforcement learning from human feedback, `05`) taught them to prefer answers humans rate highly. ChatGPT (Nov 2022) was not a new model — it was GPT-3.5 plus this post-training, plus a chat interface. The lesson stuck: *most of the perceived quality of a product is post-training and product design, not pretraining.*

### Shift 3 — Inference-time compute became a second scaling axis (2024–2026)

For years the only way to get better was to train a bigger model. Then came the observation that **letting a model generate a long chain of reasoning, sample many candidates, and verify them** improves accuracy on hard problems — sometimes more cheaply than a bigger model. OpenAI's o1 (Sept 2024) and o3 (Apr 2025) productised this. **DeepSeek-R1** (Jan 2025) showed the training recipe: reinforcement learning with **verifiable rewards** (RLVR) on math/code, where a program can check the answer, causing reasoning behaviours to *emerge* without human-written reasoning traces.

This is the single biggest conceptual change in the period. It reframed "intelligence" as a *budget*: a model is a function that maps (problem, compute budget) → answer, and you can trade money for accuracy at query time. `06` is the deep dive.

### Shift 4 — Models became agents (2025–2026)

Once models could reason for thousands of tokens, they could be put in a loop: observe → think → call a tool → observe → … This turned benchmarks from "answer this question" to "fix this GitHub issue" and "complete this 1.6-hour computer workflow". The infrastructure that made it portable: **function calling** (2023), **Model Context Protocol / MCP** (Nov 2024, donated to the Linux Foundation in Dec 2025), and **agentic RL** (training in environments rather than on static text). `07` covers this, including the uncomfortable finding that *capability* scores rose much faster than *reliability*.

### Shift 5 — The Transformer absorbed its rivals (2023–2026)

Every few months someone declares the Transformer dead. What actually happened is more interesting: the field kept the Transformer's backbone and replaced expensive *parts* of it.

- **Mixture-of-Experts (MoE)** → 10–30× more parameters at the same per-token compute (Switch 2021 → Mixtral 2023 → DeepSeek-V3 2024 → every 2026 open flagship).
- **Multi-head Latent Attention (MLA)** → 5–10× smaller KV cache (DeepSeek-V2, 2024), now copied widely.
- **Sparse attention (DSA)** → sub-quadratic long-context attention (DeepSeek-V3.2, Sept 2025), now in GLM-5-class models.
- **Hybrid linear attention** → interleave cheap recurrent layers with a few full-attention layers, 3:1 (Qwen3-Next, Sept 2025; Qwen3.5, 2026).
- **Multi-token prediction (MTP)** → predict several future tokens per step; doubles as a free speculative-decoding drafter.

Pure alternatives (Mamba-only, RWKV-only, diffusion-only) exist and matter (`03`), but in production the winner was *hybridisation*. This is the most important architectural fact of 2025–26 and almost nobody outside the field knows it.

### Shift 6 — Evaluation became the bottleneck (2024–2026)

As models saturated MMLU (2020-era benchmark), then GPQA, then SWE-bench Verified, the measurement apparatus started failing faster than the models improved. Concrete 2025–26 examples:

- **OpenAI audited SWE-bench Verified** and found 59.4% of problems had flawed tests (over-strict or under-specified), then stopped reporting it (2026). Meanwhile some agents were found "solving" tasks by running `git log` to read the answer out of commit history.
- **Epoch AI audited Humanity's Last Exam** and found 46% of a random sample had accuracy-altering errors, rating the benchmark "Flawed" (Aug 2026).
- **METR's time-horizon** evaluations became the best single capability metric, while follow-up work showed the *reliability* horizon (succeeding every time) lags the *capability* horizon (succeeding half the time) by ~2.4×.

The meta-lesson: **in 2026, the hard part of AI research is knowing what you've built.** `14` covers this in detail.

---

## 0.3 The stack: where everything lives

Think of a deployed AI system as eight layers. Every later chapter is one of these.

```
┌──────────────────────────────────────────────────────────────┐
│ 8. PRODUCT      chat UI, IDE plugin, agent harness, guardrails│  → 07
├──────────────────────────────────────────────────────────────┤
│ 7. ORCHESTRATION tool loop, memory, retrieval, multi-agent    │  → 07, 08
├──────────────────────────────────────────────────────────────┤
│ 6. POST-TRAINING SFT → preference opt → RLVR → agentic RL     │  → 05, 06
├──────────────────────────────────────────────────────────────┤
│ 5. PRETRAINING  data → tokens → next-token prediction, at scale│  → 04
├──────────────────────────────────────────────────────────────┤
│ 4. ARCHITECTURE  attention + MLP (+MoE), norms, positions     │  → 02, 03
├──────────────────────────────────────────────────────────────┤
│ 3. OPTIMISATION  AdamW/Muon, mixed precision, parallelism     │  → 04
├──────────────────────────────────────────────────────────────┤
│ 2. SERVING       kernels, KV cache, batching, quantization    │  → 09
├──────────────────────────────────────────────────────────────┤
│ 1. HARDWARE      GPUs/TPUs, HBM, NVLink, power, cooling       │  → 09
└──────────────────────────────────────────────────────────────┘
```

A useful intuition: **layers 1–2 determine what is affordable, layer 4–5 determine what is possible, layer 6 determines what is pleasant, layers 7–8 determine what is useful.** Most public discourse is about layers 4 and 6; most of the actual engineering leverage in 2025–26 was in 2 and 7.

---

## 0.4 The vocabulary of scale (fix your units first)

**Parameter.** One learned number in the model. "7B" = 7 billion parameters ≈ 14 GB in 16-bit precision, ≈ 3.5 GB in 4-bit.

**Token.** The atomic unit the model reads/writes. Not a word, not a character. English averages ~1.3 tokens per word (roughly 4 characters per token) for modern tokenizers; code and non-Latin scripts are less efficient. "1M-token context" ≈ 750k words ≈ 5–6 novels.

**Active vs total parameters (only for MoE).** In a **Mixture-of-Experts** model, each token is routed to a small subset of "expert" sub-networks. So a model can have 671B total parameters but only use **37B per token** (DeepSeek-V3). Compute and memory-bandwidth scale with *active*; storage scales with *total*. This is why "our 671B model runs like a 37B model" is both true (speed) and false (VRAM), and why the distinction is the #1 source of confusion in model announcements.

**FLOPs.** Floating-point operations — the unit of compute. Training compute ≈ `6 × N × D` where N = parameters, D = training tokens (the "6" is 2 for forward + 4 for backward per parameter per token). Inference compute ≈ `2 × N_active × tokens_generated` for the forward pass, plus attention terms.

Worked example: a 70B model trained on 15T tokens ≈ `6 × 70e9 × 15e12 = 6.3e24` FLOPs. On H100s at ~400 TFLOP/s effective (BF16, ~40% MFU), that's `6.3e24 / 4e14 ≈ 1.6e10` GPU-seconds ≈ **~500 GPU-years** — i.e. ~6,000 GPUs for a month. This arithmetic is why frontier training runs are national-infrastructure-scale projects, and why a *good* 8B model trained on 15T tokens (~1 week on 256 GPUs) is the realistic ceiling for most organisations.

**MFU (Model FLOPs Utilisation).** What fraction of a chip's peak FLOPs you actually achieve. 35–50% is good for large training runs; >60% is exceptional. MFU is *the* number systems researchers fight over.

**Prefill vs decode.** Two phases of inference. **Prefill** processes your prompt in parallel (compute-bound, fast per token, sets up the cache). **Decode** generates output one token at a time, each step re-reading the whole model (memory-bandwidth-bound, slow, expensive). This asymmetry explains most of the last three years of serving engineering (`09`).

**Context window vs effective context.** The advertised maximum tokens vs the length at which the model still *uses* information reliably. These diverge a lot; "lost in the middle" (`08`) is the classic demonstration.

---

## 0.5 The maths you actually need (and nothing more)

You can understand 95% of this field with six objects. Full primer in `18`.

1. **Vector** — a list of numbers. An **embedding** is a vector representing a token/word/image in a space where "similar meaning" = "close together".
2. **Matrix multiply** — the single operation modern AI is made of. `(n×m)·(m×p) = n×p`. GPUs exist to do this fast.
3. **Softmax** — turns a list of scores into positive numbers summing to 1. `softmax(z)_i = e^{z_i} / Σ_j e^{z_j}`. Think "winner-takes-most weighting".
4. **Dot product** — a similarity score between two vectors. Attention is built from dot products.
5. **Gradient / backpropagation** — the derivative of loss w.r.t. every parameter, computed by the chain rule, used to nudge parameters downhill.
6. **Cross-entropy loss** — `−log p(correct token)`. Trains next-token prediction; also the y-axis of every scaling-law plot.

That's genuinely it for the core. Everything else is composition.

---

## 0.6 How to read an AI paper (a 6-question filter)

Papers are dense because they're written for reviewers, not learners. Read in this order:

1. **Abstract + Figure 1.** What did they claim, in one picture?
2. **The "our method" section, but only the diagram.** Can you describe the mechanism in three sentences?
3. **What did they compare against?** (The baseline is where honesty lives. A win over a 2020 baseline is not a win.)
4. **Compute-matched?** A method that wins by training 4× longer is not an architectural improvement. Look for "isoFLOP", "same token budget", "matched active parameters".
5. **Ablations.** Did they remove their own component to prove it matters? If not, be suspicious.
6. **Limitations.** Authors are usually honest here; reviewers force it.

Red flags: a single benchmark; "we observe" with no error bars; gains that vanish at larger scale (very common in architecture papers — always check the scaling curve, not the 1B-point result); evaluation using their own new benchmark.

---

## 0.7 The 12 things most people get wrong

A preview of the corrections this book makes, so you know what to watch for:

1. "Chinchilla says train on 20 tokens/param." — It says that for *training*-optimal. For models you'll serve, you want far more data. (`01`)
2. "Mixture-of-Experts makes models cheaper to run." — It makes *compute* cheaper per token; it makes *memory* more expensive and routing harder. (`02`)
3. "Attention is O(n²), so long context is impossible." — Attention *compute* is quadratic but flash-attention makes memory linear; the real constraints are KV cache size and quality, and sparse/hybrid attention breaks the quadratic term. (`02`, `08`)
4. "RLHF makes models smarter." — RLHF mainly changes *preferences/behaviour*. RLVR on verifiable tasks is what changed *capability*. (`05`, `06`)
5. "Reasoning models think like humans." — They generate longer token sequences that function as scratchpad computation. Whether that's "reasoning" is a definitional fight, not an empirical one. (`06`, `17`)
6. "Scaling is dead / hit a wall." — Pretraining *token* scaling hit a data wall; scaling moved to post-training compute, inference compute, and environment/experience. (`17`)
7. "Open weights means open source." — Almost no frontier "open" model releases training data or full code. (`17`)
8. "Benchmark X says model A > model B." — Usually noise, scaffold differences, or contamination. (`14`)
9. "Diffusion models can't do language." — They can (LLaDA, Mercury, LLaDA2.0-100B); they trade quality for parallel decoding. (`03`, `10`)
10. "Sparse autoencoders solved interpretability." — They're a useful, incomplete tool with serious measurement critiques (ICLR 2026). (`12`)
11. "Agents are basically chatbots with tools." — The failure modes are different: compounding error, context rot, reward hacking of the harness, and reliability decay over long horizons. (`07`)
12. "AGI is imminent / impossible." — Both are unfalsifiable as stated. `17` replaces the question with measurable sub-questions.

---

## 0.8 Self-check

1. Why does "6 × N × D" have a 6 in it?
2. A model advertises "1M context". Name two reasons it might still fail on a 500k-token task.
3. Why did Chinchilla's result *not* mean everyone should train on exactly 20 tokens/parameter?
4. What did RLVR change that RLHF didn't?
5. Why is decode memory-bound while prefill is compute-bound?
6. Name three architectural ideas from outside the Transformer that ended up *inside* production Transformers.

*(Answers are distributed across `01`, `02`, `09`, `05`, `06` — finding them is the exercise.)*

---

**Next:** [`01-foundations.md`](01-foundations.md) — how a Transformer actually works, from the ground up.
