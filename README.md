# The AI Research Book (2022 – 2026)

**A self-contained, beginner-friendly field guide to everything that mattered in modern AI — architectures, breakthroughs, training methods, reasoning, agents, safety, and the open problems.**

Written: **November 2026**. Author: principal-researcher mode (Buffy). Audience: curious beginners → working engineers.

> **The promise of this book:** you should not need to open another tab. Every term is defined the first time it appears, every formula is explained in plain English *and* written out, and every claim about "who did what when" is dated. Where something is contested, you'll see the actual argument, both sides, and what evidence would settle it.

---

## How to read this

There are three honest ways in:

| You are… | Read this path | Time |
|---|---|---|
| **Completely new** | `00-start-here` → `01-foundations` → `02-modern-llm-architecture` → `05-post-training` → `06-reasoning` → `07-agents` → `13-safety` → `17-open-problems` | ~6 h |
| **A working engineer** who wants the 2024–26 delta | `02` → `03` → `04` → `06` → `07` → `09` → `16-timeline` → `14-evals` | ~4 h |
| **A researcher** mapping a new area | `17-open-problems` first (it links everywhere), then the chapter you need, then `19-reading-list` | as needed |

Each chapter is standalone but cross-linked, and ends with **"If you remember five things"** plus **self-check questions**. The glossary in `18` covers ~260 terms; use it as your lookup table.

---

## The map

### Part I — Foundations (what everything is built on)
- [`docs/00-start-here.md`](docs/00-start-here.md) — the six shifts that define 2022–2026, notation, how to think about scale
- [`docs/01-foundations.md`](docs/01-foundations.md) — tokens, embeddings, the Transformer block, attention math, the training objective, **scaling laws** (Kaplan & Chinchilla), emergent abilities and the fight over whether they're real

### Part II — Architecture
- [`docs/02-modern-llm-architecture.md`](docs/02-modern-llm-architecture.md) — every part of a 2026 model: RMSNorm, SwiGLU, RoPE, GQA/MQA, **MLA**, **MoE**, MTP, **sparse attention (DSA)**, FlashAttention, KV cache — with worked numbers
- [`docs/03-beyond-transformers.md`](docs/03-beyond-transformers.md) — the recurrent/linear-attention family: S4 → Mamba → Mamba-2 → DeltaNet → **Gated DeltaNet**, RWKV, xLSTM, hybrids (Jamba, Qwen3-Next), test-time training, and **diffusion language models** (LLaDA, Mercury)

### Part III — Making a model smart
- [`docs/04-scaling-and-training.md`](docs/04-scaling-and-training.md) — data curation, tokenizers, parallelism (DP/TP/PP/EP/CP), ZeRO/FSDP, FP8, muP, Muon, synthetic data, the **data wall**, distillation, training instability, what a frontier run actually costs
- [`docs/05-post-training.md`](docs/05-post-training.md) — SFT, **RLHF/PPO**, reward models, **DPO** and its family (IPO/KTO/ORPO/SimPO), Constitutional AI, RLAIF, rejection sampling, **RLVR/GRPO**, agentic RL, self-play, reward hacking
- [`docs/06-reasoning.md`](docs/06-reasoning.md) — chain-of-thought, self-consistency, Tree of Thoughts, **inference-time scaling laws**, PRMs vs ORMs, verifiers, o1/o3, **DeepSeek-R1-Zero**, s1, test-time RL, latent reasoning, and what "thinking" really buys

### Part IV — Systems that do things
- [`docs/07-agents.md`](docs/07-agents.md) — the agent loop, tool calling, **MCP**, computer use, coding agents, memory, multi-agent systems, SWE-bench/Terminal-Bench/METR horizons, and why reliability is the real bottleneck
- [`docs/08-retrieval-and-long-context.md`](docs/08-retrieval-and-long-context.md) — RAG, dense retrieval (DPR/ColBERT), rerankers, hybrid search, RoPE scaling/**YaRN**/LongRoPE, ring attention, KV compression, "lost in the middle"
- [`docs/09-efficiency-and-inference.md`](docs/09-efficiency-and-inference.md) — quantization (GPTQ/AWQ/GGUF/FP8/FP4/BitNet), LoRA/QLoRA, pruning, **speculative decoding**, PagedAttention/vLLM, prefill-decode disaggregation, hardware, and the arithmetic of serving cost

### Part V — Other modalities
- [`docs/10-generative-media.md`](docs/10-generative-media.md) — diffusion from DDPM to **flow matching**, latent diffusion, DiT, guidance, video models (Sora/Veo), **world models** (Genie), speech, and evaluation
- [`docs/11-vision-and-multimodal.md`](docs/11-vision-and-multimodal.md) — ViT, CLIP/SigLIP, DINOv2, SAM 2, VLM recipes, **native omni models**, and vision-language-action robotics models

### Part VI — Understanding, trusting, measuring
- [`docs/12-interpretability.md`](docs/12-interpretability.md) — features vs circuits, superposition, **sparse autoencoders**, transcoders, attribution graphs, activation patching, steering, and the critiques
- [`docs/13-safety-and-alignment.md`](docs/13-safety-and-alignment.md) — jailbreaks, constitutional classifiers, **alignment faking**, scheming, scalable oversight, weak-to-strong, RSPs and system cards, model welfare
- [`docs/14-evals-and-benchmarks.md`](docs/14-evals-and-benchmarks.md) — how to read a leaderboard, contamination, saturation, the benchmark catalogue, and why **everything you've heard about benchmark scores is partly wrong**
- [`docs/15-ai-for-science-and-math.md`](docs/15-ai-for-science-and-math.md) — AlphaFold 1→3, GNoME, GraphCast, AlphaGeometry/AlphaProof, IMO 2024→2026, **AlphaEvolve**, DeepSeekMath-V2, and the honest limits

### Part VII — The big picture
- [`docs/16-timeline.md`](docs/16-timeline.md) — an annotated chronology, 2017 → Nov 2026, with *why each item mattered*
- [`docs/17-open-problems.md`](docs/17-open-problems.md) — the live debates: data wall, scaling vs inference compute, reasoning, world models, continual learning, agent reliability, interpretability, safety, economics, AGI definitions
- [`docs/18-math-and-glossary.md`](docs/18-math-and-glossary.md) — a 10-page math primer (only what you need), notation table, and a ~260-term glossary
- [`docs/19-reading-list.md`](docs/19-reading-list.md) — ~90 annotated papers/links, grouped by topic, with "read this first" markers

---

## Companion volumes

This book is the foundation. Three shorter works build on it:

- **[`models/`](models/README.md)** — six great models reverse-engineered: annotated `config.json`, block diagrams, component deep-dives, hand parameter counts, and eight contradictions between the marketing and the maths. Plus [`07-insights-and-patterns`](models/07-insights-and-patterns.md), the cross-model pattern analysis.
- **[`experiments/`](experiments/README.md)** — three research proposals with a thesis, not a topic: a state-capacity law for hybrid linear attention, a fractional NoPE ablation that isolates *which* layers need position, and the router as the MoE sparsity ceiling. Each has numbered predictions with falsifiers, a pre-registered analysis plan, a budget, and a one-week pilot.
- **[`future/`](future/README.md)** — AI 2027–2036: the four curves (compute, algorithms, price, energy), what models become once the architecture stops changing, learning after data, the agent reliability tax, the economics, and four scenarios with dated predictions and falsifiers.

---

## The one-paragraph summary of the whole field

Between 2022 and 2026, AI moved through four regimes. **(1) Scale (2020–2023):** Transformers + more data + more compute produced *predictable* capability gains, formalised as scaling laws; GPT-3, Chinchilla and GPT-4 were the landmarks. **(2) Alignment (2022–2024):** instruction tuning and reinforcement learning from human feedback turned raw predictors into usable assistants; DPO made alignment cheap; Constitutional AI made it partly automatic. **(3) Reasoning (2024–2026):** the discovery that *spending compute at inference time* — generating long chains of thought, searching, verifying — is a second scaling axis, trained with reinforcement learning on verifiable rewards (RLVR). This produced o1/o3, DeepSeek-R1, and gold-medal IMO performance. **(4) Agency (2025–2026):** models started acting — calling tools, using computers, editing repositories, running for hours — via the Model Context Protocol, agentic RL, and long-horizon harnesses. Architecturally, the Transformer didn't die; it absorbed everything: Mixture-of-Experts for cheap capacity, latent/sparse/hybrid attention for cheap context, and multi-token prediction for cheap decoding. The unresolved problems are no longer "can it?" but "how reliably, how cheaply, how safely, and how do we know?"

---

## Conventions in this book

- **Bold** on first use of a term = it's in the glossary (`18`).
- `O(n²)` means "cost grows with the square of the input length" — see `01` §1.3 for the primer.
- Dates are the public announcement date unless noted. Model **version numbers go stale weekly**; the *mechanisms* in this book do not. Where a 2026 claim is from a vendor blog and not independently replicated, it says so.
- Numbers like "671B / 37B active" mean **total parameters / parameters used per token** — a distinction that matters enormously (§`02` MoE).
- Every chapter has a **"Common misconceptions"** block. Those are the parts most people get wrong.

---

## Honesty section (read this)

This book is a synthesis of public papers, model cards, engineering blogs and independent evaluations, written by an AI agent. Some things to keep in mind:

1. **Vendor benchmark numbers are marketing until replicated.** Self-reported scores use private scaffolds, best-of-N sampling, and sometimes training data that overlaps the test set. `14` is a whole chapter on this.
2. **Late-2026 frontier models are a moving target.** I've dated what I could verify and hedged what I couldn't. If a model name or number here contradicts a primary source, trust the primary source.
3. **Mechanistic claims are provisional.** Statements like "layer 12 computes X" are always simplifications of a system nobody can fully read (`12`).
4. **Where the field disagrees, you'll see the disagreement, not a verdict.** That's not hedging; it's the actual state of knowledge.

Now start at [`docs/00-start-here.md`](docs/00-start-here.md).
