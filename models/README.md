# Six Models, Taken Apart

A reverse-engineering course in six files. Each one takes a real released model, opens its `config.json`, counts its parameters by hand, draws one transformer block on paper, and explains every design decision — starting from a 124-million-parameter model you can run on a laptop and ending at a 2026 frontier architecture.

No prior knowledge is assumed. Every term is defined the first time it appears, every number is either computed in the file or quoted from the model's own published config, and every chapter ends with a lab you can actually run.

## The six

| # | Model | Released by | Params (total / active) | The one idea it exists to teach | Difficulty |
|---|-------|-------------|------------------------|--------------------------------|-----------|
| 1 | [GPT-2 Small](01-gpt2-small.md) | OpenAI, 2019 | 124M / 124M | The skeleton every other model is a variation of | ★☆☆☆☆ |
| 2 | [Llama 3 8B](02-llama3-8b.md) | Meta, 2024 | 8.03B / 8.03B | The modern dense layer: RoPE, RMSNorm, SwiGLU, GQA | ★★☆☆☆ |
| 3 | [Mixtral 8x7B](03-mixtral-8x7b.md) | Mistral, 2023 | 46.7B / 12.9B | Sparsity: more knowledge without more compute per token | ★★★☆☆ |
| 4 | [Flint](04-flint.md) | Recursive, 2026 | 39.8B / 2.5B | Routing + selective attention at extreme sparsity | ★★★½☆ |
| 5 | [DeepSeek-V3](05-deepseek-v3.md) | DeepSeek, 2024 | 671B / 37B | Compressing the KV cache (MLA) + fine-grained experts + MTP | ★★★★½ |
| 6 | [Qwen3-Next 80B-A3B](06-qwen3-next.md) | Alibaba, 2025 | 80B / 3.2B | Replacing most attention with a linear-time recurrence | ★★★★★ |

Then: [**07 — Insights and Patterns**](07-insights-and-patterns.md), the audit file. What falls out when you put all six side by side — the six knobs, the parameter-budget law, the attention-rationing trend, sparsity convergence, and the reverse-engineering method that works on any model you have never seen.

## Why these six

The ordering is by **difficulty, not chronology** (Mixtral is 2023 but sits after Llama 3, because you need GQA before top-2 routing makes sense).

Together they cover the whole design space:

- **Dense → sparse → hybrid-sparse.** GPT-2 and Llama 3 activate every parameter for every token. Mixtral, Flint, DeepSeek and Qwen3-Next activate a small slice.
- **Full attention → windowed attention → compressed attention → linear attention.** One model per rung of that ladder.
- **2019 → 2026.** Every idea appears the year it became mainstream, so the six files double as a history of architecture.

What I deliberately left out, and where it lives instead: Kimi K2 (a 1T-parameter extrapolation of DeepSeek-V3's recipe — covered as a scaling case study in file 07), pure SSM/Mamba models and diffusion language models (real, but not yet the shape of production models), and closed models like GPT-5-class or Gemini 3, which cannot be reverse-engineered because no weights or configs are published. Everything here is open-weight, which is the whole point: you can check my work.

## How to read a model

Each file follows the same eight-part shape, so by file three you will know where to look:

1. **Identity card** — who, when, how big, what it is for.
2. **The config, annotated** — the real `config.json`, every field explained.
3. **One block, drawn** — the layer diagram with tensor shapes.
4. **Component deep-dives** — attention, feed-forward, norms, positions.
5. **Count the parameters yourself** — a table you can reproduce with a calculator.
6. **The forward pass** — shapes from token IDs to logits.
7. **Reverse-engineering lab** — runnable code that proves the claims.
8. **What it does the old way** — the constraints that motivated the next model.

## Conventions used throughout

- **"Total" vs "active" parameters.** Total = everything stored in the weights file (what you must fit in memory). Active = the subset actually multiplied by a given token (what determines compute per token). For dense models they are the same number; for mixture-of-experts models they differ by 5–50×, and confusing them is the single most common error in reading AI news.
- **Sizes.** 1B = 10⁹ parameters. A parameter is one number stored in the model, usually 2 bytes in BF16, so 8B parameters ≈ 16 GB of weights.
- **"Vendor number"** marks a figure taken from a model card or paper. Unmarked numbers are computed in these files from the published config, and they agree with the vendor numbers to within counting conventions (I flag the one place they do not).
- **FLOPs.** Roughly `2 × active_parameters` floating-point operations per token, plus an attention term that grows with context length. That factor of 2 = one multiply + one add per weight.

## A note on honesty

Parameter counts, layer counts and tensor shapes here are exact — they come from configs published by the model's own authors and are reproducible with the code in each file. Everything about *why* a design works is my explanation, and where the field genuinely does not know, the file says so. Benchmark scores are quoted with their source and are the fastest-decaying numbers in this folder: treat them as evidence of a trend, never as a property of an architecture.
