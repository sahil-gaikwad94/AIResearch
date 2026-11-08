# 2 — What Models Become

> The architecture is converging, and that is the most under-reported fact in AI. Six knobs are now exhausted; every 2026 frontier model is a variation on the same block. What happens next is not a new block. It is the addition of a seventh knob — **memory** — and a change in what the word "model" refers to.

---

## 2.1 The design space is nearly closed

From the six reverse-engineered models, the entire architectural variation reduces to six dials:

| Knob | 2019 | 2023 | 2026 consensus |
|---|---|---|---|
| Mixing operator | full attention | full attention | hybrid: 70–80% local + 20–30% global |
| KV sharing | MHA | GQA | MLA or aggressive GQA |
| FFN sparsity | dense | top-2 of 8 | top-6..10 of 256–512 |
| Norm | LayerNorm | RMSNorm | RMSNorm + QK-norm + zero-centred |
| Position | learned table | RoPE everywhere | conditional; absent on recurrent/global layers |
| Extra heads | none | none | MTP depth 1 |

Four independent labs converged on sparsity ratios of 32–51× (Mixtral 4× in 2023; DeepSeek 32×; Kimi K2 48×; Flint 43×; Qwen3-Next 51×) and on 25–29% global attention layers. That is a strong signal: the space has been searched, and the optimum is broad and flat.

**[P] Prediction (80%):** no fundamentally new *block* architecture reaches frontier open weights before 2030. What ships is refinement: sparsity 64–128×, MTP depth 2–4, sinks everywhere, FP4 training, and better kernels.

That sounds boring. It is also the most consequential statement in this document, because it means **capability gains after 2027 come from data, environments, post-training and memory — not from the block.**

---

## 2.2 What refinement looks like, concretely

**Attention gets rationed further.** If experiment 3 in the companion folder succeeds, sparsity goes to 64–128×, which pushes attention's share of per-token compute above 50%. That guarantees more of the model becomes local: expect 85% local layers by 2029, with the global 15% using *sparse* attention (top-k over a learned index, as in DeepSeek's DSA) rather than dense attention.

**Context stops being a headline number.** 1M tokens is already commodity. The interesting number becomes what experiment 1 calls *association capacity*: how many distinct facts the model can hold and retrieve. Context length grows 4×; useful capacity grows ~1.5×, because capacity is set by state dimension and attention sparsity, not by the window.

**Numerics go to FP4.** Training in FP8 is standard by 2026; FP4 with per-block scaling is plausible by 2028 and is worth ~2× throughput. The cost is a new class of silent quality loss, which is why it will be adopted first in inference and last in pretraining.

**The vocabulary stops growing.** 128k–160k is the plateau. Above that, embedding and softmax costs outweigh the tokenisation gains.

**Depth grows slowly, width tracks active compute.** Nothing in the 2026 record suggests deeper is better past ~60 layers; the width is set by per-token FLOPs, which is set by latency targets.

---

## 2.3 The seventh knob: memory

This is the real frontier, and 2026 evidence says it is unsolved.

The problem, stated plainly: **a deployed model does not get better at your job.** It has a context window and (sometimes) a retrieval store, but its weights are frozen at training time. Everything it learns about your codebase, your preferences, your recurring mistakes, dies with the session.

**What exists today (all weak):**

1. **Retrieval as memory** — files, vector stores, conversation summaries. Works for facts, fails for skills. A model that has read your codebase 100 times is not better at your codebase.
2. **Per-user adapters** — LoRA fine-tunes on a user's data. Real gains, but they do not compose, cannot be evaluated cheaply, and are a support burden.
3. **Long context** — the brute-force answer. Expensive at scale and, per experiment 1, capacity-limited anyway.
4. **In-context learning** — the default. Free, but forgotten at the end of the session.

**What is missing, and why it is hard.** The 2026 benchmark built to measure this — CL-Bench (NeurIPS 2026) — is damning: across six domains with learnable latent structure (codebase layout, disease dynamics, opponent strategies), frontier agents "frequently overfit to immediate observations or fail to reuse knowledge across instances", and **dedicated memory systems did not help — naive in-context learning outperformed them.** A separate ICML 2026 position paper argues the answer is modular memory: in-context learning for fast adaptation, in-weight learning for stable skill, with the two designed together.

The technical obstacles are all real:

- **Forgetting.** Gradient updates on new experience degrade old capabilities (catastrophic forgetting), and nobody has a scalable fix at frontier scale.
- **Evaluation.** "Did it learn?" requires held-out tasks that share latent structure — expensive to build, easy to game.
- **Safety.** A model that updates its own weights from deployment data can be poisoned by users, and can drift in ways that evals miss six months later.
- **Economics.** Continuous updates break the "train once, serve many" cost model that makes inference cheap.

**[P] Prediction (60%):** by 2028 at least one major lab ships a product whose quality on a user's recurring tasks measurably improves with use — but via *external* memory plus per-user adapters, not weight updates. **[P] (35%):** genuine test-time weight updates in a frontier product before 2030, gated behind a "consolidation" step (offline, reviewed) rather than online.

**Why this is the highest-value knob.** Everything else is copyable. A model that improves from deployment accumulates a data advantage that competitors cannot buy, and it converts usage into capability — the only compounding moat in the stack.

---

## 2.4 "Model" stops meaning checkpoint

By 2028 the thing you call a model is a **system** with at least five parts:

```
  base weights          (pretrained, replaced every 6–12 months)
+ memory store          (per-user, persistent, versioned)
+ tool layer            (files, browser, shell, APIs, MCP-style protocols)
+ verifier stack        (tests, type checkers, proof checkers, simulators)
+ policy                (RL-tuned behaviour: when to ask, when to stop, how to plan)
```

The consequence is that **model cards become system cards**, and benchmark comparisons become less meaningful every year. A "72% on SWE-bench" number tells you about a scaffold as much as a model. Evaluation migrates to task-level, cost-adjusted, reliability-adjusted measures — which is exactly what chapter 4 argues is the only number that matters economically.

---

## 2.5 World models: the most plausible reset

If something replaces the language-model block, the most likely candidate is not a better sequence mixer. It is a model trained primarily on **video, 3D, and interaction** rather than text.

Why it could reset things:

- **Supervision density.** A minute of video contains far more physical information than a minute of text, and it is abundant.
- **A different objective.** Predicting the next *frame* (or the next latent state) forces an internal model of objects, physics, and causation in a way that next-token prediction on text does not.
- **Embodiment.** Robots need a forward model to plan; a language model's world knowledge is the wrong shape for motor control.

Why it might not: world models have repeatedly looked impressive in demos and failed on long-horizon consistency, and the training signal is much noisier than text. The honest state as of 2026 is that interactive video world models are good enough to be useful for simulation and content, and not yet good enough to be the substrate for general reasoning.

**[P] Prediction (45%):** by 2031, a frontier lab ships a model whose pretraining is majority non-text (video/3D/action) and which is competitive with text-first models on physical reasoning benchmarks. **[P] (15%):** such a model replaces the text-first paradigm as the main scaling axis before 2033.

---

## 2.6 Other resets, ranked by plausibility

| Candidate | What would have to be true | My probability by 2033 |
|---|---|---|
| **Memory/continual learning as the main axis** | solves forgetting and poisoning | 40% |
| **World models become the substrate** | video pretraining beats text pretraining per FLOP | 15% |
| **Diffusion / parallel-decoding LMs** | quality matches autoregressive at 10× lower latency | 20% (as a latency product, not a paradigm) |
| **Non-transformer recurrences with exact recall** | DeltaProduct-style models close the recall gap cheaply | 25% (as a hybrid component) |
| **Analog / optical / neuromorphic compute** | 10× energy advantage at production yield | <5% by 2033 |
| **A genuinely new learning rule** (not backprop) | scales to frontier and beats it | <10% by 2033 |

The pattern: the plausible resets are about *what the model learns from*, not *how the block is wired*.

---

## 2.7 Sizes and tiers in 2031

The distribution of model sizes will be more extreme than today, not less:

| Tier | Parameters | Where it runs | What it is for |
|---|---|---|---|
| **Frontier** | 10²⁸–10²⁹ FLOP training; 1–10T total, 30–100B active | 5 GW campuses | hardest reasoning, research, agent orchestration |
| **Workhorse** | 100–500B total, 3–10B active | single 8-GPU node, or a rented slice | 95% of production traffic |
| **Edge** | 1–4B active, 4-bit | phone, laptop, car | privacy, latency, offline, personalisation |
| **Specialist** | 1–30B, often distilled | anywhere | code, vision, speech, robotics control |

Two observations. First, **the workhorse tier is where the money is**, and it is filled by distilled versions of last year's frontier — the 6–12 month lag is structural, not a failure of open source. Second, **the edge tier is underrated**: a 3B-active model at 4-bit running at 40 tokens/s locally is enough for most personal tasks, and it removes the latency, privacy, and cost objections in one move.

---

## 2.8 A speculative 2031 model card

Written as an artefact, to make the abstraction concrete:

```
Name              Frontier-class system, 2031 vintage
Base               hybrid: 82% gated-linear layers, 18% sparse global attention
                   MoE: 512 experts, top-8, 1 shared; 2.1T total / 48B active
Context            4M native; association capacity ~30k distinct facts (measured)
Memory             persistent per-user store, consolidated nightly into a
                   reviewed adapter; opt-in, auditable, revertible
Modalities         text, image, audio, video in; text, image, audio, action out
Action             computer use, code execution, robot control (via VLA adapter)
Post-training      RL on ~50k verifiable environments; 3 reward models;
                   constitutional layer; deception probes at inference
Serving            FP4, speculative decode (MTP depth 3), disaggregated
                   prefill/decode, 3.1M tokens/s per GW
Price              ~$0.12 per M input, $0.40 per M output (2031 dollars)
Reported limits    long-horizon reliability; unverifiable domains; memory poisoning
```

Everything in that card is an extrapolation of something already measured in 2026. That is the point: the 2031 model is not alien. It is today's model with the seventh knob turned up and the price divided by a hundred.

---

## 2.9 Predictions

| Claim | Probability |
|---|---|
| No new block architecture in frontier open weights before 2030 | 80% |
| Hybrid local/global attention in >80% of new open models by 2028 | 85% |
| Frontier sparsity >64× by 2029 | 60% |
| A product that measurably improves with use (external memory) by 2028 | 60% |
| Test-time weight updates in a frontier product before 2030 | 35% |
| World-model-first frontier model by 2031 | 45% |
| "Model card" replaced by "system card" as the standard artefact by 2029 | 55% |

---

## 2.10 What would change this chapter

- **A recall breakthrough in linear attention** (closing the gap to full attention at 10× lower cost) would make 95% local layers standard and collapse the KV-cache problem entirely.
- **A continual-learning solution that survives poisoning** would move the seventh knob from research to product overnight, and would make the "model as checkpoint" framing obsolete faster than anything else on this list.
- **A demonstrated scaling failure** — loss curves flattening at 10²⁹ FLOPs — would push everything toward inference-time compute and small-model specialisation, and would make this chapter's tier table wrong in an interesting way.

**Next:** [3 — Learning after data](03-learning-after-data.md).
