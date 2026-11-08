# 06 — Reasoning Models: The Inference-Time Scaling Revolution

*Read time: 75 minutes. Prerequisites: `05`.*

---

## 6.1 The single most important idea of 2024–2026

For a decade, the recipe was: **train a bigger model to get a better answer.**
The new recipe is: **let the model think longer to get a better answer.**

That sounds like a small change. It is not. It means:

- Capability is a **function of a budget**, not a fixed property: `answer = f(problem, compute)`.
- You can trade dollars for accuracy *at query time*, per problem — spend 10× on the hard one, nothing on the easy one.
- The training objective changes: instead of "produce the best single response", it becomes "produce a *search process* that finds and verifies the best response".

This chapter covers the mechanisms: chain-of-thought, sampling, search, verifiers, process reward models, and the RL recipes (RLVR/GRPO) that made models do this by default.

---

## 6.2 Chain-of-Thought (CoT): the founding observation

**Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."** Show the model a few examples where the answer is preceded by intermediate reasoning steps, and accuracy on arithmetic/commonsense/logic jumps — but *only* for sufficiently large models (this is one of the "emergent ability" cases from `01` §1.9 that is arguably genuine, because it requires sustaining multi-step computation).

Why does it work? A Transformer computes a **fixed amount of computation per token** (one forward pass through L layers). A hard problem may require more serial computation than one forward pass can supply. Generating intermediate tokens gives the model **more forward passes**, each conditioning on the previous ones — i.e. **the context window becomes external working memory, and generation becomes a loop of computation.**

That is the whole theory of reasoning models, in one paragraph. Everything else is engineering that loop.

**Zero-shot CoT** ("Let's think step by step", Kojima et al. 2022) showed you don't even need examples — the phrase alone triggers the behaviour. This was a shock at the time and is now a baseline.

---

## 6.3 Test-time strategies (the pre-RL toolkit)

Before RL, people improved accuracy by *spending inference compute* on a fixed model. The main techniques, in increasing sophistication:

**(1) Sampling + majority vote ("self-consistency")**, Wang et al. 2022. Sample `k` reasoning traces at temperature > 0, take the **majority answer**. Big gains on math (e.g. +17% on GSM8K at the time). Intuition: independent errors are unlikely to agree.

**(2) Best-of-N with a verifier.** Sample `k` candidates, score each with a reward model or checker, take the best. Works extremely well *when the verifier is good* (code with unit tests, math with a symbolic checker). Collapses when the verifier is a weak neural scorer.

**(3) Tree search.** 
- **Tree of Thoughts** (Yao et al. 2023): generate multiple candidate "thoughts", evaluate them, expand the promising ones, backtrack. Treats reasoning as search over partial solutions.
- **Graph of Thoughts**, **Least-to-most prompting**, **Self-refine** (critique and revise), **Debate** (multiple agents argue; a judge decides).

**(4) ReAct** (Yao et al. 2022): interleave **Reason** and **Act** — the model thinks, calls a tool, observes the result, thinks again. This is the direct ancestor of every modern agent (`07`).

**(5) Program-aided reasoning.** Instead of doing arithmetic in tokens, emit code and execute it (PAL, PoT). Trivially better at arithmetic; the model becomes a planner and the interpreter does the computation. This is why "tool use" is not a bolt-on but a reasoning strategy.

---

## 6.4 Inference-time scaling laws

**The empirical claim (2024–25):** on hard reasoning tasks, accuracy rises approximately **log-linearly** with inference compute (tokens generated, or candidates sampled), and the slope can be steeper than the slope of pretraining scaling.

Key papers:
- **Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally can be More Efficient than Scaling Model Parameters"**: with an optimal compute allocation policy, test-time compute can beat a 14× larger model on problems where the smaller model has non-trivial success probability. Crucially, they show the *optimal strategy depends on difficulty*: for easy problems, best-of-N; for hard ones, sequential revision; and **a smaller model with good test-time compute can beat a bigger model**.
- **"Large Language Monkeys" (Brown et al. 2024)**: repeated sampling covers a wide range — with enough samples, a small model solves problems it "can't" solve, because the per-sample success probability is nonzero. This quantifies the gap between **pass@1** (one attempt) and **pass@k** (k attempts) and shows coverage scales smoothly even when accuracy looks flat.
- **s1 (2025)**: a 1,000-example "test-time scaling" dataset plus **budget forcing** (literally appending "Wait" to make the model keep thinking, or truncating to force it to answer) produced strong reasoning with minimal SFT.

**The important caveat:** inference scaling works *because* the base model already has a nonzero probability of solving the problem. It cannot manufacture capability that isn't there. There's a hard ceiling: `1 − (1−p)^k` with `p ≈ 0` stays ≈ 0.

---

## 6.5 Verifiers: ORM vs PRM

**Outcome Reward Model (ORM)**: scores the final answer. Simple, cheap, but gives one bit of feedback for a 10,000-token trajectory.

**Process Reward Model (PRM)**: scores each intermediate step. Richer signal, enables step-level search (beam search over reasoning steps) and better credit assignment. Costs: you need step-level labels (expensive), and PRMs are hackable (the model learns to produce steps that *look* correct), and the "correctness of an intermediate step" is philosophically murky — a step can be wrong in a way that doesn't matter, or right for the wrong reason.

DeepSeek's R1 paper explicitly *rejected* PRMs for their large-scale RL: "its advantages are limited compared to the additional computational overhead… neural reward models are susceptible to reward hacking." Their alternative: **rule-based rewards** (exact answer match + format compliance). This decision is one of the most consequential in the period, and it set the direction: **when you can check, check; don't model.**

**The verifier hierarchy (best to worst):**
1. Formal proof checker (Lean/Coq) — sound, complete for the formal statement.
2. Unit tests / execution — sound for the tests you wrote, gameable if tests are visible.
3. Symbolic solver / exact match — good for math with closed-form answers.
4. Learned PRM — dense signal, hackable, expensive.
5. LLM-as-judge with a rubric — flexible, biased, cheap.
6. Human preference model — what RLHF uses; encodes taste not truth.

---

## 6.6 o1 and the "thinking model" product shift

OpenAI's **o1** (Sept 2024) was the first widely released model trained to produce a long internal **reasoning trace** before answering, with the trace hidden from the user. Reported behaviour: large gains on competition math (AIME), science (GPQA), and code, scaling with "thinking time". **o3** (Apr 2025) extended this and, notably, achieved strong results on ARC-AGI-1 at high compute — the first system to do so — by spending thousands of dollars of inference compute per task.

What was publicly known/claimed about the recipe: RL on reasoning traces with reward signals from verifiable domains, plus a mechanism to allocate variable compute (the model learns *how long* to think). OpenAI did not publish the algorithm; the open replication came from DeepSeek.

**Product implications that are underappreciated:**
- Latency and cost become *variable and user-visible*. A "reasoning effort" dial becomes a product surface.
- Hidden reasoning traces create a **monitoring problem**: you cannot audit what you cannot see. Anthropic and others have argued for *summarised* traces; OpenAI moved toward exposing summaries. This is a live safety debate (`13`).
- The economics change: output tokens dominate cost, and output tokens can be 10–100× the input for hard problems.

---

## 6.7 DeepSeek-R1: the recipe, published

**DeepSeek-R1** (Jan 2025, later in *Nature*, Sept 2025) is the paper that made reasoning training reproducible. The core result:

### R1-Zero: pure RL, no SFT

Start from the base model (DeepSeek-V3-Base). Apply RL with **GRPO** and **rule-based rewards only**:
- **Accuracy reward**: does the final answer match (math) / do tests pass (code)?
- **Format reward**: is the reasoning inside ` thinking...</think>` tags?

No SFT, no human-written reasoning traces, no process reward model. Result: the model *spontaneously* developed long chains of thought, self-verification, backtracking, and "aha" moments — behaviours nobody taught it. AIME accuracy rose from ~15% to ~70% over training.

But R1-Zero had problems: mixed languages in the trace, unreadable reasoning, weak general (non-reasoning) capabilities.

### R1: the four-stage pipeline

1. **Cold start**: fine-tune on a few thousand *long CoT* examples (produced by R1-Zero and filtered for readability) to establish a clean format.
2. **Reasoning RL**: GRPO + rule-based rewards (same as Zero) on reasoning data.
3. **Rejection sampling + SFT**: sample many responses, keep the correct ones, add ~600k reasoning samples plus ~200k non-reasoning (general) samples, and SFT the base model on the mix. This restores general capability and readability.
4. **Final RL**: another RL round covering both reasoning (rule rewards) and general helpfulness (preference rewards).

### GRPO, precisely

**Group Relative Policy Optimisation** (Shao et al. 2024, from DeepSeekMath). The insight: you don't need a critic if you can compare a *group* of samples for the same prompt.

For each prompt `q`, sample `G` responses (e.g. 8–64). Compute reward `r_i` for each. Use the **group mean and standard deviation** as the baseline:

```
Â_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)
```

Then the objective (PPO-style clipped ratio, but with the group advantage and an explicit KL term):

```
J = E[ (1/G) Σ_i min( ratio_i · Â_i , clip(ratio_i, 1−ε, 1+ε) · Â_i ) − β·KL(π_θ ‖ π_ref) ]
```

**Why it's a big deal:** it removes the value network (critic), which was the largest memory and tuning burden in PPO. You go from 4 models to 2 (policy + reference), and the baseline is estimated from samples you were generating anyway.

**GRPO's known weaknesses:** if all `G` samples get the same reward (all correct or all wrong), the advantage is zero and there's **no gradient** — wasted compute. Remedies: curriculum/difficulty filtering, dynamic sampling, and variants that add a global or leave-one-out baseline. It's also noisy at small `G`.

### Distillation

DeepSeek then distilled R1's reasoning into Qwen-7B/32B and Llama-8B/70B by SFT on ~800k R1-generated samples. The distilled 7B/32B models became the most widely deployed reasoning models of 2025. **Key finding: distillation beat running RLVR directly on the small models** — cheaper and better.

---

## 6.8 What RLVR actually teaches (and doesn't)

This is the most contested empirical question in the field, so here's the state of evidence:

**The optimistic reading:** RLVR teaches genuine *reasoning behaviours* — longer chains, self-checking, backtracking, exploration of alternative approaches — that transfer to unseen problems.

**The skeptical reading** (several 2025 papers, e.g. "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"): RLVR mostly **sharpens the sampling distribution** — it raises pass@1 by concentrating probability on solutions the base model could already produce, while *narrowing* the set of problems solvable at pass@k. In other words, it improves *efficiency at finding* solutions more than it expands *the space of solvable problems*. This is consistent with the observation that RL-trained models lose output diversity and that pass@k curves can be worse than the base model's at large k.

**A synthesis most researchers would accept:** RLVR reliably improves (a) the model's ability to *stay on task* for long traces, (b) its willingness to verify, (c) its calibration on the trained domain, and (d) pass@1 at fixed budget. It does *not* reliably create new capabilities out of nothing, and it narrows diversity. Whether "learning to reason" is more than sharpening is unresolved (`17`).

**Related phenomena:**
- **Length explosion**: RLVR on reasoning makes traces longer over training; some of the gain is just "think longer", and some is genuinely better structure.
- **Domain spillover**: training on math improves code and some science, but not uniformly — transfer is real but narrower than early hype suggested.
- **Forgetting**: narrow RLVR can degrade general chat quality, hence the multi-stage mixed pipelines.

---

## 6.9 Test-time RL, self-play, and self-verification

Beyond "train, then sample", several 2025–26 lines make inference itself a learning process:

- **Test-time RL / TTRL**: run the RL update on the test problem itself, using majority vote as a pseudo-label. Improves hard-problem accuracy; expensive.
- **Self-verification (DeepSeekMath-V2, Nov 2025)**: train a *verifier* as well as a generator, and use the verifier as the reward model for proofs. Crucially, they **scale verification compute to auto-label new hard-to-verify proofs**, so the verifier improves as the generator does — closing the "generation–verification gap". Result: gold-level IMO 2025, near-perfect 118/120 on Putnam 2024 with scaled test-time compute. This is arguably the most important follow-up to R1, because it points at the real bottleneck: *the verifier is the ceiling*.
- **Self-play SWE-RL (2025–26)**: inject bugs and fix them, with a test patch as the formal spec; no human labels; +10.4 SWE-bench Verified.
- **Latent reasoning (Coconut and successors)**: do the reasoning in continuous hidden states instead of tokens, avoiding the tokenisation bottleneck and the cost of emitting every step. Promising, not yet at frontier parity.
- **Budget forcing (s1)**: control thinking length by truncating or appending "Wait" — a crude but effective demonstration that length itself is a dial.

---

## 6.10 The economics of thinking

A concrete example of why this matters commercially:

Suppose a model gets 40% on a hard benchmark at ~2k output tokens, and 70% at ~30k output tokens. At $15/M output tokens, that's $0.03 vs $0.45 per problem. If your task is worth $50 when solved, thinking is obviously worth it. If it's worth $0.50, it isn't. **Every serious deployment in 2026 does per-request compute routing**: a cheap model for classification, a mid model for drafting, a reasoning model with a budget for the hard 5%.

This is why:
- Providers expose "reasoning effort" / "thinking budget" parameters.
- **Distillation of reasoning into small models** is commercially enormous (you get 80% of the reasoning at 5% of the cost).
- **Speculative decoding and MTP** matter more, not less, when outputs are 30k tokens (`09`).

---

## 6.11 Common misconceptions

- **"Reasoning models are a different architecture."** No. Same Transformer; different post-training (and often different decoding/sampling).
- **"The model is 'thinking' in a human sense."** It's generating tokens that serve as computation. Whether that constitutes reasoning is a definitional question — see `17` for the actual debate (and why "it's just next-token prediction" is not a rebuttal: so is everything).
- **"More thinking is always better."** Overthinking causes error accumulation, loops, and degradation on easy problems. Models need to learn *when to stop*.
- **"RLVR scales indefinitely."** It's bounded by verifier quality and by the base model's coverage. Both are finite.
- **"Hidden CoT is just IP protection."** It's also a monitoring/oversight problem: hidden reasoning can't be audited (`13`).

---

## 6.12 If you remember five things

1. CoT works because generation gives the model more serial computation — the context is working memory.
2. Inference-time compute is a second scaling axis with log-linear returns, bounded by the base model's coverage.
3. GRPO = PPO without the critic, using group-relative advantages; it's what made RLVR affordable.
4. Rule-based verifiable rewards beat learned reward models wherever you can check the answer.
5. The verifier is the ceiling: self-verification (DeepSeekMath-V2) is the frontier, not the model.

---

## 6.13 Self-check

1. Why does CoT require a large model to help?
2. Derive the group-relative advantage and explain why a group of all-correct samples gives no learning signal.
3. Contrast ORM and PRM on cost, signal density, and hackability.
4. What does the "RLVR only sharpens" critique predict that the "RLVR teaches reasoning" view does not? What experiment would distinguish them?
5. Why is a formal proof checker a better verifier than unit tests? What does it cost you?
6. Given a fixed dollar budget per query, when should you scale the model vs. scale the thinking?

**Next:** [`07-agents.md`](07-agents.md)
