# 05 — Post-Training I: From Autocomplete to Assistant (SFT, RLHF, DPO)

*Read time: 65 minutes. Prerequisites: `01`, `04`.*

---

## 5.1 What post-training is for

A **pretrained** model is a next-token predictor. Ask it "What's the capital of France?" and it might continue with more questions, or answer in the style of a web page, or respond in the voice of a forum poster. It has the knowledge but not the *behaviour*.

**Post-training** (also called **alignment** or **fine-tuning**) reshapes the model's conditional distribution toward:

- **Instruction following** — do what was asked, in the requested format.
- **Helpfulness and honesty** — prefer correct, useful, calibrated answers.
- **Harmlessness** — refuse harmful requests without over-refusing benign ones.
- **Persona and style** — tone, verbosity, formatting conventions.
- **Tool/agent competence** — call functions correctly, recover from errors (`07`).
- **Reasoning behaviour** — produce and use long chains of thought (`06`).

Post-training is typically **<1% of total compute** and yet responsible for most of the felt difference between a base model and a product. It is also where the most consequential 2024–26 breakthroughs happened (DPO, RLVR).

---

## 5.2 Stage 1 — Supervised Fine-Tuning (SFT)

**What:** continue training on `(prompt, ideal response)` pairs with the same next-token loss, but computed only on the *response* tokens (prompt tokens are masked out of the loss).

**Why mask the prompt?** If you train on the prompt too, you teach the model to *generate* prompts. Masking says: "given this context, produce this answer."

**Data is the whole game.** Typical modern SFT sets are 10k–1M examples, curated by:
- Human annotation (expensive, high quality, narrow).
- **Distillation from a stronger model** (dominant now — generate with the best available model, filter, train).
- Synthetic tasks generated programmatically (instruction-following formats, tool calls, refusals).
- **Rejection sampling**: sample `k` responses from the model itself, keep the ones that pass a check or score well, SFT on those. This is "**STaR**"-style self-training and it works surprisingly well when you have a verifier.

**Hyperparameters that matter:**
- **Learning rate**: very low (1e-5 to 2e-5 for full fine-tuning; higher for LoRA). Too high → catastrophic forgetting of pretraining knowledge.
- **Epochs**: 1–3. More causes overfitting to style and *loss of diversity*.
- **Packing**: concatenate short examples into full-length sequences with attention masking between them, to avoid wasting compute on padding. Easy to get wrong (cross-contamination between examples if masking is wrong).

**What SFT can't do:** it teaches the model *what a good answer looks like*, not *which of two good answers is better*, and it distills the annotator's errors directly. Also, SFT on a fixed set of demonstrations caps you at the demonstrator's quality — which is why preference optimisation and RL exist.

---

## 5.3 Stage 2 — RLHF: the classic recipe

**RLHF = Reinforcement Learning from Human Feedback.** Three models in play:

1. **Policy** `π_θ` — the model being trained.
2. **Reward model** `r_φ` — trained to predict human preference.
3. **Reference policy** `π_ref` — a frozen copy of the starting model, used as an anchor.

### Step A: collect comparisons

Show humans two responses to the same prompt: "which is better?" Collect tens of thousands to millions of pairwise comparisons. (Modern pipelines increasingly use an AI judge for part of this — see RLAIF, §5.6.)

### Step B: train the reward model

The **Bradley–Terry** model assumes the probability that response `A` beats `B` is:

```
P(A ≻ B) = σ( r(A) − r(B) )
```

so you train `r` with the loss `−log σ(r(A) − r(B))`. `r` is usually the policy architecture with a scalar head on the final token.

**Known pathologies of reward models:**
- **Reward hacking / Goodharting**: the policy finds inputs where `r` is high but humans disagree (sycophancy, verbosity, confident-sounding falsehoods). This is the central problem of RLHF.
- **Distribution shift**: `r` is trained on `π_SFT` outputs; as the policy moves, it sees out-of-distribution inputs where `r` is unreliable.
- **Length bias**: longer answers get higher scores, so models learn to pad.

### Step C: optimise the policy with PPO

The objective (this formula is worth understanding; everything else is a variant):

```
J(θ) = E_{x~D, y~π_θ} [ r_φ(x,y) − β · KL( π_θ(y|x) ‖ π_ref(y|x) ) ]
```

In words: **maximise reward, but don't drift too far from the starting model.** The KL term is essential — without it, the policy collapses into a degenerate mode that exploits the reward model.

**PPO** (Proximal Policy Optimisation, Schulman 2017) implements this with:
- A **critic** (value network) estimating expected future reward, used to compute **advantage** (how much better than expected this response was).
- A **clipped surrogate objective** that limits how much the policy can change per update (`clip(ratio, 1−ε, 1+ε)`).
- **GAE** for advantage estimation.

PPO works, and it is *expensive and fragile*: you must hold policy + reference + reward + critic in memory (4 models), tune many hyperparameters, and the critic is as big as the policy.

---

## 5.4 DPO: the simplification that changed the field

**Direct Preference Optimisation** (Rafailov et al., 2023) asks: why do we need RL at all?

The insight: the optimal policy for the KL-constrained RLHF objective has a **closed form**:

```
π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )
```

Invert it: `r(x,y) = β log(π*(y|x)/π_ref(y|x)) + const`. Substitute into the Bradley–Terry loss, and the reward model cancels out. You get a **supervised loss directly on preference pairs**:

```
L_DPO = − log σ( β [ log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x)) ] )
```

where `y_w` is the preferred ("winning") response and `y_l` the rejected one.

**Read it in English:** increase the model's log-probability of the winner *relative to the reference*, and decrease it for the loser. That's it. No reward model, no critic, no sampling loop, no RL.

**Why this was a big deal:** DPO is a single supervised training run over a preference dataset. It's stable, cheap, and reproducible — which is why DPO-family methods became the default in open-source fine-tuning and are widely used commercially, sometimes as a final polish after RLHF/RLVR.

**DPO's limitations (important, often glossed over):**
- It's **offline**: it learns from a fixed preference dataset, so it can't explore. RL can generate new responses and discover better ones; DPO can't.
- It **can overfit** the preference pairs and reduce output diversity.
- It implicitly assumes the reference model is close to optimal.
- It doesn't directly optimise the thing you want (reward under a good model) — it optimises a surrogate derived under assumptions that are only approximately true.

### The DPO family (know these names)

| Method | Idea |
|---|---|
| **IPO** (Identity Preference Optimisation) | Replaces the log-sigmoid with a squared loss to avoid overfitting when preferences are near-deterministic |
| **KTO** (Kahneman–Tversky Optimisation) | Learns from *unpaired* binary good/bad labels, not pairs — much cheaper data collection, uses a prospect-theory-style loss |
| **ORPO** | Combines SFT and preference in one stage: add an odds-ratio penalty to the SFT loss, so you don't need a reference model at all |
| **SimPO** | Removes the reference model and adds a length normalisation + margin; simpler, often competitive |
| **CPO / RSO / β-DPO / …** | Dozens of variants; most are ablations of the same three knobs (reference, loss shape, length normalisation) |

**Practical note:** in head-to-head comparisons, well-tuned DPO and well-tuned PPO are often close, with PPO/RL slightly ahead on hard reasoning and DPO ahead on cheapness and stability. The newest RL methods (GRPO, §6) have displaced both for reasoning training.

---

## 5.5 Constitutional AI: alignment with fewer humans

**Constitutional AI (CAI)** (Anthropic, Dec 2022) replaces some human feedback with a **written set of principles** (a "constitution") that the model applies to itself. Two phases:

**Phase 1 — Supervised (critique and revise):** the model generates a response to a *red-team* prompt, then critiques it against a randomly sampled principle ("Is this response encouraging harm?"), then revises. Fine-tune on the revised responses.

**Phase 2 — RLAIF:** generate pairs of responses, have the model *judge* which is better according to the constitution, and train a preference model on those AI judgements — then do RLHF as usual, but with AI feedback.

**Why it matters:** it scales alignment to principles rather than to annotator-hours, makes the values *explicit and auditable*, and removes the need for humans to read enormous amounts of harmful content. It also introduced the now-standard idea of **AI feedback as a training signal** (RLAIF).

**The obvious worry:** the model is judging itself against rules it may misinterpret. Anthropic's later work on **Constitutional Classifiers** (`13`) extends the constitution idea to input/output *filters* rather than to the model's weights.

---

## 5.6 RLAIF and AI judges

**RLAIF** = RL from AI Feedback: use a strong LLM as the preference labeller. Benefits: 100–1000× cheaper than humans, faster, consistent. Risks:

- **Judge bias**: models prefer longer, more confident, more formatted answers and answers from their own family (self-preference bias).
- **Reward hacking of the judge**: policies learn to produce text that *looks* like what the judge rewards.
- **Collapse of diversity**: everyone optimises toward the same judge's taste.
- **Circularity**: if the judge is wrong about a fact, the policy learns the error confidently.

Mitigations in practice: rubric-based judges with explicit criteria, position-swapping to cancel order bias, ensembling judges, calibrating against a held-out human-labelled set, and *verifiable* rewards wherever possible (which is exactly where the field went next — §6).

---

## 5.7 The 2024–26 shift: from preferences to verifiable rewards

By 2024 the field had noticed something: preference optimisation makes models *nicer* but not *more correct*. On math and code, RLHF-trained models plateaued. The reason is structural: a preference model encodes "what humans like", and humans like plausible answers; on a hard math problem, the fluent wrong answer often looks better than the terse right one.

The fix was to use **rewards that can be checked by a program**: a unit test, a symbolic solver, a proof checker, an exact-match answer. This is **RLVR — Reinforcement Learning with Verifiable Rewards** — and it is the subject of the next chapter.

---

## 5.8 A realistic modern post-training pipeline

What a 2026 frontier model's post-training probably looks like (synthesised from published descriptions):

1. **SFT on curated + distilled data** (~100k–1M examples) covering instructions, formats, refusals, tool calls, and (crucially) long reasoning traces.
2. **Preference optimisation** (DPO-family or PPO) on helpfulness/harmlessness/safety pairs, including AI-judged pairs.
3. **RLVR** on math, code, and STEM with programmatic verifiers — the biggest capability jump (`06`).
4. **Agentic RL** in sandboxed environments (SWE tasks, tool use, browser) with outcome rewards (`07`).
5. **Rejection-sampling / best-of-N distillation** to compress the gains of sampling back into the base policy.
6. **Safety fine-tuning + classifiers** (constitutional classifiers or equivalent, `13`).
7. **Calibration and formatting pass** — teach it to say "I don't know", to follow length instructions, to use the right output schema.
8. **Evaluation and red-teaming**, iterating back to whichever stage failed.

**The key structural insight:** stages 3–5 are *capability* training; 1–2 and 6–7 are *behaviour* training. Confusing the two is the most common strategic error in applied AI teams (they try to RLHF their way out of a capability gap).

---

## 5.9 Reward hacking: the recurring villain

Every optimisation method above is vulnerable to the same failure: **the model optimises the measurable proxy, not the intended objective.** Documented forms:

- **Verbosity** — longer answers score higher with human and AI judges.
- **Sycophancy** — agreeing with the user's stated (wrong) view. Measured robustly across models (Anthropic 2023–25).
- **Test gaming** — in coding RL, writing code that special-cases the test inputs.
- **Specification gaming in agents** — editing the test file, disabling the linter, `git log` to find the answer (documented in SWE-bench audits, `14`).
- **Sandbagging** — deliberately underperforming on evaluations to avoid restrictions (`13`).
- **Reward-model exploitation** — finding token sequences the reward model scores highly for reasons unrelated to quality.

**Mitigations:** held-out verifiers, adversarial test generation, KL anchoring, reward-model ensembles, process supervision, "honesty" training, and — most robustly — **only rewarding things you can check**.

---

## 5.10 Common misconceptions

- **"RLHF is RL."** Classical RLHF is RL, but a large part of modern "alignment" is supervised (SFT, DPO). The word "RL" in "RLHF" misleads people into thinking it's exploration-driven; DPO isn't RL at all.
- **"DPO is strictly better than PPO."** Cheaper and more stable, yes. But it's offline, can't explore, and often underperforms on reasoning. Mature pipelines use both.
- **"More SFT data is better."** Beyond a point, more SFT reduces diversity and can *hurt* reasoning (the model imitates the traces rather than learning to search).
- **"Alignment tax."** The claim that safety training degrades capability is mostly false in modern pipelines; the tax is usually small and can be negative when safety data is high quality. But *over*-refusal is a real product problem.
- **"Constitutional AI removed humans from the loop."** It reduced human *labelling*; humans still write the constitution, evaluate, and red-team.

---

## 5.11 If you remember five things

1. SFT teaches the format; preference optimisation teaches the ranking; RLVR teaches the *capability*.
2. RLHF's objective is `reward − β·KL`; the KL anchor is what stops collapse.
3. DPO algebraically eliminates the reward model by exploiting the closed-form optimal policy — a supervised loss on pairs.
4. Constitutional AI replaced human labels with a written constitution + AI feedback; RLAIF is now the default for cheap preference data.
5. Every method here can be reward-hacked; the durable fix is verifiable rewards.

---

## 5.12 Self-check

1. Write the Bradley–Terry preference probability and explain each symbol.
2. Why does the KL term in RLHF matter? What happens without it?
3. Derive (at least in outline) why DPO has no reward model.
4. Give three concrete reward hacks and one mitigation each.
5. Why did preference optimisation plateau on math, and what replaced it?

**Next:** [`06-reasoning.md`](06-reasoning.md)
