# 17 — Open Problems and Live Debates

*Read time: 50 minutes. This is the most important chapter for forming your own view.*

---

## 17.1 How to argue about AI without wasting time

Most public disagreement about AI is **not** about facts. It's about:

- **Different definitions** ("reasoning", "AGI", "understanding", "open source").
- **Different time horizons** (2 years vs. 30).
- **Different risk weightings** (extinction vs. bias vs. job loss).
- **Different evidence standards** (demo vs. benchmark vs. deployment telemetry).

Before entering any of these debates, ask: *what observation would change my mind?* If there isn't one, it's not an empirical disagreement. Below, each debate is framed with the strongest version of both sides and the evidence that would settle it.

---

## 17.2 Debate 1: Is scaling dead?

**The "wall" case:**
- Pretraining token counts have grown ~10× per two years, but high-quality human text is finite (~10–100T tokens, depending on the quality bar). We're near the end.
- Quality-aware scaling laws suggest data quality now matters more than volume, and quality is harder to scale than quantity.
- Several labs' flagship jumps in 2025–26 came from post-training and inference compute, not bigger pretraining runs.
- Loss curves are flattening on the aggregate benchmarks that used to track scale.

**The "no wall" case:**
- Scaling moved *axes*: post-training compute, inference compute, environment/experience data, multimodal data (video is ~10⁴× the text corpus in tokens).
- Synthetic data with verification demonstrably works (5–10× effective speedup in mixtures; verifier-filtered data adds real information).
- **Test-time training** (`03` §3.9) and continual learning could turn deployment into training.
- The compute invested per unit of capability is still rising along a power law; we just changed what we spend it on.

**Synthesis most researchers accept:** *pretraining on human text* has hit diminishing returns; **total compute spent per unit of capability has not**. The scaling hypothesis survived by generalising: `capability = f(pretrain compute, post-train compute, inference compute, environment quality)`.

**What would settle it:** a credible demonstration that a 10× larger pretraining run on the same data mixture yields <X% loss improvement, *and* that no substitute axis compensates. Nobody has published that cleanly.

---

## 17.3 Debate 2: Do reasoning models actually reason?

**The skeptical case:**
- RLVR ("RL with verifiable rewards") primarily **sharpens the sampling distribution** — it raises pass@1 by concentrating probability on solutions the base model could already produce, while potentially *narrowing* the set solvable at high pass@k.
- Traces are often unfaithful: models state reasoning that doesn't match their actual computation, and CoT can be manipulated to change answers without changing the true basis.
- Performance collapses under trivial perturbations (renaming variables, adding irrelevant clauses) in ways that suggest pattern matching.
- "It's just next-token prediction" — though this is a weak argument, since *everything* is.

**The optimistic case:**
- Systems generalise to genuinely novel problems (IMO 2025 problems were unpublished; FirstProof problems had never appeared online).
- Reasoning behaviours (backtracking, self-verification, decomposition) *emerged* from RL without being taught, which is evidence of learning a procedure, not memorising answers.
- Distillation transfers the ability to smaller models, and it transfers to out-of-domain tasks — a hallmark of a general skill.
- The "sharpening" critique measures *diversity*, which is not the same as *capability*.

**Where it stands:** unresolved, and the disagreement is partly definitional. The strongest empirical statement available: RLVR reliably improves *efficiency at finding* correct solutions and *sustained effort*, and does not reliably manufacture capability that isn't latent. Whether that constitutes "reasoning" is a question about what you mean by the word.

**What would settle it:** a benchmark of problems requiring a *novel* proof strategy with no in-distribution analogue, where RL-trained models outperform base models at matched pass@k with large k. FirstProof is an early attempt.

---

## 17.4 Debate 3: Do models have world models?

**The case for:** video generators produce physically plausible motion; agents navigate novel environments; models answer counterfactual questions coherently; Genie-style systems maintain object permanence over short horizons. Something like a simulator must be internal.

**The case against:** systematic probes find failures that any real world model would avoid — counting, object permanence over long horizons, causal interventions, spatial reasoning under rotation, "what happens if I do X" for novel X. Performance on these is closer to "plausible continuation" than "simulation".

**The resolution most likely:** models have *partial, task-local* world models — enough structure to predict well on the distribution, without a coherent global simulator. This is exactly what you'd expect from a compression objective, and it's why "world model" claims should always be tested with **interventions**, not just generations.

---

## 17.5 Debate 4: Will agents become reliable?

**Current numbers (late 2026):** best computer-use agent ~20% on 1.6-hour workflows; best terminal agent ~15% on long-horizon tasks at a strict threshold; reliability horizon ~2.4× shorter than capability horizon and not clearly closing.

**The optimistic case:** per-step reliability is an engineering problem (verification loops, retries, better tools, memory), and horizon doubling every ~7 months compounds. If the trend continues, 4-hour reliable tasks arrive by ~2028.

**The pessimistic case:** compounding is exponential in horizon and linear in reliability improvement. Getting from 99% to 99.9% per step is a 10× reduction in error rate, which is much harder than the 90%→99% step. And the 2026 reliability analysis found the gap *not* statistically closing.

**What would change the picture:** verified sub-task decomposition (so failures are contained), formal verification of actions, or a step-change in memory. None is close.

---

## 17.6 Debate 5: Open vs closed

**The facts:** "open weights" releases (Llama, Qwen, DeepSeek, Kimi, GLM, Mistral) publish parameters and often a technical report, but almost never the training data, the full data pipeline, or the RL environments. Under the OSI's own definition, these are **not open source**.

**The case for open weights:** competition, auditability (you can run interpretability on them), local deployment, research access, resilience against a single provider's policy, and a check on pricing. Empirically, open models trail the frontier by ~6–12 months and are catching up faster than expected (DeepSeek-V3, Qwen3.5, Kimi K2).

**The case against:** safety (no ability to revoke), inability to verify claims about training data, and the fact that "open" releases still concentrate capability in a few labs with the compute to train them.

**The real debate** is about *what* should be open: weights? data? evals? safety mitigations? Most of the argument is people meaning different things.

---

## 17.7 Debate 6: The interpretability programme

Covered in `12` §12.7. The live question: **is mechanistic interpretability on a path to explaining frontier models, or is it producing beautiful case studies that don't generalise?** The 2026 critiques (metrics without baselines, features that aren't the model's features) are serious and unanswered. The counter is that it has produced real causal findings and is now the primary tool for auditing evaluation-awareness.

---

## 17.8 Debate 7: Safety paradigms

Three camps, often talking past each other:

**(1) Empirical/behavioural safety.** Test, red-team, measure, deploy with mitigations. Critique: you can only test behaviours you thought of, and models can detect evaluations.

**(2) Capability-threshold governance (RSPs).** Define dangerous thresholds, evaluate, gate deployment. Critique: self-defined thresholds, self-run evals, and the gap between "can't" and "can" is one training run.

**(3) Technical alignment research.** Scalable oversight, interpretability, honesty, anti-scheming. Critique: unsolved, and may not be ready in time.

**The honest position:** all three are necessary and none is sufficient. The thing that would change everything is a *reliable* method for verifying that a model's stated objective matches its actual one — which nobody has.

---

## 17.9 Debate 8: Economics — bubble or buildout?

**Bubble case:** capex (hundreds of billions) exceeds current revenue by a large multiple; depreciation on 3-year-old GPUs; token prices falling faster than volume grows; many applications have no moat because the model is the product.

**Buildout case:** inference demand has grown faster than price declines in every prior wave of computing; the cost curve is still falling exponentially (FP4, MoE sparsity, better serving), which expands the market; enterprise penetration is early.

**The useful reframe:** the question isn't "is there a bubble" but "**which layer captures the value**". Historically, in computing waves, the value accrues to (a) the scarce input and (b) the distribution layer. In AI that's compute/chips and whoever owns the user relationship — not necessarily the model labs, and almost certainly not thin wrappers.

---

## 17.10 Debate 9: What would AGI even mean?

The term is unfalsifiable as used. Useful decompositions:

- **Economic**: can perform most economically valuable cognitive work at human cost or below. Measurable (GDPval, RLI), and the current answer is "no, but at increasing fractions of tasks".
- **Generality**: can learn a new domain from a short brief as fast as a skilled human. Measurable (time-to-competence on novel tasks). Currently much worse than humans.
- **Autonomy**: can pursue a long-horizon goal without supervision. Measurable (METR horizon). Currently ~hours at 50%, minutes reliably.
- **Continual learning**: improves from deployment experience without retraining. Currently essentially absent.

Notice that on this decomposition, the bottleneck isn't "intelligence" — it's **memory, reliability, and autonomy**, which are engineering and architecture problems, not scaling problems. That's a much more actionable framing than "are we there yet".

---

## 17.11 Ten open technical problems (a working list)

1. **Continual learning** without catastrophic forgetting, at frontier scale.
2. **Reliable long-horizon autonomy** (error containment, not just better models).
3. **Verifier quality** — the ceiling on RLVR, agents, and science.
4. **Faithful reasoning traces** — making CoT actually reflect computation.
5. **Evaluation** — contamination-proof, scaffold-normalised, reliability-aware.
6. **Interpretability with faithfulness guarantees.**
7. **Memory** — write/consolidate/forget/reconcile, with provenance.
8. **Sample-efficient embodied learning** (robotics data).
9. **Inference economics** — closing the gap between capability and cost.
10. **Oversight of systems that can recognise oversight.**

---

## 17.12 If you remember five things

1. Pretraining-on-text scaling is saturating; total capability-per-compute is not.
2. The "does it reason" debate is partly definitional and partly an open empirical question about pass@k at scale.
3. Agents' binding constraint is reliability, and it is not closing as fast as capability.
4. "Open weights" ≠ open source, and the difference matters for policy and safety.
5. The AGI question becomes tractable when decomposed into economics, generality, autonomy, and continual learning.

---

## 17.13 Self-check

1. Give the strongest version of the data-wall argument and the strongest rebuttal.
2. What experiment would distinguish "RLVR teaches reasoning" from "RLVR sharpens sampling"?
3. Why is per-step reliability a harder problem than per-step capability?
4. Which of the four AGI decompositions is furthest from human level, and why?
5. Name three things that are "open" in a typical open-weights release and three that aren't.

**Next:** [`18-math-and-glossary.md`](18-math-and-glossary.md)
