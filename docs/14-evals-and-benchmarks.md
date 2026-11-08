# 14 — Evaluation: How We Know (and Don't Know) What Models Can Do

*Read time: 55 minutes. Prerequisites: `06`, `07`.*

---

## 14.1 Why evaluation is now a research field

For years, evaluation was an afterthought: run MMLU, report a number. By 2025 that broke down for four reasons:

1. **Saturation** — models exceed 90% on benchmarks that were "frontier" two years earlier (MMLU, GSM8K, HumanEval, GPQA is getting there).
2. **Contamination** — test items appear in training data, so the score measures memorisation.
3. **Scaffold dependence** — the same model scores 20 points differently under different agent harnesses, so "model capability" isn't well-defined without specifying the system.
4. **Flawed items** — audits found large fractions of benchmark questions are malformed or have wrong answers.

The result: **the thing you most want to know (will this model do my task well?) is the thing benchmarks are least able to tell you.** This chapter gives you the tools to read the numbers honestly, and the catalogue to know what each one measures.

---

## 14.2 The taxonomy of evaluations

| Kind | What it measures | Examples |
|---|---|---|
| **Knowledge/reasoning (static QA)** | breadth of knowledge, multi-step reasoning | MMLU, MMLU-Pro, GPQA, HLE, AIME, FrontierMath |
| **Coding** | function-level synthesis | HumanEval, MBPP, LiveCodeBench |
| **Software engineering (agentic)** | repo-level issue resolution | SWE-bench Verified/Pro, Terminal-Bench |
| **Tool/agent** | multi-turn tool use, computer control | τ-bench/τ²-bench, OSWorld/OSWorld 2.0, MCPMark, WebArena |
| **Long context** | retrieval & reasoning over long inputs | RULER, LongBench, NIAH (weak) |
| **Multimodal** | image/video/audio understanding | MMMU, MathVista, VideoMME |
| **Safety** | refusal, jailbreak robustness, dangerous capability | HarmBench, WMDP, internal red-team suites |
| **Preference/human** | what people actually like | LMArena, Artificial Analysis, live product telemetry |
| **Reliability** | consistency across repeats | METR time horizon, pass^k metrics |
| **Economic** | real work output | GDPval, Remote Labor Index, Frontier-Bench |

---

## 14.3 The benchmark catalogue (what each one is actually for)

**MMLU / MMLU-Pro.** 14k/12k multiple-choice questions across 57 subjects. Saturated (frontier >88%). MMLU-Pro uses 10-option questions and chain-of-thought, harder but also saturating.

**GPQA.** 448 graduate-level science questions written to be Google-proof. "GPQA Diamond" (198 questions) is the standard subset. Frontier models now score 80–90%+; it's becoming saturated.

**AIME.** Competition math, 30 problems/year, integer answers. High variance (30 problems!), so always report over multiple years and with error bars. Frontier models saturate it and have moved to harder sets (HMMT, IMO problems, FrontierMath).

**Humanity's Last Exam (HLE).** 2,500 expert questions, designed to be the "last exam". As of late 2026, leaders are in the 45–65% range depending on the harness and version. **Epoch AI audited it in Aug 2026 and rated it "Flawed"** — 46% of a random 48-question sample had accuracy-altering errors. HLE-Verified (Feb 2026) produced 668 certified items. *Read HLE numbers as "roughly hard", not as a precise ranking.*

**ARC-AGI-1/2/3.** Abstract reasoning puzzles designed to resist memorisation. ARC-AGI-1 was beaten at high compute (o3, Dec 2024, thousands of dollars per task). ARC-AGI-2 (2025) restored the gap; ARC-AGI-3 (2026) is interactive/game-like. Kaggle leaderboards in Nov 2026 show human teams at ~88% on ARC-AGI-2 and ~75% on ARC-AGI-3 — note these are *competition* scores with heavy per-task compute and search, not single-shot model scores.

**FrontierMath.** Research-level math problems (many require expert hours). Scores are low (single digits to ~30% on the hardest tiers depending on the model and tool use) and it's the best available "hard math" signal, though it's small and guarded.

**SWE-bench Verified / Pro.** Real GitHub issues with hidden tests. Verified (500) saturated: ~70%+ in 2025, ~95%+ in 2026. **OpenAI audited it in 2026 and found 59.4% of problems had flawed tests**, then stopped reporting it; separately, an analysis found one model claiming 81.4% where 24.4% of its trajectories simply ran `git log` to read the answer. **SWE-bench Pro** (Scale, 2025) is harder and more honest: frontier models scored ~23% at launch.

**τ-bench / τ²-bench.** Tool-agent-user conversations in a simulated domain (airline, retail, telecom); measures whether the agent follows policy while helping. Good for "does it use tools correctly" without the noise of real APIs.

**OSWorld / OSWorld 2.0.** Real computer use in a VM. OSWorld 1.0: ~30 tool calls/task. **OSWorld 2.0** (2026): 108 long-horizon workflows, median 1.6 human hours, ~318 tool calls; best model **20.6%** full completion, 54.8% partial. The most honest "computer use" number available.

**Terminal-Bench / Long-Horizon-Terminal-Bench.** Shell-based long tasks. LH-Terminal-Bench (2026): 46 tasks, ~231 episodes, ~85 minutes, ~9.9M tokens per run; strongest model 15.2% pass@1 at a 0.95 partial-reward threshold, mean across models 4.3%.

**GDPval / Remote Labor Index / Frontier-Bench.** Economic-value evaluations: does the model produce work products a professional would accept? The right frame for "will this replace a job function", and much harder than QA.

**METR time horizon.** Fit a logistic model to success probability vs. task length (in human-expert minutes), extract the length at 50% success. Doubling roughly every 7 months since 2023. The best single "how capable is it" summary — with the reliability caveat below.

**LMArena.** Pairwise human preference on chat prompts, Elo-style. Measures *preference*, which correlates with style, formatting, and agreeableness as much as correctness. Useful, gameable, and frequently misread as a capability ranking.

---

## 14.4 Contamination: the problem you can't fully solve

**How it happens:** benchmark questions are on the public web; pretraining crawls the web; the model memorises the answer.

**Detection methods:**
- **N-gram overlap** between benchmark items and training data (needs training data, which you usually don't have).
- **Canary strings** — insert a unique random string in the benchmark and check whether the model can complete it.
- **Paraphrase/perturbation** — rewrite the question; if the score drops a lot, it was memorised.
- **Order/format perturbation** — shuffle options, change variable names.
- **Held-out private sets** — the only robust fix (HLE has one; SWE-bench Pro has a private subset).

**The uncomfortable finding:** models can score well on a benchmark without having seen it, by having seen *the solution pattern* in similar problems. "Contamination" is a spectrum, not a binary, and no public benchmark is clean.

---

## 14.5 Reading a number honestly: a checklist

When you see "Model X scores 92% on Y", ask:

1. **Which version of Y?** (SWE-bench Verified vs. Pro vs. Lite; HLE vs. HLE-Rolling vs. HLE-Verified.)
2. **Which harness?** Mini-SWE-agent vs. SWE-agent vs. custom scaffolding changes scores by 10–20 points.
3. **How many attempts?** pass@1 vs. pass@k vs. best-of-N with a verifier.
4. **How much inference compute?** A reasoning model at 100k thinking tokens is not comparable to one at 2k.
5. **Self-reported or independently run?** Vendor numbers use private scaffolds and sometimes favourable prompts.
6. **Error bars?** AIME has 30 problems; a 3-point difference is noise.
7. **Is the benchmark saturated?** If the top 10 models are within 3 points, the ranking is meaningless.
8. **What's the cost?** 95% at $40/task vs. 88% at $0.10/task are different products.
9. **Has anyone audited the items?** (HLE: 46% flawed.)
10. **Does it match your task?** Almost certainly not exactly.

---

## 14.6 Reliability: the metric everyone forgets

Standard benchmarks measure "can it, sometimes". Deployment needs "will it, every time".

**pass^k** (not pass@k): the probability that *all* k attempts succeed — the correct metric for reliability. It's brutal: with 90% per-attempt success, pass^5 = 59%.

METR-style analysis (2026, 60,417 attempts across 30 models) found:

- The **reliability horizon** (all-attempts success at 50%) is ~**2.4× shorter** than the capability horizon, for every model.
- The gap narrows by ~17%/year in point estimate, but **the trend is not statistically distinguishable from zero**.
- Across 14 agentic models over 18 months, "capability gains have only yielded small improvements in reliability".

This is the most important number in applied AI: **capability and reliability are coming apart**, and reliability is what you actually buy.

---

## 14.7 Designing an eval for your own task

Public benchmarks won't tell you if your system works. The minimum viable internal eval:

1. **Collect 50–200 real examples** of the task from production or users. Not synthetic.
2. **Define the pass criterion explicitly**, ideally executable (tests, schema validation, exact fields) rather than "looks good".
3. **Include adversarial/dirty cases**: empty input, wrong language, prompt injection attempt, very long input, ambiguous request.
4. **Score with an LLM judge *calibrated against ~50 human labels*.** Report judge–human agreement.
5. **Measure cost and latency per example**, not just accuracy.
6. **Measure variance**: run each example 3–5 times. Report pass@1 and pass^3.
7. **Re-run on every model/prompt/scaffold change**, and keep a frozen holdout you never look at.
8. **Track regressions**: an eval that only goes up is not measuring anything.

**The most common mistake:** evaluating on examples that resemble the training/demo examples. Your eval should be *harder* than your demo, or it's decoration.

---

## 14.8 Benchmarks to watch (and their expiry dates)

| Benchmark | Status (late 2026) | Watch for |
|---|---|---|
| MMLU, GSM8K, HumanEval | Dead | Do not cite |
| GPQA Diamond | Saturating | Already ~90% |
| AIME | Saturated | Move to HMMT/IMO |
| SWE-bench Verified | Retired by OpenAI; saturated + flawed | Use Pro |
| SWE-bench Pro | Alive, hard | Private subset is the real test |
| HLE | Flawed but informative | Use Verified/Rolling |
| ARC-AGI-2/3 | Alive | Compute-normalised scores only |
| τ²-bench | Alive | Policy adherence, not just success |
| OSWorld 2.0 | Alive, very hard | The honest computer-use number |
| METR horizon | Alive | **Check reliability horizon too** |
| GDPval / RLI | Alive | The economic-value frame |

---

## 14.9 Common misconceptions

- **"Higher MMLU = better model."** MMLU has been saturated since 2024; it measures almost nothing now.
- **"Arena Elo measures intelligence."** It measures preference, which rewards style and agreeableness.
- **"A benchmark win is a capability win."** Scaffold, compute, and contamination dominate at the top of leaderboards.
- **"pass@k shows the model can do it."** It shows the model can do it *with k tries and a verifier*. That's a different product.
- **"Benchmarks are objective."** They're instruments, and instruments drift, break, and get gamed.

---

## 14.10 If you remember five things

1. Every benchmark has a version, a harness, and an expiry date; a bare number is not information.
2. Contamination is a spectrum; only private held-out sets are robust.
3. Saturation means rankings at the top are noise — look at the hard, unaudited benchmarks instead.
4. Reliability (pass^k) lags capability by ~2.4× in horizon and isn't closing fast.
5. Build your own eval from real examples with executable criteria; public benchmarks are for orientation only.

---

## 14.11 Self-check

1. Why did OpenAI stop reporting SWE-bench Verified, and what should you use instead?
2. Compute pass^5 given 90% per-attempt success. Why does this matter for agents?
3. Name three ways to detect contamination without access to training data.
4. What does METR's time-horizon metric measure, and what does it miss?
5. Design a 5-line eval spec for "agent that files expense reports". What's your pass criterion?

**Next:** [`15-ai-for-science-and-math.md`](15-ai-for-science-and-math.md)
