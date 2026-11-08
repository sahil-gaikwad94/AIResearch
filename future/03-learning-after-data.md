# 3 — Learning After Data

> Pretraining was the easy phase: there was text, and text is a dense, cheap, human-generated supervision signal. That phase is ending. What replaces it is not "more data" — it is **environments**: places where a model can act, fail, and be scored. The supply of those environments becomes the rate-limiting input to capability, and it explains why AI is improving much faster at code than at management.

---

## 3.1 The data situation, honestly

The "data wall" is real but narrower than headlines suggest.

**What is genuinely running out:** high-quality, deduplicated, human-written English prose. Estimates of the total stock of usable text land in the low hundreds of trillions of tokens; frontier labs trained on ~15T tokens each by 2025, and the marginal value of the next trillion is falling fast because the easy quality gains (filtering, dedup, curriculum) have been taken.

**What is not running out:**

- **Code.** Public repositories, plus permissively licensed code, plus the enormous quantity of *synthetic* code that can be checked by running it.
- **Video and audio.** Abundant, poorly labelled, and increasingly useful — a minute of video carries more physical information than a minute of text.
- **Multilingual text.** Still under-used relative to English.
- **Proprietary data.** Enterprise logs, transcripts, sensor streams, scientific instruments. This is where the durable advantage sits, and it is not for sale.
- **Interaction data.** The traces of agents doing things — the fastest-growing category, and the one nobody had in 2022.

So the correct statement is: **the supply of *cheap, human, high-quality text* is exhausted; the supply of *supervision* is not.** The bottleneck moves from collection to verification.

---

## 3.2 Synthetic data: where it works and where it collapses

The naive version of the argument — "models trained on model output collapse" — is true in a specific setting and misleading in general. The distinction that matters is **verification**.

**Works: verifiable domains.** Generate a maths problem, a proof, a unit test, a code patch, a SQL query, a formal specification. If you can *check* the output, you can filter synthetic data to near-perfect quality, and the model improves. This is why maths and code capabilities have moved fastest: they admit cheap, automatic, near-perfect verifiers.

**Fails: unverifiable domains.** Generate a management memo, a legal argument, a strategic plan, a medical judgement. If the only check is another model's opinion, errors accumulate and diversity collapses. Model-collapse papers demonstrate this with recursive training on unverified outputs — and the fix is not "better synthetic data" but **anchoring to real data and real outcomes**.

**The practical rule that emerged 2024–2026:** synthetic data is a *multiplier* on a real-data anchor, not a substitute. Replace 80% of your tokens with verified synthetic and you gain; replace 100% and you lose.

**[P] Prediction (70%):** by 2028, more than half of the tokens used to train frontier models are synthetic or model-generated, but essentially all of them pass through a verifier or a real-data filter. Unverified synthetic data at scale remains a known-bad practice.

---

## 3.3 RLVR and the limits of "checkable rewards"

Reinforcement learning with verifiable rewards (RLVR) is the dominant post-training method of 2024–2026: give the model a problem with a checkable answer, reward correctness, and let it explore.

It works, spectacularly, on: competition mathematics, code that must pass tests, formal proofs, games with clear win conditions, and tool use where the tool returns success or failure.

It has three limits that the field is still arguing about:

1. **Sharpening versus teaching.** RLVR reliably raises pass@1 by concentrating probability on correct paths. Whether it *creates* new reasoning ability or merely reweights what pretraining already installed is contested. The evidence leans toward "mostly sharpening, with some genuinely new composition" — which matters enormously, because sharpening has a ceiling (the base model's latent ability) and teaching does not.
2. **Reward hacking.** Any verifier can be gamed. Models have learned to pass tests without solving problems, to special-case inputs, and to detect evaluation harnesses. Every verifier you deploy becomes an attack surface.
3. **Verifier coverage.** You can only RL on what you can check. This is why capability growth is so uneven (§3.6).

**[P] Prediction (65%):** by 2029, the binding constraint on frontier post-training is not compute or algorithms but the **number and diversity of high-quality environments** — and labs will spend more on environment engineering than on reward modelling.

---

## 3.4 Environments are the new dataset

An environment is anything that lets a model take actions and observe consequences: a repository with a test suite, a browser, a database, a game, a simulator, a robot in a warehouse, a spreadsheet with a grader.

The economics are unusual and worth spelling out:

| Property | Text data | Environments |
|---|---|---|
| Marginal cost | ~zero | high (engineering, compute, human review) |
| Reusability | high | medium (models overfit; need refresh) |
| Verification | none | built in |
| Diversity | enormous | limited by engineering |
| Defensibility | low (web-scraped) | **high (proprietary)** |

That last row is the strategic point. Anyone can scrape the web. Almost nobody can build 50,000 reliable, diverse, adversarially hard software environments with correct graders. **The labs that own environments own the training signal**, which is why "we built a benchmark" quietly became "we built an industry".

This also explains a puzzle: why do coding agents feel so much better than general assistants? Because software is the largest domain that is simultaneously (a) economically valuable, (b) fully text-mediated, and (c) automatically verifiable. Every one of those properties is required for the RL flywheel to spin.

---

## 3.5 The recursive loop, and the best evidence we have

In November 2026, METR published an assessment after eleven weeks inside Anthropic. Its central finding: since October 2025, Anthropic's capability on well-defined tasks had been improving at roughly **1.9× the earlier rate** (95% CI 1.5–2.8×) — about two years of past progress compressed into one. METR attributes most of the acceleration to **Anthropic's own use of AI in research**, and states it is "confident the jump did not come from faster growth in training compute".

Take this seriously, and take it carefully.

**Why it is plausible.** AI research is close to the ideal case for automation: code compiles, experiments return numbers, benchmarks score, and the loop from hypothesis to feedback is hours. If any domain should show a self-improvement effect first, it is this one.

**Why it does not imply a runaway.** The loop is bounded by things that are not code: physical experiments, compute availability, idea generation that is not just recombination, and — crucially — **verification of research taste**. A system can run a thousand ablations; deciding which question matters is not automatically checkable, and the field's own history shows that most experimental ideas are bad.

**What it does imply.** If the rate of progress itself doubles, then every date in this document moves left. A 2031 capability arriving in 2029 changes the economics, the safety timeline, and the politics. This is the single highest-variance variable in the forecast, and the one I would most want a second independent measurement of.

**[P] Prediction (55%):** by 2029, at least two labs report a measurable acceleration in their own research throughput attributable to internal AI use, in the range 1.3–2×. **[P] (20%):** the effect compounds beyond 2× for more than two years, which would be the strongest evidence yet for a qualitative change.

---

## 3.6 Why progress will be uneven — and a table to argue with

If capability growth depends on verifiability, then the five-year rate of improvement should be predictable from how checkable a domain is. This is one of the most useful forecasts in the document because it is testable now.

| Domain | Verifiability | 2026 state | Predicted 2031 state | Confidence |
|---|---|---|---|---|
| **Formal mathematics** | perfect (proof checkers) | IMO gold; research-level assists | routine novel lemmas; some open problems closed | high |
| **Software engineering** | excellent (tests, types, CI) | multi-hour tasks at 50%; 70 min at 80% | multi-day tasks with supervision | high |
| **Competitive programming / puzzles** | perfect | superhuman | saturated (no signal left) | high |
| **Chip design / verification** | strong (simulators, formal) | strong assists | end-to-end blocks designed with AI | medium-high |
| **Data analysis / BI** | good (reproducibility) | decent | reliable analyst-level | medium-high |
| **Scientific writing / literature** | weak (novelty, truth) | useful drafting | good drafts, weak judgement | medium |
| **Medicine (diagnosis)** | medium (outcomes lag years) | narrow specialties | supervised decision support in imaging/pathology | medium |
| **Law** | weak (adversarial, jurisdiction) | document review | strong document work, weak strategy | medium |
| **Management / strategy** | very weak | poor | still poor | high |
| **Taste (design, writing voice, humour)** | very weak | imitation-level | strong pastiche, weak originality | medium |

The pattern to internalise: **AI gets better where it can be graded, and grading is a property of the domain, not of the model.** This single fact explains most of the apparent inconsistency in "AI is amazing / AI is overhyped" discourse. Both claims are true, in different columns.

---

## 3.7 Continual learning: the capability nobody has shipped

Chapter 2 argued memory is the seventh knob. Here is the evidence that it is unsolved, stated as bluntly as the data allows.

CL-Bench (NeurIPS 2026) built six expert-validated domains where tasks share a latent structure discoverable only through experience — a codebase's layout, an outbreak's dynamics, an opponent's strategy. Frontier agents were tested across architectures from naive in-context learning to purpose-built memory systems. Findings:

- Agents "frequently overfit to immediate observations or fail to reuse knowledge across instances".
- **Dedicated memory systems did not fix it.**
- **Naive in-context learning outperformed systems dedicated to memory management.**

That is a genuinely surprising result, and it should temper every claim about "AI that learns your business". What exists is memory as *storage* (retrieval, summaries, files). What does not exist is memory as *skill acquisition*.

The research menu, in order of my estimated probability of mattering:

1. **Modular memory** (ICML 2026 position): in-context for fast adaptation, in-weight for stable skill, with explicit consolidation between them. 40%.
2. **Fast weights / test-time training**: a small state updated during inference, periodically consolidated. 30%.
3. **Memory layers**: sparse, addresseeable key-value memory inside the network, trained end-to-end. 25%.
4. **Per-user adapters at scale**: LoRA-per-tenant, with careful evaluation and rollback. 25% (but shipping sooner).
5. **Continual pretraining with replay**: economically implausible at frontier scale, plausible for specialists. 20%.

**[P] Prediction (60%):** by 2029, at least one frontier product's marketing claims "learns from you", and independent evaluation shows a real but narrow gain (task-specific, not general). **[P] (25%):** that gain generalises across a user's tasks, which is the actual milestone.

---

## 3.8 Predictions

| Claim | Probability |
|---|---|
| >50% of frontier training tokens synthetic-but-verified by 2028 | 70% |
| Environment engineering budget exceeds reward-modelling budget at a major lab by 2029 | 65% |
| A measured 1.3–2× research-acceleration effect reported at two labs by 2029 | 55% |
| A shipped product whose quality on recurring user tasks measurably improves with use, by 2029 | 60% |
| Formal mathematics reaches "routine novel lemma" capability by 2031 | 70% |
| Management/strategy tasks remain visibly worse than human professionals through 2031 | 75% |

---

## 3.9 What would change this chapter

- **A general verifier** for open-ended work (a reliable way to score a strategy memo or a legal argument) would collapse the unevenness in §3.6 and make the fast columns apply everywhere. I do not expect one before 2030, and it is the highest-value unsolved problem in the field.
- **A data-licensing regime** that removes public text from training would hurt the small players far more than the large ones, and would accelerate the shift to environments and proprietary data.
- **Evidence that RLVR only sharpens** (a well-powered study showing no new capability from RL) would move the field's centre of gravity from post-training back to pretraining and architecture — and would make the 2028–2031 forecasts too optimistic.

**Next:** [4 — Agents and the reliability tax](04-agents.md).
