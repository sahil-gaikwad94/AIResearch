# 6 — Data

> Code does not follow the same scaling law as prose. *Scaling Laws for Code* (ACL 2026) finds code is a **more data-hungry regime**, requiring a substantially higher data-to-parameter ratio than natural language. This model is trained at 965 tokens per active parameter for that reason, not by accident.

---

## 6.1 The corpus

1,784 B tokens across five stages. The base mixture, and what each slice is for:

| Slice | Share | Tokens | Why |
|---|---|---|---|
| **Code** (permissive, deduped) | 52 % | 728 B | the substrate |
| **Repo-level context** (file + its imports + callers) | 14 % | 196 B | teaches cross-file structure |
| **Execution / commit replay** | 10 % | 140 B | the only signal that shows *effect*, not just text |
| **Natural language, docs, issues** | 12 % | 168 B | the model reads more prose than it writes |
| **Math** | 4 % | 56 B | reasoning transfer |
| **Structured** (config, SQL, schema, YAML) | 3 % | 42 B | high-frequency agent work, low coverage |
| **Multilingual** | 2 % | 28 B | the long tail of a global codebase |
| **Synthetic, execution-filtered** | 3 % | 42 B | fills rare-but-real patterns |

Then: 200 B for window extension, 100 B for memory training at 64 k, 80 B of pull-request mid-training, 4 B of SFT.

### Why execution replay is 10 %

Most pretraining sees code as *text*. A patch is not text; it is a **transition** — state before, action, state after, and whether it worked. The replay slice is the only part of the corpus that contains that structure:

```
read(auth.py) → patch(auth.py) → run(tests) → FAIL(3) → patch(auth.py) → run(tests) → PASS
```

Filtered to sequences that end in a pass, ≤ 8 patches, and with the failing output retained verbatim. The supervision is free (the commit history already encodes it) and it matches serving exactly.

### Why only 3 % synthetic

*Demystifying Synthetic Data in LLM Pre-training* (EMNLP 2025) finds the good ratio of rephrased synthetic data **converges to about 30 %** and depends on model size and data budget — but that is for prose rephrasing. For code, the filter is better than any ratio heuristic: a synthetic sample either executes and passes or it does not. This design keeps the *generated* volume high and admits only execution-verified samples, which lands naturally near 3 % of tokens at ~1 in 30 survival — and that is a property of the filter, not a tuned constant.

---

## 6.2 The highest-ROI decision in the project

*Clean Pull Request* (ICML 2026) mid-trains on **10 M real GitHub pull requests across 12 languages** and reports, on SWE-bench:

| Metric | Improvement |
|---|---|
| SWE-bench Lite | **+13.6 pp** |
| SWE-bench Verified | **+12.3 pp** |

— with an *agentless* protocol. No scaffolding, no search, no multi-turn harness. The capability was internalised into the weights.

That is a larger delta than most architecture changes deliver, for 80 B tokens of data that already exists, and it is why the mid-training stage exists ([07](07-training.md) §7.3). The lesson generalises: for repository-level editing, **the training signal matters more than the architecture**, which is a humbling conclusion for a document with eleven chapters of architecture in it.

It also constrains the eval: a model trained on PRs and then evaluated on SWE-bench is contaminated unless the eval repositories are post-cutoff. [09](09-evals-and-risks.md) requires that.

---

## 6.3 Repository-level data: what actually helped

A finding worth stating because it contradicts intuition. *On Pretraining for Project-Level Code Completion* (ICLR 2025) extends a 1.5 B model from 4 k to 16 k context using only **1 B tokens** of curated repository data, and reaches performance comparable to models trained on hundreds of billions. The ablation is the interesting part:

> Various repository-processing techniques yield similarly strong results, with **the primary gain coming from adapting to a new RoPE scaling parameter**.

So the "repo-level pretraining" benefit measured in that setup is mostly *positional adaptation*, not the repository structure itself. That has two consequences here:

1. The 200 B window-extension stage is not a nicety — it is where most of the long-context capability is actually acquired, and it is cheap (2,991 GPU-h).
2. Data-ordering tricks that place related files adjacently are worth less than they appear. Do them, but do not expect the gains attributed to them.

---

## 6.4 The corpus that does not exist yet

Everything above can be scraped or replayed. One slice cannot, and it is the one that trains the [04](04-memory.md) store controller.

**Staleness-injected multi-session replay.** Take a real trajectory of `k` sessions on one repository. Between session `k` and `k+1`, mutate one symbol that session `k` established a belief about — change a return type, flip a default, rename a parameter — in a way that **still compiles** and **changes behaviour**. The correct behaviour in session `k+1` is to notice the inconsistency and re-derive.

Construction rules:

| Rule | Reason |
|---|---|
| Compiles after mutation | otherwise the model learns to detect syntax errors, not staleness |
| Behaviour changes | the mutation must be detectable only by reading or running |
| Belief is load-bearing | if nothing depended on it, there is no signal |
| ≥ 2 sessions between write and use | otherwise it is just context |

Yield estimate: ~2 % of multi-session trajectories survive all four rules, which from a 1 M-trajectory pool gives ~20 k usable sequences — small, but this is a *behavioural* dataset, not a knowledge one, and it is replayed at high weight during SFT.

This is the single most novel data construction in the design, and also the most fragile. If the yield comes in below 5 k sequences, the controller cannot be trained and the store degrades to a static index ([09](09-evals-and-risks.md), R4).

---

## 6.5 Quality gates

Applied before a token is admitted:

| Gate | Threshold | Applied to |
|---|---|---|
| Deduplication | exact + MinHash 0.8 | all |
| License | permissive only | all |
| Compile / parse | must parse | code |
| Secret scan | zero tolerance | all |
| Test presence | ≥ 1 test file in repo | repo slice |
| Execution-verified | tests pass | synthetic, replay |
| Toxicity / PII | classifier | prose |

The secret scan deserves emphasis for a code model: repositories leak keys, and a model trained on them will emit them. Zero tolerance, scanned before tokenisation, with the scanner version pinned in the run manifest.

---

## 6.6 What the data says about the architecture

Three implications, stated as conclusions rather than hedges:

1. **Capacity is worth paying for.** The breadth of the corpus (12 languages, structured formats, configs) is far larger than 1.76 B active parameters can hold in superposition. That is the MoE argument, made from the data side rather than the scaling-law side.
2. **Long context matters less than long *sessions*.** The Clean-PR result comes from *sequences of edits*, not from 256 k of static text. This is why the memory design targets turn-to-turn persistence rather than a huge window.
3. **The corpus is the moat, and it is not the architecture.** A competitor with the same PR corpus and a plain 4 B dense model will be close. The architecture buys efficiency and the memory store buys reliability; neither buys knowledge that the data does not contain.

**Next:** [07 — Training](07-training.md).
