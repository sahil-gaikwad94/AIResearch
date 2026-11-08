# 8 — Evals and Ablations

> Public coding benchmarks are contaminated by construction: every model has seen the solutions. The only honest measure is a repository the model cannot have memorised. Everything else is a regression test.

---

## 8.1 The contamination problem, briefly

SWE-bench Verified was assembled from real GitHub issues, and its solutions are in the training data of every model released since 2024. HumanEval is saturated and leaked. Even "dated" benchmarks leak as soon as a model is trained after their collection date.

The practical consequence: **public benchmarks measure whether the model has seen the answer, not whether it can produce one.** For a model trained in 2026 on 1.2 T tokens of GitHub, essentially every pre-2026 public benchmark is suspect.

So the suite is split into two tiers, and only one of them is allowed to decide anything.

---

## 8.2 Tier 1 — decision-making evals

| Eval | What it measures | Contamination status | Weight |
|---|---|---|---|
| **Private held-out repos** | End-to-end agentic task success on post-cutoff, internal codebases with real test suites | **none** — by construction | **50 %** |
| **Long-context identifier recall** | Given a 64 k-token repo, retrieve and use an exact identifier defined far from the call site | none (synthesised) | 15 % |
| **Multi-file refactor suite** | Coordinated changes across ≥3 files with a full test suite | low | 15 % |
| **Reliability-at-k** | Success on `k` consecutive attempts, not pass@1 | n/a | 10 % |
| **Token efficiency** | Tokens generated per correct patch | n/a | 5 % |
| **Cost-adjusted success** | Pass rate per dollar of inference | n/a | 5 % |

The private-repo tier is the only one that can promote or block a checkpoint. It is also the only one that requires real work to build: 200–400 tasks across 30–50 repositories, each with a verified test suite, a task description derived from a real commit, and a golden patch that is removed before evaluation.

**Why reliability-at-k and not pass@1.** Agents run in loops with verification. A model that succeeds 50% of the time *independently* is different from one that succeeds 50% of the time and then repeats its mistake — the second is unusable in a loop even though both score 0.5. Measure the probability of three consecutive successes; it is the number that predicts whether an agent session finishes.

**Long-context recall is a first-class eval, not a diagnostic.** It directly measures constraint C1, which the whole architecture is built around. A synthetic version: place a function definition at token 2,000 and a call site at token 60,000, require the exact name in the patch. A model with approximate recall fails this and passes everything else — which is exactly why it must be measured.

---

## 8.3 Tier 2 — regression canaries

Run every 20 B tokens during training; block release on regressions but never optimise against them.

| Eval | Threshold | Why |
|---|---|---|
| MMLU-Pro | within 2 points of the 50 B-token checkpoint | general knowledge is a leading indicator of over-specialisation |
| GPQA-Diamond | within 3 points | reasoning canary |
| Instruction following (IFEval-style) | within 3 points | SFT/RL erodes this |
| Refusal calibration (safety slice) | no more than 20% false-refusal increase | code RL degrades refusals because refusing never passes a test |
| Public code benchmarks (LiveCodeBench, HumanEval+) | report, do not gate | contaminated; useful only as a sanity check for catastrophic regression |

**Over-specialisation is a real risk here.** 42% code in pretraining plus code-only RL will erode general capability, and the erosion is invisible on code evals. The canaries exist to catch it before the model becomes unable to read a bug report.

---

## 8.4 The eight ablations

All run at 300 M–1 B parameters with μP transfer, 8–20 B tokens per arm. Cost reference: **one arm at 1 B for 20 B tokens ≈ 120 GPU-hours; at 300 M for 10 B ≈ 15**. Total ≈ 1,100 GPU-hours, budgeted at 1,160 in [05 §5.3](05-training.md) to allow reruns — 2.5% of the project budget for the decisions that govern the other 97%.

### A1 — Tokeniser (the cheapest possible win)

| | |
|---|---|
| **Question** | Does the code-aware 131 k tokeniser beat a prose-trained 128 k one by enough to justify its parameters? |
| **Arms** | code-aware 131 k · standard 128 k · code-aware 65 k |
| **Scale** | 3 × 20 B tokens @ 300 M ≈ 90 GPU-h |
| **Measure** | tokens/file on held-out repos, and **bits per byte** at equal compute |
| **Decision** | ship if ≥10% token reduction *and* ≥2% bpb improvement. If gain <10%: 65 k vocabulary, parameters to FFN |
| **Cost of being wrong** | ~200 M parameters misallocated (5% of the model) |

### A2 — Full attention vs hybrid (the central bet)

| | |
|---|---|
| **Question** | Does a 3:1 hybrid match full MLA at equal KV budget on *repo-level* tasks? |
| **Arms** | full MLA · 3:1 hybrid (Gated DeltaNet), KV-normalised · 3:1 hybrid, unconstrained |
| **Scale** | 3 × 20 B tokens @ 1 B ≈ 360 GPU-h + 150 for agentic eval |
| **Measure** | task success on held-out repos; identifier-recall probe; training FLOPs |
| **Decision** | **switch to hybrid if it wins by >3 points** at equal KV. Keep full attention if within 3 |
| **Why so expensive** | this is the one place the design contradicts the field; it earns a real budget |

### A3 — Depth vs width

| | |
|---|---|
| **Question** | 40 × 3072 or 28 × 4096 at equal parameters? |
| **Arms** | 2 × 10 B tokens @ ~400 M ≈ 40 GPU-h |
| **Measure** | bpb, plus a reasoning slice if the gap exceeds 5% |
| **Decision** | depth wins if ≥1.5 points on task success; otherwise width wins on cache (30% cheaper) and decode speed |

### A4 — MTP

| | |
|---|---|
| **Question** | Do 2 MTP heads improve the trunk enough to keep, beyond the decode speedup? |
| **Arms** | 0 heads · 2 heads · 2 heads with loss weight 0.5 |
| **Scale** | 3 × 10 B @ 300 M ≈ 45 GPU-h |
| **Measure** | bpb at equal tokens; measured draft acceptance on code |
| **Decision** | keep if bpb improves ≥0.5% **or** decode speedup ≥1.6×. The second alone justifies it |

### A5 — Loss masking on observations

| | |
|---|---|
| **Question** | Should the model be trained to predict tool output and file contents, or only its own actions? |
| **Arms** | full loss · mask observations · mask observations + auxiliary "summarise this file" objective |
| **Scale** | 3 × 8 B @ 300 M ≈ 35 GPU-h |
| **Measure** | task success, and *reading* accuracy on the recall probe |
| **Hypothesis** | training on observations improves reading (the 89–99% of compute) even though it is not what gets generated |
| **Decision** | keep full loss unless masking wins by ≥2 points |

This is the most under-tested question in the design and the one most likely to produce a surprise. Most code-model training masks the observations; the thesis of this document says reading is the job, which predicts the opposite.

### A6 — Attention sinks

| | |
|---|---|
| **Question** | Do 4 sinks per head help at 32 k, or are they only useful for eviction? |
| **Arms** | sinks · no sinks · sinks + synthetic eviction during training |
| **Scale** | 3 × 8 B @ 300 M ≈ 35 GPU-h |
| **Decision** | keep if neutral-to-positive at 32 k (they are nearly free) or positive with eviction |

### A7 — Position scheme

| | |
|---|---|
| **Question** | `theta` = 100 k / 500 k / 1 M at 16 k context, and does removing position from 4 mid-stack layers help? |
| **Arms** | 4 × 8 B @ 300 M ≈ 45 GPU-h |
| **Measure** | bpb at 16 k; recall probe at 32 k and 64 k |
| **Decision** | pick the `theta` with the best recall at 64 k; adopt NoPE layers only if ≥2 points better |
| **Link** | this is the cheap, small-scale version of the companion NoPE experiment |

### A8 — Dense vs MoE (gate 0, decides v2)

| | |
|---|---|
| **Question** | Does a 12 B-total / 1.9 B-active MoE beat 4 B dense at equal *active* FLOPs, once RL rollout cost is counted? |
| **Arms** | dense 4 B · MoE 12 B/1.9 B, 32 experts · MoE with 2 shared experts |
| **Scale** | 3 × 10 B @ 1.9 B active ≈ 270 GPU-h (plus rollout timing) |
| **Measure** | bpb at equal active FLOPs; measured tokens/s/GPU during generation; routing entropy |
| **Decision** | MoE only if **>8% bpb gain** — the threshold is high because the rollout and serving costs are structural |

---

## 8.5 The measurement that costs one day and could invalidate everything

Before any of the above, instrument a real agent harness and answer three questions:

1. **What fraction of prefill tokens are cache hits?** (Target: >70%. Below that, the thesis in [01](01-thesis.md) is wrong.)
2. **What is the distribution of unique tokens per session?** (Target: 150–400 k. Higher means the cache is doing more work than assumed.)
3. **How often is the context compacted?** Every compaction is a full cache flush; more than once per 10 turns changes the serving economics.

A day of instrumentation, on logs most harnesses already produce. It is the highest-value measurement in the plan because it tests the premise rather than the implementation, and because if the answer is bad, the correct response is a different architecture — not a better kernel.

---

## 8.6 What we will not measure, and why

- **Public leaderboards as a target.** They are contaminated and they reward fitting the benchmark's harness. Report them; never optimise them.
- **Perplexity on a general corpus.** It moves for reasons unrelated to coding ability and invites tuning the wrong thing.
- **"AGI-style" composite scores.** Unfalsifiable and unstable across harnesses.
- **Human preference on code (Chatbot-Arena style).** Coding preference correlates with confidence and formatting, not correctness. Use tests.

---

## 8.7 The decision that all of this feeds

One question, at the end of the ablation programme:

> **Does this architecture beat fine-tuning an existing open 4 B model by enough to justify building it?**

If the answer is no — and [09](09-decisions-and-risks.md) says plainly that it often will be — the correct outcome of the project is to ship the tokeniser, the harness constraints, and the RL environments, and put them on top of somebody else's weights. That is a success, not a failure, and the ablations are what tell you which parts to keep.

**Next:** [9 — Decisions and risks](09-decisions-and-risks.md).
