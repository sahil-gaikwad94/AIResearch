# 5 — Training

> 1.2 T tokens, four context stages, ~45,700 GPU-hours. The budget's shape is the interesting part: the long-context stage costs 16% of the 4 k stage's compute for 4% of its tokens, and the main run is *not* the schedule bottleneck — the ablations are.

---

## 5.1 Data mixture

| Share | Tokens | Content | Notes |
|---|---|---|---|
| 42 % | 504 B | Code, 40+ languages, permissive licences only | MinHash-dedup at file and 50-line-window level; language-stratified sampling so the model does not over-index on JS |
| 18 % | 216 B | Code-adjacent: docs, READMEs, API references, changelogs, issue threads, test names | The single highest-value non-code category for agentic work |
| 15 % | 180 B | Natural language, quality-filtered | Instruction-following and reasoning transfer |
| 10 % | 120 B | Structured: JSON, YAML, TOML, CSV, SQL, logs, configs | Tool output distribution |
| 8 % | 96 B | Mathematics + formal | Reasoning substrate |
| 4 % | 48 B | Multilingual | Non-English code comments and docs |
| 3 % | 36 B | Synthetic, verifier-filtered | See §5.2 — only what passes a test |

**Why 42% code and not 80%.** A model that has only seen code cannot follow an instruction, read a bug report, or explain a design. The evidence from every strong coding model is that a substantial general-data share improves code performance, because most coding work is reading and reasoning about intent in natural language.

**Licence policy.** Permissive licences only (MIT, Apache-2.0, BSD, ISC, MPL). Copyleft and unlicensed code is excluded from pretraining even where technically available. This costs maybe 8% of the corpus and removes an entire category of legal risk from the product. The filter is applied at the repository level, not the file level, and the excluded list is versioned with the corpus.

---

## 5.2 Synthetic data: only what a verifier passed

Three sources, all gated:

1. **Test-driven synthesis.** Take a function with tests, delete the body, ask a strong model to write it, keep only what passes. Yields (problem, solution) pairs with an automatic correctness label.
2. **Mutation-derived bugs.** Inject a realistic bug into working code; the diff that fixes it is the target. This directly trains debugging, which is the actual agentic skill.
3. **Documentation-derived Q&A.** Generate questions answerable *only* from a doc page, with the page as context — trains retrieval-grounded answering rather than recall.

Rules: every synthetic sample must have a verifier outcome; rejected samples are kept as negative examples for preference data, not discarded; synthetic share is capped at 3% of tokens in pretraining and up to 40% in SFT, where the filter is much tighter.

**The failure mode to avoid:** unverified synthetic code at scale teaches the model to produce plausible-looking code that does not run. It is the most expensive mistake available in this project, because it is invisible in loss curves and only shows up in evals.

---

## 5.3 Budget, with attention accounted for

This is where naive budgeting goes wrong. `6ND` assumes attention is free; at long context it is not:

| Context | Attention FLOPs/token | Share of `6N` |
|---|---|---|
| 4,096 | 6.0 × 10⁹ | 24.7 % |
| 16,384 | 2.4 × 10¹⁰ | 98.6 % |
| 65,536 | 9.7 × 10¹⁰ | **394 %** |
| 262,144 | 3.9 × 10¹¹ | 1,578 % |

So the budget must be computed per stage:

| Stage | Tokens | Context | GPU-hours |
|---|---|---|---|
| Pretrain, phase 1 | 1,000 B | 4 k | 24,238 |
| Pretrain, phase 2 | 160 B | 16 k | 6,179 |
| Pretrain, phase 3 | 40 B | 64 k | 3,846 |
| Mid-train (repo-packed) | 50 B | 32 k | 2,890 |
| SFT | 4 B | 32 k | 231 |
| RL (generation-bound) | 24 B generated | 32 k | 2,222 |
| Ablations (24 × 8 B @ 1 B proxy) | 192 B | 8 k | 1,160 |
| Restarts, loss spikes (12 %) | — | — | 4,892 |
| **Total** | | | **45,658** |

At $1.5 / $2.2 / $3.0 per GPU-hour: **$68k / $100k / $137k**.

**Two observations that change how this project should be run.**

*The 64 k stage is 16% of the 4 k stage's compute for 4% of its tokens.* Long context is not expensive because of the tokens; it is expensive because attention is quadratic and you are paying it on every token in the sequence. This is exactly the argument for hybrid attention — and the reason the design refuses it is that this cost is a one-time training cost, not an inference cost ([01 §1.2](01-thesis.md)).

*The main run is not the bottleneck.* Pretraining is ~37,000 GPU-hours: 12 days on 128 H100s, 3 days on 512. The schedule is dominated by **iteration** — ablations, failed runs, data pipeline debugging. That is why 24 ablations at 1 B parameters are budgeted explicitly, and why the proxy-model strategy below matters more than cluster size.

---

## 5.4 Context staging

| Phase | Context | Tokens | Purpose |
|---|---|---|---|
| 1 | 4,096 | 1,000 B | 83% of tokens at the cheapest context; learns syntax, semantics, short-range code |
| 2 | 16,384 | 160 B | Multi-function reasoning; attention cost doubles |
| 3 | 65,536 | 40 B | Repo-scale; the expensive phase, deliberately short |
| Mid-train | 32,768 | 50 B | **Repo packing** (§5.5) — whole repositories concatenated |

**Do not train the whole run at 64 k.** The FLOPs are 4.9× per token, and the model does not need repo-scale context until it has learned to program at all. Staging is the single largest cost lever in the budget: doing all 1.2 T at 64 k would cost ~118,000 GPU-hours instead of ~34,000.

**RoPE handling across stages.** Train at 4 k with `theta = 500000` from the start, not extended later. Extending `theta` after the fact (the YaRN/ABF approach) works but costs a fine-tuning phase and leaves a measurable quality gap at short range. Setting it high initially costs a small amount of short-context resolution, which is the right trade for a model whose target is repo-scale work.

---

## 5.5 Repo packing

Mid-training on 32 k-token documents built from *whole repositories* is where the model learns the thing that matters most: that files relate to each other.

Format, with reserved tokens from [04](04-tokenizer.md):

```
<|file|>src/api/routes.py
<|content|>
from .models import User
...
<|file|>src/models.py
<|content|>
class User(Base):
...
```

Rules:

- **Dependency order.** Files are concatenated in import order (topological sort of the import graph), so a definition usually precedes its use. This teaches the model something true about code.
- **Path tokens matter.** Keeping the path in-context lets the model learn that `tests/test_routes.py` is about `src/api/routes.py`.
- **Cap at 32 k with whole-file boundaries.** Never split a file across a boundary; drop the file instead. A model trained on truncated files learns to hallucinate closing braces.
- **Include the test files.** Half the value of repo packing is seeing the test-file relationship.

**Risk, stated:** this teaches the model to emit `<|file|>` markers when it should emit a diff. Mitigation: keep the packing format *only* in the input position (never as a generation target), and use a distinct `<|patch|>` format for outputs during SFT.

---

## 5.6 Optimiser and schedule

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimiser | AdamW (β₁ 0.9, β₂ 0.95) | β₂ 0.95 rather than 0.999 is standard for LLMs; long memory of squared gradients hurts |
| Peak LR | 3.0 × 10⁻⁴ | Standard for 4 B at a 4 M-token batch |
| Schedule | Warmup 2,000 steps → cosine → 10% of peak | |
| Global batch | 4 M tokens, ramped from 0.5 M over the first 5 B tokens | Small batches early are more sample-efficient per FLOP; large batches later are more throughput-efficient |
| Weight decay | 0.1, applied to matrices only | Never to norms, biases or embeddings |
| Grad clip | 1.0 | |
| Precision | bf16 compute, fp32 master weights, fp32 Adam state | |
| Sequence length | 4 k → 16 k → 64 k as staged | |

**μP, and an honest caveat.** The most valuable engineering decision in this section is not a hyperparameter, it is **μP-style parameterisation** so that learning rates tuned on a 100 M proxy transfer to 4 B without a fresh sweep. A 4 B LR sweep costs ~10,000 GPU-hours; a proxy sweep costs ~200.

The caveat: μP's guarantees are derived for standard residual/attention blocks, and MLA's low-rank query path is not standard. The practical compromise is μP-inspired init (`std ∝ 1/√width`) plus a *coordinated check*: tune at 100 M, then verify at 1 B that the same LR is within 1.5× of optimal, then use it at 4 B with a small sweep (±30%) early in phase 1 where checkpoints are cheap. Anyone claiming μP gives exact transfer through MLA has not tested it.

---

## 5.7 Precision: bf16 first, FP8 behind a gate

Phase 1 runs in bf16. FP8 matmuls are introduced only after 200 B tokens, and only if a 1 B controlled comparison matches bf16 loss within 0.3% at equal tokens.

FP8 is worth ~1.6–1.9× throughput and is how the budget above stays affordable at the low end. It is also the most common way to silently damage a model: per-tensor scaling fails on outlier channels, and the damage appears as a slightly worse model, never as an error. The gate exists because "slightly worse model" is indistinguishable from "slightly worse hyperparameters" once you are 300 B tokens in.

---

## 5.8 Stability playbook

Loss spikes are not an edge case at this scale; they are the normal operating condition. The playbook, in the order the responses should be tried:

1. **Detect** on a smoothed loss derivative, not raw loss. Threshold: 3 consecutive steps above +0.2 nats from the running median.
2. **Skip the batch**, roll back to the last checkpoint, and resume with the data order advanced past the offending shard. Most spikes are data.
3. **Lower LR by 20%** for 500 steps, then ramp back.
4. **Investigate** only if 1–3 recur: check grad-norm histogram by layer, attention entropy by head, and the specific shard.

Structural defences already in the design: QK-norm (bounds pre-softmax logits), zero-init residual projections (bounds early residual growth), no biases (removes one gradient pathology), fp32 master weights (removes a whole class of underflow).

**Monitoring that actually catches problems early:** attention entropy per head (a head that collapses to uniform or to a single token is broken), embedding norm drift, and the ratio of gradient norm between the first and last block.

---

## 5.9 Infrastructure

- **128–512 H100/H200-class GPUs**, single cluster, high-bandwidth interconnect. At 4 B, tensor parallelism is unnecessary: FSDP/ZeRO-3 with full sharding fits comfortably.
- **Sequence parallelism for the 64 k stage.** Activations at 64 k × 4 M-token batches are the memory constraint, not the weights. Activation checkpointing on every block for phase 3.
- **Data pipeline as a versioned artefact.** Tokenised shards, dedup indices and the sampling weights are hashed into the run manifest; a mismatch aborts rather than warns. The most common cause of an irreproducible run is a silently changed data pipeline.
- **Checkpoint every 30 minutes**, keep the last 5 plus every 10th. At 1.2 T tokens a lost week is 3% of the budget.

---

## 5.10 What would make me restart

- Bits-per-byte not tracking the 100 M proxy's predicted curve within 3% by 100 B tokens → the μP transfer failed, and the LR is wrong.
- Attention entropy collapsing in more than 10% of heads by 50 B tokens → the architecture has a pathology, not a tuning problem.
- Data-mixture ablation showing code share should be ≥60% → restart the mixture before 200 B tokens, not after.
- Loss spikes recurring more than weekly after the playbook → investigate the numerics (FP8, if enabled) before spending more compute.

**Next:** [6 — Post-training](06-post-training.md).
