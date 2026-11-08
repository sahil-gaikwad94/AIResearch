# 9 — Evals and Ablations

> Nine ablations, 1,600 GPU-hours, and each one is written so that a negative result changes a decision rather than being explained away afterwards.

---

## 9.1 The suite

Tier 1 decides; tier 2 only blocks releases. Both carried from day 1, with two additions that only exist because of day-2 decisions.

| Eval | Measures | Weight |
|---|---|---|
| **Private held-out repos** | end-to-end task success, post-cutoff | 40 % |
| **Long-context recall at 256 k / 1 M** | multi-needle + identifier retrieval at 4 depths | 20 % |
| **Protocol compliance** | 200 event-log tasks; does the model patch stale content? | 15 % |
| **Cost per solved task** | tokens × (prefill + decode) ÷ successes | 10 % |
| **`R` policy** | mean `R` on easy vs hard slices | 10 % |
| **Reliability at 3 consecutive** | loop success, not pass@1 | 5 % |

### The 1 M recall probe

Day 1's probe was one needle at 64 k. That is too easy to be informative for a model with a 1 M window, and too weak to catch the specific failure mode sparse attention has. The day-2 probe, in order of severity:

1. **Multi-key retrieval:** 8 needles, 8 distinct keys, all must be returned exactly. A model with a positional prior in its indexer passes the single-needle version and fails this.
2. **Definition-to-use:** 4 classes defined at 5–15% depth, used at 80–95%, with decoy classes of similar names. This is the failure the anchor branch exists to prevent; if it fails *with* anchors on, the anchor budget is too small.
3. **Exact-copy:** reproduce a 40-token function verbatim from 400 k tokens back. Tests whether the compressed branch has displaced real tokens in the selection.
4. **Recency-conflict:** the same identifier redefined later; the model must use the later definition.

Scores are reported per depth (32 k, 128 k, 512 k, 1 M), because a model that is perfect at 128 k and 60% at 1 M is a different product from one that degrades gracefully.

---

## 9.2 B1 — Sparse vs full attention (the central bet)

| | |
|---|---|
| **Question** | Does the four-branch sparse design match full attention on repo-level tasks at 64 k, and beat it at 256 k+? |
| **Arms** | sparse (window+compressed+selected+anchors) · full attention · sparse without anchors |
| **Scale** | 3 × 20 B tokens @ 1 B, then agentic eval ≈ 400 GPU-h |
| **Measure** | task success on held-out repos at 32 k / 128 k / 512 k; recall probe |
| **Decision** | ship sparse if ≥ full − 1 point at 64 k **and** ≥ full + 2 points at 256 k. If full wins by >1 point at 64 k, sparse is only enabled above 64 k (a runtime switch, not a retrain) |

That last clause is the useful part: attention mode can be a serving configuration. If sparse only pays at long context, the model still trains with the mask and serves densely below 32 k.

## 9.3 B2 — Sparse exact vs linear hybrid

| | |
|---|---|
| **Question** | At equal KV bytes, does keeping every token addressable beat compressing it (3:1 KDA)? |
| **Arms** | sparse exact (11.5 KiB/token) · KDA:MLA 3:1 at equal cache · KDA 3:1 unconstrained |
| **Scale** | 3 × 20 B @ 1 B ≈ 340 GPU-h |
| **Measure** | repo task success; multi-key recall; long-context copy |
| **Decision** | switch if KDA wins by ≥3 points on repo tasks **and** does not lose ≥5 points on exact-copy |

This is the ablation that would most change the document, because Kimi Linear reports a hybrid beating full MLA outright. The pre-registered asymmetry — 3 points to switch, but 5 points of exact-copy loss to veto — reflects the argument in [02 §2.3](02-the-reading-problem.md): aggregate scores and exact reproduction are different capabilities, and code is punished hard for losing the second.

## 9.4 B3 — The recurrent block

| | |
|---|---|
| **Question** | Does depth recurrence buy reasoning at 4 B, and does fixed-memory KV cost anything? |
| **Arms** | (a) 20 unique + adaptation phase (chosen) · (b) 20 unique, looped from scratch · (c) no loop, 26 unique layers at equal training FLOPs · (d) fixed-memory vs recomputed KV |
| **Scale** | 4 × 20 B @ 1 B ≈ 450 GPU-h |
| **Measure** | bpb; reasoning slice (GPQA-style); repo tasks at equal FLOPs |
| **Decision** | keep the loop if (a) ≥ (c) at equal training FLOPs and (a) ≥ (b) − 1 point. Switch to recomputed KV if (d) shows >2 points |

Arm (c) is the honest control and the one most designs omit: **more real layers at equal compute**. If 26 unique layers beat 20+loop, the loop is fashion.

## 9.5 B4 — Anchors and hyperconnections

| | |
|---|---|
| **Question** | Does the deterministic anchor branch earn its kernel complexity, and do full hyperconnections beat the loop embedding? |
| **Arms** | anchors on/off · loop-embedding vs hyperconnections (4 streams) |
| **Scale** | 4 × 10 B @ 300 M ≈ 90 GPU-h |
| **Measure** | definition-to-use probe; repo tasks |
| **Decision** | keep anchors if ≥2 points on the probe; adopt hyperconnections if `φ` improves enough to show ≥1 point |

## 9.6 B5 — Indexer design

| | |
|---|---|
| **Question** | Is the two-stage hierarchical indexer necessary, and does learning beat position? |
| **Arms** | hierarchical (chosen) · flat token-level scan · coarse only (no fine re-score) · no indexer (window+compressed only) |
| **Scale** | 4 × 10 B @ 300 M ≈ 90 GPU-h, plus latency measurement at 1 M |
| **Measure** | recall probe; **indexer latency share** at 1 M; selection entropy |
| **Decision** | keep hierarchical if flat is >15% slower at 1 M with equal recall. Keep the indexer at all if "no indexer" loses ≥3 points |

## 9.7 B6 — Muon, and the reversal

| | |
|---|---|
| **Question** | Does Muon's pretraining advantage hold at 4 B / 1.2 T, and does the fine-tuning reversal reproduce? |
| **Arms** | Muon→AdamW (chosen) · AdamW throughout · Muon throughout |
| **Scale** | 3 × 100 B @ 1 B, then a matched SFT on each ≈ 500 GPU-h |
| **Measure** | loss at 100 B; post-SFT task success |
| **Decision** | keep the hybrid unless AdamW-throughout wins on *final* post-SFT quality by ≥1 point |

This is the only ablation whose result I would bet against the literature on, because the reversal paper's crossover depends on token/parameter ratio and nobody has published 309 tokens/param at 4 B.

## 9.8 B7 — Tokeniser (carried from day 1)

Unchanged in design: 131 k code-aware vs 128 k prose-trained vs 65 k. Decision at 10% token reduction and 2% bpb. **If it fails, it changes the vocabulary, not the architecture** — the embedding is 13% of parameters, so a 65 k vocab returns ~200 M parameters to the FFN.

## 9.9 B8 — Dense vs MoE (carried, still a gate)

12 B total / 1.9 B active vs 4.1 B dense at equal active FLOPs. Decision at 8% bpb, because the rollout and serving costs are structural and a smaller margin does not pay for them. Still expected to lose.

## 9.10 B9 — The protocol

The only ablation that is not a model:

| | |
|---|---|
| **Question** | Does the append-only protocol actually deliver the 13× reduction in a real harness? |
| **Arms** | naive re-render · append-only deltas · deltas + volatile-files-last ordering |
| **Measure** | measured cache hit rate; tokens re-prefilled per turn; task success |
| **Decision** | if hit rate <80% with deltas, the protocol is not implementable against real tools and the design's largest systems claim fails |

Cost: a week of harness engineering, no GPU time. It is listed last because it is cheap, and it should be run first for exactly that reason.

---

## 9.11 The decision all of this feeds

> **Is Kestrel-2 better than day 1 for the workload we actually have?**

Concretely: at equal *served* cost, does it solve more repo tasks? The comparison is not parameter counts or benchmark points; it is

```
tasks solved per (GPU-hour × dollar) at 256 k context
```

Day 1 cannot run 256 k without extension, which is itself the answer for long-context work. For short-context work, day 1 is cheaper per token (0.61× at 4 k) and may well win — which is why [10](10-decisions-and-risks.md) contains a section on when to ship day 1 instead.

**Next:** [10 — Decisions and risks](10-decisions-and-risks.md).
