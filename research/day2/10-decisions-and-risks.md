# 10 — Decisions, Risks, and When Day 1 Was Better

> The design log, the risks with early-warning signals, and an honest answer to the question every v2 document avoids: under what conditions is the previous version the right one to ship?

---

## 10.1 Decision log

| # | Decision | Alternative rejected | Why | Reverse if |
|---|---|---|---|---|
| **D1** | Sparse exact attention (window + compressed + selected + anchors) | full attention (day 1); linear hybrid | keeps every token addressable, reads 2.7% of the context at 256 k | B1: full wins >1 pt at 64 k → serve dense below 64 k |
| **D2** | Deterministic structural anchors, 256 tokens | learned anchors; no anchors | parser cannot miss `class User`; ISSTA shows basic topology carries the benefit | B4: <2 pts on the probe |
| **D3** | Recurrent block of 16 layers, `R = 1…8` | 40 unique layers; 26 unique at equal compute | depth per parameter, and cache-free in `R` | B3: 26 unique layers win at equal FLOPs |
| **D4** | **Fixed-memory** recurrence (KV projected once) | recompute K/V each iteration | makes the shared prefix cache independent of `R`; a reader refines the question, not the record | B3(d): >2 pts for recomputed KV |
| **D5** | MLA 512 + 64, fused `W_DKV` | GQA-4; MHA | fewer parameters *and* less cache than GQA-4 (unchanged from day 1) | — |
| **D6** | 20 unique layers, `d = 4096` | 40 × 3072 | fewer layers amortise sparse attention's per-layer fixed read cost | B3 |
| **D7** | Hierarchical indexer, FP8, 128-dim | flat token scan | HISA: the flat indexer becomes the bottleneck at 1 M | B5 |
| **D8** | Append-only delta protocol | re-render files; compact | prefix hashes are cumulative, so any rewrite invalidates the suffix | B9: hit rate <80% in practice |
| **D9** | Cache published per `(repo, commit)` | per-session cache | 64 sessions/GPU instead of 3 at 400 k | — |
| **D10** | Muon for pretraining, AdamW from SFT | AdamW throughout; Muon throughout | ~2× pretraining; the advantage **reverses** on fine-tuning | B6 |
| **D11** | Coverage-delta in the reward | pass/fail only | penalises hollow success without specifying the fix | if the added-test mutation filter is too noisy |
| **D12** | `R`-budget reward with `λ_R = 0.01` | fixed `R`; no penalty | teaches when to think; penalty calibrated to 10–20% of success | if mean `R` collapses on hard slices |
| **D13** | 1,275 B tokens, 309/param | Chinchilla 80 B | inference-dominated economics, unchanged from day 1 | budget halves |

### On D4, the decision that cost the most thought

Recomputing K/V per iteration is what the literature does, and it is more expressive. Kestrel-2 refuses it for two reasons that had nothing to do with modelling:

1. A prefix cache shared across sessions must not depend on `R`. If it did, a repository would need up to eight cached copies and the entire economy in [08 §8.2](08-serving.md) evaporates.
2. The refinement loop matches the workload: attention here is retrieval, and iterating improves the query, not the record.

That is a *systems constraint selecting an architecture*, which is either the best kind of decision or a rationalisation — and the only way to tell is B3(d), which is why it is pre-registered with a 2-point switch rule rather than defended in prose.

---

## 10.2 Risk register

| # | Risk | P | Impact | Mitigation | Early warning |
|---|---|---|---|---|---|
| **R1** | Sparse attention underperforms dense at 64 k | 35 % | High | serve dense below 32 k (runtime switch) | B1 at 1 B |
| **R2** | Indexer collapse to a positional prior | 40 % | High | distillation loss; entropy monitoring | selection entropy <25% of baseline |
| **R3** | The loop does not help at 4 B | 45 % | High | keep 26-unique-layer fallback fully specified | bpb flat across `R` |
| **R4** | Fixed-memory recurrence loses expressivity | 35 % | Medium | B3(d); per-`R` cache variants as fallback | recall gap vs recomputed KV |
| **R5** | Real harnesses cannot be made append-only | 30 % | **Very high** (kills D8/D9) | wrap tools; diff their output | hit rate <80% in a week of measurement |
| **R6** | Sparse attention is 1.6× more expensive at short context | **certain** | Medium | train at `R=1`; serve dense below 32 k | — (already priced) |
| **R7** | Muon reversal arrives earlier than expected | 40 % | Medium | AdamW from SFT regardless | B6 at 100 B |
| **R8** | Coverage reward is gamed by assertion-free tests | 45 % | Medium | mutation-score added tests | coverage up, kill rate down |
| **R9** | 1 M context quality collapses despite training | 40 % | Medium | staged training to 256 k; probe per depth | >10 pt drop from 512 k to 1 M |
| **R10** | Kernel complexity delays the whole project | 50 % | Medium | mask-based reference first; fused kernel later | >6 weeks on the sparse kernel |

Two deserve more than a row.

**R5 is the one that decides whether day 2 exists.** The entire 13× prefill reduction depends on tools that append rather than rewrite. If a formatter, a package manager or an editor plugin can only express its output as a full-file rewrite, the protocol needs a diffing shim for each, and some will be wrong. A week of harness measurement — no GPU — tests this, and it should happen before the first training run, not after.

**R3 is the one I would bet on least.** Every piece of looped-depth evidence comes from models either much larger, trained much longer (7.7 T tokens), or both. At 4 B and 1.3 T with an adaptation phase rather than from-scratch training, the effect could easily be within noise. The design survives it — the fallback is 26 unique layers, and nothing else changes — but the document's headline claim about depth would be wrong.

---

## 10.3 When day 1 is the better design

This is the section most v2 documents omit, and it matters more than the ablations.

| If… | Ship day 1 | Because |
|---|---|---|
| Sessions stay under 32 k tokens | **yes** | day 2 costs 1.6× more per token there and buys nothing |
| You need on-device / laptop deployment | **yes** | 20 KiB/token × 64 k is 1.25 GiB; day 2's advantage is a server advantage |
| Your harness cannot be made append-only | **yes** | without D8, the largest gain disappears and day 1's simpler kernel wins |
| You will not build the sparse kernels | **yes** | three attention paths is real engineering; a mask-based version is 4× slower |
| You need one model for chat *and* coding | **yes** | a 1 M repo reader is a poor chat model |
| Your repositories are small (under 100 k tokens) | **probably** | the crossover is around 64 k ([02 §2.5](02-the-reading-problem.md)) |

The honest summary: **Kestrel-2 is a design for repositories, and day 1 is a design for files.** If the workload is a single file that fits in 8 k tokens, day 2 is strictly worse and should not be built.

---

## 10.4 What would make me abandon Kestrel-2

In order of likelihood:

1. **B9 fails** — real harnesses cannot be append-only, hit rate stays below 80%, and the 13× prefill reduction is unavailable. Then day 2 is a more complex model with a 1.6× short-context tax and no compensating systems win.
2. **B3 fails** — the loop does not beat 26 unique layers at equal FLOPs. The recurrent block is removed, and the design becomes "day 1 with sparse attention", which is a fine model and a much smaller claim.
3. **B1 fails at 64 k** and sparse only wins above 128 k. Not fatal: attention mode becomes a serving switch. But it means training and serving diverge, and that is a maintenance cost to weigh.
4. **B2 shows KDA winning by >3 points without losing exact-copy.** Then the third option was not better than the second, and the document's central argument about addressability is wrong.
5. **The indexer never learns content-based selection** (R2), in which case the anchor branch is doing all the work and the learned branch should be deleted.

Three of the five are testable at 1 B for under 500 GPU-hours. A design that can only be validated by spending its whole budget is not a design.

---

## 10.5 Summary judgement

**What this gets right, if the thesis holds:**

- **It reads 96× less at 1 M** while keeping every token addressable. The decode-flatness table in [08 §8.1](08-serving.md) is the strongest single result in either document.
- **It puts the cache where the value is** — one shared prefix per repository, 64 sessions per GPU, and a protocol that never invalidates it.
- **It prices its own speculation.** The loop is bought at `φ = 0.46`, paid for only in an adaptation phase, and has a fully specified fallback.

**What it gets wrong, with high probability:**

- **It is more expensive exactly where most tokens are.** 63% of pretraining runs at 4 k, where sparse attention is a 1.6× tax and the anchors are noise. Day 1 is the better model below 32 k, and that is most of the world's coding.
- **The loop will underperform its literature.** Every result it rests on comes from a larger budget, and an adaptation phase is not the same as training looped from scratch.
- **Something in the protocol will break against a real tool.** `git`, `black`, `npm` and IDE plugins all rewrite files; the shims will be the least glamorous and most fragile code in the project.

**The one-line version:** day 1 made reading cheap in FLOPs, day 2 makes it cheap in bytes and in volume; the second is the right optimisation for repositories, and the first is the right one for files.

**Back to:** [README](README.md) · [01 — What day 1 missed](01-what-day1-missed.md) · [02 — The reading problem](02-the-reading-problem.md)
