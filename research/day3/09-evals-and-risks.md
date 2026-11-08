# 9 — Evaluations, Ablations, Risks

> Six ablations, 2,500 GPU-hours, and the honest conditions under which this design should not be built at all.

---

## 9.1 The suite

| Eval | Measures | Weight |
|---|---|---|
| **Held-out repositories, post-cutoff** | end-to-end resolution | 35 % |
| **Long-session tasks** (20+ turns, 2+ sessions) | memory and staleness handling | 20 % |
| **Staleness probe** | does the model use superseded facts? | 15 % |
| **Cost per solved task** | tokens + test-minutes ÷ successes | 15 % |
| **No-verifier slice** | quality where nothing can be checked | 10 % |
| **Reliability at 3 consecutive** | not pass@1 | 5 % |

The **staleness probe** is new and is the one that decides whether [01](01-the-validity-problem.md) was right. Construction: 200 tasks where session 1 establishes a fact, the repository mutates, and session 2 requires the *new* behaviour. Score = fraction of runs that read the stale fact versus re-derived. A model with no memory scores well here trivially (it re-derives everything); a model with bad memory scores worst. So the metric is reported **alongside** resolution, never alone — a lesson from the memory literature, where "good at not being wrong" and "useful" are different things.

The **no-verifier slice** exists because [05](05-reasoning.md) §5.6 predicts this model will lose there. Measuring the loss is the point.

---

## 9.2 B1 — The activation ratio

| | |
|---|---|
| Question | Is 1:32 survivable at 1.76 B active? |
| Arms | 1:32 (96 experts) · 1:8 (24 experts, same active) · dense FFN at same active FLOPs |
| Cost | 3 × 20 B tokens @ 700 M ≈ 260 GPU-h |
| Decision | keep 1:32 if within **1.5 pp** of 1:8 on repo tasks; otherwise move to 1:16 |

The dense arm is the control nobody runs. If a dense FFN with the same active FLOPs matches 96 experts, the entire capacity argument is wrong and the model should be dense and simpler.

## 9.3 B2 — Memory vs long attention

| | |
|---|---|
| Question | Does a 4.2 MB state replace a long KV cache? |
| Arms | window 4 k + memory · window 4 k only · window 64 k (full attention) · memory rank 128 / 256 / 512 |
| Cost | 4 × 15 B @ 700 M ≈ 300 GPU-h |
| Decision | ship memory if ≥ **2 pp** over window-only at 64 k, and within 3 pp of full 64 k attention |

The rank sweep matters because rank is a tuned constant, not a derivation ([04](04-memory.md) §4.2). If rank 512 is much better, the 4.2 MB claim becomes 8.4 MB — still 280× smaller than KV, but the number in every document changes.

## 9.4 B3 — The store

| | |
|---|---|
| Question | Does a validated, versioned store beat a text file? |
| Arms | validated store · plain `MEMORY.md` · no memory |
| Cost | **no GPU** — a week of harness engineering |
| Decision | ship if ≥ **2 pp** over `MEMORY.md`, or ≥ 25 % fewer turns |

This is the cheapest experiment in the programme and the one that would most embarrass the design if it fails. It should run first, before any training.

## 9.5 B4 — Shared expert

| | |
|---|---|
| Question | Was the 2026 study right that shared experts do not matter? |
| Arms | none · 1 × 1,536 shared · 2 × 1,024 shared |
| Cost | 3 × 15 B @ 700 M ≈ 200 GPU-h |
| Decision | adopt at **≥ 1.5 pp** at equal active FLOPs |

## 9.6 B5 — Reasoning allocation

| | |
|---|---|
| Question | Is a learned difficulty policy worth 4×? |
| Arms | learned allocation · fixed best-of-4 · fixed best-of-N=1 with 4 turns · pairwise head on/off |
| Cost | RL harness + 300 GPU-h |
| Decision | ship learned if it solves ≥ **1.4×** the tasks per verifier-minute |

The ICLR 2025 result predicts ~4× over *fixed best-of-N*. This design would be happy with 1.4× against a well-tuned fixed policy, because the source result compares against a naive one.

## 9.7 B6 — PR mid-training, with a contamination control

| | |
|---|---|
| Question | Does +13.6 pp reproduce, and is any of it leakage? |
| Arms | with PR mid-training · without · with PR data from pre-2024 only |
| Cost | 3 × 40 B @ 2 B ≈ 400 GPU-h |
| Decision | keep if the pre-2024 arm retains ≥ 60 % of the gain |

The third arm exists because SWE-bench-derived training data is the most likely source of a result that evaporates on a private benchmark.

---

## 9.8 Risk register

| # | Risk | P | Impact | Mitigation | Early signal |
|---|---|---|---|---|---|
| **R1** | Memory cannot replace long attention at 64 k | 40 % | High | grow window to 8 k (B2); cost +4 % | B2 at 700 M |
| **R2** | Staleness corpus yields < 5 k sequences | **50 %** | High (kills L3 controller) | synthesise harder; bootstrap from injected prompts | corpus build, week 1 |
| **R3** | 1.76 B active is a reasoning ceiling users notice | **60 %** | High | search + verification; measure no-verifier slice | B5 |
| **R4** | MoE serving tax makes it slower than a dense 4 B in practice | 45 % | High | batch ≥ 16; hot-set residency; never batch 1 with EP | profiling before training |
| **R5** | 1:32 capacity gain does not materialise at this scale | 30 % | High | B1; fall back to 1:16 | B1 |
| **R6** | PR mid-training is contaminated | 30 % | Medium | B6 arm 3; post-cutoff evals | B6 |
| **R7** | Memory state saturates by 100 k tokens | 40 % | Medium | rank 512; decay; normalise | `‖B‖` monitoring |
| **R8** | Router collapse wastes 70 % of capacity | 35 % | Medium | bias balancing; utilisation gini gate | gini > 0.5 |

Three deserve more than a table row.

**R2 is the most likely single point of failure.** Everything about L3 — the store, the controller, the staleness probe — depends on a corpus that does not exist and must be constructed. If multi-session trajectories with load-bearing beliefs are rarer than estimated, the controller cannot be trained, and the design degrades to "a model with a text file", which §9.4 may well show is fine. That is an unpleasant possibility to sit with, so it is scheduled first.

**R3 is the one I would bet on.** A 1.76 B-active model with excellent retrieval and verification will still be worse than a 4 B dense model at holding five interacting constraints in mind. The bet is that coding work is more often *retrieval-and-checking* than *multi-constraint reasoning*. If that is wrong for the target workload, the whole thesis is wrong for that workload.

**R4 is the one that decides whether anyone can afford to run it.** A model that wins on paper and loses 2× on latency at batch 1 is a research result, not a product. The mitigation — batch aggressively, keep hot experts resident — is standard, but it means single-user latency is the worst case, and single users are how models get evaluated by hand.

---

## 9.9 When this design is wrong

| If… | Then build | Because |
|---|---|---|
| There is no test suite or verifier | a dense 4 B with long context | search buys nothing without a signal ([05](05-reasoning.md) §5.6) |
| Sessions are short (single turn, one file) | a small dense model | memory and MoE are dead weight |
| You cannot curate PR data | don't expect repo-level editing | +13.6 pp came from that data, not the architecture ([06](06-data.md) §6.2) |
| You need on-device | 4-bit dense, 4 B | 10.2 GB of weights plus expert streaming is not a phone |
| You cannot build the store | a model with a long window | text memory without validity loses (§1.1) |

The honest summary: **this is a design for repository work with an executable oracle.** Remove the oracle and it is a mediocre 1.76 B model; remove the repository and it is over-engineered.

---

## 9.10 Abandon triggers

1. **B3 fails** — a plain text file matches the validated store. Then [01](01-the-validity-problem.md) is wrong, and the project is an architecture without a thesis.
2. **B2 fails by >5 pp** — memory cannot stand in for long attention. The window must grow, and the compute advantage in [07](07-training.md) largely disappears.
3. **B1 fails** — 1:32 loses to 1:8 by >2 pp at equal active. The model becomes 1:16 at the same stored size, which is a different, weaker design.
4. **R4 confirms** — actual single-session latency exceeds 3× a dense 4 B. Then the deployment story is false regardless of quality.

Three of the four are measurable before any large training run, for under 600 GPU-hours and two weeks of harness work. A design that can only be validated by spending its whole budget is not a design.

---

## 9.11 Summary judgement

**What is right, if the thesis holds:**

- **A 20.5 B model whose decode reads 1.84 GB/token**, and which is flat in context because the cache does not grow. That is the strongest single result in the folder.
- **4.2 MB of session state** instead of 2.36 GB of KV at 256 k — a 562× reduction that makes concurrency a compute question rather than a memory question.
- **A memory tier that carries provenance**, which is the only mechanism in the design that addresses the failure mode the 2026 benchmarks actually measure.

**What is probably wrong:**

- **The reasoning ceiling.** 1.76 B active is small, and search does not manufacture capability that is not there.
- **The staleness corpus.** 50 % odds it is too small to train the controller, which would leave the most novel part of the design unbuilt.
- **The serving economics at small batch.** MoE pays a 2–3× tax, and this design leans hard on batching to amortise it.

**One line:** the previous designs made reading cheap; this one makes *remembering* cheap — and bets that the hard part of coding is knowing which of your memories is still true.

**Back to:** [README](README.md) · [01 — The validity problem](01-the-validity-problem.md) · [04 — Memory](04-memory.md)
