# Research log

Working notes from a principal-researcher/engineer mode: pick a problem, do the arithmetic, commit to a design, write down what would prove it wrong.

Unlike `../experiments/` (proposals for questions that need new measurements) and `../future/` (forecasts), the work here is **constructive**: build a specific thing, specify it precisely enough to implement, and check its numbers.

| Day | Topic | Output |
|---|---|---|
| [day1](day1/README.md) | Architecture for a small coding model | **Kestrel-Coder-4B** — full spec, reference PyTorch implementation, tokeniser design, training and RL plans, serving math, pre-registered ablations, decision log |

## Convention

Every document set here obeys three rules, because the alternative is a pile of plausible-sounding prose:

1. **Numbers are computed, not remembered.** `tools/count_params.py` produces every parameter, KV-byte and FLOP figure quoted in the text, and the reference implementation asserts that the analytic count matches the instantiated model.
2. **Every choice names the alternative it beat.** A design decision without a rejected alternative is a preference, not an engineering decision.
3. **Every design names what would falsify it.** If nothing could, it is not a design, it is a slogan.

---

## Day 2 — Kestrel-2

[`day2/`](day2/) revisits day 1 with 2025–26 evidence and replaces three of its assumptions: attention (sparse exact over a *selected subset*, not full attention), context (append-only deltas, not re-rendered files), and depth (a runtime dial via a recurrent block, not a parameter tax). It lands on **4,125,828,176 parameters, 11,520 B of KV per token, and a decode step that streams 96× less cache at 1 M context** — for 43,097 GPU-hours, slightly *less* than day 1.

Start with [`day2/01-what-day1-missed.md`](day2/01-what-day1-missed.md), which is a review of day 1 written by its author and is the reason day 2 exists.

---

## Day 3 — Kestrel-3

[`day3/`](day3/) starts from a different question. The 2026 memory benchmarks show that agent memory systems usually make coding *worse*: in VibeMemBench, 11 of 12 solver × memory-system pairings fail to beat having no memory at all, even though the same information injected directly helps every solver. The failure is **validity** — stale, unverifiable claims — not capacity.

So Kestrel-3 is built around three memory tiers: a 4,096-token exact window, a **4.2 MB delta-rule state** (562× smaller than a 256 k KV cache), and a versioned store whose entries carry evidence and a content hash. Behind it sit 96 fine-grained experts per layer at a **1:32 activation ratio** — 20,485,187,584 parameters stored, **1,761,814,528 read per token**, trained for **31,025 GPU-hours**.

Start with [`day3/01-the-validity-problem.md`](day3/01-the-validity-problem.md).

### Day 3, v2 — Kestrel-3.1

[`day3/v2/`](day3/v2/) generalises v1's memory thesis to the whole stack: **in code, generation is expensive and checkable, so every decision should move work from the expensive side to the checkable side.**

It rests on one measurement. Self-play against a reference-free judge drives the judge's pass rate from 0.72 to 0.94 while true accuracy stays at 0.20; a three-judge ensemble still accepts 55 % of false positives. Change one thing — make the judge commit its own answer *before* seeing the candidate — and the false-positive rate falls from 0.719 to 0.012.

From that: 256 latent-width experts at 1:42 sparsity (10.7× cheaper per expert than full width), a 4-layer recurrent workspace that rents depth per token at 12.3× less than a decoded token, learning driven by test execution rather than a judge, on-policy distillation of verified search trajectories, and a self-improvement loop gated by a verifier that cannot be talked into agreeing.

**20,865,312,258 parameters stored, 1,990,944,258 read per token, 28,761 GPU-hours, 11.5 GB at 4-bit.** Seven assertions in `kestrel31/model.py`, all passing.

Start with [`day3/v2/01-thesis.md`](day3/v2/01-thesis.md), or the chapter everything depends on, [`day3/v2/08-verification.md`](day3/v2/08-verification.md).

---

## Day 4 — Kestrel-4 × VEXX 1.5

[`day4/`](day4/) starts by auditing the three days before it. Seven defects are found — RoPE
constructed but never applied, a decode path that bypassed the causal mask, expert bytes
under-counted by 16×, a latent norm in every diagram and no module, a dead scorer, 256 experts
that added no routed rank, and five soundness holes in the verifier — and every one of them
pointed the same way: it made the design look better than it was.

The answer is not a better promise, it is a different unit of verification. **Kestrel-4 moves
the verifier inside the decode loop**: a four-tier stack (automaton → outward-rounded interval
contractor → exact rational Fourier–Motzkin → CDCL jumpback) masks each token, splices
deterministic fields without sampling, and rewinds to the clause that caused a conflict instead
of to the start of the patch. The measured ladder is 0.719 → 0.012 → 0.00216 → **0.000324**
false accepts, which turns self-training at a 40 % solve rate from a 1.8 % poison into a 0.049 %
one, and turns a failed 600-token patch from 1,746 GFLOP into **52.4 GFLOP**.

**21,248,719,490 parameters stored, 2,071,178,882 read per token (9.75 %), 23,530 GPU-hours,
11.69 GB at 4-bit, 18,784 tok/s at 4 k/batch 64 with KV traffic, 2.16 GB int8 KV per 1 M session
(33.6 MB on chip).** Fourteen assertions in `kestrel4/model.py`, all passing; the config is
generated from the reference implementation so the ledger cannot drift, and `tools/audit_v4.py`
tries to refute each mechanism in the prose (16 claims failed across two audits; all are fixed,
31/31 pass).

Start with [`day4/01-what-day3-missed.md`](day4/01-what-day3-missed.md), which is an audit
written after the fixes, or [`day4/04-verification.md`](day4/04-verification.md), which is the
chapter the rest depends on.
