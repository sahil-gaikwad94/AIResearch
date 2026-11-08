# 1 — The Four Curves

> Every argument about AI's future is really an argument about which constraint binds first. This chapter puts numbers on the four curves that matter, converts them into the same units, and shows where each one bends.

---

## 1.1 The unit that makes it intuitive: gigawatt-months

Training compute is usually quoted in FLOPs, which nobody can feel. Convert it to power × time instead.

A modern accelerator rack (72 GPUs, liquid-cooled) draws roughly 120 kW, so a gigawatt of IT load is about **600,000 accelerators**. At ~2.5 PFLOPS dense BF16 each and a realistic 40% model-FLOPs utilisation, one gigawatt sustains ~6×10²⁰ FLOP/s. Over a month:

> **1 GW-month ≈ 1.6 × 10²⁷ FLOPs.**

That single conversion makes the whole roadmap legible:

| System | Training FLOPs | Gigawatt-months | What it means physically |
|---|---|---|---|
| GPT-3 (2020) | 3.1 × 10²³ | 0.0002 | a few hundred GPUs for a few weeks |
| GPT-4 class (2023) | ~2 × 10²⁵ | 0.012 | ~10,000 GPUs for months |
| 2026 frontier [T] | ~4.8 × 10²⁷ | ~3 | a 1 GW campus for a quarter |
| 2028 frontier [P] | ~4 × 10²⁸ | ~27 | a 5 GW campus for five months |
| 2030 frontier [P] | ~4 × 10²⁹ | ~245 | 10 GW for two years — *or* the curve bends |

The 2026 row follows from the measured trend: frontier training compute has grown **5× per year** since 2020 (Epoch AI, doubling every 5.2 months). 2020 → 2026 is six years, so 3.1 × 10²³ × 5⁶ ≈ 4.8 × 10²⁷. The 2028 and 2030 rows assume growth decelerates to ~3×/year, for reasons in §1.4.

Notice what the bottom row implies: **by 2030, a frontier run would need a two-year, 10 GW allocation.** No announced campus is that large. Either the trend bends, or training is spread across sites, or efficiency does the work instead. This is the single most important quantitative fact in the forecast, and it is why "scaling continues" and "scaling hits a wall" can both be right — the wall is physical, not algorithmic.

---

## 1.2 Curve one: compute, and the three things propping it up

Frontier training compute has grown 5×/year. That growth decomposes into three multiplicative factors:

| Factor | Growth | Source |
|---|---|---|
| Aggregate compute of AI chips sold | 2.5×/year | Epoch AI |
| Training run duration | 1.4×/year | Epoch AI |
| Share of compute allocated to training | ~1.4×/year (early), now falling | Epoch AI |

5 ≈ 2.5 × 1.4 × 1.4. The middle term is the fragile one.

**Why duration stops growing.** A training run that takes 18 months will be beaten by one that starts six months later with better hardware and better algorithms — because algorithmic efficiency improves ~2.8×/year. Epoch's analysis puts the efficient ceiling at roughly **9 months**, and projects the trend reaches it around **2027 (90% CI: 2025–2029)**. Since duration contributed about a third of total scaling progress since 2018, the end of duration growth removes about a third of compute growth.

**Why the training share falls.** As inference demand explodes, a smaller fraction of a lab's compute goes to training. Epoch projects training share falling from ~40% in 2025–26 to ~20% by end-2027.

Combine: 5×/year ÷ 1.4 (duration) ÷ ~1.3 (share) ≈ **2.7–3×/year** after 2027. That is still fast — doubling every ~8–9 months — but it is a different regime from doubling every 5 months, and it changes every downstream number.

**[P] Prediction (75%):** the largest single training run's compute growth slows from ~5×/year to ~3×/year between 2027 and 2029, observable as the doubling time moving from ~5 months to ~9 months. **Falsifier:** a 2028 run above 10²⁹ FLOPs, which would require growth at the old rate.

---

## 1.3 Curve two: algorithmic progress

Independent of hardware, the same capability gets cheaper as methods improve. The best estimate from the pretraining era: the compute needed for a fixed level of performance **halved roughly every 8 months** (5–14 month 95% CI), i.e. ~2.8×/year. A later decomposition that separates hardware, competition and algorithms from market prices puts the pure algorithmic component at **~3×/year**, with 5–10×/year total at fixed capability (Gundlach et al., 2025).

Where the 3× comes from is worth internalising, because it tells you what to expect next:

- **Architecture** (MoE, GQA/MLA, hybrid attention): roughly one-time gains of 2–5× each, arriving every 18–24 months.
- **Data** (filtering, deduplication, mixture, synthetic): large, continuous, and the least published.
- **Post-training** (RLHF → DPO → RLVR): the dominant source since 2024. Reasoning models converted inference compute into capability, which is less an efficiency gain than a new axis.
- **Numerics and kernels** (BF16 → FP8 → FP4, FlashAttention, speculative decoding): steady 1.5–2× per generation.

The crucial structural point: **algorithmic progress compounds with compute rather than substituting for it.** The two multiply. 5×/year compute × 2.8×/year algorithms ≈ **14×/year effective compute** — which is why a model that needed 3 GW-months in 2026 needs ~0.14 GW-months to match in 2029.

That last number is the engine behind everything in chapter 5: it is *why* the frontier's price collapses and why open weights track it with a lag.

---

## 1.4 Curve three: price

At fixed capability, inference prices have fallen extraordinarily fast. Epoch's September 2026 analysis, across five benchmarks:

- **47% per quarter at fixed performance ≈ 13× per year** since 2023.
- At the moment a capability is first achieved (SOTA debut), cost falls **66% per quarter (75×/year)**; two years later the same capability falls **32% per quarter (4.7×/year)**.
- For comparison: DNA sequencing has fallen 1.84×/year since 2001; computing 1.51×/year since 1940.

Concrete: GPT-3 scored 43.9 on MMLU at $60 per million tokens in 2020. Llama-2-7B scored 45.3 at $0.20 in July 2023 — a 31×/year compound decline. In September 2026 the cheapest current-generation small model (GPT-6 Luna) listed at **$0.10 per million input tokens and $0.50 output**, while the frontier tier sat at roughly $8–20 output.

**The trap in that number.** Price *per token* and price *per task* are diverging. Reasoning models spend 10–100× more tokens per task. Epoch's worked example: hitting ~27% on FrontierMath took ~43M output tokens from o4-mini in April 2025 and ~5M from GPT-5.2 in December 2025 — about 3× cheaper in eight months, not 13×. **Your bill can rise while prices collapse**, because you keep buying harder tasks. Anyone planning a budget on "prices fall 13×/year" will be wrong by an order of magnitude.

**[P] Prediction (70%):** by 2028, fixed-capability price decline slows to the 4–6×/year regime, because the easy gains (distillation, batching, smaller models) are exhausted and the remaining cost is dominated by energy and memory bandwidth.

---

## 1.5 Curve four: energy and physical buildout

This is where the arithmetic of §1.1 meets the electricity grid.

| Quantity | Value | Source |
|---|---|---|
| Global data-centre electricity, 2030 | ~945 TWh (≈3% of world electricity) | IEA |
| AI-focused data centres, 2030 | ~465 TWh (3× 2025) | IEA |
| Data-centre share of *growth* in electricity demand to 2030 | <10% globally; ~50% in the US | IEA |
| Total data-centre investment 2026–2030 | $3.9T (IEA base case) to $5–6.5T (Bain) | IEA; Bain |
| New capacity by 2030 | ~150 GW, roughly tripling 2025 | Bain |
| Largest campuses under design | 5 GW, $150–200B each | Bain |
| Time to build a GW-scale campus | ~2.1 years | Epoch AI |
| On-site gas generation, 2024 → 2026 | 25 GW → 45 GW | IEA |
| Community opposition | ~4 GW of US projects postponed in Q1 2026 | McKinsey |

Three things follow.

**The constraint is not generation, it is interconnection and local politics.** Globally, data centres are 3% of electricity in 2030 — small. But they are 50% of *US demand growth*, and they concentrate in a handful of regions, where they meet 2–4 year grid queues and local opposition. That is why on-site gas capacity doubled in two years and why SMRs (20+ GW of announced plans, first units post-2030) are being financed by technology companies rather than utilities.

**Memory is a real, recent bottleneck.** The IEA notes that in the second half of 2025 and early 2026, **high-end memory production capacity became a binding constraint** on AI server production. This is the least-discussed limit and the most likely to cause a short-term surprise: HBM supply determines how many accelerators can be built, and it is a three-year capex cycle.

**Efficiency is improving faster than demand.** The IEA's striking finding: *power consumption per AI task is declining at a rate unprecedented in energy history*. So the honest framing is not "AI will eat the grid" but "AI is a fast-growing, geographically concentrated load with an unusually steep efficiency curve" — and the political fights will be local, not global.

---

## 1.6 What the four curves produce together

Put them on one timeline:

| Period | Dominant regime | Binding constraint | What it looks like |
|---|---|---|---|
| 2023–2027 | **Compute-rich** | chips, then memory | capability jumps every 6–12 months; price falls 13×/yr; everyone scales |
| 2028–2031 | **Verification- and power-bound** | environments, grid, reliability | capability gains concentrate in verifiable domains; frontier runs cost $1–5B; agent usefulness gated by trust |
| 2032–2036 | **Unknown regime** | depends on whether a new paradigm lands | either a plateau of diminishing returns, or a step change from world models / continual learning |

The transition in 2028 is where my forecast differs from both the accelerationist and the decelerationist stories. Not because capability stops — the curves are too strong for that — but because **the marginal dollar stops buying benchmark scores and starts buying reliability, environments, and watts.**

---

## 1.7 The arithmetic that people get wrong

Three common errors, each with the correction:

**"Effective compute doubles every 6 months, so we get 2²⁰ by 2036."** Effective compute is growing ~14×/year today, but that has never been sustained for a decade in any technology. The 5×/year compute term is already bending (§1.2), and the 2.8×/year algorithmic term is itself the *output* of research that gets harder as the easy wins are taken. Model 8–10×/year for 2028–2032 and 4–6×/year after, and say so explicitly.

**"Prices fall 13×/year, so inference will be free."** Cost per token, yes-ish. Cost per *task*, no: capability demand rises to meet supply, and reasoning multiplies tokens per task. Budget for 3–5×/year per task, not 13×.

**"945 TWh is 3% of global electricity, so power is a non-issue."** Globally correct, locally wrong. The relevant number is 50% of US demand growth landing in a dozen counties with 3-year grid queues. Constraints that are 3% globally and 50% locally bind locally.

---

## 1.8 What would change this chapter

- **A new efficiency paradigm** (e.g. training in FP2, or a 5× architectural win) would push the power wall out by years.
- **A slowdown in algorithmic progress** to ~1.5×/year would make the 2028 deceleration much sharper and move the whole timeline right by 2–3 years.
- **Cheap fusion or a grid buildout surge** would remove the 2028–2031 constraint and return us to a compute-rich regime.
- **A capability plateau** (loss curves flattening despite scale) would invalidate the premise that compute buys capability at all — the one outcome that makes every prediction here too optimistic.

**Next:** [2 — What models become](02-architecture.md).
