# AI, 2027–2036

A forecast written from the inside of the numbers, not from the mood of the discourse.

This is six chapters about what AI systems will be, how they will be built, what they will cost, and what will break — over the next five to ten years. It is not a list of predictions about product names. It is an argument about **which curves bind when**, because almost every disagreement about the future of AI is really a disagreement about which constraint arrives first.

## The thesis in one page

Four exponentials are running at once, and they are not equally fast:

| Curve | Rate (measured 2023–2026) | Source |
|---|---|---|
| **Training compute** for frontier models | 5× per year (doubling every 5.2 months) | Epoch AI |
| **Algorithmic efficiency** (compute needed for fixed capability) | ~2.8–3× per year (halving every ~8 months) | Epoch AI; Gundlach et al. |
| **Price at fixed capability** | ~13× per year (47% per quarter) | Epoch AI, Sep 2026 |
| **Agent task horizon** (length of task completed at 50%) | doubling every ~4–7 months | METR |

Multiply the first two and you get **effective compute: roughly 14× per year.** Multiply that by a price decline of 13× per year and you get the single most important economic fact of the decade: **the frontier is a moving target whose price halves every few weeks, which means the frontier's exclusivity lasts about a year.**

Three constraints then arrive in order:

1. **Power and packaging, around 2028.** Training compute has been growing 5×/year partly because training runs got longer (1.4×/year). That cannot continue: Epoch's analysis says runs longer than ~9 months are dominated by starting later with better hardware and algorithms, a limit reached around 2027. Once duration stops growing, compute growth falls to roughly the rate at which clusters can be built and powered — about 2.5–3×/year. Data centres are already ~945 TWh by 2030 (3% of world electricity), buildouts are $5–6.5 trillion for ~150 GW, and single campuses are being designed at 5 GW and $150–200 billion.
2. **Verification, around 2028–2030.** Capability gains now come from reinforcement learning in environments where success is checkable. That works spectacularly for code, mathematics, formal proof, and games — and poorly for management, medicine, law, and taste. The bottleneck stops being model quality and becomes **the supply of environments in which a model can practise and be scored.**
3. **Reliability, continuously.** METR's measurements show the 50% horizon at ~12 hours (Sep 2026) while the 80% horizon is ~70 minutes — a factor of ~10, and a separate analysis puts the reliability-adjusted horizon at ~2.4× shorter than the capability horizon, with no statistically detectable closing trend. Economic delegation needs 95–99%, not 50%, so the gap between "impressive demo" and "useful employee" is set by this ratio, not by benchmark scores.

**My central forecast:** the period 2027–2031 is one of *diffusion*, not discontinuity. Capability continues to rise, but the binding constraints move from "can a model do this" to "can we verify it", "can we afford the power", and "can we trust it repeatedly". The organisations that win are those that own environments, verification, distribution, and power — not those with the best benchmark score.

A secondary forecast, with lower confidence and much higher stakes: **the rate of progress itself may accelerate**, because AI research is one of the most verifiable domains that exists (code compiles, benchmarks score). In November 2026 METR reported that Anthropic's rate of capability gain on well-defined tasks had roughly **doubled** since late 2025, attributing most of the speed-up to the lab's own use of AI in research, and explicitly *not* to faster compute growth. If that generalises, every timeline in this document compresses by roughly a factor of two, and the 2031 column is where the 2033 predictions land.

## How to read the claims

Every substantive claim is tagged:

- **[F] Fact** — measured, dated, sourced. Verifiable today.
- **[T] Trend** — extrapolation of a measured curve, with its historical range stated.
- **[P] Prediction** — my forecast, with a probability where I can defend one.
- **[S] Speculation** — a mechanism I find plausible but cannot yet support with numbers.

Where a [P] rests on a [T] that has already bent (agent horizons, for instance), I say so. Where forecasters have been systematically wrong, I say that too — the record is worth knowing: benchmark-driven capability has repeatedly surprised **upward** on two-year horizons, while deployment, reliability and economics have repeatedly surprised **downward**. My forecasts are deliberately asymmetric to compensate: I expect capability sooner than intuition suggests, and usefulness later.

## The chapters

1. **[The four curves](01-four-curves.md)** — compute, capital, energy, algorithmic progress and price. Where each one bends, with arithmetic. Includes the 2027–28 deceleration case and what it does to every other number.
2. **[What models become](02-architecture.md)** — the six knobs are exhausted; the seventh is memory. What a 2031 model actually is (a system, not a checkpoint), what happens to sparsity, context, multimodality, and world models, and a speculative 2031 model card.
3. **[Learning after data](03-learning-after-data.md)** — the data wall, synthetic data, RL environments, verifier scaling, and the recursive loop. Why progress becomes uneven across domains, with a table of which fields move fast.
4. **[Agents and the reliability tax](04-agents.md)** — horizons, the 50%-versus-99% problem, why verification is the real bottleneck, what an agent org chart looks like, and the security problem nobody has solved.
5. **[Form factors, price, and who captures value](05-economics.md)** — three tiers, distillation lag, on-device, open weights, the collapse in price, and the uncomfortable question of where the margin ends up.
6. **[Scenarios, timeline, and how to check me](06-scenarios.md)** — four scenarios with probabilities, a dated prediction table to 2035, the indicators to watch, and the specific observations that would prove this document wrong.

## What this is not

It is not a claim that scaling continues forever, nor that it has stopped. It is not a claim about when "AGI" arrives — the term has no agreed operational definition, and forecasts that depend on it are unfalsifiable by construction. And it is not a claim that the current architectural consensus is permanent: chapter 2 names the specific things that would reset it.

The useful test of a forecast is not whether it feels right. It is whether you can name the observation that would make you abandon it. Chapter 6 does that for every major claim here.
