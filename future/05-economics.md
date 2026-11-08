# 5 — Form Factors, Price, and Who Captures Value

> A frontier model is a perishable asset. Its price falls 13× per year, its capability is matched by an open model within 6–12 months, and it costs billions to make. Understanding who survives that squeeze is most of understanding the industry's next decade.

---

## 5.1 Three tiers, and the lag that defines the market

| Tier | 2026 | 2031 [P] | Serving model |
|---|---|---|---|
| **Frontier** | $8–20 per M output | $0.40–1.00 | owned datacentres, 5 GW campuses |
| **Workhorse** | $0.50–3 per M output | $0.05–0.15 | rented slices, single nodes |
| **Edge** | n/a (mostly cloud) | free at point of use | phone/laptop, 4-bit, 1–4B active |

The workhorse tier is filled by **last year's frontier, distilled**. This lag is structural:

1. A new frontier model demonstrates capability on hard tasks.
2. Distillation data is generated from it (often by the frontier lab itself, as a product).
3. A 5–30B model is trained on that data.
4. Six to twelve months later, the small model is at 90–95% of the frontier's capability on the tasks that matter, at 1–3% of the price.

**[P] Prediction (80%):** through 2031, the open/cheap tier stays 6–12 months behind the frontier on benchmark-style tasks, and *further* behind (12–24 months) on long-horizon agentic reliability, which does not distil well.

That last clause matters and is under-appreciated: **distillation transfers knowledge, not judgement.** A distilled model can answer like the teacher; it cannot run a 12-hour task with the teacher's error recovery, because that behaviour comes from RL on environments, not from imitating outputs.

---

## 5.2 The price collapse and its one real consequence

Fixed-capability prices fall 47% per quarter (13×/year). Compound that forward:

| Year | Cost of 2026-frontier-class capability (relative) |
|---|---|
| 2026 | 1× |
| 2028 | 0.006× |
| 2030 | 0.00004× |
| 2032 | 0.0000002× |

Obviously this cannot run forever — it hits the cost of electricity and memory bandwidth, which fall at 1.1–1.5×/year. The floor is roughly **the energy and depreciation cost of the computation**, which for a workhorse-tier task is fractions of a cent. So the correct long-run statement is: **intelligence becomes a cheap utility whose price is set by joules.**

The consequence that matters strategically: **the frontier's exclusivity window is about one year.** Any capability that is a pure function of "a model can do X" is commoditised within 12 months. Therefore durable advantage must come from something that does not distil:

- **Memory and personalisation** (chapter 2): a model that knows your context cannot be copied from outputs.
- **Environments and verification** (chapter 3): proprietary graders and simulators.
- **Distribution and trust**: being the thing people already use, with the audit trail and the integration.
- **Proprietary data and closed loops**: sensors, transactions, outcomes.
- **Power and silicon**: the physical layer.

---

## 5.3 On-device is the underrated tier

A 3B-active model at 4-bit needs ~2 GB of weights and runs at 30–60 tokens/s on a 2025-era phone. That is enough for: summarisation, drafting, transcription, translation, screen understanding, personal search, on-device agents with app access, and continuous personalisation that never leaves the device.

Why it matters beyond privacy:

- **Latency.** Round-tripping to a datacentre adds 200–800 ms. Local inference is instant, which changes interaction design.
- **Cost.** Zero marginal cost per user, which changes product economics for high-volume features.
- **Availability.** Works offline, on a plane, in a hospital basement.
- **Regulation.** Data that never leaves the device sidesteps a large fraction of compliance burden.

**[P] Prediction (65%):** by 2029, the majority of consumer AI interactions run at least partly on-device, with the cloud reserved for hard reasoning. **[P] (40%):** by 2031, phones ship with a persistent on-device agent that has its own memory and permissions, and users think of it as a feature of the OS rather than an app.

---

## 5.4 Who captures the value

Four candidates, with the case for and against each.

**Compute landlords (hyperscalers, neoclouds).** They own the scarce input: power, shells, interconnect, and the capital to build 5 GW campuses at $150–200B each. The case: scarcity rents, and $3.9–6.5T of investment through 2030 has to earn a return. The risk: enormous fixed costs, 2-year build times, and the possibility of overbuilding into a demand air pocket — the classic telecom bubble shape. **[P] (55%):** at least one major overbuild correction (a >30% drawdown in AI-infrastructure equities) before 2031.

**Model labs.** They have the talent and the training pipelines. The risk is brutal: each generation costs more, the price of the previous generation collapses, and the customer can switch in a week. The case for survival: labs that own *memory* and *environments* build moats that distilling cannot breach. **[P] (60%):** the number of labs capable of training a frontier model falls from ~8 in 2026 to 4–5 by 2031, because the capital and power requirements exceed what mid-size players can raise.

**The application layer.** Historically where the durable margins live (see: the internet). AI applications own workflow, data, and trust. The risk: model providers absorb their features. The case: distribution and domain depth beat generic capability, and the price collapse means the model is a cheap input rather than a toll. **[P] (65%):** the largest AI-derived enterprise value in 2032 sits in applications and infrastructure software, not in model labs.

**Data and outcome owners.** Hospitals, banks, manufacturers, logistics firms, and anyone with proprietary process data and a closed feedback loop. They are the only group whose asset does not fall in price. The risk: they are bad at building software.

**My summary judgement:** value accrues to whoever owns a **closed loop** — data in, action out, outcome measured, loop repeats. Labs own it in research; applications own it in workflows; data owners own it in operations. Pure capability provision is the worst position in the stack, because it is the only part that gets commoditised on a 12-month cycle.

---

## 5.5 The energy and politics of it

The physics from chapter 1 has political consequences that will shape the decade more than any model release.

- Data centres are **3% of global electricity in 2030** but **~50% of US demand growth**. That mismatch is the whole story: globally fine, locally contested.
- **~4 GW of US projects were postponed in Q1 2026** because of community opposition. Local permitting is now a first-order constraint on capability.
- **On-site gas went from 25 GW to 45 GW in two years.** Technology companies are becoming power companies because the grid cannot move fast enough.
- **SMRs are post-2030** (>20 GW announced, first units after 2030). Anyone promising nuclear-powered training this decade is selling a timeline that does not exist.
- **Memory (HBM) was a binding constraint in H2 2025–2026**, and remains the least-discussed chokepoint. Silicon capacity, not chip design, sets the ceiling on deployment.

**[P] Prediction (65%):** by 2029, at least one major AI buildout is publicly blocked or relocated by local opposition or water/land constraints, delaying a frontier training run. **[P] (50%):** by 2030, electricity cost is a larger share of inference cost than accelerator depreciation for the workhorse tier.

---

## 5.6 Labour: the task-versus-job error

Almost all public argument on this is confused by a category error. AI automates **tasks**, and jobs are bundles of tasks with verification, accountability, and coordination attached.

What the 2026 evidence supports:

- Tasks that are text-mediated, verifiable, and repetitive are being automated or heavily assisted now: boilerplate code, first-pass document review, data cleaning, tier-1 support, translation, routine analysis.
- **Junior work is disproportionately affected**, because junior work is precisely the verifiable, well-specified part of a profession. This has a second-order effect that is rarely discussed: if the junior rung disappears, the pipeline that produces senior judgement breaks, and the shortage appears 5–10 years later.
- Jobs requiring accountability (a named human who signs), physical presence, or unverifiable judgement are barely touched and will remain so through 2031.

**[P] Prediction (70%):** by 2031, measurable employment effects concentrate in entry-level knowledge work (a 10–25% reduction in junior hiring in affected roles in advanced economies), while aggregate unemployment shows no large signal. **[P] (55%):** by 2032, the "missing junior cohort" becomes a recognised bottleneck in at least two professions.

---

## 5.7 Predictions

| Claim | Probability |
|---|---|
| Open-weight lag stays 6–12 months through 2031 | 80% |
| Frontier-capable labs fall to 4–5 by 2031 | 60% |
| Majority of consumer AI interactions partly on-device by 2029 | 65% |
| >30% drawdown in AI-infrastructure equities before 2031 | 55% |
| Electricity > accelerator depreciation as a share of workhorse inference cost by 2030 | 50% |
| Largest AI enterprise value in applications/infra software by 2032 | 65% |
| Junior knowledge-work hiring down 10–25% in affected roles by 2031 | 70% |

---

## 5.8 What would change this chapter

- **A demand air pocket** (enterprises failing to realise ROI at scale) would deflate infrastructure spending faster than capability, producing a 2001-style correction in which the technology keeps improving while the equities collapse.
- **A regulatory liability regime** that makes deployers responsible for agent errors would slow deployment sharply in high-stakes domains and accelerate it in low-stakes ones — a bifurcation rather than a slowdown.
- **A memory/product breakthrough** would break the commoditisation cycle by making switching costs real, and would move value decisively toward whoever ships it first.

**Next:** [6 — Scenarios and how to check me](06-scenarios.md).
