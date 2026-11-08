# 6 — Scenarios, Timeline, and How to Check Me

> A forecast that cannot be wrong is not a forecast. This chapter states four scenarios with probabilities, a dated table of predictions, the indicators that distinguish the scenarios, and the specific observations that would falsify the main claims of this document.

---

## 6.1 The four scenarios

They are distinguished by which constraint binds, not by how good the models are.

### Scenario A — Smooth Diffusion (45%)

Compute decelerates to ~3×/year, algorithmic progress holds at ~2.5×/year, effective compute grows ~8×/year. Capability improves steadily; the binding constraint becomes **deployment** — verification, integration, trust, regulation — and value accrues to applications, data owners, and infrastructure. Agents become genuinely useful in verifiable domains by 2029 and remain supervised everywhere else. Prices collapse; intelligence becomes a utility; the labour effects are real but concentrated in junior, verifiable work.

*What it feels like from the inside:* less like a singularity, more like the 1995–2005 internet — extraordinary, uneven, and mostly about adoption rather than invention.

### Scenario B — Reliability Plateau (25%)

The 50% horizon keeps growing but the 99% horizon stalls, because errors stay correlated and calibration does not improve. Agents remain impressive demos and narrow production tools. Economic impact concentrates in **assistive** uses (copilots, drafts, search, code review) rather than autonomous ones. Capex continues but returns disappoint; there is a correction in infrastructure equities; capability keeps improving while deployment value lags by years.

*Distinguishing feature:* benchmark scores keep rising while enterprise ROI surveys stay flat — the exact pattern of 2026.

### Scenario C — Compute Crunch (20%)

Power, memory, and permitting bind harder than expected. Training compute growth falls to ~2×/year or below after 2028; frontier runs cost $5–10B and only three players can attempt them; a memory shortage lasts through 2028. Capability growth slows to ~1.5×/year effective (still fast in absolute terms, but visibly slower). Open weights close the gap because the frontier cannot pull away. Regulation tightens around a smaller number of very large actors.

*Distinguishing feature:* the largest announced training run in 2029 is not much bigger than 2027's, and the constraint cited is megawatts, not ideas.

### Scenario D — Recursive Breakout (10%)

The METR-inside-Anthropic finding generalises: research throughput compounds, algorithmic progress rises from 2.5×/year to 5–10×/year, and the effective-compute curve steepens rather than bending. Capability timelines compress by roughly half. Everything in chapters 3–5 arrives 2–4 years early, and the binding constraint becomes verification and governance rather than capability — because the systems outrun our ability to check them.

*Distinguishing feature:* algorithmic progress measurably accelerating in independent benchmarks, and labs reporting internal research throughput gains above 2× sustained for more than a year.

**Probabilities sum to 100%.** If I had to name my biggest single uncertainty, it is whether B or A is right — i.e. whether reliability is a solvable engineering problem or a structural property of current methods.

---

## 6.2 The dated table

| Date | Prediction | Confidence |
|---|---|---|
| **2027** | Largest training run $1–3B; frontier training compute ~10²⁸ FLOPs | 75% |
| | Hybrid local/global attention in >80% of new open models | 85% |
| | 1M-token context standard at the frontier | 80% |
| | 50% agent horizon 24–48h on METR-style suites | 65% |
| | Open-weight model within 8 months of frontier on standard benchmarks | 80% |
| **2028** | >50% of frontier training tokens synthetic-but-verified | 70% |
| | First product whose quality on recurring user tasks improves with use | 60% |
| | Price per successful complex agent task <$1 (workhorse tier) | 70% |
| | Training compute growth visibly slowed (doubling >7 months) | 60% |
| | Public agent-caused data/money incident | 60% |
| **2030** | Frontier training compute ~2–4×10²⁹ FLOPs, or the curve has bent | 70% |
| | Multi-day supervised software tasks routine | 55% |
| | 99% horizon reaches 8 hours | 50% (by 2031: 65%) |
| | Frontier-capable labs down to 4–5 | 60% |
| | Majority of consumer AI interactions partly on-device | 65% |
| | Humanoid robots in commercial logistics deployment (thousands of units) | 55% |
| **2032** | Continual learning in a frontier product (narrow, real gains) | 50% |
| | "Model" officially a system: weights + memory + tools + verifier + policy | 70% |
| | First AI system producing a verified novel result in a narrow research subfield | 60% |
| | Recognised "missing junior cohort" bottleneck in ≥2 professions | 55% |
| **2035** | Systems that do most remote knowledge work at human level *with supervision* | 45% |
| | A fundamentally new architecture replacing the transformer block at the frontier | 20% |
| | Frontier capability matched by open weights within 6 months | 65% |
| | Per-token price within 100× of the energy cost of producing it | 70% |

---

## 6.3 The indicators to watch

Six numbers, each cheap to track, that discriminate between the scenarios:

1. **Epoch's frontier training compute** — is the doubling time still ~5 months, or has it moved to 8–10? (A at 8–9, C at 12+.)
2. **METR's 80% horizon** — is the gap to the 50% horizon narrowing, holding, or widening? (A if narrowing, B if holding or widening.)
3. **Price per successful complex task**, not per token — is it falling 3×/year or 13×? (A if 5×+, B if <3×.)
4. **HBM and advanced packaging lead times** — six months of lead time above two quarters means a physical ceiling.
5. **Grid interconnection queue times** in the top five US data-centre markets — the cleanest measure of the 2028 crunch.
6. **Open-weight lag on agentic tasks** (not benchmarks) — if it stays above 12 months, distillation is not transferring judgement, which supports the memory/environment moat thesis.

And one qualitative indicator that is more informative than any of them: **whether the top labs' published evals shift from benchmark scores to task-completion and reliability curves.** When that happens, the field has internally accepted that capability is no longer the bottleneck.

---

## 6.4 What would falsify this document

Stated plainly, so it can be checked:

- **The central thesis dies** if, by 2029, the 99% agent horizon exceeds 8 hours *and* continual learning ships in a general product *and* training compute keeps growing at 5×/year. That combination means the constraints I identify were not binding, and the accelerationist reading was correct.
- **Chapter 1 dies** if frontier training compute grows faster than 5×/year through 2030, or if a single run exceeds 10²⁹ FLOPs before 2029 (which would require the old growth rate to persist past the power wall).
- **Chapter 2 dies** if a new block architecture reaches frontier open weights before 2030 with a >2× efficiency advantage, or if world-model-first pretraining becomes the main scaling axis before 2031.
- **Chapter 3 dies** if a general verifier for open-ended work is demonstrated before 2030, which would collapse the unevenness in §3.6 and make the "fast columns" universal.
- **Chapter 4 dies** if the reliability gap (50% vs 80% horizon) narrows by more than 2× before 2030.
- **Chapter 5 dies** if the open-weight lag exceeds 24 months on general capability, or if a single lab captures >50% of AI application revenue, which would mean commoditisation did not happen.

---

## 6.5 What I would and would not bet on

**Would bet on:**
- Effective compute keeps growing faster than 5×/year through 2028. (The curves are too strong and too well-measured.)
- Intelligence gets dramatically cheaper per unit of capability. (13×/year has held for three years across five benchmarks.)
- Code and formal mathematics remain the fastest-improving domains. (Verifiability is structural.)
- The frontier's exclusivity stays around one year. (Distillation plus open weights.)
- Reliability, not capability, is the binding constraint on economic impact through 2030.

**Would not bet on:**
- Any specific date for "AGI", because the term has no operational definition and forecasts using it are unfalsifiable.
- Which lab leads in 2032. (Three years is roughly the half-life of a frontier lead.)
- The size or direction of labour-market effects. (The task/job distinction makes aggregate forecasts nearly useless.)
- Whether the transformer block survives. (I give 20% to replacement by 2035 — high enough to matter, low enough that planning around it is unwise.)

---

## 6.6 The honest uncertainty

Three things I genuinely do not know, and that would change the most:

**Whether reliability is solvable.** If correlated errors and poor calibration are properties of current training methods rather than of the problem, then Scenario B is right and the decade is one of assistive AI. If they are engineering problems that fall to better RL and better verification, Scenario A dominates and agents arrive on schedule. This is the single largest fork.

**Whether the research loop compounds.** The November 2026 METR finding is the most consequential datum in this document, and it rests on one lab, eleven weeks, and a self-report. If it replicates, everything moves left. If it does not, the 2027–2030 period looks considerably calmer.

**Whether a new paradigm lands.** World models, continual learning, and non-transformer recurrences are all plausible; none is demonstrated at frontier scale. A single success in any of them resets the architecture chapter, and possibly the whole timeline.

What I am confident about is the *shape*: capability will keep improving faster than most institutions can absorb, the constraints will move from algorithms to physics to verification, and the organisations that win will be those that own closed loops rather than those that own the best checkpoint.

---

## 6.7 Closing

The most common error in thinking about AI's future is to treat it as one question — "how good will models get?" — when it is at least four: how much compute can be built and powered, how much capability each unit of compute buys, what can be verified well enough to train and trust, and who captures the value when the capability is cheap.

Those four curves run at different speeds and bend at different times. The interesting decade is the one where they cross.

**Back to:** [README](README.md) · [1 — The four curves](01-four-curves.md) · [2 — What models become](02-architecture.md) · [3 — Learning after data](03-learning-after-data.md) · [4 — Agents](04-agents.md) · [5 — Economics](05-economics.md)
