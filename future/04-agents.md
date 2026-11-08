# 4 — Agents and the Reliability Tax

> The most important number in AI is not a benchmark score. It is the ratio between the task length a model completes *half* the time and the task length it completes *almost always*. That ratio decides whether AI is a tool or a colleague, and in 2026 it was roughly 10× and not closing.

---

## 4.1 What the horizon curve actually says

METR measures the length of task (in human-expert time) that a model completes with a given success probability. The 50% horizon has doubled roughly **every 7 months since 2019**, and faster recently — about **4.2 months since 2023**, or ~89 days since 2024.

| Date | Model | 50% horizon | Notes |
|---|---|---|---|
| Feb 2025 | Claude 3.7 Sonnet | ~1 hour | |
| Nov 2025 | Opus 4.5 | ~5.3 hours (320 min) | |
| Sep 2026 | Opus 4.6 class | ~12 hours | and ~70 min at 80% |
| Apr 2026 | Claude Mythos Preview | 17 hours | longest measured; tasks >16h unreliable |
| late 2026 | frontier | ~1 week | *trend prediction, not achieved* |

Two things to notice. First, the trend has been extraordinarily reliable — this is one of the few capability curves with a decade of measurements and a stable slope. Second, **it has not yet delivered the one-week agent** that a naive extrapolation from 2024 predicted for 2026. The frontier is at 12 hours at 50%, and the "1 week" point keeps sliding.

That is the first genuine bending of an AI capability curve, and it deserves more attention than it gets.

---

## 4.2 The reliability tax, with arithmetic

Here is why 50% is nearly useless economically. If a task takes `n` steps and each step succeeds with probability `p`, the end-to-end success rate is `pⁿ`:

| Steps | p = 0.90 | p = 0.95 | p = 0.99 | p = 0.999 |
|---|---|---|---|---|
| 5 | 59% | 77% | 95% | 99.5% |
| 10 | 35% | 60% | 90% | 99.0% |
| 25 | 7% | 28% | 78% | 97.5% |
| 100 | 0.003% | 0.6% | 37% | 90% |
| 500 | ~0 | ~0 | 0.7% | 61% |

An eight-hour workday is on the order of 100–500 discrete decisions. **Delegating a day of work requires per-step reliability near 99.9%, not 95%.** This is not a nitpick about benchmarks; it is the arithmetic that separates "impressive" from "employed".

Now the measured gap. In October 2026, a preprint (Merlet, 60,417 attempts) estimated that reliability-adjusted horizons are about **2.39× shorter** than capability horizons, and found **no statistically detectable trend toward closing**. METR's own numbers are starker at higher confidence: 12 hours at 50%, 70 minutes at 80% — a **10× gap**.

Extrapolating the gap to 99% (labelled [T], because nobody has measured it cleanly): the 99% horizon today is plausibly **10–30 minutes of human work**.

**[P] Prediction (65%):** the 99% horizon reaches 8 hours — the "trust it with a day's work" threshold — between **2029 and 2031**, not before. **[P] (25%):** it arrives by 2028, which requires either the 50% horizon to reach ~100–800 hours or a genuine narrowing of the reliability gap.

**Why the gap may be structural.** Errors are not independent. A model's failures cluster around the same ambiguous situations, so the "per-step" model is optimistic in good cases and pessimistic in others. Worse, models are bad at knowing when they are uncertain, which is precisely the signal you need to insert a human check. Calibration is improving, but a model that is 99% accurate and 70% calibrated still requires review of everything.

---

## 4.3 Verification is the hinge

Agents work where errors are cheap to catch. That single sentence explains almost all observed variance in agent usefulness:

| Domain | Cost of catching an error | Agent usefulness 2026 |
|---|---|---|
| Code with tests | seconds, automatic | high and rising fast |
| Data analysis with reproducible pipeline | minutes | high |
| Customer support with policy checks | seconds, scripted | high for narrow intents |
| Research with citations | hours, human | medium |
| Legal drafting | days, expert | medium for documents |
| Medical decisions | months to years, harm | low (supervised only) |
| Strategy / management | never cleanly | low |

The consequence for deployment design is under-appreciated: **you do not need a better model to get value from agents, you need a cheaper verifier.** Teams that build strong automated verification around a mediocre model beat teams with a great model and no verification. This is the single most actionable claim in this document.

---

## 4.4 What an agent actually is by 2029

The scaffold matters as much as the model, and scaffolds are converging on an organisational shape rather than a chatbot shape:

```
  Orchestrator          plans, decomposes, allocates budget, decides when to stop
  ├─ Specialist workers  code, search, analysis, browser, shell
  ├─ Verifier            tests, linters, type checkers, simulators, cross-checks
  ├─ Memory              task state, prior attempts, user preferences, project facts
  ├─ Policy              permissions, spend limits, escalation rules, audit log
  └─ Review loop         retry on failure; escalate on ambiguity; report uncertainty
```

The parts that are still bad: **knowing when to stop**, **knowing when to ask**, and **recovering from its own mistakes without repeating them**. Those are the same three failures that make junior employees expensive, and they are not obviously solved by more scale.

**[P] Prediction (70%):** by 2029, the standard enterprise agent deployment includes a dedicated verification layer and a spend limit, and the job title "agent reliability engineer" exists at large firms. **[P] (55%):** by 2030, an agent completes a multi-day software task end-to-end with human review only at checkpoints, in a codebase it has memory of.

---

## 4.5 The security problem nobody has solved

An agent with credentials is a new class of vulnerability, and prompt injection is not fixed.

The core issue: a language model cannot reliably distinguish **instructions from data**. If an agent reads a web page, an email, a PDF, or a code comment, that content can contain instructions, and the model may follow them. Every mitigation to date is probabilistic — input classifiers, delimiters, "never follow instructions in tool output" — and all have been bypassed.

Combine that with: agents holding API keys, access to production systems, the ability to spend money, and the ability to send email. You get a system where a malicious document can exfiltrate data or move funds, and where the standard security model (authenticate the human, authorise the action) does not apply because the *model* is choosing actions.

**What will actually happen, in my estimation:** capability restrictions (allowlists, sandboxing, spend caps, human approval for irreversible actions) rather than solved injection. Agents will be deployed with the security posture of an untrusted contractor: least privilege, audit logs, no unattended access to irreversible operations. That is workable, and it caps autonomy well below what demos suggest.

**[P] Prediction (75%):** a publicly reported incident in which an agent — not an attacker directly — moved data or money because of injected content, before 2029. **[P] (20%):** a robust general solution to instruction/data separation is demonstrated before 2030.

---

## 4.6 The unit economics of a task

Token costs are falling so fast that cost is no longer the binding constraint for most agent work — reliability is. A rough calculation for a one-hour software task:

- 200k–2M tokens of context, tool output, and reasoning
- at $1/M blended (2026 workhorse tier): **$0.20–$2.00 per task**
- versus 1 hour of a developer: $50–$150 fully loaded

So the ratio is 25–500× cheaper *when it works*. That asymmetry is why the market tolerates 50% success: at 2× the attempts, you still win by an order of magnitude. It also predicts where agents land first — tasks where retrying is cheap and failure is visible. And it predicts where they land last: tasks where one failure is catastrophic, regardless of price.

**[P] Prediction (70%):** by 2028, price per successful complex agent task falls below $1 for the workhorse tier, and reliability — not cost — is cited as the top blocker in enterprise surveys.

---

## 4.7 Predictions

| Claim | Probability |
|---|---|
| 50% horizon exceeds 24h (measured, replicated) by end-2028 | 70% |
| 50% horizon exceeds 1 week by 2030 | 45% |
| 99% horizon reaches 8h by 2031 | 65% |
| Reliability gap (50% vs 80% horizon) narrows by >2× before 2030 | 30% |
| Public agent-caused data/money incident before 2029 | 75% |
| General prompt-injection defence demonstrated before 2030 | 20% |
| Multi-day supervised software tasks routine by 2030 | 55% |

---

## 4.8 What would change this chapter

- **A calibration breakthrough** — models that reliably know when they don't know — would be worth more than another 10× in capability, because it converts 50%-reliable agents into safely-supervised ones.
- **A general verifier** (chapter 3, §3.9) would move every "low" row in §4.3 up.
- **A plateau in horizon growth** would make 2028–2031 a period of consolidation, with most gains coming from scaffolding and verification rather than models. Note that this is not a pessimistic scenario for *value*: it is the scenario where deployments finally get safe enough to scale.

**Next:** [5 — Form factors and economics](05-economics.md).
