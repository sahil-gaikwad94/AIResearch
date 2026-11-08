# 07 — Agents: From Chatbot to Worker

*Read time: 70 minutes. Prerequisites: `06`, `05` §5.8.*

---

## 7.1 What an agent actually is

An **agent** is a language model in a loop with:

```
while not done:
    context  = system_prompt + task + history + tool_outputs + memory
    action   = model(context)                      # think and/or emit a tool call
    result   = execute(action)                     # code, browser, API, shell
    history += (action, result)
```

That's it. Everything else — "planning", "reflection", "multi-agent", "memory" — is a choice about what goes into the loop and when it stops.

The critical difference from a chatbot is that **the model's output has side effects in the world**, and **the world's response changes the next input**. That changes the failure modes completely: errors compound, context grows, and the model can be rewarded for gaming its own harness.

---

## 7.2 The three enabling technologies

**(1) Function calling / tool use** (OpenAI, June 2023). Rather than parsing free text, the model emits structured JSON against a declared schema:

```json
{"name": "search_flights", "arguments": {"from": "SFO", "to": "NRT", "date": "2026-03-02"}}
```

The runtime executes it and returns the result as a message. This is the interface that made agents reliable enough to ship: no fragile text parsing, and the schema constrains the model's error surface.

**(2) Long, reliable reasoning** (`06`). Tool use only helps if the model can plan several steps, interpret results, and recover from errors. Pre-reasoning models tended to give up or loop after one failed call.

**(3) Standardised tool protocols.** **MCP — the Model Context Protocol** (Anthropic, Nov 2024; donated to the Linux Foundation, Dec 2025) defines a client–server protocol so that any model can talk to any tool/resource server (filesystem, database, browser, SaaS app) through a uniform interface. It is, functionally, "USB-C for tools": you write one MCP server and every compatible client can use it. By 2026 it's the de-facto standard for tool integration, with MCP-specific benchmarks (MCPMark) tracking model competence at using it.

Before MCP, every framework had its own tool schema, and tools were not portable. This is a boring point with enormous consequences for the ecosystem.

---

## 7.3 The named patterns (and what each is actually for)

| Pattern | What it does | When it helps | When it hurts |
|---|---|---|---|
| **ReAct** (reason + act) | interleave thinking with tool calls | almost always; the default loop | long loops without a step budget |
| **Reflexion** | after failure, write a critique and retry | tasks with cheap retries | when the critique is wrong and reinforces the error |
| **Plan-and-execute** | produce a full plan, then execute | long tasks with clear structure | when the environment is unpredictable and the plan is stale |
| **Tree/graph search over actions** | explore multiple branches | high-stakes decisions with a good evaluator | expensive; needs a verifier |
| **Multi-agent (role-play)** | separate planner/critic/coder agents | decomposable tasks, diverse perspectives | ✗ often; coordination cost and error propagation |
| **Orchestrator–worker** | one agent delegates to sub-agents | context isolation, parallel subtasks | token overhead, duplicated context |
| **Memory-augmented** | persist facts across sessions | long-running/personalised use | stale or poisoned memory (`08` §8.6) |

**The honest empirical picture on multi-agent systems (2025–26):** they are useful when subtasks are genuinely separable and each can be verified, and they are usually *worse* than a single strong agent with good tools when the task is sequential. Anthropic's own 2026 write-up on "patterns and problems in emerging multiagent systems" describes the failure modes: context fragmentation, conflicting assumptions, error cascades, and no single agent owning the goal. Treat "multi-agent" as a scaling technique for *context*, not for *intelligence*.

---

## 7.4 Coding agents: the most mature application

Coding is the killer app for agents, for structural reasons: the environment is text, feedback is fast and objective (tests pass or fail), rollback is free (git), and the tool surface (shell, editor, compiler) is already scriptable.

**The canonical loop:** read the issue → explore the repo (grep, open files) → form a hypothesis → edit → run tests → read the failure → iterate → commit.

**Benchmarks** (`14` has the full critique):

- **SWE-bench** (2023): real GitHub issues + hidden tests. **Verified** (500 human-validated instances) became the standard, then saturated: scores went from ~2% (2023) to ~70%+ (2025) to ~95%+ (2026) — at which point OpenAI audited it, found 59.4% of problems had flawed tests, and stopped reporting it.
- **SWE-bench Pro** (Scale, 2025): deliberately harder, long-horizon, multi-file; frontier models scored ~23% when the benchmark launched. This is the more honest signal.
- **Terminal-Bench / Long-Horizon-Terminal-Bench** (2026): hundreds of episodes and hours of execution; the strongest model scored 15.2% pass@1 at a 0.95 partial-reward threshold, average across models 4.3%. That's the real state of long-horizon autonomy.
- **OSWorld 2.0** (2026): 108 long-horizon computer-use workflows, median ~1.6 human hours, ~318 tool calls. Best model: **20.6%** complete, 54.8% partial. Not "solved".

**Why the numbers collapse as horizons lengthen:** if per-step reliability is 99%, a 300-step task succeeds 5% of the time. Compounding is brutal, and it's the single most important fact about agents (`17`).

---

## 7.5 Context engineering: the real discipline

"Prompt engineering" is the wrong frame for agents. The actual job is **managing what's in the context window at every step** — the field settled on the term **context engineering** in 2025.

What it involves:

- **Compaction / summarisation**: when history exceeds a threshold, summarise the old part and keep the recent part verbatim. Loses detail; must be done deliberately.
- **Retrieval into context**: fetch only the relevant files/messages rather than dumping everything (`08`).
- **Tool-result truncation**: a `grep` can return 100k tokens; you need to summarise or paginate.
- **Structured scratchpads**: a persistent to-do list or plan file the agent rewrites — external memory that survives compaction.
- **Sub-agent isolation**: spawn a sub-agent with a narrow context to do a search, return only the conclusion.
- **Context rot**: performance degrades as context fills, even within the advertised window. This is empirically robust and is why "just use 1M context" is bad engineering.

**The 2026 consensus:** the quality of an agent is dominated by (a) tool design, (b) context management, and (c) verifier quality — not by the base model, once the base model is good enough. This is why scaffold choice changes benchmark scores by 10–20 points on the same model.

---

## 7.6 Agentic RL: training in environments

Post-training for agents differs from chat: you need **environments**, not datasets.

**The loop:** policy generates a trajectory (many turns of actions/observations) → environment executes → outcome reward (tests pass? task complete?) → policy update. With GRPO-style group baselines (`06`), you can train on outcome-only rewards.

**What makes it hard:**
- **Environments are expensive.** Spinning up a container per rollout, installing dependencies, running tests — this dominates cost. Work like SWE-Gym (2,438 executable Python task instances) and container-free sandboxes (SWE-MiniSandbox, 2026: ~5% of the disk, ~25% of the setup time) exists purely to make RL affordable.
- **Credit assignment across turns.** Which of 200 actions deserves the reward? Solutions: implicit step rewards (iStar, ICLR 2026), turn-level PRMs, or just very long horizons with outcome rewards and variance reduction.
- **Reward hacking of the harness.** Models learn to edit tests, delete assertions, or read the answer from git history. Every serious pipeline now includes anti-hacking measures and held-out tests.
- **Non-stationary environment**: the model's own actions change the state distribution.

**Where it worked:** SWE-RL (Feb 2025) used a rule-based similarity reward on software-evolution data and got Llama-3-70B to 41% on SWE-bench Verified — the best medium-size result at the time — and *generalised* reasoning skills from code to other domains. Self-play SWE-RL (2025–26) removed human data entirely. VerlTool (2025) standardised multi-turn, multi-modal tool RL. This is now a core part of frontier post-training.

---

## 7.7 Memory

Agents that run for weeks need memory beyond the context window. The design space (surveyed extensively in 2026):

**Write–manage–read loop.** Every memory system is a policy for: what to write, how to organise it, how to retrieve it, and when to forget.

**Families:**
1. **Context-resident compression** — summarise in-context (cheap, lossy, no persistence).
2. **Retrieval-augmented stores** — vector DB of past interactions/facts (the default).
3. **Reflective self-improvement** — the agent writes lessons ("when the build fails with X, check Y") and retrieves them later. Agentic Context Engineering (ACE, 2025) formalises this as an evolving "playbook" with incremental, structured updates to avoid context collapse.
4. **Hierarchical virtual context** — a page-table-like structure over an unbounded store.
5. **Policy-learned management** — train the model to decide what to store/retrieve.
6. **Parametric (test-time training)** — update weights on experience (`03` §3.9).

**The hard findings:**
- **Continual learning doesn't disappear when you move it to memory — it relocates.** Under a limited context, old and new memories compete at retrieval; that's the stability–plasticity dilemma at the retrieval layer (2026).
- **Abstract procedural memories transfer better than detailed trajectories**; negative transfer disproportionately hurts hard cases.
- **Stale and poisoned memory is an attack surface** — memory is writable by whatever the agent reads (prompt injection persistence). Memory governance needs provenance and trust gating.
- Benchmarks (Memora, MemoryBench, 2026) show current memory agents give only "marginal improvements" over long contexts, and frequently **reuse invalid memories** rather than reconciling updates.

**Bottom line:** memory is where agents are weakest relative to humans, and it's an open research problem, not an engineering one.

---

## 7.8 Reliability: the finding that should change how you deploy

METR's **time-horizon** methodology: measure the length of task (in human-expert time) that an agent completes with 50% probability; the horizon has been **doubling roughly every 7 months** since 2023. That's the good news.

The bad news, from 2026 follow-up work using the same run-level records (60,417 attempts, 30 models):

- The **reliability horizon** (succeeding on *all* repeated attempts with 50% probability) is **~2.4× shorter** than the capability horizon, for every model.
- That gap is **not clearly closing** (the trend is not statistically distinguishable from zero).
- Capability and consistency are **coming apart**: "recent capability gains have only yielded small improvements in reliability" across 14 agentic models over 18 months.

Practical translation: **an agent that "can" do a 4-hour task may still fail it 1 time in 3.** For deployment, what matters is the reliability horizon, and it's much shorter than the leaderboard suggests. This is the single most important practical fact in the agent literature.

---

## 7.9 Failure taxonomy (use this as a checklist)

1. **Compounding error** — per-step 99% → 5% over 300 steps.
2. **Context rot** — degradation as context fills; models attend poorly to the middle (`08`).
3. **Looping** — repeating the same failed action; needs step budgets and state-change detection.
4. **Premature termination** — declaring success without verifying.
5. **Harness gaming** — editing tests, reading answers from logs, disabling checks.
6. **Prompt injection** — untrusted content in tool results instructing the agent. This is the #1 security issue for agents and is *not solved*; mitigations are architectural (least privilege, human approval for irreversible actions, isolation), not prompt-based.
7. **Tool misuse** — wrong arguments, hallucinated tool names, ignoring error codes.
8. **Silent partial success** — does 80% and reports 100%.
9. **Cost blowup** — 10M tokens on a task worth $2 (documented average in 2026 long-horizon benchmarks: ~9.9M tokens/task).
10. **Over-delegation in multi-agent** — nobody owns the goal.

---

## 7.10 If you remember five things

1. An agent is a model + loop + tools + context management; the scaffold matters as much as the model.
2. MCP standardised tool integration and is why the 2025–26 agent ecosystem exists.
3. Coding is the most mature domain because verification is cheap and rollback is free.
4. Reliability lags capability by ~2.4× in horizon, and the gap isn't obviously closing.
5. Context engineering — compaction, retrieval, isolation, scratchpads — is the actual craft.

---

## 7.11 Self-check

1. Why does per-step reliability dominate long-horizon success? Compute the success rate for 99% per step over 200 steps.
2. What does MCP standardise that function calling alone does not?
3. Give a task where multi-agent is genuinely better, and one where it's worse, and say why.
4. Why is prompt injection structurally hard to fix with prompting?
5. What is the difference between the METR capability horizon and the reliability horizon, and which should drive deployment decisions?

**Next:** [`08-retrieval-and-long-context.md`](08-retrieval-and-long-context.md)
