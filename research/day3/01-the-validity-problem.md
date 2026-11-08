# 1 — The Validity Problem

> Every memory system built for coding agents in 2025–26 has the same failure, and it is not the one we designed for. It is not capacity, it is not retrieval quality, and it is not latency. **It is that the memory does not know whether it is still true.**

---

## 1.1 The result that should stop you

*VibeMemBench* (2026) is the first benchmark that isolates memory from coding ability. 111 targets from 90 real repositories, 3,634 history trajectories, executable tests as ground truth. The construction is careful: a target only enters the set if *injecting the relevant prior experience demonstrably improves the executable outcome* in a reference setting. Every item therefore carries a remembered fact whose usefulness has been verified by running code.

The results split in two:

| Setting | Effect |
|---|---|
| **Verified experience injected directly into the prompt** | +1.1 to +4.5 pp task resolution on 4 of 5 solvers, **and fewer agent steps on all 5** |
| **Four existing memory systems, building and retrieving experience themselves** | **11 of 12 solver × system pairings fail to beat the memory-off baseline** |

Read that again. The *information* is provably useful — when it is handed over, every solver gets better and faster. The *systems* that are supposed to produce that information cannot recover it from the same history without making things worse.

The gap between those two rows is the subject of this document. It is not a retrieval bug. It is not an embedding-quality problem. It is that the systems store **text**, and text has no opinion about its own truth.

---

## 1.2 Three failure modes, all the same failure

**Staleness.** *MemTrace* (2026) names the pattern: a refreshed context retrieves a test failure from twenty turns ago, after the fix and after a passing rerun. The agent reads "this test fails", and it is true — in the past tense. The paper calls the mismatch between recovered history and current repository state **state-memory mismatch**, and finds it across three long-horizon benchmarks and two agent harnesses.

**Forgetting.** *Memora/FAMA* (2026) measures whether an agent's response reflects the *current* memory state, penalising reliance on obsolete or deleted memory. The dominant error category is not missing information — it is confidently using information that has been superseded.

**Over-answering.** The same benchmark finds recommendation errors driven by "failure to forget outdated memory and partial memory retrieval". The agent has a memory, the memory is out of date, and the agent has no mechanism that would tell it so.

**Recording the wrong things.** Anthropic's *components of agent memory* (Nov 2026) reports a clean ablation: instruct the agent to "record only failures" and the memory gain collapses — Opus 5 loses **0.234** of a 0.828 score, Opus 4.7 loses **0.150**. What is worth remembering is not what a naive heuristic says is worth remembering.

These are the same failure seen from four angles:

> **A memory entry is treated as a fact. It is actually a claim, made at a time, about a version of a repository, supported by some evidence.**

Text files flatten that structure away, and once it is gone no amount of retrieval quality can recover it, because the information needed to judge validity was never stored.

---

## 1.3 What a memory entry has to be

If the diagnosis is right, the fix is a schema, not a better index. An entry must carry:

| Field | Question it answers | Cheap to compute? |
|---|---|---|
| `key` | *what is this about?* — a canonical symbol id, not a file path | yes (parser) |
| `claim` | *what do we believe?* — one sentence | no (model) |
| `evidence` | *why?* — a test id, a trace, a read at a commit | yes (harness) |
| `validity_key` | *is it still about the same thing?* — content hash of the symbol body | **yes (hash)** |
| `confidence` | *how much should this steer behaviour?* | learned |

The `validity_key` is the whole trick, and it is nearly free. A symbol's body hashes in microseconds. When the body changes, the hash changes, and every claim attached to it is *known* stale — not suspected, known — before it is ever shown to the model.

This is a one-line idea with a large consequence:

> **Invalidation should be a hash comparison, not a judgement call.**

Systems that make it a judgement call ask a model to notice that its own memory is out of date, in context, under time pressure. That is exactly the task at which §1.1 shows models fail.

---

## 1.4 The three things the architecture must therefore do

The evidence above constrains the design more tightly than any benchmark table.

**1. Separate capacity from computation.** The knowledge a coding model needs is large (APIs, idioms, error signatures, per-repository conventions) and almost never all relevant at once. That argues for enormous stored capacity and a tiny active footprint — which is a mixture-of-experts argument, and the 2026 scaling study settles the numbers ([03](03-moe.md)).

**2. Make the model's own memory compressed, not cached.** Exact recall of recent tokens and cheap compression of distant ones are different jobs. Attention over a window does the first; a fixed-size learned state does the second at constant cost. A 256 k-token KV cache is 2.36 GB; the same session's compressed state here is **4.2 MB**, a factor of 562 ([02](02-architecture.md)).

**3. Spend the savings on verification, not parameters.** *Scaling test-time compute optimally* (ICLR 2025) finds compute-optimal allocation beats best-of-N by **4×**, and lets a small model outperform one **14×** larger — but only on problems where the small model has non-trivial pass rates, i.e. where a verifier can separate good from bad. The same paper's negative results matter: naïve strategies lose to plain sampling. Verification is the scarce resource, so the architecture should minimise verifier calls.

Each of those is a chapter. The rest of this folder is the design that follows from them.

---

## 1.5 What this thesis would be wrong about

Written before the experiments, so it cannot be retrofitted:

1. **If a plain text-file memory matches a validated store**, the schema is overhead. Test: same agent, same tasks, one with `MEMORY.md`, one with the store. If the gap is under 2 pp, delete the store.
2. **If staleness is rare in practice**, invalidation is solving a problem that does not occur. Test: measure the fraction of retrievals whose validity key changed since write, on real sessions. Below 5% and the mechanism is not worth its complexity.
3. **If the win comes entirely from injection** (row 1 of §1.1) and not from *self-generated* entries, then the model does not need a memory controller — it needs a better prompt. Test: compare injected vs self-written entries at matched content.
4. **If verifier calls dominate cost and cannot be reduced**, reasoning-by-search is unaffordable regardless of how small the model is.

**Next:** [02 — Architecture](02-architecture.md).
