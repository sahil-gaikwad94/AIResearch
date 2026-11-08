# 08 — Retrieval and Long Context: Getting the Right Information In

*Read time: 60 minutes. Prerequisites: `02` §2.4–2.5.*

---

## 8.1 The two ways to give a model knowledge it wasn't trained on

1. **Put it in the context window** (long context / RAG).
2. **Put it in the weights** (fine-tuning, continual pretraining).

In 2026 the practical answer is: **retrieval into context for facts, fine-tuning for behaviour/format/domain style, and long context for documents you actually need whole.** They're complementary, and the engineering question is which is cheaper per unit of accuracy.

This chapter covers both, plus the uncomfortable findings about how models actually use long inputs.

---

## 8.2 RAG: the pipeline

**Retrieval-Augmented Generation** (Lewis et al. 2020) = retrieve relevant documents, put them in the prompt, generate an answer conditioned on them.

The pipeline is more moving parts than people expect, and each is a failure mode:

```
[1] INGEST      parse PDFs/HTML/tables → clean text
[2] CHUNK       split into passages (256–1024 tokens), with overlap + metadata
[3] EMBED       encode each chunk into a vector
[4] INDEX       store in a vector DB (+ keyword index for hybrid)
[5] QUERY       user question → rewrite/expand → embed
[6] RETRIEVE    top-k by similarity (dense) ∪ top-k by BM25 (sparse)
[7] RERANK      cross-encoder scores (query, chunk) pairs → keep top-n
[8] ASSEMBLE    order, dedupe, add citations, fit the budget
[9] GENERATE    answer with instructions to cite and to say "not found"
[10] EVALUATE   did it retrieve the right thing? did it answer correctly?
```

**Where it breaks, in order of frequency:**

1. **Chunking** — a fact split across a boundary is unretrievable. Tables and code are especially bad.
2. **Retrieval miss** — the embedding doesn't capture the query's intent (e.g. "what's our refund policy for EU customers?" vs. a doc titled "Returns — EMEA").
3. **Reranking skipped** — top-k dense retrieval has poor precision; a cross-encoder reranker is the single highest-ROI component.
4. **Context assembly** — putting 20 chunks in and hoping. Models attend to the beginning and end (`lost in the middle`, §8.7); ordering matters.
5. **The model ignoring the retrieved text** and answering from parametric memory (which is where hallucinations come from in RAG systems).
6. **Stale index** — the doc changed; retrieval returns the old version. Versioning is essential.

---

## 8.3 Embeddings and retrieval models

**Bi-encoder (dense retrieval):** embed query and document *independently* with the same model, compare with cosine/dot product. Fast (precompute all doc vectors), but the query and doc never "see" each other — a real expressiveness limit.

- **DPR** (Karpukhin et al. 2020) established the dual-encoder + hard-negative training recipe.
- **Contriever**, **E5**, **BGE**, **GTE**, **jina**, and the 2025–26 leaderboard leaders are all variations with better data and training objectives.
- **Matryoshka embeddings**: one vector whose prefixes are themselves usable embeddings, so you can truncate dimensions to trade accuracy for memory.

**Late interaction — ColBERT** (Khattab & Zaharia 2020): embed each *token* of query and doc, and compute **MaxSim**: for each query token, take the max similarity over doc tokens, then sum. Much more expressive than a single vector, with precomputation (docs are embedded offline). Storage is ~10–100× larger; modern variants (ColBERTv2, PLAID) compress heavily. Good default when precision matters and you can afford the index.

**Sparse/lexical — BM25** is not obsolete. It is exact-match-sensitive, cheap, and unbeatable for names, IDs, error codes, and rare terms. **Hybrid retrieval (dense ∪ BM25, fused with RRF)** is the production default, not a compromise.

**Cross-encoder rerankers**: take `(query, chunk)` jointly through a Transformer and output a relevance score. Far more accurate than bi-encoders, far slower — so you use them on the top 50–100 candidates only. This two-stage (recall then precision) pattern is the standard architecture.

---

## 8.4 Evaluation: the part everyone skips

You cannot improve a RAG system you can't measure. You need two separate metrics:

- **Retrieval quality**: recall@k (is the right chunk in the top k?), MRR, nDCG. Build a set of ~100 questions with known answer chunks. This is cheap and catches 80% of problems.
- **Answer quality**: correctness, faithfulness (is every claim supported by the retrieved text?), completeness, citation accuracy. LLM-as-judge is acceptable *if* calibrated against ~50 human labels.

**RAGAS-style metrics** (faithfulness, answer relevance, context precision/recall) are the common vocabulary. The failure mode of LLM-judged RAG evals is that the judge prefers fluent answers regardless of grounding.

**The diagnostic that matters:** when an answer is wrong, was the right chunk retrieved? If yes → generation/prompting problem. If no → retrieval problem. Teams routinely spend weeks on prompts when the problem is chunking.

---

## 8.5 Long context: how far it has actually come

| Year | Typical max | Mechanism |
|---|---|---|
| 2020 (GPT-3) | 2k | absolute learned positions |
| 2023 (GPT-4) | 8k–32k | RoPE + fine-tuning |
| 2024 (Claude 3, Gemini 1.5) | 200k–1M | RoPE scaling, ring attention |
| 2025 (Llama 4, Gemini 2.5) | 1M–10M (advertised) | CP/ring attention at inference, sparse attention |
| 2026 (Qwen3.5, Gemini 3-class) | 262k–1M native, 10M experimental | hybrid linear + sparse attention, NoPE |

Mechanisms (detail in `02` §2.4, `03`):
- **RoPE scaling** (PI, NTK-aware, YaRN, LongRoPE/LongRoPE2) — make the position encoding behave at unseen lengths.
- **Continued pretraining on long documents** — without this, the encoding fix doesn't generalise.
- **Ring attention / context parallelism** — split the sequence across GPUs, pass K/V (or Q) around a ring, compute attention blockwise. Meta reported <1 minute for 1M tokens on one H100 host, and 10M tokens across 32 hosts using pass-KV / pass-Q variants.
- **Sparse attention** — DSA-style top-k selection makes long context affordable at inference (`02` §2.9).
- **KV cache compression** — quantize, evict, or share across layers.

---

## 8.6 The uncomfortable findings

**(1) "Lost in the Middle"** (Liu et al. 2023). For multi-document QA and key-value retrieval, performance is highest when the relevant information is at the **beginning or end** of the context, and degrades significantly in the middle — *even for models explicitly built for long context*. The curve is a U-shape. This has been replicated many times and is the single most important practical result about long context.

**Implication:** how you order retrieved chunks matters as much as how many you include. Put the most important evidence first (or last).

**(2) "Effective context" < "advertised context".** Benchmarks like RULER and LongBench show that most models degrade well before their claimed limit, especially on multi-hop and aggregation tasks (as opposed to needle retrieval).

**(3) Needle-in-a-haystack is a weak test.** A model can retrieve a single sentence from 1M tokens and still fail to reason over a 50k-token contract. NIAH measures *retrieval*, not *comprehension*. Vendors love it because models pass it.

**(4) Context rot.** Performance degrades as context length grows even when the task is unchanged and the relevant text is present — plausibly due to attention dilution and training-data length distribution. Agents feel this constantly (`07` §7.5).

**(5) Long context ≠ free.** Attention compute is quadratic (mitigated but not eliminated), KV cache is linear and large, and prefill latency at 1M tokens is tens of seconds to minutes.

---

## 8.7 RAG vs long context: how to actually choose

| Situation | Better choice |
|---|---|
| Fact lookup in a large corpus | RAG (index once, retrieve per query) |
| Whole-document reasoning (contract review, codebase understanding) | Long context, possibly with retrieval to select the document |
| Data that changes hourly | RAG (no retraining) |
| Style/format/domain behaviour | Fine-tuning (LoRA), not retrieval |
| A handful of stable documents, high query volume | Long context with **prefix caching** (`09`) — cache the document's KV once, reuse across queries |
| Millions of documents | RAG, always |
| Multi-hop questions across 50 docs | Hybrid: retrieve → long-context window over the shortlist → reason |

**The economic crossover** is roughly: if the corpus fits in context and you'll ask many questions about it, **prompt caching** makes long context cheaper than building a retrieval stack. If the corpus is large or dynamic, retrieval wins. Most production systems in 2026 do both: retrieve a shortlist, then reason over it in a long window.

---

## 8.8 Memory vs retrieval vs context (they're different)

- **Context** = what's in the window right now (ephemeral).
- **Retrieval** = fetching from an external store into context (stateless, query-driven).
- **Memory** = a persistent, agent-managed store that is *written* by the system over time (`07` §7.7).

Conflating them causes bad architecture. A vector DB is not a memory system; memory requires write policies, consolidation, forgetting, trust/provenance, and reconciliation of contradictions. The 2026 memory benchmarks show current systems failing at exactly those.

---

## 8.9 Common misconceptions

- **"1M context makes RAG obsolete."** No: cost, latency, lost-in-the-middle, and dynamic corpora keep RAG necessary. But prompt caching has genuinely eaten part of RAG's territory.
- **"Dense retrieval beats BM25."** They fail differently. Hybrid is standard.
- **"Chunking is a detail."** It's the most common root cause of RAG failure.
- **"If it passes needle-in-a-haystack, long context works."** It measures retrieval, not reasoning.
- **"Embeddings understand meaning."** They capture distributional similarity; negation, numeric comparison, and rare entities are classic failure cases.

---

## 8.10 If you remember five things

1. RAG is recall-then-precision: hybrid retrieval + a cross-encoder reranker is the high-ROI architecture.
2. Order your context: the middle is where information goes to die.
3. Long context is a position-encoding problem *and* a long-document training-data problem.
4. Effective context is much shorter than advertised; measure it on your task, not on NIAH.
5. Retrieval, context, and memory are three different things; don't build one and call it the others.

---

## 8.11 Self-check

1. Why does a bi-encoder lose expressiveness compared to a cross-encoder, and how does the two-stage pipeline exploit that?
2. Sketch the "lost in the middle" experiment and its result.
3. You have 200 stable PDFs and 10k queries/day. RAG or long context? What does prompt caching change?
4. Give two tasks where needle-in-a-haystack success would tell you nothing useful.
5. Why is BM25 still in production alongside dense retrieval?

**Next:** [`09-efficiency-and-inference.md`](09-efficiency-and-inference.md)
