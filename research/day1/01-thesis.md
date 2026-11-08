# 1 — The Thesis

> Most architecture documents start from "what do good models look like". This one starts from a measurement: **what does a coding agent actually spend compute on?** The answer inverts several defaults, and that inversion is the whole design.

---

## 1.1 The workload is not chat

Take a realistic agentic coding session and count tokens. Not a benchmark — an actual loop: read files, run tests, read the failure, edit, repeat.

**Session model used throughout this document:**

| Quantity | Value |
|---|---|
| Repo digest + system prompt at turn 1 | 30,000 tokens |
| New content per turn (file reads, tool output, diffs) | 8,000 tokens |
| Context window | 64,000 tokens |
| Turns | 20 |
| Output per turn (patch + explanation) | 600 tokens |
| Thinking per turn | 300–16,000 tokens (varies) |

Total **unique** tokens read: 30,000 + 20 × 8,000 = **190,000**.
Total tokens **served** as prefill without caching: the context is re-sent every turn, so roughly Σ(context size) ≈ **1.04 M**.
Total tokens **generated**: 20 × (600 + thinking) ≈ **18 k–340 k**.

Now price that in FLOPs, using the model's real numbers (prefill at 64k context costs 1.21 × 10¹¹ FLOPs/token; decode costs 2N = 8.2 × 10⁹ FLOPs/token):

| Thinking / turn | Tokens generated | Prefill FLOPs | Decode FLOPs | Prefill : decode | **Prefill share** |
|---|---|---|---|---|---|
| 300 | 18 k | 2.30 × 10¹⁶ | 1.5 × 10¹⁴ | 156 : 1 | **99.4 %** |
| 1,000 | 32 k | 2.30 × 10¹⁶ | 2.6 × 10¹⁴ | 88 : 1 | 98.9 % |
| 4,000 | 92 k | 2.30 × 10¹⁶ | 7.5 × 10¹⁴ | 31 : 1 | 96.8 % |
| 16,000 | 332 k | 2.30 × 10¹⁶ | 2.7 × 10¹⁵ | 8.5 : 1 | 89.5 % |

**The model spends 89–99% of its compute reading.** Even with 16,000 tokens of reasoning per turn — longer than most models ever produce — reading dominates nine to one.

And this is the *cached* case. Without prefix caching the prefill term is 5.5× larger (1.04 M tokens instead of 190 k), and the prefill share is 99.9% regardless of how much the model thinks.

Whatever the model is, it is overwhelmingly an instrument for **ingesting repository state**. Design accordingly.

---

## 1.2 The consequence that inverts the 2026 consensus

The field's answer to "long context is expensive" is hybrid linear attention: replace 70–80% of full-attention layers with gated linear attention or a DeltaNet variant, cutting per-token cost from O(context) to O(1) and the cache from O(context) to a fixed state.

That reasoning is correct for **training** and for **uncached inference**, and it is the wrong optimisation for **cached agentic inference**, for a reason that is easy to miss:

> **Prefix caching converts a compute problem into a memory problem.**

The repository prefix is byte-identical across turns. A production harness caches its KV, so each unique token is prefilled *once per session*, not once per turn. The quadratic cost is paid on 190 k unique tokens and then amortised over 20 turns of reuse. What you cannot amortise is the **cache**: 64 k tokens of KV must stay resident for the whole session, for every concurrent session.

So the binding resource is:

> **bytes of KV per token** — not attention FLOPs per token.

Which means hybrid linear attention is buying you something you already have (cheap prefill of cached prefixes) and charging you something you cannot afford (exact recall). §2.3 does the arithmetic; §2.4 shows MLA making the cache 24× smaller than MHA and 4× smaller than GQA-4 *while also using fewer parameters than GQA-4*.

**The second-order insight, which is a systems fact rather than an architectural one:** compaction destroys prefix caches. If your harness summarises the context every five turns, the prefix changes, the cache is invalidated, and you are back to paying full quadratic prefill on every turn. The memory saving that makes full attention affordable *depends on the harness being append-only between compactions*. That is a design constraint on the serving stack, and it belongs in this document because a model that is right and a harness that is wrong is a system that is slow.

---

## 1.3 The five constraints

Everything in the design descends from these.

| # | Constraint | Why it binds for code | What it rules out |
|---|---|---|---|
| **C1** | **Exact recall over long spans** | An identifier defined 40 k tokens ago must be retrieved *exactly*. A paraphrase is a compile error. | Aggressive recurrence, small fixed state, sliding-window-only attention |
| **C2** | **Read-dominant compute profile** | 89–99% of FLOPs are prefill (§1.1) | Optimising decode throughput; large per-token cost; anything that inflates sequence length |
| **C3** | **Cheap rollouts** | RL needs thousands of generations per update; rollouts must be affordable *per token generated* | Huge active parameter counts; models that need 8 GPUs to sample |
| **C4** | **Automatic verification** | The training signal is tests, not preferences. This is a gift — use it | Training objectives that cannot consume a pass/fail signal |
| **C5** | **Long-horizon reliability** | A 100-step task needs ~99.9% per-step accuracy to complete | Anything that improves capability while worsening calibration |

C1 and C3 shape the architecture. C4 shapes the data and RL plan. C5 is *not* an architecture problem, and pretending otherwise is the most common mistake in documents like this one (§1.5).

---

## 1.4 Seven design principles

**P1 — Optimise the bound resource, and say which one it is.**
Here: KV bytes per token. Every architectural choice is scored against it. MLA exists to make it 20 KiB instead of 480 KiB. The tokeniser exists to make the token count (and therefore the cache) smaller for the same code.

**P2 — Never trade exact recall for throughput.**
This is a deliberate rejection of the consensus, and it is falsifiable: if a hybrid model with a fixed 37 MB state matches full attention on repo-level tasks at equal KV budget, P2 is wrong and the design should change. Ablation A2 in [08](08-evals-and-ablations.md) tests exactly this.

**P3 — Depth over width at equal parameters.**
Reasoning quality in small models tracks depth more than width, and depth costs no extra KV (each layer has its own fixed-size-per-token cache; width multiplies it). 40 × 3072 beats 28 × 4096 at equal parameters, and the cache is identical.

**P4 — Every token of sequence length is a tax you pay 40 times.**
At 64 k context, training FLOPs are 4.9× the parameter FLOPs (§5.3). A tokeniser that emits 18% fewer tokens cuts the entire budget by ~15% *and* shrinks every cache. This is why [04](04-tokenizer.md) exists and why it was designed before the block was finalised.

**P5 — Make the model a good RL substrate.**
RL is where planning comes from. A good substrate is: small enough to sample at 3,000 tok/s/GPU, stable enough that 20 different reward shapes do not destabilise it, and deterministic enough that a seed reproduces a rollout. This is the real argument against MoE at this size.

**P6 — Serve the loop, not the message.**
Output format is an architectural surface. The model should emit diffs, not file rewrites (10× fewer tokens, cache-friendly, reviewable). Tool calls should be a fixed grammar, not free text parsed by regex. Every one of these choices is worth more than 20% of parameters.

**P7 — No exotic block without an ablation that could kill it.**
There is no looped depth, no mixture-of-depths, no diffusion LM, no learned positional table, no per-layer attention variant. Each is a plausible win; none is measured at this scale. They are in [08](08-evals-and-ablations.md) as ablations, not in the spec.

---

## 1.5 What architecture can and cannot buy

The most useful table in this document, because it prevents the design from over-claiming.

| Capability | Mostly bought by | Architecture's contribution |
|---|---|---|
| **Knowing libraries and APIs** | pretraining data + parameters | moderate (capacity) |
| **Writing syntactically valid code** | tokeniser + data | high (tokenisation is the single largest lever) |
| **Reading a 100 k-token repo accurately** | context + attention | **high — this is what the design optimises** |
| **Multi-file reasoning** | context + depth | high |
| **Planning a 20-step task** | **RL in environments** | low |
| **Recovering from a failed test** | **RL + verification** | low |
| **Knowing when to stop / ask** | **RL + calibration** | very low |
| **Long-horizon reliability (99%)** | post-training + harness | **none** |

Read that bottom half carefully. A model can be architecturally excellent and still be a bad agent, because planning and reliability are bought with environments and RL, not with blocks. This design spends its architectural budget on C1 and C2 — the two constraints architecture can actually move — and spends its training budget on C4, where the leverage is.

---

## 1.6 The one number

Everything in §2 collapses into a single figure:

> **20,480 bytes of KV per token, at FP8, across 40 layers.**

Which yields, immediately:

- 64 k context = **1.25 GiB** of cache per session;
- 256 k context = **5.0 GiB**;
- an 80 GB GPU holding 4 GB of FP8 weights can keep **~40 concurrent 64 k sessions** resident;
- the same model with MHA would need 480 KiB/token — 24× more — and would fit **1.6** sessions.

That is the entire argument for MLA, and the reason this model can afford full attention at every layer while a GQA-based model of the same size cannot.

---

## 1.7 What would falsify the thesis

Stated now, before any of the design is defended:

1. **If real agent harnesses have low prefix-cache hit rates** — because they compact, reorder, or re-render context every turn — then prefill compute matters again, hybrid attention becomes correct, and §1.2 is wrong. Testable immediately: instrument a real harness and measure bytes of prefill actually served versus unique tokens. **This is the highest-value measurement in the entire design, and it costs a day.**
2. **If a hybrid model with a 37 MB fixed state matches full attention on repo-level tasks**, C1 and P2 are wrong.
3. **If RL turns out to be the dominant capability lever even at 4 B** — i.e. a 4 B model with heavy RL beats a 30 B model with light RL on real repo tasks — then the size choice is vindicated but the architectural emphasis is misplaced, and the next iteration should spend everything on environments.
4. **If the tokeniser gain is under 8%**, P4 is weaker than claimed and the vocabulary should shrink to 65 k to save embedding parameters.

Each of these is an experiment, not an opinion, and three of the four are cheap.
