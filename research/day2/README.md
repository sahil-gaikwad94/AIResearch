# Kestrel-2 — Day 2

> Day 1 asked *how do we make reading cheap?* and answered with caching. Day 2 asks the harder question: **cheap relative to what?** A cached prefix still has to be *streamed* on every token, and day 1 never measured that. Kestrel-2 keeps every byte of the record and reads it sparsely instead of compressing it away.

**Spec:** 4,125,828,176 parameters · 20 unique layers (2 prelude + 16 recurrent + 2 coda) · effective depth 20–68 · MLA (512 + 64) · 11,520 B of KV per token at FP8 · 256 k native context, 1 M extended · sparse attention with deterministic structural anchors.

Every number below is produced by [`tools/count_params_v2.py`](tools/count_params_v2.py) and independently asserted against the instantiated model in [`kestrel2/model.py`](kestrel2/model.py). Both run.

---

## The three inversions

**1. Day 1 optimised the wrong half of the cost.** Caching makes prefill *compute* free. It does not make *bandwidth* free: at 256 k context, day 1's decode step streams **5 GiB of cache per generated token** — more than the 4 GB of weights. The model is not compute-bound at long context; it is memcpy-bound. Measured against day 1, Kestrel-2 streams **65× less cache per decode token at 256 k**, and **96× less at 1 M**.

**2. "Full attention or lossy recurrence" was a false dichotomy.** Day 1 rejected hybrid attention because linear layers compress the record into a fixed state. True — but the third option, *exact attention over a selected subset*, keeps every token addressable and skips the rest. NSA reports matching or exceeding full attention with perfect needle-in-a-haystack, and DSA ships it in a frontier model. Day 1 bet against sparsity without knowing it was a bet.

**3. Depth should be a runtime dial, not a parameter tax.** Day 1 spent 40 unique layers to get depth 40. Kestrel-2 spends 20 unique layers and buys depth at generation time (R = 1…8 → depth 20…68). At 4.1 B parameters it can compute like a 52-layer model on hard steps and a 20-layer model on a formatting fix. The iso-depth literature puts a number on the discount: a loop is worth roughly `φ = 0.46` of a real layer in parameter efficiency — a real win, not a free one, and priced as such in [06](06-training.md).

---

## What changed, and what did not

| | Day 1 | Day 2 | Why |
|---|---|---|---|
| Attention | full, every layer | window + compressed + top-k blocks + **structural anchors** | 65× fewer bytes read at 256 k; exact on what is selected |
| KV per token | 20 KiB | **11 KiB** | 20 unique layers instead of 40 |
| Unique layers | 40 | **20** (16 looped) | depth becomes a dial |
| Effective depth | 40 | 20–68 (52 at default) | reasoning capacity without parameters |
| Re-prefill per turn | ~8 k tokens | **~300–800** | append-only delta protocol ([04](04-context-protocol.md)) |
| Optimiser | AdamW | Muon → AdamW at SFT | ~2× pretraining; the advantage **reverses** on fine-tuning |
| Training cost | 45,658 GPU-h | **43,097 GPU-h** | sparse attention makes long-context stages cheap |
| MLA, decoupled RoPE, dense-not-MoE, diff output, verifier-first RL | — | **unchanged** | day 1 got these right |

That last row matters. This is not a rebuttal of day 1; it is day 1 with three assumptions replaced by measurements.

---

## Verified

```
[1] parameters (inference) : 4,125,828,176  (4.1258 B)
    analytic expected      : 4,125,828,176
    + MTP (training only)  : 33,562,624  -> 4,159,390,800

[2] sparse(k=all,w=all) vs dense  : max|diff| = 0.000e+00
[3] incremental decode vs prefill  : max|diff| = 1.4e-06
[4] cache bytes r=1 … / r=4 ...    identical=True
[5] r=1 vs r=4 differ              : finite=True
```

Two implementations, one number, and four properties that are design claims rather than unit tests — including the one that matters most: **when the sparsity drops nothing, the model is bit-identical to dense attention.** That is what separates this from a heuristic.

---

## The reading order

| File | What it argues |
|---|---|
| [01 — What day 1 missed](01-what-day1-missed.md) | Six specific errors, each with the 2025–26 evidence that overturns it |
| [02 — The reading problem](02-the-reading-problem.md) | The thesis: compress the *reading*, never the *record*. Full arithmetic. |
| [03 — Architecture](03-architecture.md) | The four attention mechanisms, the indexer, the recurrent block, the block diagram |
| [04 — The context protocol](04-context-protocol.md) | Why append-only deltas are worth more than any architectural choice here |
| [05 — Parameters](05-parameters.md) | Hand-derived count, config, and the ratios that say whether the shape is sane |
| [06 — Training](06-training.md) | Dense→sparse warmup, loop adaptation, Muon's reversal, the 43 k GPU-h budget |
| [07 — Post-training](07-post-training.md) | Coverage-delta rewards, loop-budget RL, and the verifier as the product |
| [08 — Serving](08-serving.md) | The cache as a published artifact; concurrency, latency, and the 15× tiering effect |
| [09 — Evals and ablations](09-evals-and-ablations.md) | Nine pre-registered ablations that can each kill a decision |
| [10 — Decisions and risks](10-decisions-and-risks.md) | The decision log, the risk register, and **when day 1 was the better design** |

---

## The one number

> **11,520 bytes of KV per token, of which a decode step reads ~4,096 tokens' worth and skips the rest.**

Which yields: 2.81 GiB per 256 k session, 38 such sessions per H100 *after* the shared repo prefix is paid once, and a decode step that is bound by weights rather than by cache until roughly 1 M tokens — where day 1 had been cache-bound since 128 k.

**Next:** [01 — What day 1 missed](01-what-day1-missed.md).
