# 1 — What Day 1 Missed

> A design review of day 1, written by the person who wrote day 1, with the 2025–26 evidence that has since arrived. Six errors, three of them load-bearing.

Day 1's documents are internally consistent: the numbers cross-check, the constraints propagate, the falsifiers are stated. That is exactly why its errors are worth writing down — a coherent design built on a wrong premise is more dangerous than an incoherent one, because nothing inside it can catch the mistake.

Here is what the outside world caught.

---

## The six errors

| # | Day 1's claim | Status | The evidence | Day 2's answer |
|---|---|---|---|---|
| **E1** | "The binding resource is **bytes of KV per token**" | Half right | It is bytes **streamed per decode step**, which grows with context | Sparse reading ([02](02-the-reading-problem.md)) |
| **E2** | "Never trade exact recall for throughput" | **False dichotomy** | NSA matches/exceeds full attention with perfect NIAH; DSA ships at frontier | Exact attention over a *selected subset* ([03](03-architecture.md)) |
| **E3** | "Prefix caching converts a compute problem into a memory problem" | True, but it stops one step early | vLLM/SGLang hash *prefix + block*; any mutation invalidates the whole suffix | Append-only delta protocol ([04](04-context-protocol.md)) |
| **E4** | "Compaction destroys prefix caches" | Correct diagnosis, no remedy | Day 1 told the harness not to compact; the harness compacts because context grows | Remove the *reason* to compact: cheap 1 M reading |
| **E5** | "Looped depth: no evidence at 4 B" | **Overtaken** | 3.5 B recurrent-depth reaching 50 B-equivalent compute; LoopLM 1.4 B ≈ 12 B | Depth as a runtime dial ([03 §3.5](03-architecture.md)) |
| **E6** | "Muon is not the place to discover optimiser bugs" | **Overtaken, with a twist** | Kimi K3 and DeepSeek V4 use Muon; ~2× pretraining efficiency — but the advantage **reverses** after large pretraining budgets | Muon for base, AdamW from SFT ([06 §6.6](06-training.md)) |

Three of these (E2, E5, E6) are facts that changed. Three (E1, E3, E4) were analytical misses that the 2025–26 systems literature makes obvious in hindsight. The analytical ones are the more embarrassing, so they go first.

---

## E1 — Day 1 measured the wrong quantity

Day 1's central quantity is `20,480 bytes per token`. It is correct, it is well derived, and it is the wrong number to optimise.

Here is the number day 1 never computed. **Decoding one token requires streaming the entire attention cache**, because every query attends to every key:

| Context | Day 1 cache read per decode token | Time at 3.35 TB/s | Day 1 weights read | Which dominates? |
|---|---|---|---|---|
| 8 k | 156 MB | 0.05 ms | 4.08 GB → 1.22 ms | weights |
| 64 k | 1.25 GB | 0.37 ms | 1.22 ms | weights |
| 256 k | **5.00 GB** | **1.49 ms** | 1.22 ms | **cache** |
| 1 M | **20.0 GB** | **5.97 ms** | 1.22 ms | **cache, 5×** |

At 64 k, day 1's design is fine and its reasoning holds. At 256 k — the context a monorepo actually needs — **the cache is the bottleneck, and it grows linearly forever.** Day 1 said "residency is the constraint" and sized the deployment for 64 k sessions, which is precisely the context at which the error is still invisible.

The correction is not "use a smaller cache". Day 1 already made the cache as small as MLA allows (11–20 KiB/token). The correction is that **you must not read all of it**, which is E2.

> The general lesson, worth more than the specific fix: *a resource bound must be expressed in the units the hardware charges for.* Day 1 counted bytes resident. The GPU charges for bytes moved.

---

## E2 — "Exact recall" was never a choice between two options

Day 1's principle P2 — *never trade exact recall for throughput* — framed the design space as:

- full attention: exact, O(n) reads;
- linear/recurrent attention: cheap, lossy.

Hybrid attention (the field's 3:1 answer) was rejected because the linear part compresses the record into a fixed state, and a compressed state cannot reproduce `_build_attention_mask` exactly.

That reasoning is sound **and the dichotomy is false**, because there is a third option that day 1 did not consider: keep every token in the cache, and attend exactly to a *selected subset* of it.

Native Sparse Attention (ACL 2025) does exactly this with three branches — compressed global, selected blocks, sliding window — and reports that the resulting model "maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning", with **perfect needle-in-a-haystack accuracy** and 11.6× decoding speedup at 64 k. DeepSeek's DSA variant ships it in a frontier model with a learned indexer selecting 2,048 tokens per query.

The distinction day 1 missed is the one that decides everything:

| Approach | What happens to token *i* | Exact recall of token *i*? |
|---|---|---|
| Linear / recurrent | folded into a fixed-size state, irreversibly | **no** — approximate by construction |
| Full attention | always read | yes |
| **Sparse exact** | **kept, addressable, read when selected** | **yes, when selected** |

Sparse attention's failure mode is "the indexer did not select it", which is a *ranking* problem with a measurable rate — not a *capacity* problem with a hard ceiling. That is a completely different risk profile, and day 1's plot collapsed the two.

Day 2's answer goes one step further than NSA, because code has something prose does not: **a deterministic structural index** (§E2a).

### E2a — The anchor idea that a general model cannot have

An indexer's job is to guess which 12 blocks out of 4,096 matter. For prose that is the whole problem. For code it is half-solved before the model runs: if the current context mentions `User`, then the block containing `class User` matters, and a parser knows where it is.

ISSTA 2026's *How Much Static Structure Do Code Agents Need?* measures exactly this: injecting deterministic structure (call and inheritance relations) inline, with no embedding index and no reranker, improves issue localisation — and the benefit **saturates at basic topology**, with denser graphs adding cost and little gain.

So Kestrel-2 adds a fourth, *non-learned* branch: a 256-token budget of always-attended anchor lines (definition signatures, imports, file outlines). It costs nothing to compute, cannot be trained away, and converts the indexer's worst failure ("I never saw the class definition") into an impossibility.

---

## E3 — The cache is an artifact, not a session detail

Day 1 treats the prefix cache as a runtime optimisation: a thing the serving stack does. Both major engines implement it as **content-addressed, prefix-hashed blocks**:

- vLLM: `hash(prefix tokens + block tokens) → KV block`, shared across requests, copy-on-write;
- SGLang: a radix tree with copy-on-write branching, extended to a three-tier hierarchy (HBM → host → external L3).

The prefix hash is the crucial mechanical detail. Because each block's key includes *all* tokens before it, **a single changed byte invalidates every block after it**. Not "degrades hit rate" — invalidates. Day 1 knew this in spirit ("compaction destroys the cache") but treated it as a harness etiquette problem.

It is not etiquette; it is a *contract*, and the model can be designed to satisfy it. That is what [04](04-context-protocol.md) does: never rewrite history, append deltas, and the cache survives by construction.

The payoff is large and already reported in production: SGLang's unified radix cache documents multi-turn hit rates near **98%**, and effective input throughput rising from 9.4 K to 145.5 K tokens/s (**15.5×**) once a third cache tier is added. Day 1 assumed this worked; day 2 designs for it.

---

## E4 — Day 1 diagnosed compaction and then asked the harness to stop

Day 1's sharpest systems insight was that compaction destroys the cache, and it followed with a constraint: *be append-only between compactions*.

That is a request, not a design. The harness compacts because context grows without bound, and it grows because every turn adds tool output. Asking it to stop is asking for OOM.

The alternative is to make growth *affordable*: if reading a 500 k-token prefix costs 4,096 tokens of attention per layer instead of 500,000, there is no reason to compact. Compaction exists to control a cost that sparse reading has already removed.

> Day 2's protocol makes context monotonic and cheap. Day 1's makes it small and expensive. The first is a design; the second is a negotiation with the runtime.

---

## E5 — The looped-depth evidence arrived

Day 1 listed looped depth as "attractive for reasoning-per-parameter, but no evidence at 4 B" and relegated it to an ablation. That was the right call in early 2025 and the wrong one in late 2026:

| Work | Result |
|---|---|
| Geiping et al., *Recurrent Depth* (2025) | 3.5 B model, 800 B tokens; iterating the block at test time improves reasoning "sometimes dramatically", up to compute equivalent to 50 B |
| Ouro / LoopLM (2025–26) | 1.4 B and 2.6 B looped models match up to 12 B; 7.7 T tokens; gains come from **knowledge manipulation, not capacity** |
| Iso-depth scaling laws (NeurIPS 2026) | Quantifies the discount: recurrence-equivalence exponent **φ = 0.46**; truncated BPTT makes it worse (0.38), hyperconnections better (0.65) |

Two details from that literature decided Kestrel-2's design more than the headline numbers:

1. **Looping buys manipulation, not knowledge.** Exactly the right trade for a 4 B model whose weakness is reasoning, not facts.
2. **φ = 0.46 is a discount, not a free lunch.** A loop recovers roughly half of what a real layer would give at equal compute. So the design must *price* it — and it does, in [06](06-training.md), by paying for depth only in a short adaptation phase rather than across all 1.2 T tokens.

The second point is the one day 1 would have got wrong by being enthusiastic. Day 2's recurrent block is a bet with a stated exchange rate.

---

## E6 — Muon, and the reversal nobody quotes

Day 1 rejected Muon with "1.2 T tokens is not the place to discover optimiser bugs." By late 2026 that is no longer defensible: Kimi K2 used MuonClip, Kimi K3 and DeepSeek V4 use Muon, and reported pretraining gains are around **2× compute efficiency** over AdamW.

But the interesting part is the part that is not in the marketing. *Pretraining Data Scale Reverses the Muon–AdamW Fine-Tuning Gap* (Nov 2026) finds that beyond a sufficiently large pretraining budget, **AdamW-pretrained models fine-tune better than Muon-pretrained ones**, and that the crossover happens at progressively smaller token/parameter ratios as models get larger.

A design document that adopted Muon everywhere because "frontier labs use it" would walk straight into that. Kestrel-2 uses Muon for the base run and **switches to AdamW at SFT**, and [06](06-training.md) pre-registers an ablation for the crossing point. This is the single best example in either document of why "what do frontier labs do" is not a substitute for reading the ablations.

---

## What day 1 got right

Not everything needs replacing, and saying which parts survive is part of the review:

- **MLA, and the parameter-dominance argument.** Fewer parameters than GQA-4 *and* 4× less cache. Still true, still the reason the cache is 11 KiB instead of 80.
- **Decoupled RoPE with a fused `W_DKV`.** Mandatory, for the reason day 1 gave (you cannot rotate a cached vector), and the fused projection is kept exactly as the torch disagreement forced it.
- **Dense, not MoE.** Rollout cost and routing stability still dominate at 4 B. Kestrel-2 changes the *recurrent* axis, not the expert axis.
- **Diff output, not file rewrites.** A 33× token reduction that also happens to be the cache-friendly format.
- **Verifier-first post-training and the reward-hacking catalogue.** Unchanged in substance; extended in [07](07-post-training.md).
- **The falsifier discipline.** Every document ends with what would kill it. That habit is why this document could be written at all.

---

## The uncomfortable summary

Day 1's architecture is right about *what* to optimise and wrong about *how the hardware charges for it*. It made reading cheap in FLOPs and left it expensive in bytes, at exactly the context lengths where the product needs to operate. Fixing that changes the attention mechanism, the context protocol, and the deployment shape — but leaves the model's skeleton, its tokeniser, and its entire post-training plan intact.

Which is the useful outcome of a review: three assumptions replaced, not a redesign.

**Next:** [02 — The reading problem](02-the-reading-problem.md).
