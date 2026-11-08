# Day 1 — Kestrel-Coder-4B

**A small coding model designed as a high-recall, prefill-cheap, RL-trainable reader.**

| | |
|---|---|
| **Parameters** | 4.083 B (inference) / 4.102 B (training, MTP heads included) |
| **Shape** | 40 layers × 3072 wide, 24 heads × 128 |
| **Attention** | MLA, latent 448 + 64 rope, full attention **at every layer** |
| **FFN** | SwiGLU, 7680 wide |
| **Context** | 64k native, 256k extended |
| **KV cache** | 20 KiB/token at FP8 → 1.25 GiB at 64k |
| **Pretraining** | 1.2 T tokens, staged 4k → 16k → 64k |
| **Budget** | ~45,700 GPU-hours ≈ $68k–137k |
| **Serving** | 2.1 GB at 4-bit; ~8 s per agent turn on one H100 |

Nothing here is exotic on its own. The design is the *combination*, and the combination follows from one measurement about what coding agents actually spend compute on.

---

## The five-line version

1. **A coding agent is a reader, not a writer.** Across a session, 89–99% of the model's FLOPs go into ingesting repository content and the rest into emitting code. Every leaderboard optimises the small part.
2. **Prefix caching turns that reading problem from compute into memory.** The repo is re-read every turn but only ever prefilled once, so what binds is *KV bytes per token*, not attention FLOPs.
3. **Therefore the 2026 consensus is wrong for this workload.** Hybrid linear attention exists to make long context cheap in compute; with caching, long context is cheap in compute anyway. What it costs you is exact recall — and mis-remembering an identifier is a fatal error in code.
4. **So: full attention at every layer, compressed 12× by MLA, cached at FP8.** MLA turns out to be *cheaper in parameters* than GQA-4 here (21.2 M vs 22.0 M per layer) while using 4× less cache. That is not a trade-off, it is a domination, and it is the reason the design is shaped this way.
5. **Planning is not an architectural property.** Depth helps; planning comes from RL in verifiable environments. The architecture's job is to be a *cheap, stable RL substrate* — which is why 4 B dense beats 10 B-sparse-1.6 B-active for this use, and why the tokeniser got as much attention as the attention.

---

## Files

| File | What it contains |
|---|---|
| [01-thesis.md](01-thesis.md) | The argument from first principles: what a coding model actually does, the five binding constraints, seven design principles, and an honest table of what architecture can and cannot buy |
| [02-architecture.md](02-architecture.md) | The full spec: block diagram, MLA math, why decoupled rope, cache arithmetic, depth-vs-width, MTP, and the rejected alternatives with reasons |
| [03-config-and-parameters.md](03-config-and-parameters.md) | Hand-derived parameter count, layer-by-layer, and the annotated `config.json` |
| [04-tokenizer.md](04-tokenizer.md) | A code-aware tokeniser: whitespace, digits, identifiers, structural tokens, and how to measure whether it paid off |
| [05-training.md](05-training.md) | Data mixture, curriculum, compute budget with attention accounted for, μP, stability, infrastructure |
| [06-post-training.md](06-post-training.md) | SFT and RL: environments, verifiers, and a serious treatment of reward hacking (the model *will* edit the test) |
| [07-serving.md](07-serving.md) | KV math, prefix caching, prefill/decode arithmetic intensity, quantization, latency and cost per turn, deployment shapes |
| [08-evals-and-ablations.md](08-evals-and-ablations.md) | The eval suite that is not contaminated, and eight pre-registered ablations with decision rules |
| [09-decisions-and-risks.md](09-decisions-and-risks.md) | Decision log with rejected alternatives, risk register, and what would make me abandon this design |

## Code

| File | Purpose | Status |
|---|---|---|
| [`kestrel/config.json`](kestrel/config.json) | Canonical config | — |
| [`kestrel/model.py`](kestrel/model.py) | Reference PyTorch implementation (MLA, sinks, MTP, KV cache) | **runs**, all assertions pass |
| [`tools/count_params.py`](tools/count_params.py) | Analytic parameter / KV / FLOP / budget model | **runs**, dependency-free |

Reproduce every number in these documents:

```bash
python3 research/day1/tools/count_params.py        # params, KV, FLOPs, budget
python3 research/day1/kestrel/model.py             # needs torch; runs self-checks
```

The torch reference builds the real 4.08 B model on a `meta` device to count parameters, then runs forward, loss and KV-cache-equivalence tests on a shrunken config that uses the identical code path. Verified output:

```
params (inference) : 4.0834 B
params (training)  : 4.1023 B
  of which MLA     : 0.849 B
  of which FFN     : 2.831 B
cache equivalence  : max|full - incremental| = 1.91e-06
full cache/token   : 512 floats/layer x 40 layers = 20480 B/token at FP8
```

That last line is the design in one number.

---

## What this is not

It is not a claim that a 4 B model competes with a frontier model on hard reasoning. It is a claim that for **repo-scale coding with tools**, capability per dollar is dominated by recall, cache and rollout cost — and that a 4 B model built around those constraints is a better instrument than a 30 B model built for chat.

It is also not a recommendation for most teams. [09](09-decisions-and-risks.md) says plainly when fine-tuning an existing open 4 B model is the better decision, and why this design exists only for the cases where it is not.

---

## Read in this order

If you have ten minutes: [01-thesis.md](01-thesis.md) → the spec table above → [02 §2.3](02-architecture.md) (why full attention) → [09](09-decisions-and-risks.md) (what would falsify it).

If you are implementing it: [03](03-config-and-parameters.md) → [`config.json`](kestrel/config.json) → [`model.py`](kestrel/model.py) → [05](05-training.md) → [06](06-post-training.md).
