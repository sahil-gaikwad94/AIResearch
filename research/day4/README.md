# Day 4 — Kestrel-4 × VEXX 1.5

Day 3.1 asked how to build memory whose claims can be checked. Day 4 asks the next question,
and it is a different one:

> What changes when the verifier is **inside the decode loop** instead of after it?

Kestrel-3.1 verified a *document*. If verification fails, the sample is thrown away and the
tokens are re-spent. Kestrel-4 verifies a *token*: the constraint system is a live object
that masks the next distribution, splices deterministic fields without sampling, and rewinds
to the clause that caused a conflict instead of to the start of the patch. The neural side
gets the state capacity to hold that system; the symbolic side gets the authority to say no.

**21,248,719,490 parameters stored, 2,071,178,882 read per token (9.75 %).** Both numbers are
asserted twice: analytically in `kestrel4/model.py` and by instantiating the model on the
`meta` device and summing `p.numel()`. `kestrel4/config.json` is generated from `FULL` and
checked to round-trip, so the config file cannot drift from the ledger.

```bash
python research/day4/kestrel4/model.py        # 11 assertions — ledger, absorption, routing, jumpback, proof
python research/day4/tools/count_params_v4.py # params, FLOP/token, GPU-hours, serving bytes
python research/day4/tools/sim_verify_v4.py   # retry correlation, poison rates, CDCL vs whole-patch retry
```

## The ledger

| Component | Per layer | × layers | Stored | Active |
|---|---:|---:|---:|---:|
| Local windowed MLA (w = 4,096) | 28,574,752 | 12 | 342,897,024 | 342,897,024 |
| Global NoPE sparse MLA | 27,919,392 | 4 | 111,677,568 | 111,677,568 |
| Householder memory (n_h = 2) | 12,722,176 | 16 | 203,554,816 | 203,554,816 |
| Subspace-grouped MoE | 1,245,839,624 | 16 | 19,933,433,984 | 755,893,376 |
| Code-JEPA workspace | — | 1 | 120,084,482 | 120,084,482 |
| Embedding (tied) + final norm | — | — | 536,875,008 | 536,875,008 |
| **Total** | | | **21,248,719,490** | **2,071,178,882** |

Per-token active share is 9.75 %, of which 36.5 % is the MoE (6 of 256 experts per layer),
31.8 % is attention and memory, 25.9 % is the tied embedding, and 5.8 % is the workspace.

## Design commitments

1. **Soundness is a mechanism, not a number.** `FPR = 0.000` is not a measurable claim; it is
   a statement about sample size. Day 4 replaces it with four tiers of rejection, each of
   which fails closed, and reports the residual false-accept rate of the *whole stack*
   (0.000324 at the measured gate) rather than of one component.
2. **Always-on capacity must buy state, not just scores.** The Householder memory (203.6 M)
   and the workspace (120.1 M) are 1.5 % of parameters but carry the session state (4.19 MB)
   and the hypothesis codebook. Everything else is rented per token.
3. **Every choice names the alternative it beat.** A decision without a rejected alternative
   is a preference.
4. **Every claim names its falsifier.** §08 pre-registers C1–C9 with metrics, thresholds and
   kill criteria before any training run.

## Chapters

| # | File | Question it answers |
|---|---|---|
| 01 | [what-day3-missed](01-what-day3-missed.md) | Which claims in day 1–3.1 did not survive an audit, and why did the assertions not catch them? |
| 02 | [architecture](02-architecture.md) | 12 local MLA + 4 global NoPE: what each layer type is for, and the absorption identity |
| 03 | [memory](03-memory.md) | Why Householder reflections and not the delta rule; what 4.19 MB of state can represent |
| 04 | [verification](04-verification.md) | The four tiers, the measured FPR ladder, and the five runtime holes closed |
| 05 | [execution-driven-learning](05-execution-driven-learning.md) | Verified execution as the only training signal that cannot be argued with |
| 06 | [sparse-moe](06-sparse-moe.md) | Why 8 × 32 with a two-stage router beats 256 flat, and the rank argument |
| 07 | [reasoning-and-decoding](07-reasoning-and-decoding.md) | CRANE dual mode, 0-DoF splicing, CDCL jumpback: the decode budget |
| 08 | [safety-and-falsifiers](08-safety-and-falsifiers.md) | Pre-registered C1–C9, failure modes, and what would kill the design |
| 09 | [training-serving-and-ablations](09-training-serving-and-ablations.md) | 25,067 GPU-hours, 11.69 GB at 4-bit, 33,330 tok/s, and what each ablation costs |
| 10 | [day5-preview](10-day5-preview.md) | Three directions for Kestrel-5 — forecast, labelled |

## What this is not

No weights were trained. Every throughput, GPU-hour and false-accept figure here is either
an analytic model (`count_params_v4.py`) or a simulation with its assumptions stated
(`sim_verify_v4.py`); §09 lists which numbers are measured, which are modelled, and which are
forecast. The VEXX 1.5 runtime claims in §04 are the only ones backed by executed tests:
71 unit tests and 12 soundness tests in Rust, 429 in Python.
