# Kestrel-3.1 — day 3, v2

**20.9 B parameters stored. 2.0 B read per token. Depth rented by the token. Signal from the test runner.**

v1's thesis was about memory: what you remember is only as good as your ability to tell whether it is still true. v2 generalises it to the whole stack, because the 2026 evidence makes the general form quantitative:

> **In code, generation is expensive and checkable. Every design decision should move work from the expensive side to the checkable side.**

That single sentence produces all five requested subsystems, and it also produces their ordering — verification first, because everything else is downstream of whether the check can be trusted.

## The result that organises everything

Self-play against a reference-free judge drives the judge's pass rate from **0.72 to 0.94** while true accuracy stays flat at **0.20**. A three-judge ensemble still accepts **55 %** of false positives.

Then one protocol change: **make the judge commit to its own answer before it sees the candidate.** False-positive rate goes **0.719 → 0.012**, discrimination to **0.96**.

Same model. Different order of operations. 60× less error.

That is why v2 spends its design budget on verification rather than on a better judge — and why `tools/sim_verify.py` can show, in arithmetic, that best-of-16 with a blind judge scores **0.280, below the 0.300 you get by not searching at all**.

## The model

```
embed → 16 × [ windowed MLA │ delta-rule memory │ 256 latent-width experts, top-6 ]
      → workspace × r  (4 layers, d_w=1024, r ∈ 1..8)
      → confidence head → head (tied)
```

| | |
|---|---|
| Total parameters | **20,865,312,258** |
| Active per token | **1,990,944,258** (9.54 %) |
| Expert sparsity | **1:42** (256 routed, 6 active, latent width 384) |
| Per-expert cost vs full width | **10.7× cheaper** |
| Session state | **4.2 MB** (562× smaller than a 256 k KV cache) |
| Forward compute, 4 k | **4.65 GFLOP** at r=1, **5.59** at r=8 |
| One deliberation pass | **0.236 GFLOP** — **12.3×** cheaper than a decoded token |
| Training | **28,761 GPU-h** + **12,000 CPU core-h**, 9.4 days on 128 GPUs, $43k–$86k |
| Weights at 4-bit | **11.5 GB** — fits 24 GB with 12.5 GB spare |

## The five subsystems, and the one sentence each

| | Subsystem | Claim | Evidence |
|---|---|---|---|
| 1 | [Sparse MoE](03-sparse-moe.md) | capacity is cheap if experts are narrow; 1:42 at the same layer budget | K2 sparsity law (48), K3 latent experts (56, 2.5× efficiency) |
| 2 | [Recurrent reasoning](04-recurrent-reasoning.md) | depth is a rental, decided by the verifier | Huginn 5× GSM8K, Looped Transformers, LOTUS 2.5–6.9× latency |
| 3 | [Execution-driven learning](05-execution-driven-learning.md) | the failure text is the most valuable token in the batch | FC-SWE +2.8/+4.3 pts, mini-harness −23.1 pp without execution |
| 4 | [Distillation](06-distillation.md) | distil the search, not the answer; don't train a teacher | GKD on-policy reverse KL, distillation scaling laws (ICML 2025) |
| 5 | [Self-improvement](07-self-improvement.md) | a loop is self-improving only if something outside can say no | rise-then-collapse, Echo Gap, CARE, Mind the Gap |
| | [Verification](08-verification.md) | the gate every other chapter depends on | commit-first FPR 0.719 → 0.012 |

## Verified, not asserted

Seven assertions in `kestrel31/model.py`, all passing:

```
[1] total   torch=20,865,312,258  analytic=20,865,312,258  expected=20,865,312,258
    active  analytic=1,990,944,258  expected=1,990,944,258  ratio=9.54%
[2] latent  sparsity=42.7  experts=256  top_k=6
    per-expert  latent=4,718,592  full=50,331,648  10.7x cheaper -> same layer budget, 2.67x the experts of a 96-expert design
[3] depth   params(r=1)=20,865,312,258  params(r=8)=20,865,312,258  identical
    flops/tok  r=1: 4.65 GFLOP   r=6: 5.32 GFLOP   (+14%)
[4] cost    one workspace pass=0.236 GFLOP  one CoT token=2.908 GFLOP  -> 12.3x cheaper per step
[5] moe     routed=24  expected=24  shared_always_on=True  grads=True
[6] loop    init drift=8.05e-05  p_pass=0.492  16-pass norm ratio=1.00  finite=True
[7] forward logits=(1, 8, 128)  states=2  p_pass=(1,)  ws=(1, 8, 32)
    session state 2,097,152 floats = 4.2 MB fp16, independent of context
all assertions passed
```

Test [6] is the one that would have been easy to skip and is the most informative: it proves the recurrent path starts as an **8.05e-05 identity** and stays stable over 16 unrolls (norm ratio 1.00). A recurrent model that starts at scale 1.0 trains for a few hundred steps and then diverges.

Test [2] caught a wrong claim during development: I had written that latent experts were `(d/d_lat)² = 114×` cheaper. They are `d/d_lat = 10.7×` cheaper, because the inner width is unchanged. The assertion computes the ratio from the config and refuses to let the documentation drift from it.

## Files

| | |
|---|---|
| [01 — Verification is the scarce resource](01-thesis.md) | the thesis, the five-part loop, three inversions, falsifiers |
| [02 — Architecture](02-architecture.md) | trunk + workspace + controller, the full ledger, three claims |
| [03 — The sparse MoE](03-sparse-moe.md) | 256 latent experts, routing, the serving tax, rejected designs |
| [04 — Recurrent reasoning](04-recurrent-reasoning.md) | depth as a rental, the ladder, why halting gates are avoided |
| [05 — Execution-driven learning](05-execution-driven-learning.md) | reward, failure-conditioned recovery, patch slicing |
| [06 — Distillation](06-distillation.md) | search→policy, verifier→head, teacher→data |
| [07 — Self-improvement](07-self-improvement.md) | the loop and the four gates that stop it eating itself |
| [08 — Verification](08-verification.md) | commit-first, the GV gap, error independence |
| [09 — Training](09-training.md) | 28,761 GPU-h, data mixture, stability, gates, restarts |
| [10 — Serving and risks](10-serving-and-risks.md) | 24 GB, latency, R1–R9 with probabilities, abandon triggers |
| [`kestrel31/model.py`](kestrel31/model.py) | reference implementation, 7 assertions |
| [`kestrel31/config.json`](kestrel31/config.json) | canonical configuration |
| [`tools/count_params_v31.py`](tools/count_params_v31.py) | parameters, FLOPs, budget, serving |
| [`tools/sim_verify.py`](tools/sim_verify.py) | the verification arithmetic — labelled as arithmetic |

```bash
.venv/bin/python research/day3/v2/kestrel31/model.py        # 7 assertions
.venv/bin/python research/day3/v2/tools/count_params_v31.py # cost model
.venv/bin/python research/day3/v2/tools/sim_verify.py       # verification arithmetic
```

## How this differs from v1, and from day 1/2

Nothing here is inherited as a mechanism. Three components coincide with v1 — windowed latent attention, delta-rule memory, fine-grained MoE — and each was re-derived, then changed:

| | v1 | v2 |
|---|---|---|
| Experts | 96 full-width, top-3, 1:32 | 256 latent-width, top-6, **1:42** |
| Depth | fixed 16 layers; recurrence rejected | 4-layer workspace, **r ∈ 1..8**, verifier-conditioned |
| Learning | offline data then RL on tests | execution → distillation → self-training, gated |
| Verification | a schema for memory entries | the organising principle of the system |
| Cost | 31,025 GPU-h | **28,761 GPU-h** + 12,000 CPU core-h |

The MoE and the memory carried over in spirit and were re-specified; the recurrence, the execution loop, the distillation pipeline, and the verification protocol are new, and the last one changes the meaning of the other three.

Day 1 and day 2 solved a different problem — cheap reading — and their mechanisms (product-key memory, sparse indexer, anchors) were re-examined and rejected again for the same reason as before: none of them can carry a `git commit`.

## Honest limits

- **The workspace may be 118 M wasted parameters (R1, 40 %).** Latent CoT underperforming explicit CoT beyond 1 B is a measured result; we are betting code-with-a-verifier is the exception.
- **The loop may not be safe (R2/R4).** Rise-then-collapse is real, KL does not prevent it, and our gates are an argument, not a measurement.
- **The verifier arithmetic is arithmetic.** `sim_verify.py` states its assumptions at the top of the file and in its output banner. Only the FPR values are measured — by someone else.
- **Batch-1 latency on older hardware is 1.6× worse at r=8.** Stated in §10 rather than hidden behind the B300 number.
- **Nothing is trained.** Every capability claim is an argument from another paper's measurement, plus counts from our own code.

**Start at:** [01 — Verification is the scarce resource](01-thesis.md).
