# Kestrel-3

**20.5 B parameters stored. 1.76 B read per token. 4.2 MB of memory per session.**

A coding model built on a thesis that came out of the 2026 memory literature rather than out of an architecture diagram:

> **The bottleneck is not what a model knows or how long it thinks. It is whether what it remembers is still true.**

---

## The finding that started this

*VibeMemBench* (2026) is the first benchmark to isolate memory from coding ability: 111 targets from 90 real repositories, judged by executable tests, where every target carries prior experience whose usefulness was verified by running code.

| Setting | Result |
|---|---|
| Verified experience **injected** into the prompt | **+1.1 to +4.5 pp**, fewer steps on all 5 solvers |
| Four real memory systems, building experience themselves | **11 of 12 solver × system pairings fail to beat no memory** |

The information is provably useful. The systems that produce it make things worse. MemTrace (2026), FAMA/Memora (2026) and Anthropic's agent-memory report all locate the same failure: **stale, unverified, over-generalised memory**, used confidently.

Everything else in this design follows from treating a memory entry as a *claim with a validity key* rather than a fact.

---

## The model

```
embed → 16 × [ windowed MLA │ delta-rule memory │ 96-expert MoE ] → head
```

| | Value |
|---|---|
| Total parameters | **20,485,187,584** |
| Active per token | **1,761,814,528** (8.60 %) |
| Layers | 16, `d = 4096` |
| Attention | MLA, window **4,096**, 576 B/token/layer |
| Long-term memory | delta-rule fast weights, `d_m = 512`, rank 256 |
| **State per session** | **4.2 MB** (vs 2.36 GB of dense KV at 256 k — **562× smaller**) |
| Experts | 96 per layer, top-3, `d_ff = 1024`, **1:32** activation |
| Compute | **13.82 GFLOP/token — at 4 k or at 256 k** |
| Decode | 1.84 GB/token, 0.55 ms, **1,821 tok/s** (batch 1) |
| Training | **31,025 GPU-h**, 1,784 B tokens, 10.1 days on 128 GPUs |
| Cost | $47k / $68k / $93k at $1.5 / $2.2 / $3.0 per GPU-hour |

### Three tiers of memory, three jobs

| Tier | Mechanism | Holds | Fails by |
|---|---|---|---|
| L1 | window attention | last 4 k tokens exactly | losing the distant past |
| L2 | fast weights | everything older, compressed | blurring detail |
| L3 | **versioned store** | claims + evidence + content hash | never — it revalidates |

Only the first two are neural. The third is a schema, because the evidence says the failure is bookkeeping, and bookkeeping does not improve with a larger matrix.

---

## Verified, not asserted

Every number above is produced by code and checked against a second implementation.

```
[1] total   torch=20,485,187,584  analytic=20,485,187,584  expected=20,485,187,584
    active  analytic=1,761,814,528  expected=1,761,814,528  ratio=8.600%
[2] window  max_row=8  causal=True  ok
[3] moe     tokens_out=24  expected=24  k=2  dropless=True
[4] memory  1 association: cos=+0.70   6 associations: cos=+0.72   (rank 16/32 caps recall near 0.71)
[5] state   2,097,152 floats = 4.2 MB fp16 per session, independent of context
            (dense KV@256k = 2.36 GB fp8, 562x larger)
[6] forward logits=(2, 12, 128)  states=2  ok
all assertions passed
```

Test [4] is the one that matters. It failed twice before it passed, and both failures were real bugs: a transposed update (`Aᵀ err` instead of `err A`) and a read in the wrong order (`q M` instead of `M q`, which silently makes the memory non-associative and stores nothing). A memory layer that is *almost* right returns plausible vectors and remembers nothing — which is why the test measures recall rather than shapes.

Test [1] caught two counting errors the same way: the assertion is `==`, not a tolerance, so a missing 4 MB matrix could not hide.

---

## Files

| | |
|---|---|
| [01 — The validity problem](01-the-validity-problem.md) | Why memory systems fail, and what a memory entry has to be |
| [02 — Architecture](02-architecture.md) | The three tiers, the parameter ledger, the bets |
| [03 — The mixture of experts](03-moe.md) | 2,000-run evidence for 1:32, routing, serving tax |
| [04 — Memory](04-memory.md) | Delta rule, the 4.2 MB state, the versioned store |
| [05 — Reasoning](05-reasoning.md) | Budget allocation, pairwise verification, RL recipe |
| [06 — Data](06-data.md) | 1,784 B tokens, PR mid-training, the staleness corpus |
| [07 — Training](07-training.md) | 31,025 GPU-h, stage order, stability gates |
| [08 — Serving](08-serving.md) | 20 B on a 24 GB card, expert hot sets, costs |
| [09 — Evaluations and risks](09-evals-and-risks.md) | Six ablations, eight risks, abandon triggers |
| [`kestrel3/model.py`](kestrel3/model.py) | Reference implementation, 6 assertions |
| [`kestrel3/config.json`](kestrel3/config.json) | Canonical configuration |
| [`tools/count_params_v3.py`](tools/count_params_v3.py) | Cost model behind every table |

```bash
python3 research/day3/tools/count_params_v3.py     # parameters, budget, serving
.venv/bin/python research/day3/kestrel3/model.py   # 6 assertions
```

---

## Research basis

Each mechanism is chosen from a specific measured result, not from fashion.

**Capacity.** *Slicing and Dicing* (2026): 2,000+ pretraining runs, quality improves **monotonically with total parameters even at 128:1 activation**, and optimal expert size depends on *active* parameters. *Scaling Laws for Fine-Grained MoE* (2024), *Beyond 50B* (2025) corroborate.

**Memory.** *Titans* (NeurIPS 2025): attention is short-term, neural memory is long-term. *TNT* (ICLR 2026): hierarchical chunking makes TTT **17× faster** to train. *TTT-E2E* (2026) and *Titan-Llama* (2026) show the long-context payoff.

**Reasoning.** *Scaling Test-Time Compute Optimally* (ICLR 2025): compute-optimal allocation beats best-of-N by **4×**, and a small model can beat one **14×** larger. *Without Verification or RL is Suboptimal* (2025). *V1* (ICLR 2026): pairwise self-verification, **+10 % Pass@1**. *GFPO* (ICLR 2026): shortest-correct filtering.

**Data.** *Scaling Laws for Code* (ACL 2026): code is more data-hungry than prose. *Clean Pull Request* (ICML 2026): PR mid-training gives **+13.6 / +12.3 pp** on SWE-bench Lite/Verified — agentless.

**Systems.** *Demystifying the MoE Serving Tax* (MLSys 2026): MoE pays **2–3×** at equal per-token compute. *Mira* (2026): predictive prefetch, **5.71×** on constrained GPUs. *METRO* (2025): balance activated experts, not tokens.

---

## What this design is not

- **Not a reasoning model.** 1.76 B active is a 1.76 B reasoning ceiling. Search and verification recover a lot where an oracle exists, and nothing where one does not. The no-verifier eval slice measures the loss rather than hiding it ([09](09-evals-and-risks.md) §9.1).
- **Not deployable on a phone.** 10.24 GB at 4-bit before expert streaming.
- **Not proven.** Nothing here is trained. Every capability claim is an argument from someone else's measurement.
- **Probably wrong about its most novel part.** The store controller needs a staleness corpus whose yield is estimated at ~2 % of multi-session trajectories; if it comes in under 5 k sequences, the L3 tier degrades to a static index (R2, 50 %).

## The honest one-liner

The previous designs made **reading** cheap in FLOPs, then in bandwidth. This one makes **remembering** cheap — 4.2 MB instead of 2.36 GB — and bets that the hard part of coding is knowing which of your memories is still true.

**Start at:** [01 — The validity problem](01-the-validity-problem.md).
