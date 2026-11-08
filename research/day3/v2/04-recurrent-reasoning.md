# 04 — Recurrent reasoning

**A 4-layer, 1024-wide workspace applied `r` times, where `r` is decided by a verifier and not by a halting gate.**

## Why depth is rented

Parameter count and compute per token are usually the same dial: to think harder you buy a bigger model. MoE already breaks that for *capacity* — 20.9 B stored, 2.0 B read. Recurrence breaks it for *depth*: the same 4 layers are applied again and again, so the effective depth at inference is 16 + 4r while the parameter count is fixed.

From `kestrel31/model.py`, test [3]: `params(r=1) == params(r=8) == 20,865,312,258`. Depth is not a property of the weights.

The cost of a pass, from test [4]:

| | GFLOP |
|---|---|
| One decoded token through the trunk | **2.908** |
| One workspace pass | **0.236** |
| Ratio | **12.3× cheaper** |

That ratio is the entire argument. A chain-of-thought step is a token: it costs a full forward, a vocabulary projection over 131,072 entries, and 576 bytes of new KV in every layer. A latent step costs a quarter of a gigafop and **zero new KV** — the workspace cross-attends to context the trunk already computed. Six latent steps cost 1.42 GFLOP; six CoT steps cost 17.4 GFLOP and 3.5 KB of cache per token.

Compute per token is **flat in context length** — 4.65 GFLOP at 4 k and 4.65 at 64 k — because attention is windowed at 4,096 and the memory tier is O(1). A dense-attention model of this size at 64 k would pay **28.4 GFLOP**, 6.2× more, and that is the comparison the window exists to make invalid. The 12.3× advantage of a latent step over a decoded token is therefore context-independent: it is a property of the two operations, not of the sequence around them.

What does scale with context is the **cache**. A decoded token appends 576 bytes in each of 16 layers; a workspace pass appends nothing, because it cross-attends to context the trunk has already computed. At 4,096 tokens that is 37.7 MB of KV versus zero — and it is the reason depth can be spent freely on a long-context task but chain-of-thought cannot.

## The evidence that recurrence buys capability

| Study | Finding |
|---|---|
| *Recurrent Depth* (NeurIPS 2025, Huginn) | 3.5 B params, 800 B tokens; extra loops reach a **~50 B-equivalent** compute load; **5×** over a fixed-depth model of the same size on GSM8K |
| *Looped Transformers* (ICLR 2025) | a `k`-layer model looped `L` times nearly matches a `kL`-layer model on addition, p-hop induction, and math — "many reasoning problems need depth, not parameters" |
| *LOTUS* (2026) | looped transformer with supervision on each latent block; first latent-CoT method to close the gap with explicit CoT at **3 B**; **2.5–6.9×** lower thought-phase latency |
| *Trading Depth for Time* (2026) | one latent thought token recovers **67 % / 81 %** of the double-depth gain at **~48 % fewer parameters** on 16- and 20-layer MoE backbones |
| *Adaptive Loops and Memory* (ICLR 2026 workshop) | looping helps **mathematical** reasoning; gated memory banks recover commonsense; combined beats an iso-FLOP baseline with **3× the layers** on math |
| *LSRL* (EMNLP 2025) | Huginn at r=8 scores only **13.5 %** on GSM8K; process-supervised GRPO on latent states adds **+2.06** on MathQA |

Two of those rows are warnings, not endorsements, and they shaped the design.

**Warning 1 — the gains are uneven across task types.** Looping helps math and multi-step logical structure; it does not automatically help everything. Code is closer to math than to commonsense, which is why the bet is plausible — but the same table shows Huginn at 13.5 % GSM8K, which is not a model you would ship. Depth is a multiplier on a capability that has to exist first.

**Warning 2 — latent states need supervision or they drift opaque.** LSRL's contribution is not that latent recurrence works; it is that GRPO on latent states *needed process supervision to work at all*. LOTUS reaches parity with explicit CoT only by putting a cross-entropy loss on **every** latent block. We do not do that. §04.7 states the consequence.

## Training: a fixed prior, not a learned gate

The naive design is a halting head: at each loop, a small network decides whether to stop, trained with a ponder cost. It is elegant, and the 2026 evidence says it is the fragile part.

2607.20519 diagnoses why: the same exit distribution is used both as the inference-time stopping rule and as the training-time weighting of per-depth losses, so a failure is ambiguous between the readout, the induced trajectory, and their interaction. The finding that matters: **fixed-prior depth supervision** — shaping the trajectory with a schedule, no input-dependent gate — produces difficulty-aware trajectories on its own, and matches or beats learned gates on Ouro-1.4B/2.6B.

So we train with:

- `r` sampled per batch from a prior, weighted toward shallow: 60 % at r=1, 25 % at r∈{2,3,4}, 15 % at r∈{5,6,8}. Mean r ≈ 2.2.
- The workspace layer-scale initialised at **1e-4**, so the loop is an identity at step 0 (measured drift **8.05e-05**) and the model has to discover that looping helps.
- No ponder loss. The compute budget is set by the sampler, and the sampler is set by us.

What this buys: the model sees every depth during training and cannot degenerate into "always r=2" the way a halting gate can collapse. What it costs: no input-dependent depth at training time, so the confidence head has to learn the mapping from states at various depths to pass probability from execution labels rather than from the loop itself.

## Inference: a ladder, checked at each rung

```
r = 2
answer = decode(r)
if verifier_available:
    while not pass(answer) and r < 8:
        r = next_rung(r)          # 2 -> 4 -> 6 -> 8
        answer = decode(r)
else:
    r = 2 if P(pass) > 0.5 else 8   # confidence head only
```

This is the "verification is the scarce resource" thesis applied to compute. Depth is spent **only where the check fails**, which is the one signal that is not the model's own opinion.

The arithmetic, from `tools/sim_verify.py` (stated assumptions: `p(r) = 0.62 − 0.32 · 0.55^(r−1)`, `p(1) = 0.30`):

| Policy | Solved | Cost (GFLOP) |
|---|---|---|
| fixed r=2 | 0.444 | 0.47 |
| fixed r=4 | 0.567 | 0.94 |
| fixed r=8 | 0.615 | 1.89 |
| **ladder 2→4→6→8** | **0.963** | **1.52** |

The ladder reaches 96 % solved for **80 % of the cost of always running at maximum depth**, because most tasks exit at r=2 and the ones that don't were going to fail anyway.

**Read that table carefully.** The 0.963 is not a capability claim — it is what you get if each rung is an independent draw at `p(r)`, and independent draws are the single most generous assumption in this document. Real retries at higher depth are correlated with the first attempt. The honest reading is: *the ladder dominates a fixed depth by construction, and the size of the gap depends on how decorrelated a deeper attempt is.* B2 measures that directly.

## Why the workspace is narrow, and what it costs

The workspace is 117.98 M parameters: four layers, each with cross-attention to the trunk, windowed self-attention, and a SwiGLU FFN at 4096 inner width.

At batch 1, decode is bandwidth-bound, so the question is whether those weights can stay in cache:

| Part | L2 | Workspace at 4-bit | Consequence |
|---|---|---|---|
| B300-class | 126 MB | 65 MB | resident; passes 2..r cost FLOPs only |
| H100 | 50 MB | 65 MB | does not fit; each pass re-reads 65 MB ≈ 19 µs |

On the H100 path, going from r=2 to r=8 adds ~115 µs to a step whose trunk cost is ~170 µs. That is a real 68 % latency increase, and an agent that makes six tool calls per task feels it. §10 gives the latency budget per deployment class rather than pretending this is free.

The alternative — looping the full 19.7 B-parameter trunk — would make each pass cost 100× more in bytes. Narrow core, many passes, is the only version of this that is affordable.

## The limit that could kill it

**We do not supervise latent steps, and LOTUS says that is how latent CoT loses to explicit CoT.** Their result: latent CoT underperforms explicit CoT beyond 1 B parameters, and the gap *widens* with scale, until you put per-latent-block supervision in. We supervise only the final output plus the confidence head.

Predicted consequence, stated in advance so it can be checked: on tasks where the reasoning is a short chain with a verifiable endpoint, the ladder will do well, because the verifier supplies the step signal externally. On tasks with no verifier — design questions, ambiguous specs — recurrence will add latency and not much else. That is why §08 insists the verifier is not optional and why §09 reports the no-verifier slice separately rather than averaging it away.

If B2 shows latent depth is flat on code, the fix is known and cheap: add gold intermediate-state supervision (patch-level: which file, which function, which hunk) at each latent block, which is exactly what the execution traces in [05](05-execution-driven-learning.md) already provide for free.

## Rejected alternatives

| Option | Why not |
|---|---|
| **Learned halting gate (ACT/PonderNet)** | 2607.20519: conflates stopping with loss weighting; we use a prior + verifier instead |
| **Full-width recurrent block** | pass cost becomes weight traffic for 19.7 B parameters; depth would be 100× more expensive |
| **Looping the whole trunk** (Universal Transformer style) | same, plus it destroys the "knowledge store / scratchpad" separation |
| **Explicit CoT as the primary depth mechanism** | 12.3× more expensive per step and grows the KV cache; still used for what it is good at — communicating, and text-verifiable reasoning |
| **Chain-of-thought plus latent, both at max** | no controller; doubles cost for a gain nobody has measured |

## What would falsify this

1. **B2** (1,000 GPU-h): does `p(r)` rise on code? If accuracy at r=8 − r=2 is under 3 pp on held-out tasks, the workspace is decoration.
2. **Latency**: if r=6 on the deployment class adds more than 30 % to agent wall-clock, the ladder's default moves to r=2 with escalation only on hard tasks.
3. **Correlation of retries**: measure whether a second attempt at higher depth succeeds more often than a fresh sample at the same depth. If it doesn't, the ladder should re-sample rather than deepen — a one-line policy change worth knowing about.

## Next

[05 — Execution-driven learning](05-execution-driven-learning.md): where the signal that drives all of this comes from.
