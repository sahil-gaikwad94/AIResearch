# 02 — Architecture

Two engines and a controller.

```
            ids
             │
        ┌────▼─────┐
        │ embedding│  131,072 × 4,096         537 M
        └────┬─────┘
             │
   ┌─────────▼──────────┐
   │  trunk × 16        │   windowed MLA (4,096)
   │                    │   delta-rule memory
   │                    │   256 latent-width experts, top-6
   └─────────┬──────────┘   20.21 B
             │
      x ─────┼──────────────────────────────┐
             │                              │
   ┌─────────▼──────────┐          ┌────────▼────────┐
   │ workspace × r      │          │  confidence head │
   │ 4 layers, d_w=1024 │          │  P(pass) ∈ [0,1] │
   │ cross-attn + self  │          └────────┬────────┘
   └─────────┬──────────┘                   │
             │  h                           │
             └──────────► + scale·proj_out ─┘
             │
        ┌────▼─────┐
        │  head    │  tied to embedding
        └──────────┘
```

## The parameter ledger

Every number below is asserted by `kestrel31/model.py` test [1], which computes the model twice — once with PyTorch on the `meta` device, once analytically — and requires both to equal a constant.

| Component | Parameters | Share |
|---|---|---|
| Attention, per layer | 27,787,552 | |
| Memory, per layer | 10,616,832 | |
| MoE, per layer | **1,224,737,024** | 97 % of a layer |
| Norms, per layer | 12,288 | |
| **Layer total** | **1,263,153,696** | × 16 = 20,210,459,136 |
| Embedding | 536,870,912 | tied head |
| Final norm | 4,096 | |
| Workspace | 117,978,114 | 0.57 % |
| **Total** | **20,865,312,258** | |
| **Active per token** | **1,990,944,258** | **9.54 %** |

Active per layer is 83,505,696: attention 27.8 M, memory 10.6 M, and 45.1 M of FFN — six routed experts at 4.72 M each, plus the 12.58 M shared expert that every token uses.

## Component 1 — the trunk: capacity, bought cheaply

The trunk is v1's block with a reshaped mixture of experts. Sixteen layers at `d=4096`: windowed latent attention (576 bytes of KV per token per layer, as in v1), a delta-rule memory giving 4.2 MB of session state independent of context, and 256 routed experts of latent width 384.

The reshape is the point. A full-width expert at this inner width costs 50.3 M parameters; a latent-width expert costs 4.72 M, **10.7× less**, because the expert reads and writes a 384-dim vector instead of a 4096-dim one. Holding the layer budget fixed, that buys **2.67×** more experts — 256 instead of 96 — and lifts expert sparsity from 1:32 to **1:42**. Routing still happens on the full-width token, so the router sees everything. (Evidence and the competing designs: [03](03-sparse-moe.md).)

## Component 2 — the workspace: depth, rented by the token

A 4-layer, 1024-wide recurrent core that reads the trunk's hidden state once, is applied `r` times, and writes back through a layer-scale initialised at **1e-4**.

That initialisation is not cosmetic. Test [6] measures it: at step 0, the logits with `r=1` and `r=8` differ by **8.05e-05** relative. The recurrent path begins as an identity, so the model has to *learn* that thinking helps rather than being forced to route signal through an untrained loop. The same test unrolls the workspace 16 times and requires the hidden norm to stay within 0.5–2× of its input; measured ratio **1.00**.

Why narrow? Because the whole efficiency argument depends on a pass being cheap, and at batch 1 decode, cost is bytes moved, not FLOPs. The workspace is 117.98 M parameters — **65 MB at 4-bit**, small enough to stay resident in the 126 MB L2 of a B300-class part. Passes 2..r then cost arithmetic, not memory traffic. On an H100 (50 MB L2) it does not fit, and each pass re-reads 65 MB; §10 states both cases rather than quoting the favourable one.

Why not loop the whole trunk? Because the trunk is 19.7 B parameters of MoE weights, and re-reading them per pass would make depth cost 100× what it costs in a narrow core. The trunk is a knowledge store; the workspace is a scratchpad. Conflating them is how recurrent models become slow. (Evidence: [04](04-recurrent-reasoning.md).)

## Component 3 — the confidence head: the controller is distilled, not learned from scratch

One head on the workspace's pooled state: 1024 → 512 → 1, **525,313 parameters**, output `P(pass)`.

It is trained against **execution outcomes**, not against its own consistency, and it does two jobs:

1. **At inference**, it is the cheap first opinion in the depth ladder — if `P(pass)` is high, stop at r=2; if low, escalate to r=4, 6, 8, checking with the real verifier at each rung when one is available.
2. **In the self-improvement loop**, it ranks candidates before the expensive test run, which is what makes 64 samples affordable instead of 6.

This is a deliberate inversion of the standard adaptive-computation design. The literature on learned halting gates is not encouraging: 2607.20519 finds that a single exit distribution is typically used both as the stopping rule *and* as the training-time weighting of per-depth losses, and that this conflation makes poor adaptive-compute behaviour hard to attribute — with fixed-prior depth supervision producing difficulty-aware trajectories without any input-dependent gate. So we train with a **fixed prior over r** and put the input-dependent decision at inference, where a verifier can supervise it.

## What the architecture deliberately does not have

| Not present | Why | Cost of being wrong |
|---|---|---|
| A learned halting gate | 2607.20519: conflated exit distribution; fragile | depth is a fixed ladder, slightly less elegant, measurable |
| A shared full-width *attention* path in the workspace | would double pass bytes | narrow core may underfit; B2 measures it |
| Routing by RL (MoE-GRPO) | adds an RL loop to the hot path | router stays a linear layer, 1.05 M params |
| A dense verifier network as a model component | execution is the verifier; the head is auxiliary | if tests are absent, only the head remains — and §08 says that is not enough |

## Three claims, and what each rests on

**C1 — capacity per active FLOP is better than v1.** Rests on the latent-expert arithmetic (10.7×) and on K2/K3 reporting scaling gains from higher sparsity at fixed active budget. *Not* verified for code at this scale. Falsified by B1 if sparsity 42.7 loses to 32.

**C2 — depth is cheap enough to allocate per token.** Rests on measured FLOP arithmetic in our own implementation: 0.236 GFLOP per pass versus 2.908 per decoded token, **12.3×**. This is the most solid claim in the document, because it is a count, not a benchmark.

**C3 — a verifier-conditioned ladder beats a fixed depth.** Rests on arithmetic (§08) plus the assumption that `p(r)` rises with r. If `p(r)` is flat — if latent depth does nothing on code — C3 is vacuous and the workspace is 118 M wasted parameters. That is what B2 exists to detect.

## Bets, in order of how much they cost if wrong

1. **Latent recurrence transfers from math to code.** Huginn's 5× GSM8K gain and LOTUS's 2.5–6.9× latency reduction are the evidence; neither is a code result. (118 M params, 1,000 GPU-h to test.)
2. **Verifier independence is achievable on our own task distribution.** If not, the loop in [07](07-self-improvement.md) cannot run. (No GPU; one week.)
3. **The 24 GB deployment story holds at batch 1.** 11.5 GB of weights leaves 12.5 GB, which is generous — but 0.57 GB of traffic per token is the constraint, not capacity.

## Next

[03 — The sparse MoE](03-sparse-moe.md) for why 256 latent-width experts and not 96 full-width ones, then [04 — Recurrent reasoning](04-recurrent-reasoning.md) for why depth is rented.
