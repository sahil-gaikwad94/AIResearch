# 03 — The sparse mixture of experts

**256 routed experts, top-6, latent width 384, one shared full-width expert. Sparsity 1:42.**

## The evidence that sets the shape

Three numbers from 2025–26 do most of the work.

**1. At fixed active parameters, more experts keeps helping.** The Kimi K2 report runs a controlled sparsity sweep: hold activated parameters and activated experts fixed at 8, vary only the total expert count, and quality improves monotonically with sparsity. They chose **sparsity 48** (384 experts, 8 active) for a 1.04 T model with 32.6 B active — and the resulting model beats DeepSeek-V3's 671 B/37 B with **13 % fewer activated parameters**. That is a direct measurement that the *ratio* is a real lever, not a bookkeeping choice.

**2. The gain survives into latent-width experts.** Kimi K3 pushes to **896 routed experts, top-16, sparsity 56**, and reports **2.5× scaling efficiency over K2**. The mechanism is the one used here: routed experts operate in a **compact latent space**, while shared experts keep a full-width path. Narrow experts are what make sparsity 56 affordable at all.

**3. Sparsity is cheap on paper and expensive on interconnect.** The MLSys 2026 serving-tax study is unambiguous: at equal per-token compute, MoE pays **2–3×** in serving. The tax is all-to-all traffic and straggler imbalance, not arithmetic.

Those three facts point at a specific design: **be as sparse as the interconnect allows, and make each expert small so a single all-to-all moves less data.**

## Why latent width

An expert's cost is dominated by its input and output projections. Full width: `3 × 4096 × 4096 = 50,331,648` parameters. Latent width 384 with the same inner width: `3 × 384 × 4096 = 4,718,592`.

**10.7× cheaper per expert** — exactly `d / d_lat`, asserted in test [2]. Note what is *not* squared: keeping the inner width at 4096 means the saving comes only from the outer dimensions. Squeezing the inner width too would save more parameters and lose more capacity; that trade is not free, and Kimi's latent design keeps the inner width intact.

Holding the per-layer budget at ~1.22 B, the saving converts into experts:

| Design | Experts | Active | Sparsity | Per-expert |
|---|---|---|---|---|
| v1 | 96 | 3 | 1:32 | 50.3 M (full width) |
| **v2** | **256** | **6** | **1:42** | **4.72 M (latent)** |

Active FFN per layer: 6 × 4.72 M routed + 12.58 M shared + 3.15 M projections + 1.05 M router = **45.09 M**, against v1's 38.14 M. So v2 pays **18 % more active FFN compute** for **2.67× the experts** and a routing decision spread over 256 options instead of 96. That is the trade, stated plainly: v2 is not cheaper per token than v1, it is differently shaped.

## The shared expert, and a reversal

v1 argued *against* a shared expert, on the grounds that it is always-on compute in a design whose whole purpose is to avoid always-on compute. v2 keeps one. Three reasons, and the honest cost:

1. **Both reference designs keep one.** K2 and K3 each have exactly one shared expert. When two independent frontier labs land on the same exception, it is probably absorbing something real — common transformations that every token needs and that routed experts waste capacity re-learning.
2. **It protects the latent bottleneck.** With every expert reading a 384-dim vector, the model has no full-width path through its FFN at all. The shared expert is that path.
3. **It stabilises early training**, when routing is near-uniform and the routed path is effectively a noisy average of 6 of 256 experts.

Cost: **12.58 M of 45.09 M active FFN parameters, 28 %**. If B1 shows the shared expert contributes under 0.5 pp, it comes out and active parameters drop to ~1.79 B. That is a cheap experiment and it is scheduled.

## Routing

```
logits = W_r · x + b          # b is the balancing bias, not a gradient term
w, idx = topk(logits, 6)
w      = softmax(w)
y      = shared(x) + W_up( Σ_{e ∈ idx} w_e · Expert_e(W_down(x)) )
```

Three details that matter:

**Dropless.** Every token gets exactly six experts. No capacity factor, no token dropping. Test [5] asserts `routed == batch × seq × 6` exactly, which is the property that a capacity-based router silently violates under load.

**Auxiliary-loss-free balancing.** The bias `b` is updated by a sign rule toward uniform load rather than by adding a gradient term to the loss. The reason is specific: an auxiliary load-balancing loss fights the language-modelling objective, and at 256 experts the pressure is large enough to distort routing rather than merely equalise it. The bias decouples the two.

**Group-limited all-to-all.** Experts are partitioned into 8 groups of 32; a token's six experts are drawn from at most 3 groups. This caps inter-node traffic at 3 round trips instead of 6 and is what makes sparsity 42.7 affordable on a 128-GPU cluster. The cost is a small restriction on which expert sets are reachable, which the K2 ablations suggest is worth paying.

## Serving: what sparsity 42.7 actually costs

From `tools/count_params_v31.py`, 4-bit weights, one decode step:

| Batch | Experts touched | Bytes/token | Bandwidth ceiling |
|---|---|---|---|
| 1 | 5.9 | 0.57 GB | 5,918 tok/s |
| 16 | 80.2 | 0.05 GB | 70,642 tok/s |
| 64 | 199.0 | 0.02 GB | 200,876 tok/s |

The batch-1 number is the honest one. At batch 1, **always-on weights dominate**: 1.0 B parameters that are not experts cost 0.551 GB per step no matter what the router does. Sparsity buys nothing at batch 1 — a 20.9 B model at batch 1 behaves like a 1 B model with a big disk. The tax the MLSys study measures is exactly this, and it is why the deployment story in §10 is about **batch ≥ 16**, where 80 of 256 experts are touched and per-token traffic falls 11×.

The 70k and 200k figures are bandwidth ceilings, not predictions: past batch ~16 the step becomes FLOP-bound and the real limit is arithmetic.

Two serving levers we do not use in the base model but keep on the shelf:

- **ASET-style entropy thresholding** (ACL 2026): skip experts when the router distribution is peaked enough, trading average activation for quality along a Pareto frontier. Training-free, so it can be applied per deployment. We do not adopt it because it changes the model's behaviour post hoc and our cost argument doesn't need it.
- **Expert offload with a hot set**, following Mira's predictive prefetch (5.71× on constrained GPUs). The 24 GB story in §10 assumes this.

## Rejected alternatives

| Option | Evidence | Why not |
|---|---|---|
| **Routing-free MoE** (2604.00801) | eliminates router, softmax, top-k and load balancing; better perplexity at all scales; survives higher learning rates | No code-domain evidence at scale, and it removes the one place the model can express "this token is different". Promising; parked, revisit at 1 B-scale ablation |
| **Routing optimised by RL** (MoE-GRPO, CVPR 2026) | routing as a sequential decision raises expert diversity and reduces expert overfitting | Puts an RL loop inside the forward pass. We already have one expensive RL loop; two interacting is unanalysable |
| **Expert-choice routing** | balances load by construction | breaks causality for autoregressive decoding — a token's expert set would depend on future tokens |
| **Sparsity 64+** (K3 territory) | K3 sustains 56 with latent experts | our interconnect budget (128 GPUs) and the batch-1 tax both get worse; revisit after B1 |
| **No shared expert** | v1's choice; DeepSeekMoE has two | see above; 28 % of active FFN is too much to abandon without a measurement |

## What would falsify this

1. **B1**: a 1 B-active ablation at sparsity 32 vs 42.7 vs 56 on code. If 42.7 does not beat 32, the latent width is a serving optimisation and should be described as one. (~1,200 GPU-h.)
2. **Shared-expert removal**: if held-out loss moves less than 0.5 pp, delete it and bank 28 % of active FFN compute.
3. **Router collapse under length**: at 64 k contexts, does the effective number of experts per token drop? Measured by expert entropy on a fixed eval set. If entropy falls below ~4.0 effective experts, the model is silently becoming dense-routed and the balance bias needs retuning.

## Next

[04 — Recurrent reasoning](04-recurrent-reasoning.md): the other half of the "cheap compute" argument.
