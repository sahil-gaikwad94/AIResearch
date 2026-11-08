# 3 — The Mixture of Experts

> 96 experts per layer. 3 fire. One parameter in 32 is touched, and the model stores twenty billion of them. This chapter is the evidence that the ratio is not reckless, and the engineering that makes it survivable.

---

## 3.1 The finding that licenses the ratio

*Slicing and Dicing: Configuring Optimal Mixtures of Experts* (2026) is the largest controlled study of MoE configuration to date: **over 2,000 pretraining runs**, models up to 6.6 B total, varying expert count, expert dimension, heterogeneous sizing, shared-expert size and load balancing — and measuring the interactions between them, which earlier work did not.

Four results, in order of how much they matter here:

**1. Quality improves monotonically with total parameters, at every activation ratio tested — including 128:1.** There is no cliff where extra capacity stops helping. This is the licence for 20.5 B stored behind 1.76 B active.

**2. Optimal expert size is nearly invariant to total parameter count, and depends on active parameter count.** So you do not scale an expert to match an FFN. You scale it to the active budget, then add experts. With 1.76 B active, a 1,024-wide expert is the right order of magnitude — a *twelfth* of the 12,288 a dense FFN of this width would use.

**3. Dropless routing gives a consistent gain.** No token dropping, no capacity factor, no random discard under load.

**4. Shared experts, heterogeneous expert sizing, and most load-balancing choices have small effects** compared to expert count and granularity.

Result 4 is the one that contradicts conventional wisdom, and it is why this design has **no shared expert** (§3.4).

Corroborating work: *Scaling Laws for Fine-Grained MoE* (2024) finds a U-shaped loss curve in granularity — more, smaller experts help until routing cost dominates — and *Scaling Fine-Grained MoE Beyond 50B* (2025) confirms fine-grained wins at 56 B total / 17 B active. *Towards Greater Leverage* (2025) independently derives an optimum near 7 activated experts and finds the optimal shared-expert ratio *decreases* as compute grows.

---

## 3.2 The configuration

| | Kestrel-3 | DeepSeek-V3 | OLMoE | Why ours differs |
|---|---|---|---|---|
| Experts/layer | 96 | 256 | 64 | 16 layers only; total bounded by storage |
| Active | 3 | 8 | 8 | 1:32 activation, per the 2026 study |
| Expert `d_ff` | 1,024 | 2,048 | 1,024 | scaled to active budget, not FFN width |
| Shared expert | **none** | 1 | none | 2026 study: small effect; see §3.4 |
| Routing | dropless top-3 | dropless top-8 | top-8 | dropless is the measured gain |
| Balancing | loss-free bias | loss-free bias | aux loss | removes the conflict gradient |
| Active/total | 8.6 % | 5.5 % | 18.8 % | between the two |

Per layer: `96 × 3 × 4096 × 1024 = 1,207,959,552` expert parameters, plus `4096 × 96 + 96` for the router.

Across 16 layers that is **19.33 B of experts** — 94 % of the model. Everything else, attention and memory and embeddings, is 1.15 B.

---

## 3.3 Routing

```python
logits = W_r x + b        # b is the balancing bias, NOT part of the gate weight
w, idx  = logits.topk(3)
w       = softmax(w)
y       = Σ_i w_i · Expert_i(x)
```

Two details carry the design.

**The bias participates in selection only.** Following the auxiliary-loss-free strategy (2024) used by DeepSeek-V3, `b` is added to the logits *before* the top-k, but the gate weights come from the affinities, not from the biased scores. This keeps the bias from distorting the mixture: it moves which experts are chosen, not how much each contributes.

**The bias is updated from load, with no gradient:**

```
after each step:  b_i += γ · (target_load − load_i)      γ = 0.001
```

No auxiliary loss, therefore no interference gradient fighting the language objective. The 2024 paper's contribution was showing this beats aux losses at up to 3 B scale; the 2026 study independently finds load-balancing *mechanism* is second-order, which makes the choice low-risk either way.

**Dropless means the batch shape is ragged.** Each expert receives a variable number of tokens, so the GroupGEMM kernel pads to the maximum. This is where MoE serving overhead comes from, and it is not small:

> *Demystifying the MoE Serving Tax* (MLSys 2026) measures MoEs at **2–3× worse than dense models at equal per-token compute**, with a "straggler tax" that adds **40–80%** to prefill kernel latency when the token distribution is uneven.

We accept that tax for two reasons: decode is memory-bound and barely affected (§3.6), and the 2026 study says the quality gain from capacity is the largest effect available. But the tax is real, and [08](08-serving.md) budgets for it rather than pretending it away.

---

## 3.4 Why there is no shared expert

DeepSeekMoE introduced shared experts so that "common knowledge" would not have to be relearned by every routed expert, and reported a clear win at fixed *activated* parameters. That evidence is real and it is in the opposite direction from this design.

Three reasons to go the other way anyway:

1. **The 2026 study varied it directly, at scale, across 2,000 runs, and found the effect small relative to expert count and granularity.** It is the newer and larger experiment, and it is a controlled comparison rather than an ablation inside one model.
2. **A shared expert costs active FLOPs on every token** — it is a dense FFN by another name. At a 1:32 ratio, the whole point is that active compute is small; a 1,536-wide shared expert would add 18.9 M parameters of *always-on* compute per layer, roughly +24% to active FFN.
3. **The mechanism it protects against is routing collapse**, and loss-free balancing plus dropless routing already handle that without spending compute.

None of this is certain, which is why B4 ([09](09-evals-and-risks.md)) adds a shared expert and measures. The pre-registered switch: adopt it if it buys **≥1.5 pp** on repo tasks at equal active FLOPs.

---

## 3.5 What the experts learn — and whether language is a real prior

An untested hypothesis worth stating plainly: experts in a code model may specialise by **language and by role** (data access, control flow, error handling, tests) rather than by whatever latent clusters the router finds.

The cheap experiment is a warm start: for the first 5 B tokens, bias the router so that expert groups receive tokens whose file extension maps to a language class, then decay the prior to zero over 20 B tokens. Cost: an index, not a model. If specialisation by language is real, the router rediscovers it faster; if it is not, the prior decays and nothing is lost.

Anthropic-style caution applies: the fact that a prior *feels* right is not evidence. B1 measures `bpb` with and without, and the prior ships only if it saves ≥0.5 % of tokens to a fixed loss.

---

## 3.6 What 1:32 does to serving

Because only 3 of 96 experts fire, the working set is a small fraction of the model at any batch size — and it grows slowly:

| Batch | Distinct experts touched (expected) | Weights + KV read | Step time |
|---|---|---|---|
| 1 | 3.0 | 1.84 GB | 0.55 ms |
| 16 | 38.2 | 2.28 GB | 0.68 ms |
| 64 | 83.6 | 3.44 GB | 1.03 ms |

At batch 16 you touch 40 % of the experts for 16 tokens — which is why the decode cost per *token* falls from 1.84 GB to 143 MB. This is the opposite of the usual MoE problem (all experts needed at large batch): at 1:32, even batch 64 leaves 13 experts cold.

Two systems consequences:

- **The hot set fits in VRAM.** Real workloads are skewed; a repository or a language uses a subset. Recent systems exploit exactly this: *Mira* (2026) prefetches predicted experts two layers ahead and reports **5.71×** throughput on a memory-constrained GPU; *DynaExq* (2026) keeps hot experts resident at higher precision and gets **2.73×** over offloading.
- **A 20.5 B model fits a 24 GB card at 4-bit** (10.24 GB of weights) if the cold experts stream. That is the deployment story [08](08-serving.md) is written around.

---

## 3.7 Rejected, with reasons

| Option | Why not |
|---|---|
| **1:8 activation** (24 B active-equivalent) | 4× the compute for maybe +1 pp; [02](02-architecture.md) is a compute-first design |
| **256 experts at 1:64** | better capacity per FLOP on paper, but VRAM and router cost grow while gain is sub-linear in the tested range |
| **Shared expert** | §3.4 — measured small effect, costs always-on FLOPs |
| **Expert-choice routing** | requires future tokens; cannot be causal |
| **ReLU routing (ReMoE)** | differentiable and elegant, and best in its paper's small-scale tests — but no evidence at 20 B total behind 1.7 B active, and dropless top-k is the safer bet |
| **Hash / router-free** | the ReMoE comparison finds deterministic hash routing worse than learned across the board |
| **MoE only in some layers** | the 2026 study finds monotone gains in total parameters; making 16 of 16 layers sparse is the simple version of that |

---

## 3.8 What would falsify this chapter

1. **bpb at 1:32 tracks worse than 1:8 by >2 %** at 200 B tokens → the ratio is too aggressive at this active size.
2. **Expert utilisation stays skewed after 50 B tokens** (max/mean load >2.5 with bias updates on) → the balancing is not working and the effective expert count is smaller than 96.
3. **Most experts never win a token on a fixed probe set** after 100 B → capacity is being wasted; granularity is wrong.
4. **Shared expert buys ≥1.5 pp** → the 2026 result does not transfer, and §3.4 is wrong.

**Next:** [04 — Memory](04-memory.md).
