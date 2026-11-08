# 7 — Training

> **31,025 GPU-hours** for 1,784 B tokens on a 20.5 B-parameter model. That is 10.1 days on 128 GPUs, and it is cheaper than training a 4 B dense model for the same number of tokens — because only 8.6 % of the parameters are ever read.

---

## 7.1 The budget

| Stage | Tokens | Context | Window | GPU-h |
|---|---|---|---|---|
| Pretrain | 1,400 B | 4 k | 4 k | 16,979 |
| Window extension | 200 B | 8 k | 8 k | 2,991 |
| **Memory training** | 100 B | **64 k** | **4 k** | **1,213** |
| PR mid-training | 80 B | 32 k | 4 k | 970 |
| SFT | 4 B | 32 k | 4 k | 49 |
| RL (generation-bound) | — | 32 k | 4 k | 3,000 |
| Ablations | — | — | — | 2,500 |
| **Subtotal** | **1,784 B** | | | **27,701** |
| Restarts (12 %) | | | | 3,324 |
| **Total** | | | | **31,025** |

At $1.5 / $2.2 / $3.0 per GPU-hour: **$47k / $68k / $93k**.

### The line worth staring at

Memory training at **64 k context costs 1,213 GPU-hours** — less than the 8 k window-extension stage. That is not a typo and it is not an estimate error: the attention window stays at 4,096, so per-token FLOPs are identical to the 4 k stage (13.82 GFLOP), and 100 B tokens at that rate is exactly what the arithmetic says.

The previous design's 256 k stage cost 553 GPU-h for 12 B tokens — 65× more expensive per token. That is the difference between paying for reach with attention and paying for it with a 4.2 MB state.

### Sanity check on the FLOPs

```
6 × 1,761,814,528 (active)  = 10.57 GFLOP
attention, 4 k window       =  3.22 GFLOP
memory read + delta update  =  0.02 GFLOP
                              ─────────
                              13.82 GFLOP/token
```

× 1,400 B tokens = 1.93 × 10²² FLOPs. At 32 % MFU on 989 TFLOP/s peak = 16,979 GPU-h. The model has no hidden quadratic: at 4 k, attention is 23 % of forward compute, and at 64 k it is still 23 %.

---

## 7.2 Optimiser and schedule

| | |
|---|---|
| Optimiser | **Muon** on 2-D matrices; AdamW on embeddings, norms, router bias |
| Muon LR | 0.02, momentum 0.95, 5 Newton–Schulz steps |
| AdamW LR | 3 × 10⁻⁴, β = (0.9, 0.95) |
| Weight decay | 0.1 on matrices |
| Batch | 4 M tokens, ramped from 0.5 M over 5 B |
| Warmup | 2,000 steps |
| Decay | cosine to 10 % of peak |
| Grad clip | 1.0 |

Muon is chosen for the same reason as everywhere else in this family of models: roughly 2× pretraining efficiency on matrix parameters. Unlike a dense model, the parameter groups are uneven — 19.3 B of expert matrices, 0.98 B of everything else — so the optimiser state is dominated by experts.

**Sharding.** Muon momentum for 20.5 B parameters is 82 GB in bf16; weights 41 GB, gradients 41 GB → 164 GB total, sharded across 64 GPUs at 2.6 GB each. The constraint is not memory but **collective volume**: 96 experts × 16 layers means expert-parallel all-to-all on every layer, and that is where the MFU goes (see the 32 % assumption).

**No model-parallel tensors.** At `d = 4096` and 16 layers, nothing needs to be split within a layer. Expert parallelism is the only sharding axis that matters, and it is set to 8 (12 experts per GPU).

---

## 7.3 Stage order, and why

```
1,400 B @ 4k   ──── language, code, structure
   200 B @ 8k  ──── positional adaptation (the ICLR 2025 finding)
   100 B @ 64k ──── the memory learns to compress
    80 B @ 32k ──── PR replay: the biggest single delta
     4 B SFT    ──── behaviour, protocol, staleness handling
```

**Window extension before memory training.** The memory learns to compress what attention cannot see. If the window is 4 k and sequences are 4 k, there is nothing to compress and the state learns nothing. Sequences must be longer than the window before the layer has a job.

**PR mid-training late.** It is the most valuable data (+13.6 pp on SWE-bench Lite in the source paper) and the most format-specific; putting it last, after the model can already read a repository, means it teaches editing rather than basic comprehension. The cost is forgetting, mitigated by replaying 10 % base data throughout.

**Memory training is short (100 B).** The memory is a small module learning a simple objective — compress what you cannot attend to. It does not need 500 B tokens, and long sequences are the expensive ones even here.

---

## 7.4 Memory-specific training mechanics

The delta-rule state is not a parameter, so nothing in §7.2 applies to it. Its training has its own rules, taken from the TNT result (ICLR 2026) that decoupled training efficiency from inference quality:

| Knob | Training | Inference |
|---|---|---|
| Chunk size | 512 | 1 (token by token) |
| State resets | every 8 chunks | never |
| Parallelism | chunks across devices | sequential |
| Final phase | 10 B tokens at chunk 1, no resets | — |

The reset is what makes the layer trainable: without it, chunk `k` depends on chunk `k−1`, and the sequence cannot be split. The final no-reset phase re-teaches the dependency. TNT reports up to **17×** training speedup from this scheme, and the fine-tune is what keeps the quality.

**Read-before-write is enforced in training too.** The reference implementation reads the state, then writes, at every position. A version that wrote first would show a suspiciously low training loss and would fail eval — which is why the config carries the flag explicitly rather than relying on implementation discipline.

---

## 7.5 Stability

Three quantities are logged from step 0, because each has a failure mode that is invisible in the loss curve until it has cost a week:

| Quantity | Healthy | Failure | Response |
|---|---|---|---|
| `‖B‖` per layer | plateaus | grows >5 %/1 k tokens | normalise `k`, decay `B` |
| Expert load max/mean | < 1.6 | > 2.5 after 50 B | raise γ, check bias updates |
| Router entropy | stable | collapses to <25 % of step-10 k | the experts have merged; restart |
| Attention entropy | stable | falls below 0.5 nats | sinks absorbing everything |

Standard spike handling applies: detect on a 3-step smoothed derivative, skip the batch, roll back, lower LR 20 % for 500 steps.

**The MoE-specific risk is not instability, it is silent capacity waste.** A router that sends 70 % of tokens to 12 of 96 experts trains fine and loses most of the model's capacity. That is why expert utilisation is a first-class metric rather than a diagnostic:

```
utilisation_gini  = Gini(load per expert)     # target < 0.35
dead_experts      = #experts with < 0.1% of tokens over 1 B
```

Both are gated in [09](09-evals-and-risks.md).

---

## 7.6 SFT: what behaviour is actually installed

4 B tokens, the smallest stage, doing the most visible work:

| Data | Share | Purpose |
|---|---|---|
| Agent trajectories (generate 12 : keep 1) | 55 % | tool use, diff format |
| PR-replay instructions | 20 % | editing style |
| **Staleness-injected sessions** | 15 % | the L3 controller's only training signal |
| Refusals / clarifications | 5 % | calibration on underspecified tasks |
| Long-context synthesis | 5 % | multi-file reasoning |

The third row is the one that cannot be bought: 20 k sequences in which a remembered fact has silently changed, and the correct action is to notice. Without it the store is a text file with extra steps ([04](04-memory.md) §4.4).

---

## 7.7 What would make me restart

1. `‖B‖` diverging before 200 B tokens with normalisation on → the delta-rule learning rate is wrong for the real `d_m`, and the rank/`η` pair must be re-derived.
2. Expert utilisation Gini above 0.5 after the first 100 B → the router is collapsing and the bias updates are not reaching it.
3. bpb at 1:32 more than 2.5 % behind a 1:8 control at 200 B → the activation ratio is too aggressive, and the fix is more active parameters, not fewer experts.
4. Memory ablation (B2) showing the state contributes < 0.5 % loss improvement at 64 k → the memory is not doing the job and the window must grow.

**Next:** [08 — Serving](08-serving.md).
