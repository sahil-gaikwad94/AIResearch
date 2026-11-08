# 8 — Serving

> The 24 GB problem: how to serve a 20.5 B-parameter model on a card that holds 10.2 GB of it at 4-bit, without the expert-shuffling that made earlier MoE deployments slower than dense models of a third the size.

---

## 8.1 One decode step

Bytes are what decode pays for, and at 1:32 there are surprisingly few of them.

| Batch | Experts touched | Active weights | Window KV | State | Total read | Step | Throughput |
|---|---|---|---|---|---|---|---|
| 1 | 3.0 | 1.76 GB | 36.9 MB | 4.2 MB | **1.84 GB** | **0.55 ms** | 1,821 tok/s |
| 16 | 38.2 | 1.76 GB | 36.9 MB | 4.2 MB | **2.28 GB** | **0.68 ms** | 23,480 tok/s |
| 64 | 83.6 | 1.76 GB | 36.9 MB | 4.2 MB | 3.44 GB | 1.03 ms | 62,000 tok/s |

Three things hold this together, and each is a design decision:

- **The window does not grow.** At 1 M tokens of conversation the KV read is still 36.9 MB, because there is no 1 M KV. This is the single largest difference from a conventional long-context deployment, where cache bytes overtake weights at ~200 k.
- **The state does not grow.** 4.2 MB per session, at any turn count.
- **Expert activation saturates slowly.** At batch 64 you still touch only 87 % of experts. Compare a 1:8 model, where batch 16 already touches all of them — the standard MoE throughput cliff.

The 4.3× jump from batch 1 to batch 16 is the whole economics of the deployment: latency-bound at small batch, throughput-bound at large.

---

## 8.2 Memory: three deployments

| Target | Weights | Window + state | Cold experts | Fits? |
|---|---|---|---|---|
| **H100 80 GB, FP8** | 20.48 GB | 41 MB | all resident | yes, with 55 GB spare |
| **L40S 48 GB, FP8** | 20.48 GB | 41 MB | hot set resident | yes |
| **RTX 4090 24 GB, 4-bit** | 10.24 GB | 41 MB | streamed from host | yes, 13 GB headroom |

The coherent story is the third. A 20.5 B-parameter model on a consumer card, because the model only ever reads 1.76 B of it per token and the hot set — the experts a given repository or language actually uses — is a small neighbourhood.

### Making the hot set pay

Measurement from the systems literature, not from us:

| Technique | Reported gain | Source |
|---|---|---|
| Predictive prefetch (2 layers ahead) + hot/stage cache | **5.71×** throughput on a constrained GPU | Mira, 2026 |
| Hot experts resident at higher precision, cold at lower | **2.73×** over offloading | DynaExq, 2026 |
| Balance *activated experts per GPU*, not tokens, in decode | **11–22 %** lower decode latency | METRO, 2025 |

The third is the counter-intuitive one and it applies directly here: in the memory-bound decode regime, balancing tokens across GPUs *increases* the number of distinct experts touched per GPU, which increases bytes read. Balancing experts instead is strictly better. With 1:32 activation this matters more than usual, because the whole bet is on few experts per step.

### Expert parallelism

96 experts across 8 GPUs = 12 per GPU, 2.6 GB per GPU at FP8. Every token needs 3 experts; on average 1.9 of them will be remote, so there is an all-to-all on every layer. At batch 16 that is 48 remote dispatches per layer — small messages, and the tax is latency, not bandwidth.

That is why the deployment default is **batch 16–64 on one node**, not batch 1 across eight: at batch 1 the all-to-all latency dominates and the model is slower than a dense 4 B would be. The MoE tax is real (MLSys 2026 measures 2–3× worse than dense at equal per-token compute), and the way to pay it is to batch.

---

## 8.3 A turn, and what it costs

800 new tokens in, 600 out, 32 k of conversation:

| Stage | Work | Time |
|---|---|---|
| Prefill (window 4 k) | 800 × 2 × 1.76 GFLOP ≈ 2.8 TFLOP | **2.4 ms** |
| Decode 600 tokens | 600 × 0.55 ms | **0.33 s** |
| Test run | pytest | 10–60 s |
| **Model time per turn** | | **~0.33 s** |

A 20-turn session generates ~12,000 tokens:

| | Value |
|---|---|
| GPU-seconds per session (batch 16, amortised) | **0.51 s** |
| Cost per session at $2.20/GPU-h | **$0.0003** |
| Cost per 1 M generated tokens | **$0.026** |

The model is 0.4 % of the wall clock of a turn and 0.03 % of its cost. **Everything expensive about this product is the verifier**, which is the argument [05](05-reasoning.md) is built on: spend model compute freely, because it is nearly free relative to running a test suite.

---

## 8.4 Session state

4.2 MB of fast weights per session, plus 36.9 MB of window KV. An 80 GB H100 with 20.5 GB of weights has ~55 GB free:

```
55 GB / (36.9 MB + 4.2 MB) ≈ 1,300 resident sessions
```

In practice the window KV is shared across sessions with the same conversation prefix, and the limit is compute, not memory: 16 concurrent sessions is what batch 16 serves at full throughput. **Memory is no longer the binding constraint on concurrency** — which it was in every previous design, where a 256 k session cost 2.36 GB.

Practical limit: 32 concurrent sessions on one GPU before latency degrades, at which point the scheduler should queue rather than thrash the expert cache.

---

## 8.5 Quantisation

| Format | Weights | Quality | Use |
|---|---|---|---|
| BF16 | 40.96 GB | reference | eval only |
| **FP8 (E4M3)** | 20.48 GB | < 0.3 % | server default |
| 4-bit AWQ | 10.24 GB | 0.5–1.5 % | 24 GB cards, edge |

One component must not be quantised aggressively: **the router**. A 0.1 % shift in router logits flips top-3 selections, and a flipped selection changes which expert's weights are read — a discrete, discontinuous error. Router stays FP8 at minimum, and never 4-bit. It is 0.4 % of parameters, so there is no reason to economise.

The memory state (`B`) is FP16. It is 4.2 MB; quantising it would save nothing and cost recall.

---

## 8.6 Failure modes

| Failure | Symptom | Fix |
|---|---|---|
| **Expert cache thrash** | latency spikes to 10× when two repositories alternate | pin per-tenant hot sets; partition by repo |
| **Straggler expert** | one GPU 40–80 % slower in prefill | METRO-style balance of activated experts |
| **Batch-1 latency cliff** | single user sees 8 ms/token | never serve batch 1 with EP > 1 |
| **State corruption across sessions** | stale facts bleed between users | state is per-session, never pooled; assert isolation |
| **Window truncation** | model "forgets" the current file mid-edit | window is 4 k — a very long file exceeds it; chunk and rely on state |
| **Router drift after fine-tune** | utilisation shifts, hot set changes | re-profile hot set after every SFT run |

The fifth row is a real product constraint: a 4,096-token window means a 3,000-line file cannot be held whole. The store and the state are supposed to carry it, but this is the scenario B2 must test, because it is where the design is most likely to fail in practice.

---

## 8.7 Why this deployment is unusual

Most MoE deployments fight the same two problems: all experts must be resident, and large batches activate all of them. At 1:32 the model dodges both — 24 GB holds a 20 B model, and batch 64 still leaves experts cold.

The price is paid twice: once in the serving tax (2–3× vs dense at equal compute) and once in quality risk (the model may not be as good as a dense model of the same *active* size). Both are budgeted, and both are measured in [09](09-evals-and-risks.md).

**Next:** [09 — Evaluations and risks](09-evals-and-risks.md).
