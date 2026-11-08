# 4 — Memory

> Two of the three memory tiers are neural and one is not. The neural ones are 4.2 MB per session; the non-neural one is the reason the other two are allowed to be lossy.

---

## 4.1 L1 — the window

4,096 tokens, exact, every layer. `2 × 4,096 × 4,096 × 4 bytes` per head-batch is not the number that matters; this is:

```
KV per token per layer = 512 (latent) + 64 (RoPE) = 576 floats
× 16 layers × 4,096 window = 36.9 MB at FP8 — fixed, forever
```

Attention within the window costs `4 · w · d` FLOPs per layer per token = **3.22 GFLOP**, and it does not grow. That is the entire reason the window is 4,096 and not 256 k.

One non-obvious addition: **one learned attention sink per head**, appended as an extra always-attendable column. Without it, the softmax in early positions has nowhere to put probability mass when the window is nearly empty, and the first tokens of every sequence get distorted. Cost: 32 parameters per layer.

---

## 4.2 L2 — the delta rule, and why not a bigger state

The long-term memory is a **fast-weight matrix**, low-rank, updated online by a gradient step, read by a query:

```
M = A B,   A ∈ R^{512×256} learned,   B ∈ R^{256×512} per session

read :  y_t = q_t · A · B
write:  err = A B k_t − v_t
        B  ← B − η · (err · A) k_tᵀ
```

That update is one gradient step on `½‖ABk − v‖²` with respect to `B`. Three properties make it the right primitive at this scale:

**No backward pass.** The write is a closed-form outer product. The model does not backpropagate through its own memory update, so training cost per token is unchanged by having a memory at all.

**Constant cost.** `4 · d_m · r = 524,288` MACs per token per layer, 16.8 MFLOP across the model — **0.12 %** of the 13.82 GFLOP forward pass.

**Constant size.** `256 × 512 × 16 layers = 2,097,152` floats = **4.2 MB** at FP16, for a session of any length.

Compare the thing it replaces:

| | This design | Dense KV |
|---|---|---|
| 256 k context | 4.2 MB | 2.36 GB (FP8) |
| 1 M context | 4.2 MB | 9.4 GB (FP8) |
| 500 turns | 4.2 MB | grows every turn |
| **Ratio at 256 k** | | **562× smaller** |

### Why low rank

A full 512 × 512 state would be 8× larger (33 MB/session) and, more importantly, would make the write `O(d_m²)` instead of `O(d_m · r)`. Rank 256 is the smallest power of two that keeps the recall test in §4.5 above 0.5 cosine. It is a tuned constant, not a derivation, and B2 varies it.

### Why read-before-write

The read at position `t` uses the state produced by tokens `< t`. If the write happened first, the model could read the answer it is about to be asked for — a subtle form of leakage that would show up as a suspiciously good training loss and a mysteriously bad eval. `read_before_write: true` in the config exists to make that impossible by construction.

### Why the state must not be trained by backprop

If `B` were a parameter, it would be a 4.2 MB per-sample activation gradient — which is what makes TTT-style layers expensive. Making it a *closed-form update* means the memory is part of the forward pass, not part of the graph. The trade: no credit assignment through the memory. If a write at token 100 harms a prediction at token 1,000, nothing can teach it not to. That is the design's main modelling weakness and is listed as such in [09](09-evals-and-risks.md).

---

## 4.3 Training a memory that is not a parameter

The naive version is a sequential loop over every token — unusable. The practical version comes from *TNT* (ICLR 2026), which decouples training efficiency from inference quality:

- **Chunk the sequence** (512 tokens). Update the state once per chunk boundary using the chunk's summed delta.
- **Reset periodically** (every 8 chunks) to break the sequential dependency, so chunks can be processed in parallel across devices.
- **Two-stage**: train most tokens with large chunks and resets; finish with a short phase at the inference chunk size and no resets.

TNT reports **up to 17× faster training** than the most accurate sequential baseline while improving quality. Without it, this layer would not be trainable at [06](07-training.md)'s budget.

The chunked objective is not identical to the sequential one — resetting state discards long-range credit. The fine-tuning phase at the end is what closes that gap, and B2 measures the cost.

---

## 4.4 L3 — the store, which is not a neural network

The evidence in [01](01-the-validity-problem.md) says memory systems fail on validity. No amount of matrix size fixes that, because the fix is a schema:

```jsonc
{
  "key":       "pkg.auth.Session.refresh",       // canonical symbol, parser-derived
  "claim":     "returns None when the token is expired; callers must check",
  "evidence":  {"kind": "test", "id": "tests/test_auth.py::test_expired", "passed_at": "a91f2c3"},
  "validity":  {"content_hash": "b7e1...", "commit": "a91f2c3"},
  "confidence": 0.82,
  "writes": 3, "hits": 11, "last_hit": "turn 47"
}
```

### Write path

The model emits a `remember` tool call. The harness — not the model — fills `evidence` and `validity` by hashing the symbol body and recording the commit. **A write without evidence is rejected.** That single rule implements the Anthropic finding that unconstrained recording degrades rather than helps.

### Read path

1. Retrieve by symbol key (exact) and by embedding (fuzzy), merge.
2. **Recompute `content_hash` for each candidate.** Mismatch → mark stale, do not surface, log.
3. If `evidence.kind == test` and the file changed since `passed_at`, optionally re-run the test (budgeted).
4. Surface survivors with their confidence and evidence, in that order.

Step 2 costs one hash per candidate. It is the cheapest and highest-value operation in the whole system, because it converts "the model must notice its memory is stale" into "the harness tells the model its memory is stale".

### The controller is trained

The model learns three decisions, and all three are trained on a corpus that does not otherwise exist:

| Decision | Signal |
|---|---|
| when to write | later-session tasks that need the fact |
| when to trust | staleness-injected sessions where the fact was silently changed |
| when to invalidate | tasks where using the stale fact fails the tests |

**Staleness-injected replay.** Take a real multi-session trajectory. Between session `k` and `k+1`, mutate a symbol that session `k` depended on, in a way that changes behaviour but keeps the file compiling. The correct behaviour in session `k+1` is to notice and re-derive. This is synthetic data of a kind that cannot be scraped, and it is the training signal for the only skill the failure literature says models lack.

---

## 4.5 Correctness, executed

The memory module is not asserted to *exist*; it is asserted to *learn*. From `model.py`:

The module is tested on the property that defines it: **does repeated writing of one association converge, and does it degrade gracefully as you load it?** Two measurements from `model.py`, both cosine similarity between what the memory returns and the value it was given.

There is a ceiling, and it is worth knowing before reading the numbers: with `A` random and fixed at test time, the state can only represent the **rank-`r` projection** of the value, so expected recall is bounded near `√(r/d_m)`. At `r=256, d_m=512` that is **0.71** — a memory that recovers 71 % of what it stored is behaving correctly, not broken. In the trained model `A` is learned, so the subspace is chosen rather than random, and the ceiling moves; the test measures the update rule, not the trained ceiling.

The first run of this test failed, and the failure was informative. Probing with `mem.q(x)` rather than the normalised key returned a cosine of **−0.61**: the update rule had converged on the value, but the read used an *unrelated* random projection, so `kᵀq ≈ 0` and the returned vector was numerical noise. Two real bugs came out of it — a transposed update (`Aᵀ err` versus `err A`) and a read in the wrong order (`q M` instead of `M q`, which makes the memory non-associative). A memory layer that is *almost* right returns plausible-looking vectors and quietly stores nothing; this is why the test measures recall rather than shape.

The other memory assertions: the state is the same shape for every sequence length (`[5]`), and the state is 4.2 MB rather than 2.36 GB at 256 k.

---

## 4.6 Failure modes

| Failure | Looks like | Detection | Response |
|---|---|---|---|
| **State saturation** | recall plateaus early, then degrades as `‖B‖` grows | track `‖B‖` per 1 k tokens | normalise `k`, decay `B` |
| **Recency overwrite** | good at 4 k, poor at 64 k | probe with needles at depths 10/50/90 % | increase rank or add per-layer state |
| **Silent staleness** (L3) | model confidently uses superseded facts | hash mismatch rate, task failures on injected sessions | this is what L3 exists to prevent |
| **Write spam** | store grows 10× faster than hits | hits/writes ratio per entry | confidence threshold on `remember` |
| **Memory inversion** | reads dominate, model stops using attention | ablation: zero the state, measure delta | regularise the read gate |

The fourth row is the one that killed several published agent-memory systems: more memory is not better if most of it is never used and dilutes retrieval.

---

## 4.7 Rejected

| Option | Why not |
|---|---|
| **Full KV cache to 256 k** | 2.36 GB per session, and it is the dominant cost at every batch size |
| **Titans-style full MLP state** | more expressive, but the per-layer state would be 10–30 MB and the write needs backprop |
| **TTT-E2E (train the whole network at test time)** | strongest long-context results in 2026, but requires an inner training loop at inference — unusable inside a coding agent's turn budget |
| **RAG only** | no state at all; every turn pays full prefill and nothing is learned from the session |
| **Product-key memory (PEER / Memory Layers)** | a genuine option — near-zero FLOPs per token, and Memory Layers reports beating dense models with 2× compute — but it stores *facts in weights*, which cannot be invalidated per-repository. Wrong tool for the validity problem |

That last rejection is the expensive one: product-key memory is the most efficient capacity mechanism available, and this design leaves it on the table because its entries cannot carry a `commit`.

**Next:** [05 — Reasoning](05-reasoning.md).
