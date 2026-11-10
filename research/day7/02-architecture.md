# 02 — The 3:1 hybrid: what it is, what it beat, how to refute it

## The layout

Kestrel-7 arranges layers in repeating four-layer macro-blocks:

```
[ gdn, gdn, gdn, full ] [ gdn, gdn, gdn, full ] ...
  4k   4k+1  4k+2  4k+3
```

* **Gated DeltaNet** (`GatedDeltaNetLayer`) keeps an O(1) recurrent state per head.
* **Full/sliding gated attention** (`GatedCausalSelfAttention`) every fourth layer performs the
  exact lookup.

This is the Qwen3.5/Qwen3-Next layout, and the Nov-2026 Encode–Address–Reader account gives it a
mechanism: the recurrent layers build a retrieval query in their state; the attention layer reads
it. The design is only worth its complexity if both halves do their job, so each has a standalone
test.

**Alternative it beat:** all-full attention at equal width/depth. It loses on KV memory by 4× for
the same number of KV heads, and the hybrid matches it at equal tokens in the cited work.

**Falsifier:** hybrid validation loss worse than dense at equal tokens *and* compute, or streaming
decode disagreeing with a full forward pass by more than `1e-4`. The second is asserted for
`TINY_HYBRID`; the first needs a real pretraining run this repository does not contain.

## Gated DeltaNet

Per head, per token, with `q̃ = q/||q||₂`, `k̃ = k/||k||₂`:

```
g_t     = exp(-exp(A_log) · softplus(a_t + dt_bias))       retention in (0, 1]
β_t     = sigmoid(b_t)                                     write strength in (0, 1)
S_t     = g_t · S_{t-1} + k̃_t ⊗ (β_t · (v_t − (g_t · S_{t-1})ᵀ k̃_t))
o_t     = S_tᵀ q̃_t / sqrt(d_head)
```

Before the update, a depthwise causal Conv1D (K=4) with SiLU gives local token order without RoPE.
The output is `W_o(RMSNorm_zc(o) ⊙ SiLU(z))`.

**Why the delta rule.** A pure decayed state (Mamba-2 style) can only add outer products; writing
the same key twice leaves both values smeared together. The delta rule first subtracts what the
state already retrieves, so a repeated key is *corrected* rather than accumulated. **Falsifier:**
associative recall at lag 64 below cosine 0.80 (the Day-5 simulator checks this separately).

**Document-boundary reset.** On packed streams the state `S_{t-1}` is zeroed whenever
`doc_id_t ≠ doc_id_{t-1}`. The depthwise convolution needs the same treatment or it smuggles the
previous document's last three tokens into the new document — the first version of this file did
exactly that, and the cross-document test caught a nonzero difference. Conv taps are now masked by
document ID as well, and the test asserts `diff == 0.0` for both the dense and hybrid stacks.

## Attention stability

* **Initialization and output scale:** embeddings are `N(0, 1/√d)`, linear/convolution weights are
  `N(0, 0.02)`, and the final logits are multiplied by a non-parameter `1/√d`. Beat the audit's
  default `N(0,1)` embedding init, whose initial logit std was 21.55 and CE 252 against 5.61 for a
  uniform distribution. The fixed run now starts at CE 5.94 versus uniform 5.95 (TINY_HYBRID,
  384-token tokenizer).
* **Zero-centered RMSNorm:** `y = x · rsqrt(mean(x²)+eps) · (1+γ)`, `γ=0`, fp32 reduction. Beat
  `γ=1` init, which forces every layer to re-learn the residual scale.
* **Per-head QK-RMSNorm** before RoPE. Beat unnormalized QK, whose logit scale grows with
  `head_dim`.
* **Sigmoid output gate:** `y = W_o(Attn(q,k,v) ⊙ σ(W_gate x))`. Beat an ungated output, which lets
  low-rank sink directions dominate early training.

**Falsifier for all three:** an attention-logit or activation-RMS explosion during a long run. None
of these can be validated by a 50-step smoke run; they are stated as invariants with a test of
their *initialization* (γ = 0, gate at 0.5, retention near 0.94), not as measured wins.

## MTP head

Depth-1 multi-token prediction: `h_t` is concatenated with `embed(x_{t+1})`, projected, passed
through a SwiGLU residual, and used to predict token `t+2`:

```
L = L_CE + 0.2 · L_MTP
```

**Alternative it beat:** pure next-token training, which gives no planning signal beyond `t+1`.
**Falsifier:** zero gradient in the MTP head, or MTP loss not decreasing. The gradient is asserted
(`||grad|| > 0`); the loss trend needs a real run.

At inference the same head is a self-speculative draft. That is **planned, not implemented** in this
repository: there is no draft/verify loop, so no speedup is claimed or measured. The training head
itself is implemented and gradient-tested.

## The computed ledger

From `metrics.json` (regenerate with `python3 research/day7/tools/param_ledger_v7.py`):

| Preset | Parameters | KV bytes (fp16) | All-full counterfactual | GDN state (fp32, b=1) |
|---|---|---|---|---|
| `TINY_DENSE` | **6,032,128** | **1,048,576** | 1,048,576 | 0 |
| `TINY_HYBRID` | **6,250,192** | **262,144** | 1,048,576 | **125,952** |
| `SMALL_HYBRID` | **37,844,704** | **2,097,152** | 8,388,608 | **897,024** |
| `BASELINE_100M` | **105,062,040** | **12,582,912** | 50,331,648 | **2,018,304** |

KV arithmetic for `BASELINE_100M`: 3 full layers × 2 (K,V) × 4 kv heads × 64 head_dim × 4096 ctx ×
2 bytes = 12,582,912 B. Replacing the 9 GDN layers with attention would cost 50,331,648 B, so the
hybrid removes **75.0%** of per-sequence KV traffic. The recurrent state is fixed in sequence length
— it is charged once, not per token.

`TINY_DENSE` is the control: same width and depth as `TINY_HYBRID`, no recurrent layers, no KV
saving. It exists so "hybrid helps" can be tested against an equal-parameter dense sibling rather
than against nothing. The two differ by 218,064 parameters (the GDN mixers versus the three extra
attention layers); the parameter ledger prints both, because an unequal comparison is not a
comparison.

## What is *not* claimed

No pretraining-scale result is claimed. The only measured architecture facts in this repository are:
streaming equivalence (`< 1e-4`), exact zero cross-document leakage, MTP gradient flow, initial CE
within 0.02 of uniform, and the computed parameter/KV ledger. Everything else is a specification
with a falsifier. The 50-step run is a mechanism check and memorizes its 38-document corpus; it is
not a capability result.
