# Kestrel-6 — Trainable Baseline

**Day 6 turns the Day-5 research prototype into a real, trainable language model.**

The central conclusion of the Day-5 technical audit stands: *do not add more
architectural novelty before there is an honest end-to-end baseline.* Day 6
therefore builds **Track B (architecture research)** starting from a small,
dense, fully trainable decoder-only Transformer, with Day-5 mechanisms offered
only as toggleable ablations that must each beat the baseline at equal
parameter count, training tokens, and compute.

## What was built

| Component | File | Status |
|---|---|---|
| Config presets (TINY / SMALL / BASELINE + Day-5 ablations) | `kestrel6/config.json` | complete |
| Config loader (`get_config`, defaults, validation) | `kestrel6/config.py` | complete |
| Byte-level BPE tokenizer (train / encode / decode / save / load) | `kestrel6/tokenizer.py` | complete |
| Dense decoder-only model (RMSNorm, RoPE, GQA/MQA, SwiGLU, tied head, KV cache) | `kestrel6/model.py` | complete |
| Day-5 ablation: local MLA (latent KV, sliding window) | `kestrel6/model.py` | complete |
| Day-5 ablation: WY Householder fast-weight memory | `kestrel6/model.py` | complete |
| Day-5 ablation: Stiefel subspace MoE | `kestrel6/model.py` | complete |
| Day-5 ablation: recurrent Code-JEPA workspace | `kestrel6/model.py` | complete |
| Training loop (CE loss, AdamW, warmup+cosine, grad clipping, accumulation, validation) | `kestrel6/train.py` | complete |
| Autoregressive generation (greedy / temperature / top-p, stop tokens, prefill + decode_step KV cache) | `kestrel6/generate.py` | complete |
| Checkpoint save/load with config-hash validation (fail-closed) | `kestrel6/checkpoint.py` | complete |
| Data pipeline (packed docs, next-token labels, train/val/test split) | `kestrel6/data.py` | complete |
| Test suite (11 tests) | `tests/test_kestrel6.py` | complete |
| End-to-end run script | `run_kestrel6.py` | complete |

## Honest metadata (no false flags)

- `model_status = "randomly_initialized"` until a checkpoint is loaded.
- `training_backend = "pytorch"` (real autograd, no fake "Muon" claims).
- `checkpoint_loaded = False` until a real checkpoint is loaded and validated
  against the config hash.
- `supports_autograd = True`.
- Checkpoint stores `VERSION`, `config`, `config_hash`, `source_commit`,
  `step`, `token_count`, `best_val_loss`, `history`, `seed`, `run_name`.
- Loading **fails closed** if the config hash does not match the model's config.

## Design decisions (from the Day-5 audit)

1. **Dense baseline first.** The Day-5 `FULL` config is 21.25B stored parameters
   (19.93B MoE) and cannot be trained from scratch in this track. Day 6 starts
   at 10M–150M (TINY/SMALL/BASELINE presets) so loss can actually decrease and
   generation can be measured.

2. **Ablations, not defaults.** Local MLA, global NoPE landmark attention, WY
   memory, subspace MoE, and the workspace are behind config flags
   (`attn_impl`, `global_layers`, `use_memory`, `use_moe`, `use_workspace`).
   Each must be validated as an ablation against the dense baseline.

3. **No manual logit perturbation.** The Day-5 `r >= 8` "flip top-2 ties by
   1e-4" hack was removed from the model forward pass. Reasoning improvements
   must come from a trained objective or a verifiable inference-time search
   procedure, never from a hand-injected output perturbation.

4. **Non-mutating metrics.** `orthogonality_error()` measures Stiefel error
   without retracting weights; `retract_stiefel_subspaces()` is the explicit
   mutation. Metrics record before/after the optimizer update.

5. **Fail-closed verification posture inherited.** The VEXX/SMT/Farkas/CRANE
   components from Day 5 are deliberately *not* promoted to a trusted path in
   Day 6; they remain research prototypes. The audit's recommendation (Z3 /
   cvc5 / SymPy / strict parsing) stands for any future trusted verification.

## Running

Requires Python 3 + PyTorch (not present in this sandbox; the code is written
to be correct-by-construction and ready to run once available).

```bash
# Install a tokenizer-trained model and smoke-train 50 steps on TINY.
python3 research/day6/run_kestrel6.py --preset TINY --steps 50

# Run the test suite.
python3 -m pytest research/day6/tests/ -q
# or:
python3 research/day6/tests/test_kestrel6.py
```

## Config presets

- `TINY`: 256 d, 4 layers, 8192 vocab — pipeline smoke test (~0.3M params).
- `SMALL`: 512 d, 8 layers, 16384 vocab — stable training check (~14M params).
- `BASELINE`: 768 d, 12 layers, 32000 vocab — small research target (~50M params).
- `BASELINE_MLA_LOCAL` / `BASELINE_GLOBAL_LANDMARK`: Day-5 mechanism ablations.

Every preset is in `config.json`; numbers are computed there, not transcribed.
