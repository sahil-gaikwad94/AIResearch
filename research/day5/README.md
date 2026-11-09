# Kestrel-5 × VEXX 2.0 — Day 5

The Day 5 deliverable: a neuro-symbolic architecture (spectral-gated
Compact-WY Householder fast-weight memory, two-tier NoPE sparse-landmark
attention, Stiefel-orthogonal subspace MoE, a KV-cached Code-JEPA recurrent
workspace) with a verification layer (VEXX 2.0: tiered masks, exact-rational
Farkas/CDCL cuts, L3 proof certificates) designed to be *ceiling-capped in
code*, not in prose.

Every number in this directory is computed by code from the canonical config
and asserted. Nothing here is a guess with units attached.

## Verification gate

| command | what it proves |
|---|---|
| `python3 kestrel5/model.py` | 15 self-assertions over the reference implementation |
| `python3 tools/count_params_v5.py` | 19 ledger/KV/FLOP/serving/VRAM checks |
| `python3 tools/sim_verify_v5.py` | disjoint FPR split (3 arms, MC-checked) + Farkas/CDCL ladder |
| `python3 tools/audit_v5.py` | 77 adversarial checks incl. an independent FLOP re-derivation |

All four exit 0 with NumPy/CPU only (no PyTorch dependency).

## Documents

| file | subject |
|---|---|
| `01-what-day4-missed.md` | the seven defects (F1–F7) and their mechanisms |
| `02-architecture.md` | full trunk, parameter ledger, KV footprint, FLOP/serving model |
| `03-fast-weight-memory.md` | WY Householder memory and the spectral gate |
| `04-attention-and-sparsity.md` | two-tier NoPE landmark attention and the KL-aligned indexer |
| `05-moe-and-upcycling.md` | Stiefel subspace MoE, 2×3 vs 6×1, zero-shock upcycling |
| `06-workspace-and-jepa.md` | Code-JEPA recurrent workspace and the cache-once fix |
| `07-verification-and-farkas.md` | VEXX 2.0 controller, exact Farkas cuts, certificates |
| `08-code-fpr.md` | the disjoint code false-positive-rate split and calibration protocol |
| `09-serving-and-free-tier.md` | serving model, NANO tier, VRAM budgets |
| `10-day6-preview.md` | what Day 6 must build and what it must not claim |

## Delta list: the uploaded reference draft vs the code here

The uploaded reference draft was treated as a draft, not gospel. The genuine
defects found in it and what this build does instead:

| # | draft claim / behaviour | status in this build |
|---|---|---|
| 1 | Farkas cut supports only an upper bound (`a_c > 0`) | **fixed** — a negative culprit coefficient implies a *lower* bound; the controller now routes it correctly |
| 2 | `Kestrel5Torch` class | **removed** — no such class exists; the reference is pure NumPy |
| 3 | "sparse gather execution" | **removed** — the shipped implementation runs dense matmuls against an exact sparse *mask*; equivalence is the claim, and the audit proves the sparsity budget |
| 4 | "exposes a 3-depth MTP draft tree" | **removed** — the ledger contains no neural MTP draft heads; MTP is specified in doc 06 and lands in Day 6 |
| 5 | indentation tracking | **removed** — not present in the automaton |
| 6 | a magic `-0.05` cut | **kept only in the simulator** as the Day 4 baseline to be falsified |
| 7 | `0.218 GFLOP` per extra recurrent step | **reproduces exactly** (`2·4·27,266,048 = 218,128,384`) |
| 8 | `0.307 GFLOP` pre-cache saving | **does not reproduce** from any per-step decomposition we can build — flagged in doc 02 and the audit |
| 9 | `12,216 tok/s` | **a target, not a measurement** — under fp8 it is ≈74.9% MBU; under fp16 it exceeds the hardware ceiling (147% MBU). Labelled everywhere |
| 10 | 1M-session footprint `144.70 MB` | **specified ledger kept** and asserted; separately reported a *window-complete* `161.48 MB` that accounts for the 4,096-key window the global layers actually attend over |
| 11 | zero-shock upcycling `top_k_per_group` doubles | **kept and documented** — the honest transient 2× routed cost at the split step |
| 12 | config key `mtp_Spec_depth` | renamed to `mtp_spec_depth` |

## Literature groundings (design provenance, not measured results)

These are the published techniques the design borrows from; this directory
does not re-prove them.

- **NSA** (ACL 2025): compression / selection / sliding-window branches → the
  two-tier sparsity budget and the dense-mask equivalence argument in doc 04.
- **DeepSeek-V3.2 DSA** (Lightning Indexer, KL supervision, block size 64,
  top-2048): the indexer-KL objective and block granularity in doc 04.
- **Grazzi et al.** (ICLR 2025): negative eigenvalues in recurrent state
  tracking → the Householder reflection gate in doc 03.
- **DeltaNet / WY chunkwise** (Yang et al. 2024; DeltaProduct, NeurIPS 2025):
  the Compact-WY product and `n_h` Householders per token in doc 03.
- **DeepSeekMoE** (2024): fine-grained + shared experts → doc 05.
- **ALF-LB sign rule** (Wang et al., arXiv 2408.15664): the load-balance bias
  update that never enters the LM gradient → doc 05.
- **Mündler et al.** (PLDI 2025), **SynCode**, **XGrammar**: type/grammar-
  constrained decoding → the tier-0/1 mask family in doc 07.

## Determinism and honest unknowns

- All seeds are pinned; identical runs are bit-identical (the audit checks it).
- Per-class FPR residuals in doc 08 are *calibrated operating points*, not
  measurements — the protocol for measuring them is stated there.
- GPU-real measurement (kernel timings, real tok/s) is deferred to Day 6; the
  serving numbers here are bandwidth arithmetic, labelled as such.
