# 04 — Scaling and Training: How These Things Actually Get Built

*Read time: 70 minutes. Prerequisites: `01` §1.8, `02`.*

---

## 4.1 The shape of a frontier training run

A frontier pretraining run in 2026 has roughly this pipeline, and every stage is a research area:

```
1. DATA ACQUISITION      crawl, license, scrape code, books, papers, video
2. FILTERING             quality classifier, dedup (exact + fuzzy), safety filters,
                         decontamination against benchmarks
3. MIXING                domain weights (web/code/math/multilingual), curriculum
4. TOKENIZATION          train BPE, verify compression rate per language
5. ARCHITECTURE          pick the stack (`02`, `03`), size via scaling laws
6. INFRASTRUCTURE        cluster, parallelism plan, checkpointing, fault tolerance
7. PRETRAINING           1–40T tokens, weeks–months, thousands of GPUs
8. MID-TRAINING          long-context extension, annealing on high-quality data
9. POST-TRAINING         SFT → preferences → RLVR → agentic RL (`05`, `06`)
10. EVALUATION           internal + external, red-team, contamination checks (`14`)
```

Most people underestimate stages 2, 3, 8. **Data quality and the annealing recipe are, by most accounts, worth more than architecture choices** at equal compute. This is the least-published and most-valuable part of the field.

---

## 4.2 Data: the real bottleneck

### Scale

- GPT-3 (2020): 300B tokens.
- Chinchilla (2022): 1.4T tokens.
- Llama 2 (2023): 2T. Llama 3 (2024): **15T**. Llama 4: ~30T+ (reported).
- DeepSeek-V3 (2024): 14.8T. Qwen3: ~36T. 2026 flagships: some report 30–40T.

Meanwhile, the stock of *high-quality* human text is estimated at 10–100T tokens depending on how generously you define quality. **We are approaching the end of the easy supply** — hence "the data wall" (`17`).

### Filtering: what actually matters

1. **Deduplication** is the single highest-value filter. Exact-hash dedup (URL, then document) plus **fuzzy dedup** (MinHash/LSH near-duplicate detection) removes the enormous boilerplate redundancy of the web. Repeated documents cause memorisation and waste compute.
2. **Quality classifiers.** Train a small classifier to distinguish "high-quality" text (e.g. Wikipedia, books, curated web) from random web text; keep the top-scoring fraction. This is what "FineWeb-Edu", "DCLM", and similar datasets do. A classifier score threshold is a compute/quality tradeoff, not a moral one.
3. **Heuristic filters**: length, symbol-to-word ratio, presence of boilerplate ("lorem ipsum", "enable javascript"), stop-word density, n-gram repetition.
4. **Decontamination**: remove text overlapping benchmark test sets. Increasingly seen as *necessary but insufficient* — see `14` on why models still seem to know the answers.
5. **Safety filtering**: remove PII, CSAM, and (policy-dependent) harmful content. In practice this is a large-scale classifier pipeline with hard tradeoffs on multilingual coverage.

### Mixture and curriculum

You don't just concatenate everything. The **data mixture** (what fraction is code, math, multilingual, academic, web) strongly affects downstream capability — code and math improve reasoning, high-quality prose improves writing, multilingual data improves transfer. Modern practice:

- **Upweight** code and math beyond their natural web frequency (often 10–30% of tokens).
- **Curriculum / annealing**: in the final ~5–10% of pretraining, switch to a much higher-quality mixture (curated web, textbooks, instruction-like data), lower the learning rate, and extend context. This "mid-training" phase is where a lot of the perceived quality is won. DeepSeek-V3, Llama 3, and Qwen all describe variants.
- **Repeat policy**: high-quality sources are repeated 2–4×; low-quality sources are repeated never.

---

## 4.3 Parallelism: the six axes

You cannot fit a frontier model in one GPU's 80–288 GB. So you split it. Each axis splits a different thing, and each has a different communication pattern (the expensive part).

| Axis | Splits | Communication | Use when |
|---|---|---|---|
| **Data parallel (DP)** | the *batch* | gradient all-reduce per step | always (base case) |
| **ZeRO / FSDP** | optimizer state + gradients + params across DP ranks | sharded all-gather / reduce-scatter | DP with models too big for replication |
| **Tensor parallel (TP)** | individual matrices (columns/rows) | all-reduce inside every layer — **very chatty**, needs NVLink | within a node |
| **Pipeline parallel (PP)** | *layers* into stages | point-to-point activations between stages; bubble overhead | across nodes |
| **Expert parallel (EP)** | MoE experts | **all-to-all** per MoE layer | MoE models |
| **Context parallel (CP)** | the *sequence* | ring attention (pass K/V or Q between ranks) | long context |

**The standard recipe** for a large dense model: TP within a node (8 GPUs), PP across nodes, DP/FSDP across everything else → "3D parallelism". For MoE, add EP. For 1M-token training, add CP.

**ZeRO stages** (Rajbhandari et al. 2020) are worth knowing precisely:
- Stage 1: shard optimizer states (Adam has 2 moments + fp32 master weights = 12 bytes/param, huge).
- Stage 2: shard gradients too.
- Stage 3: shard parameters too (→ this is **FSDP** in PyTorch: parameters are all-gathered per layer just before use, then freed).

Rule of thumb: a 70B model in BF16 with Adam needs `70e9 × (2 weights + 12 optimizer) = ~980 GB` of state — more than 12 H100s just to *hold*, before activations. This is why sharding is mandatory, not optional.

**Checkpointing and faults.** At 10k-GPU scale, hardware failures happen hourly. Training stacks checkpoint asynchronously to remote storage every few hundred steps and resume automatically; the engineering of *not losing progress* is a substantial fraction of the infra work. Meta's Llama 3 paper reported 419 unexpected interruptions in a 54-day run — and that's normal.

---

## 4.4 Numerics: BF16, FP8, FP4

**Precision** determines both speed and stability.

- **FP32** (32-bit float): the old default. Accurate, slow, memory-hungry.
- **BF16** (bfloat16): 8 exponent bits (same range as FP32), 7 mantissa bits. The 2022–2024 default for training. Range matters more than precision for gradients.
- **FP16**: more mantissa (10 bits), less range — needs **loss scaling** to avoid underflow. Mostly historical.
- **FP8** (E4M3 for weights/activations, E5M2 for gradients): Hopper-era. First demonstrated at frontier scale by **DeepSeek-V3**, which trained a 671B MoE in FP8 with careful **per-block scaling** and periodic BF16 promotion for sensitive operations, reporting no loss degradation and ~2× throughput. FP8 training is now standard.
- **FP4 / NVFP4**: Blackwell-era. 4-bit floats with per-block microscaling. Currently used mostly for *inference*; FP4 *training* is an active research area with reported successes in 2026 but is not yet the default for frontier runs.

**The stability playbook** (this is where practitioners earn their money):
- QK-norm or logit soft-capping to stop attention logits exploding.
- Careful initialisation (and **muP**, below).
- Gradient clipping (typically 1.0).
- **z-loss** on the softmax denominator to prevent logit drift.
- Monitoring: grad-norm spikes, loss spikes, activation outliers. When a loss spike occurs, the standard fix is to roll back to the last checkpoint, skip the offending data batch, and continue — which is exactly what several lab papers describe.

---

## 4.5 muP: hyperparameter transfer

**Problem:** you tune learning rate on a 100M model, then train a 70B model, and it diverges. Tuning at scale costs a fortune.

**Maximal Update Parametrisation (µP)** (Yang et al. 2021, "Tensor Programs V") gives a prescription for how to scale initialisation, learning rate, and output multipliers with width/depth so that the **optimal hyperparameters are invariant to scale**. Result: tune once on a small proxy, transfer to the big model, and it works.

µP is now standard practice in serious labs because it converts hyperparameter search from "impossible at scale" to "do it small, transfer". The Cerebras-GPT paper (2023) demonstrated the scaling laws it enables. 2025 work extended this logic to matrix-preconditioned optimisers (Muon, SOAP, Shampoo): with µP-style learning-rate scaling plus blocking/spectral normalisation, these achieve ~**1.4× speedup over AdamW** consistently from 190M to 1.4B, whereas the speedup vanishes under incorrect scaling. That's a clean example of "the scaling rule *is* the result".

---

## 4.6 Optimisers beyond AdamW

- **AdamW** (2017/2019): decoupled weight decay; the workhorse.
- **Lion** (2023): sign-based momentum; memory-efficient, sometimes better, less stable.
- **Sophia**: second-order-ish with a diagonal Hessian estimate; reported ~2× speedups, mixed adoption.
- **Muon** (Keller Jordan, 2024): applies **Newton–Schulz orthogonalisation to the momentum matrix** before the update — i.e. it takes the gradient's update direction and makes it "as orthogonal as possible" across the matrix's singular directions. Gives ~1.4× compute efficiency at matched loss in multiple independent studies, now used in production by at least one frontier lab and reproduced widely.
- **Distributed Muon / SOAP / Shampoo**: matrix-preconditioning at scale, with the µP-style scaling rules above.

The general trend: **the optimiser is now a first-class scaling lever** alongside data and architecture, after a decade where Adam was untouchable.

---

## 4.7 Synthetic data: what works, what doesn't

**Why:** high-quality human text is finite; models can generate more.

**What the evidence says (2025 systematic study, >1,000 models, >100k GPU-hours):**

1. **Rephrased synthetic data alone is not faster than natural web text.** Generating paraphrases of your corpus doesn't add information.
2. **A mixture of ~1/3 rephrased synthetic + 2/3 natural web text can accelerate pretraining 5–10×** to reach a given loss. The mixture is the trick, not the synthetic data.
3. The optimal synthetic ratio **depends on model size and data budget**, empirically converging around **~30%** for rephrased data.
4. **Pure generated "textbook" data degrades at scale** — it teaches the model the generator's style and biases, and quality collapses in later epochs (the "model collapse" concern, which is real but *conditional* on naive recursive training without fresh data or filtering).

**What works in practice:**
- **Distillation from a stronger model** (generating high-quality reasoning traces, code, or explanations) — this is now a core part of post-training (`05`, `06`).
- **Verifier-filtered generation**: generate many candidates, keep only those that pass a checker (unit tests, symbolic solver, proof assistant). This is the highest-value synthetic data because correctness is *checked*, not assumed.
- **Environment rollouts**: agent trajectories in sandboxes (`07`).
- **Self-play with a formal spec** (e.g. Self-play SWE-RL, 2025–26: agent injects bugs then fixes them, +10.4 on SWE-bench Verified with no human-labelled data).

**The rule:** synthetic data is worth what your verifier is worth. Unverified synthetic data mostly recycles; verified synthetic data adds.

---

## 4.8 Distillation

**Knowledge distillation** trains a small **student** to match a large **teacher**. Variants:

- **Logit distillation**: match the teacher's full soft distribution (KL divergence). Richer signal than hard labels; needs teacher logits.
- **Sequence distillation**: train on teacher-generated text (the dominant form now — "generate 1M high-quality solutions with the big model, SFT the small one").
- **On-policy distillation**: student generates, teacher scores/corrects — closes the distribution mismatch.
- **Reasoning distillation**: DeepSeek showed that distilling R1's *reasoning traces* into Qwen/Llama 7B–70B transferred much of the reasoning ability — cheaper and often better than running RLVR on the small model directly.

2025–26 also saw **quantization-aware distillation** (QAD) and **disaggregated quantization** work: train separate prefill and decode weights at different bit-widths (compute-native FP4 prefill, 2–4-bit decode) so each phase gets the format it needs. Reported gains over weight-only quantization are large at 1–2 bits.

---

## 4.9 What a frontier run actually costs

Rough 2026 arithmetic (public estimates; treat as order-of-magnitude):

- **Compute**: `6 × N_active × D`. [Note: for MoE, FLOPs scale with *active* parameters, but memory with *total*.]
- **Hardware**: H100 ≈ $25–40k; GB200 NVL72 rack ≈ $3M for 72 GPUs + 36 Grace CPUs; GB300 similar. Cloud rental: $2–4/GPU-hour for H100-class.
- **A 70B dense model on 15T tokens**: `6 × 7e10 × 1.5e13 = 6.3e24` FLOPs. At 400 TFLOP/s effective = 1.6e10 GPU-seconds ≈ 500 GPU-years ≈ **$15–40M** at cloud rates, less if owned.
- **A 1T-parameter MoE with 30B active on 30T tokens**: `6 × 3e10 × 3e13 = 5.4e24` FLOPs — *similar compute* to the 70B dense, but the *memory* requirement is 100× larger. This is the MoE bargain in one line.
- **Total frontier run (2025–26)**: credible public estimates are $0.5–2B for the very largest, including data, staff, failed runs, and evaluation. Training compute for the largest runs is estimated in the 10^25–10^26 FLOP range.

**Power**: a 100k-H100 datacenter draws ~100+ MW. This is why "AI factories" and grid interconnection queues became a policy topic (`17`).

---

## 4.10 Training dynamics worth knowing

- **Warmup + cosine decay**: standard schedule — linearly ramp LR over the first ~0.1–1% of steps, then cosine-decay to ~10% of peak. Warmup prevents early divergence.
- **WSD (Warmup–Stable–Decay)**: hold LR constant for most of training, then decay sharply at the end. Advantage: you can *branch* — checkpoint at the stable phase and fork multiple decayed variants (different data mixtures, different lengths) cheaply. Adopted widely in 2024–26 because it makes experimentation affordable. LLaDA2.0 uses a block-level WSD for its AR→dLLM conversion.
- **Batch size**: grows with model size and LR; too small = noisy, too large = worse generalisation per token (there's a critical batch size). Gradient accumulation simulates large batches.
- **Loss spikes**: sudden jumps, usually from outlier data or numerical instability. Diagnose, roll back, skip batch.
- **Memorisation vs generalisation**: repeated data and small datasets induce verbatim memorisation, which is both a privacy and a benchmark-contamination problem.

---

## 4.11 Common misconceptions

- **"Training is the expensive part."** For a deployed model, lifetime *inference* usually dominates total cost. This is why small, heavily overtrained models win commercially.
- **"More data always helps."** Beyond ~4 epochs of the same data, returns collapse; and low-quality data actively hurts by consuming capacity.
- **"Synthetic data will solve the data wall."** Verified synthetic data helps; unverified synthetic data mostly recycles the teacher's errors.
- **"FP8/FP4 training is just a flag."** It requires per-block scaling, selective promotion, and careful validation; a bad scaling scheme silently degrades quality.
- **"Parallelism is a solved infra problem."** Every axis has failure modes; MoE all-to-all and long-context CP are still active engineering.

---

## 4.12 If you remember five things

1. Data filtering, dedup, and the annealing mixture matter more than most architecture choices.
2. `6ND` is the only cost formula you need; MoE decouples compute (active) from memory (total).
3. 3D parallelism + ZeRO/FSDP + EP is the standard topology; communication, not FLOPs, is the constraint.
4. muP lets you tune small and transfer; Muon-class optimisers give ~1.4× at matched loss with the right scaling rules.
5. Synthetic data is only as good as the verifier that filters it.

---

## 4.13 Self-check

1. Why does ZeRO stage 3 need to all-gather parameters per layer rather than keeping them?
2. A 1T MoE with 30B active costs the same training FLOPs as a 30B dense — so why is it 20× more expensive to serve?
3. Why does WSD scheduling make experimentation cheaper?
4. Give an example of synthetic data that *is* clearly worth generating, and one that isn't, and say what distinguishes them.
5. Why does BF16 have better range but worse precision than FP16, and why does range win for training?

**Next:** [`05-post-training.md`](05-post-training.md)
