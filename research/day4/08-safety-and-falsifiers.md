# 08 — Safety, failure modes, and the pre-registered falsifiers

Everything above is an argument. This chapter is the part that can be wrong in public.

## C1–C9, registered before training

| ID | Claim under test | Metric | Kill criterion | Cost |
|---|---|---|---|---|
| C1 | MLA absorption survives training | max nats between absorbed and standard paths, post-training | > 1e-3 nats or > 0.01 perplexity ⇒ drop absorption at serving | 0.5 GPU-h |
| C2 | Subspace-grouped MoE beats flat at equal budget | pass@1 HumanEval+/MBPP+ | < +1.5 pts and < 0.02 nats ⇒ flat router | 1,400 GPU-h |
| C3 | Per-head RoPE in local layers is load-bearing | 256 k needle recall, syntax pass@1 | within 2 pts of local NoPE ⇒ remove RoPE | 900 GPU-h |
| C4 | Householder eigenvalues beat the delta rule | parity + S5 at length 64, exact | < 95 % or < +3 pts over delta rule ⇒ revert memory | 120 GPU-h |
| C5 | NoPE global layers beat RoPE global at 1 M | 1 M exact recall | RoPE within 1 pt ⇒ use RoPE (simpler) | 400 GPU-h |
| C6 | Hypothesis codebook breaks retry correlation | solved-rate at equal GFLOP, measured `ρ` | < +5 pts or `ρ ≥ 0.4` ⇒ drop codebook, re-price ladder | 260 GPU-h |
| C7 | 0-DoF splicing pays at batch 1 | wall-clock on 600-token patch | < 1.2× ⇒ gate at ≥ 40 % deterministic | 6 CPU-h |
| C8 | Verifier is sound and complete enough | 10⁶ adversarial documents | any accepted-invalid; rejected-valid > 1e-4 | 40 CPU-h |
| C9 | Self-training is safe at 40 % solve | poison rate of the stack | > 0.1 % ⇒ no unattended distillation | 180 GPU-h |
| C10 | Chunked BPTT beats per-step detach | parity / validation loss at equal tokens | < 1 point gain ⇒ detach every step | 60 GPU-h |
| C11 | The indexer prior is used, not just differentiable | logit sensitivity + ablation loss | no gain ⇒ drop the prior and the indexer path | 40 GPU-h |
| C12 | A block-summary KV hierarchy preserves recall at 1 M | exact recall, bytes/sequence | < 90 % recall ⇒ keep the 4.29 GB cache | 20 GPU-h |

Total cost of the falsification programme ≈ 3,427 GPU-h, 14.6 % of the 23,530 GPU-h budget. That
is the price of being allowed to say the numbers are real, and it is paid *before* the large
runs, in the order C8 → C11 → C7 → C10 → C4 → C6 → C1 → C3 → C5 → C2 → C9. The cheap ones come
first because they can delete a mechanism before it costs a training run — and because the
first audit already showed that the mechanisms most likely to be inert are the cheapest ones
to test (C11 costs 40 GPU-h and caught a path with zero influence on the model's output).

## Failure modes that are not in the table

**Verifier starvation.** If tier 2 rejects most partial documents on real traffic, the decoder
spends its budget rewinding and the runtime becomes a tax. Trigger: rejection rate > 30 % over
a 10 k-patch sample. Response: the constraint language is too expressive for the model's
competence, and the fix is to *narrow the schema*, not to loosen the verifier.

**Mode collapse onto verifiable tasks.** A model trained only on signals it can check will
drift toward tasks that are cheap to check. Mitigation: hold 15 % of SFT tokens as
execution-unverifiable but human-curated data, and track the ratio of verified to unverified
tokens per domain. If any domain falls below 5 % verified, its weight is capped rather than
raised — a domain that cannot be checked should not dominate the mixture.

**fp16 state drift.** The 4.19 MB session state is cast per step. Trigger: > 1e-2 relative norm
drift over 1 M steps. Response: fp32 state (8.4 MB) and re-run C4, because drift can masquerade
as a missing eigenvalue.

**Adversarial indexer recall.** The global layers read 8 blocks chosen by a 128-dim indexer.
A document can be built whose retrieval target scores below the top-8 blocks. Trigger: recall
below 90 % on a targeted-adversarial needle set. Response: raise `sel_topk` to 16 (adds 0.29
GFLOP/token at 1 M) or add a second indexer head with a different pooling; do not raise the
window.

**Mask-induced NaN in third-party samplers.** A dead-end prefix masks every logit, including
EOS, to `-inf`. HF's sampler then has no legal token; depending on backend this yields NaN,
uniform sampling, or an exception. This is a *protocol* failure, not a VEXX failure: the
correct behaviour is to treat the row as dead and return control to the orchestrator, which is
what `generate()` does internally. Any serving bridge that applies the mask without handling
the dead row is unsound regardless of what tier 0 decided. Documented here because it is the
one place where a correct mask can still produce an incorrect token.

**Certificate forgery.** Before A5, `verify_certificate` accepted a signed certificate with no
trusted key. The L3 memory store's SHA-256 witness is now only meaningful when the verifier is
given a key; without one, verification fails closed. Any deployment that wants content-addressed
memory to mean something must provision the key. The same failure was found one level up in
`kestrel4/model.py`: `emit_l3_proof_certificate` returned `verified: True` without checking
anything (audit A6). It now evaluates linear checks with exact `Fraction` arithmetic and
returns `False` when no checks are supplied.

**Signed-container residual checks.** The tier-1 volume twist masks `v > C` where `C` is the
remaining container capacity. That is correct only when container elements are non-negative;
with signed elements a large negative value passes a check it should fail. Kestrel-4's schemas
are non-negative by construction, but the runtime does not enforce it, so this is a documented
limitation rather than a proven-safe case.

## What would falsify the whole thesis

Three outcomes would be enough, and any one of them is sufficient:

1. **C2 and C6 both fail.** If grouped experts add no measurable width and conditioned retries
   add no measurable diversity, then Kestrel-4 is a 21 B model that behaves like a 2 B model
   with expensive bookkeeping, and the Day 3 thesis ("move work to the checkable side") has no
   neural implementation.
2. **C8 fails with a single accepted-invalid document.** Then the runtime's authority is
   unfounded, every poison figure in §04 is invalid, and the design has to fall back to
   execution-only filtering at 33× the retry cost.
3. **Verifier starvation above 30 % on real patches.** Then the constraint system is
   unlearnable at this scale, and the honest move is to shrink the schema until the model can
   satisfy it, which is a different product.

The design is deliberately arranged so that each of these is cheap to discover relative to the
cost of the full training run. That is what the 3,427 GPU-h is for.

## Residual risks that no ablation covers

- **Benchmark contamination.** Execution-verified corpora are built from merged PRs, which are
  in the pretraining data of every strong model. The mitigation is temporal (post-cutoff PRs
  only) and it costs data, not correctness.
- **Verifier overfitting by the model.** A model trained against one verifier learns that
  verifier's blind spots. The mutant gate exists to make the blind spots move, but it is not a
  proof; C8 and C9 bound the exposure rather than eliminating it.
- **Serving assumptions.** The 4-bit byte model assumes 0.55 B/param including scales and
  metadata, 3.35 TB/s HBM, and no sparsity in the kernels. Each is stated in §09; if the
  deployment uses 8-bit weights, every throughput figure halves and the splicing crossover
  falls from 6.4 % to 3.2 % deterministic fields (the 16 µs CPU cost is amortized over a step
  that is twice as long).
