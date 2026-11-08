# 9 — Decisions, Risks, and When Not to Build This

> A design document that only contains reasoning is unfalsifiable. This one ends with the decisions and what would reverse each of them, the risks with early-warning signals, and the honest conditions under which the right move is to fine-tune somebody else's model.

---

## 9.1 Decision log

| # | Decision | Alternative rejected | Reason | Reverse if |
|---|---|---|---|---|
| **D1** | Full attention at every layer | 3:1 hybrid linear (the 2026 consensus) | Prefix caching makes prefill cheap; exact recall is required for code | A2 shows hybrid +3 points at equal KV; or cache hit rate <70% |
| **D2** | MLA, latent 448 + 64 rope | GQA-4, MHA | Fewer parameters *and* 4× less cache than GQA-4 — a domination, not a trade | Kernel maturity makes it >1.3× slower to serve than GQA |
| **D3** | 40 layers × 3072 | 28 × 4096 (equal params) | Reasoning tracks sequential depth | A3 shows <1.5-point gain; width wins on 30% smaller cache |
| **D4** | 4.08 B dense | 12 B / 1.9 B-active MoE | RL rollouts 3× cheaper; one-GPU serving; no routing instability across 20 reward shapes | A8 shows >8% bpb gain |
| **D5** | 131 k vocab, tied | 65 k + wider FFN | Token efficiency cuts *every* cost, not just parameters | A1 shows <10% token reduction |
| **D6** | MTP-2 | no MTP | Gradient density, plus free speculative decoding | A4 shows no bpb gain and <1.6× speedup |
| **D7** | Fused `W_DKV` (no separate rope-key projection) | DeepSeek's separate `W_KR` | Identical math, one fewer matmul | — (correctness-neutral) |
| **D8** | `theta` = 500 k from step 0 | extend later with YaRN/ABF | Avoids a fine-tuning phase and a short-range quality gap | Recall at 64 k is worse than an extend-later baseline (A7) |
| **D9** | QK-norm + zero-init residual projections | standard init | Cheapest available stability at 40 layers | — (costs 512 params) |
| **D10** | GRPO with constant loss divisor | per-sequence normalisation | Per-sequence normalisation systematically rewards longer outputs | — (this is a known bug, not a preference) |
| **D11** | Diff output format | whole-file rewrites | 33× fewer generated tokens; cache-friendly; reviewable | Harnesses cannot apply diffs reliably |
| **D12** | Read-only tests + impossibility canaries | trust the harness | The model will edit the test if it can | — |
| **D13** | 1.2 T tokens (294/param) | Chinchilla 80 B | Inference dominates lifecycle cost; small models are deliberately overtrained | Training budget halves and capability is acceptable |

### On D7, specifically

An earlier draft of this document specified both a fused `W_DKV` *and* a separate `W_KR`, and the analytic script printed 4.091 B for inference. The torch implementation reported 4.083 B. The discrepancy was 197 k parameters per layer — exactly `d_model × d_rope` — and it revealed that the analytic script was double-counting a projection the model did not have.

The lesson is not "check your arithmetic" but something more specific: **two independent implementations of the same model are worth more than one careful one**, because the disagreement localises the error immediately. That is why `tools/count_params.py` exists separately from `kestrel/model.py` and why the model asserts against the script's number.

---

## 9.2 Risk register

| # | Risk | P | Impact | Mitigation | Early warning |
|---|---|---|---|---|---|
| **R1** | The central bet is wrong — hybrid attention wins on real repo tasks | 25 % | High (redesign) | A2 is funded and pre-registered | recall probe flat across arms at 1 B |
| **R2** | Real harnesses have low prefix-cache hit rates | 30 % | High (thesis invalid) | Instrument before building (§8.5) | hit rate <70% in logs |
| **R3** | MLA kernels are immature | 40 % | Medium | Fall back to GQA-4: 1.3× cache, same quality | serving throughput <60% of GQA baseline |
| **R4** | Tokeniser overfits Python | 35 % | Medium | Language-stratified eval; per-language token counts | >25% token reduction on Python, <8% on Go/Rust |
| **R5** | 4 B is simply too small for hard reasoning | 50 % | High (positioning) | Sell it as an instrument, not an oracle; route hard tasks to a bigger model | GPQA canary flat while code improves |
| **R6** | Reward hacking beyond the catalogue | 45 % | High | Canaries in the training loop; read-only tests; assertion-count differential | canary pass rate >2% |
| **R7** | Test contamination inflates results | 60 % | High (false confidence) | Private held-out repos are the only gating tier | public and private scores diverge by >15 points |
| **R8** | Data licensing | 20 % | High (legal) | Permissive-only filter at repo level | — (prevented by construction) |
| **R9** | Long-context quality collapses past 32 k | 40 % | Medium | Staged training; sinks; recall eval at 64 k | recall probe drops >10 points from 32 k to 64 k |
| **R10** | Sandbox infrastructure dominates RL cost | 50 % | Medium | Budget CPU alongside GPU ([06 §6.9](06-post-training.md)) | >20% of rollouts lost to infra failure |

Two of these deserve more than a table row.

**R5 is the honest one.** A 4.08 B model will not match a 30 B model on hard multi-step reasoning, and no architectural choice in this document changes that. What the design claims is narrower and more defensible: *for repo-scale coding with a verifier in the loop, capability per dollar is dominated by recall, cache and rollout cost* — and on that axis a 4 B model built around those constraints is a better instrument than a 30 B model built for chat. If the private-repo evals do not show that, the design has failed on its own terms.

**R7 is the dangerous one**, because contamination produces *good* numbers. The defence is structural (private repos are the only gating tier) rather than statistical, and the divergence check between public and private scores is the tripwire.

---

## 9.3 When not to build this

Most teams should not. The honest decision tree:

| If… | Then… |
|---|---|
| You need a good coding model next month | **Fine-tune an existing open 4 B.** Qwen/Llama-class weights plus your data will beat this design for at least six months of engineering |
| Your workloads are chat, not agents | **Do not build this.** The design is optimised for a read-heavy, cacheable, verifier-driven loop; in chat, decode latency dominates and these choices are wrong |
| You cannot run 128 GPUs for two weeks | **Do not pretrain.** Use the tokeniser, the harness constraints and the RL recipe on someone else's base model |
| You do not have private repositories with test suites | **Do not build the RL stage.** Without a verifier you are training on preference, and the whole post-training plan is void |
| You need one model for everything | **Do not build this.** Specialisation is the point; a 4 B code model is a poor general assistant |

The design is justified only when at least three of these hold: you need a code-specific tokeniser, you control the serving stack and can exploit prefix caching, you have proprietary repositories with reliable tests, and you intend to do serious RL. If two or fewer hold, the correct engineering decision is to fine-tune.

---

## 9.4 What would make me abandon this design

In order of how likely each is to actually happen:

1. **A2 shows hybrid attention winning by more than 3 points at equal KV budget.** The central bet dies; the design reverts to a conventional 3:1 hybrid and keeps everything else.
2. **Prefix-cache hit rates in real harnesses are below 70%.** The prefill thesis dies. Attention becomes a compute problem again, and hybrid attention is then correct for the original reason.
3. **A4 shows MTP gives neither bpb gain nor speedup.** Small, but it would remove the only decode-side optimisation.
4. **The 1 B proxy's loss curve is worse than a well-tuned public 4 B at equal tokens.** This is the "you are not going to out-train the field" signal, and it should be checked at 100 B tokens, not at the end.
5. **The tokeniser gain is under 10%.** Not fatal — it changes the vocabulary, not the architecture — but it removes P4 from the thesis.

Four of these five are cheap to test before committing the main run. That is deliberate: a design that can only be validated by spending its entire budget is not a design, it is a bet.

---

## 9.5 Summary judgement

What this design gets right, if the thesis holds:

- **It reads better per dollar than anything else at this size.** 20 KiB/token of cache, full attention, MLA — the serving math is the strongest part of the document.
- **It is cheap to iterate on.** 4 B dense means a 20 B-token ablation costs 300 GPU-hours and a full RL iteration fits on one node.
- **The post-training plan is where the capability actually is**, and it has the most specific treatment of reward hacking I could write.

What it gets wrong, with high probability:

- **It will underperform on hard reasoning** relative to bigger models, and no amount of architecture fixes that.
- **The RL stage will be harder than budgeted**, because verifier quality is the real bottleneck and test suites are worse than anyone believes until they are audited.
- **Something in the ablation list will overturn a decision here.** That is what the ablations are for; the document is written so that being wrong is cheap and detectable rather than expensive and silent.

**Back to:** [README](README.md) · [1 — Thesis](01-thesis.md) · [2 — Architecture](02-architecture.md) · [8 — Evals and ablations](08-evals-and-ablations.md)
