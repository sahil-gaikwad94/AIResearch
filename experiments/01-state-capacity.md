# Experiment 1 — How Much State Does a Model Need?

**The association-capacity law of hybrid linear attention**

> **Thesis.** In a hybrid model, recall fails when the number of *distinct competing associations* `m` approaches the per-head value dimension `d_v` — not when the context gets long, and not when the needle gets far away. If that is right, the cliff location is predictable from the config, doubles when you double `d_v`, and predicts which real agent tasks break better than token count does.

| | |
|---|---|
| **Question** | What determines the point at which a fixed-state layer stops being able to recall? |
| **Intervention** | None — measurement plus small controlled training runs |
| **Primary metric** | Recall accuracy as a function of `m`, at fixed `T` and fixed distance |
| **Decisive statistic** | Partial correlation of accuracy with `m` given `T`, versus with `T` given `m` |
| **Compute** | ~3,000 GPU-hours (~$6k) |
| **Duration** | 6–8 weeks with two people |
| **Kill condition** | Cliff tracks `T` at fixed `m` (§9) |

---

## 1. Where the question comes from

Qwen3-Next replaced 36 of its 48 attention layers with Gated DeltaNet and advertises 262,144 tokens of context. The vendor's own justification is that linear attention is "fast but weak at recall", so they keep 12 exact layers. That is an honest description of a trade-off — and it leaves the important question unanswered: **weak at recall of what, exactly?**

The published evidence is consistent but coarse:

- **Zoology** (Arora et al., ICLR 2024) showed that gated-convolution models need model dimension to grow at least linearly with sequence length to solve multi-query associative recall, while attention solves it at near-constant dimension. A 70M attention model beat a 1.4B gated-convolution model on associative recall.
- **Based** (Arora et al., 2024) turned that into a Pareto frontier: recurrent state size trades directly against recall capacity, and you can dial it.
- The **Gated DeltaNet** paper's own S-NIAH table shows the split cleanly: at 8K context, DeltaNet scores 98.8 on single-needle retrieval and 14.4 on the two-needle variant; Mamba-2 goes 30.4 → 17.0. Multi-key retrieval is where fixed state dies.
- **"Scaling up the State Size of RNN LLMs"** (ACL 2025) defines a "maximum recall context length" and shows it grows with state size — and that Gated DeltaNet is still less competitive on multi-key NIAH.

So: state size sets recall. Everyone knows this. What nobody has done is turn it into a *quantity you can compute from a config and use to predict failures on real work*.

## 2. The mechanism, in numbers

A linear-attention layer's state is a matrix `S ∈ R^{d_k × d_v}` built by summing outer products:

```
S = Σ_i v_i k_iᵀ            o = Sᵀ q
```

Retrieving the value for key `k_j` gives `Σ_i (q·k_i) v_i`. With random unit-norm keys in `d_k` dimensions, the target contributes `≈ v_j` and the other `m−1` associations contribute noise with standard deviation `≈ sqrt((m−1)/d_k).`

**Signal-to-noise ratio ≈ sqrt(d_k / (m−1)).**

That single line generates the whole experiment:

- Retrieval degrades smoothly once `m` exceeds roughly `d_k/100`, and fails when `m ≈ d_k` (SNR ≈ 1).
- The cliff should be **independent of `T`** and **independent of distance**, because neither appears in the expression.
- Doubling the head dimension should roughly double the number of associations the layer can hold.

For Qwen3-Next: 32 value heads × 128 dimensions = 4,096 per layer, with soft degradation from ~40 per head. Across the layers a fact must survive, the model-level cliff lands in the low thousands of distinct associations. And 262k tokens at a realistic density of one fact per ~100 tokens is **~2,600 facts** — the advertised context length sits right at the predicted capacity edge. That coincidence is the thing worth testing.

### Two competing explanations to separate

| Explanation | Prediction | How it differs |
|---|---|---|
| **Capacity** (this thesis) | Cliff versus `m` at fixed `T`; flat versus `T` at fixed `m` | Doubling `d_v` doubles `m*` |
| **Decay** (gate forgetting) | Cliff versus distance, at fixed `m` | Changing the gate's decay range moves the cliff; changing `d_v` does not |
| **Expressivity** (Merrill et al. 2024) | Failure on state-tracking (S₅, parity) independent of `m` | Present even with `m = 10` |

These are not mutually exclusive, and separating them is most of the experiment's value.

## 3. The expressivity sub-hypothesis (and its literature)

Merrill, Petty & Sabharwal (ICML 2024) proved that linear SSMs cannot solve permutation composition (`S₅`), because they are in `TC⁰`. That result is often over-generalised to "linear attention can't track state", which is too strong:

- **Grazzi et al. (ICLR 2025)** showed the limitation is the *eigenvalue range*: finite-precision linear RNNs with only positive eigenvalues cannot solve parity, and DeltaNet's generalized-Householder transitions can learn any regular language **if** the eigenvalues are allowed into `[−1, 1]` (achieved by scaling `β` up to 2, giving reflections).
- **DeltaProduct** (Siems et al., NeurIPS 2025) showed that products of `k` Householder matrices (k = 1 is DeltaNet) can solve the `S₅` word problem, putting them outside `TC⁰`.

Stock DeltaNet uses `β ∈ (0,1)`, i.e. eigenvalues in `[0,1]`. So the theory predicts it should be **bad at parity and permutation composition**, and good at associative recall. Gated DeltaNet adds decay on top, which does not fix eigenvalues.

**This gives the experiment a free theoretical falsifier**: if a production hybrid like Qwen3-Next can reliably compose permutations in-context at length, then either the deployed `β` range is effectively wider than documented, or the "capacity" story is doing work the expressivity story cannot.

## 4. Predictions

Written before running. `m` = number of distinct key→value associations; `T` = context length; `D` = distance from query to target.

| # | Prediction | Quantitative form | If it fails |
|---|---|---|---|
| **P1** | Capacity dominates | At fixed `T`, accuracy falls from ≥0.95 to ≤0.30 as `m` goes 10 → 1,000 | Capacity is not the binding constraint |
| **P2** | Length is nearly irrelevant once `m` is fixed | At fixed `m = 50`, accuracy varies < 5 points as `T` goes 8k → 256k | The field's "long context" framing is correct after all |
| **P3** | The cliff scales with head dimension | `m*` for `d_v = 128` is 1.8–2.5× the `m*` for `d_v = 64` | The mechanism is wrong (some other parameter sets capacity) |
| **P4** | Failure is instance-level predictable | On real tasks, `m` predicts per-instance failure with AUC ≥ 0.75, beating token count (`m` adds ≥ 0.10 AUC over `T`) | Capacity is a synthetic artefact |
| **P5** | Gating trades retention for stability | Gated variant has ~20–40% lower `m*` but a longer usable `D` than ungated DeltaNet | The gate is doing something else entirely |
| **P6** | Stock GDN fails S₅, and this is separable from capacity | S₅ accuracy stays at chance for `m` small; improving when `β` range is widened to 2 | Expressivity theory does not bind in practice |

P3 and P4 are the load-bearing ones. P3 is a mechanism test that a wrong theory cannot pass by accident; P4 is the claim that changes practice.

## 5. Design

### 5.1 Synthetic factorial (mechanism)

A modified MQAR protocol with three **independent** knobs, which the original does not fully separate:

- `m ∈ {10, 30, 100, 300, 1000}` distinct associations
- `T ∈ {8k, 32k, 128k, 256k}` tokens of filler
- `D ∈ {1k, 10k, 100k}` distance from the query to its target (target at a fixed absolute offset)

The key design trick: **vary `m` and `T` orthogonally.** Standard NIAH confounds them (more filler usually means more distractors). Filler here is generated to contain no additional key–value pairs, so `m` is exactly controlled while `T` grows.

Additional arms:

- **S₅ word problem** (compose a sequence of permutations, report the final mapping) at lengths 8–512. This is the expressivity test.
- **Parity / mod-3** (the Grazzi protocol), cheap and diagnostic.

### 5.2 Small-model training (causal, not just correlational)

Evaluating an existing model cannot distinguish "the model learned to use its state efficiently" from "the state is the limit". Train from scratch:

- 3 architectures × 2 state sizes × 2 gate settings = 12 configs
- 300M parameters, 20B tokens, 3 seeds = 36 runs
- state size varied by `d_v ∈ {64, 128}` at fixed everything else — this is the P3 test
- gate varied by using Gated vs plain DeltaNet layers

### 5.3 Production models (external validity)

Inference-only, at native long context: Qwen3-Next 80B-A3B (hybrid), a full-attention baseline (Llama 3.1 8B/70B), an MLA model (DeepSeek-V3), and a Mamba-family model if a strong open checkpoint exists.

### 5.4 Real tasks (the P4 test)

Three task families where `m` can be counted exactly from the prompt without heuristics:

1. **Repo symbol table** — `m` = number of distinct function/class definitions in context; query asks for a definition or a call site.
2. **API documentation** — `m` = number of distinct endpoint names in context.
3. **Meeting transcript** — `m` = number of distinct person→decision bindings.

For each instance, fit a logistic model of failure on `(m, T, D, model)` and compare AUC of `{T}` versus `{m}` versus `{m, T}`. The pre-registered claim is that `m` adds ≥ 0.10 AUC.

## 6. Measurements beyond accuracy

- **State saturation**: extract `S` from GDN layers at query time; measure its stable rank and Frobenius norm versus `m`. Prediction: rank saturates at the cliff.
- **Write interference**: cosine similarity between successive `k` vectors; at high `m`, keys collide and the delta rule's correction becomes destructive. Prediction: mean pairwise |cos| rises sharply at `m*`.
- **Gate statistics**: distribution of `α` per head as a function of distance — the decay explanation predicts `α → 1` for far tokens in layers that must retain them.
- **Head specialisation**: do some heads carry all the associations? Prediction: a few heads dominate at low `m`, and the load spreads (then saturates) at high `m`.

## 7. Budget

| Item | GPU-hours |
|---|---|
| 36 training runs at 300M / 20B tokens (25 GPU-h each) | 900 |
| Production-model long-context inference (~5k prompts at 256k) | 1,800 |
| Probing / state extraction passes | 300 |
| **Total** | **~3,000** (≈ $6k at $2/GPU-hour) |

## 8. Analysis plan

1. **Main figure**: accuracy versus `m`, one line per `T`, at fixed `D`. If P1 and P2 hold, the lines lie on top of each other and fall together.
2. **Scaling figure**: `m*` (the `m` at which accuracy crosses 0.5) versus `d_v`, log-log. P3 predicts slope ≈ 1.
3. **Held-out prediction**: fit `m*` on synthetic data, then predict the real-task failure threshold without refitting. This is the honest test.
4. **Decision rules, pre-registered**:
   - P1 confirmed if accuracy drops ≥ 0.5 across the `m` range at every `T`.
   - P2 confirmed if the max deviation across `T` at fixed `m` is < 0.05.
   - P3 confirmed if the fitted slope is in [0.8, 1.3] with a CI excluding 0.5 and 2.
   - P4 confirmed if ΔAUC ≥ 0.10 with a bootstrap CI excluding 0.
5. **Error bars** from seeds (training) and from bootstrapping over task instances (evaluation), not over prompts.

## 9. What would kill it

- **If accuracy tracks `T` at fixed `m`**, capacity is not the constraint; the honest conclusion is that the field's token-count framing is right and this thesis is wrong. That is a publishable negative result and it would redirect effort to decay/distance mechanisms.
- **If `m*` does not move with `d_v`**, the mechanism is not "dimensions as memory slots" — maybe it is precision, or the number of heads, or the chunk size in the kernel.
- **If real-task failure is uncorrelated with `m`** (P4 fails), the synthetic result is a toy and the practical advice ("measure associations, not tokens") is unjustified.

## 10. If it works, what changes

- **Evaluation**: long-context benchmarks get a second axis. Reporting "256k context" without an association-capacity number would be like reporting a car's top speed without its range.
- **Design**: head dimension becomes a decision made from expected association density, not a default. A coding-agent model expects thousands of symbols; a chat model expects dozens.
- **Serving**: routing decisions ("does this prompt need the full-attention layers?") become computable from `m` rather than from token count.
- **Theory**: connects the Zoology/Based line (small-scale recall) to the expressivity line (S₅, TC⁰), which currently do not talk to each other.

## 11. The one-week pilot

Train 6 tiny models (50M, 2B tokens, `d_v ∈ {32, 64, 128}`) on MQAR with `m ∈ {10, 30, 100, 300}` at a single `T = 8k`. Cost: ~40 GPU-hours. If `m*` does not visibly move with `d_v` in that pilot, the mechanism is wrong and the full study should not be run.
