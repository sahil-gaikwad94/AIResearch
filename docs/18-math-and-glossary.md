# 18 — Math Primer, Notation, and Glossary

*Reference chapter. Skim the maths, use the glossary as a lookup.*

---

## Part A — The maths you actually need

### A.1 Scalars, vectors, matrices

- **Scalar**: a single number, `3.7`.
- **Vector**: an ordered list, `x = [0.2, −1.4, 0.7]` — a point in space. Dimension = length.
- **Matrix**: a 2D grid, `W ∈ R^{m×n}` — a linear map from n-dim space to m-dim space.

**Dot product**: `a · b = Σ_i a_i b_i`. Measures alignment. Large positive = same direction.

**Matrix multiply**: `(AB)_{ij} = Σ_k A_{ik} B_{kj}`. Cost is `m·n·p` multiply-adds — the reason GPUs exist.

### A.2 Softmax

```
softmax(z)_i = e^{z_i} / Σ_j e^{z_j}
```

Turns arbitrary scores into a probability distribution. Properties: positive, sums to 1, differentiable, **shift-invariant** (adding a constant to all `z` changes nothing — which is why numerical implementations subtract the max). **Temperature** `τ` divides `z` before softmax: `τ→0` = argmax (deterministic), `τ→∞` = uniform (random).

### A.3 Cross-entropy and KL divergence

**Cross-entropy** between true distribution `p` and model `q`:

```
H(p,q) = −Σ_i p_i log q_i
```

For one-hot targets (next-token prediction), this reduces to `−log q(correct)`. Minimising it maximises the likelihood of the data.

**KL divergence**: `D_KL(p‖q) = Σ p_i log(p_i/q_i)`. How much `q` differs from `p`. Not symmetric. Used everywhere in alignment as an anchor: `KL(π_θ ‖ π_ref)` penalises drifting from the reference policy.

**Perplexity** = `exp(cross-entropy)`. "How many equally-likely options is the model effectively choosing between." PPL 10 ≈ 10-way uncertainty per token.

### A.4 Gradients and backpropagation

The **gradient** `∇_θ L` is the vector of partial derivatives of the loss w.r.t. each parameter: the direction of steepest increase. **Gradient descent** steps against it: `θ ← θ − η∇_θ L`.

**Backpropagation** computes that gradient efficiently via the chain rule, reusing intermediate results from the forward pass. Cost ≈ 2× the forward pass (hence the "6×N×D" training FLOPs: 2 forward + 4 backward per parameter per token).

**Adam** keeps a running mean (`m`) and variance (`v`) of gradients and updates `θ ← θ − η · m̂/(√v̂ + ε)`, giving each parameter an adaptive step size. **AdamW** decouples weight decay from the gradient update.

### A.5 Expectations and sampling

`E_{x~p}[f(x)] = Σ_x p(x) f(x)` — the average of `f` under distribution `p`.

**Monte Carlo**: estimate an expectation by sampling. This is what "sample 64 completions and take the mean reward" does (GRPO).

**Variance reduction** is why baselines exist: `E[r]` estimated from few samples is noisy; subtracting a baseline `b` (like the group mean) doesn't change the expectation but massively reduces variance. That is the entire justification for GRPO's group-relative advantage.

### A.6 The attention formula, once more, with shapes

```
Input:  X ∈ R^{n×d}                  (n tokens, d model dim)
Q = X W_Q, K = X W_K, V = X W_V      (W ∈ R^{d×d_k})
S = Q Kᵀ / √d_k    ∈ R^{n×n}         (scores)
S ← S + mask                          (−∞ above diagonal for causal)
A = softmax(S)     ∈ R^{n×n}          (row-wise)
O = A V            ∈ R^{n×d_v}
```

Cost: `O(n² d)` for scores and `O(n² d)` for the weighted sum. Memory (naive): `O(n²)`.

### A.7 A minimal Transformer block (pseudocode)

```python
def block(x, mask):
    h = rmsnorm(x)
    q, k, v = project_qkv(h)          # with RoPE applied to q, k
    a = flash_attention(q, k, v, causal=True)
    x = x + out_proj(a)
    h = rmsnorm(x)
    x = x + swiglu_mlp(h)             # or MoE routing
    return x
```

That's the whole architecture. Everything else in `02` is a choice inside `project_qkv`, `flash_attention`, and `swiglu_mlp`.

---

## Part B — Notation table

| Symbol | Meaning |
|---|---|
| `N` | number of model parameters (total, or active for MoE — always check) |
| `D` | number of training tokens |
| `C` | training compute in FLOPs ≈ `6ND` |
| `L` | sequence length (or number of layers — context-dependent; check) |
| `d` / `d_model` | hidden dimension / model width |
| `h` | number of attention heads |
| `d_head` | dimension per head |
| `h_kv` | number of key/value heads (GQA) |
| `T` | temperature; also diffusion timesteps |
| `β` | KL coefficient in RLHF/DPO; also inverse temperature |
| `π_θ`, `π_ref` | policy (model being trained), reference policy |
| `r` | reward; also LoRA rank |
| `σ` | sigmoid; also standard deviation (context) |
| `s` | guidance scale (diffusion) |
| `k` | top-k (MoE routing, sparse attention, sampling) |
| `ε` | clipping epsilon; noise (diffusion) |

---

## Part C — Glossary (~260 terms)

**A**
- **Activation**: a value flowing through the network (usually post-nonlinearity).
- **Activation patching**: swapping an activation from a clean run into a corrupted run to test causality.
- **AdamW**: adaptive optimiser with decoupled weight decay.
- **Agent**: a model in a loop with tools and environment feedback.
- **Agentic RL**: RL over multi-turn environment trajectories.
- **Alignment**: making a system pursue intended objectives.
- **Alignment faking**: behaving well in training to avoid having values modified.
- **Attention**: relevance-weighted averaging of values, computed from query-key dot products.
- **Attention head**: one (Q,K,V) projection + attention computation; heads run in parallel.
- **Autoregressive (AR)**: generating one token at a time, conditioned on previous ones.
- **Aux-loss-free balancing**: MoE load balancing via bias updates instead of an auxiliary loss.

**B**
- **Backpropagation**: chain-rule gradient computation through the network.
- **Batch size**: number of sequences per optimisation step.
- **Beam search**: keep the k best partial sequences at each step.
- **BF16**: 16-bit float with 8 exponent / 7 mantissa bits; training default.
- **BitNet**: ternary-weight (≈1.58-bit) model family trained from scratch.
- **Block diffusion**: diffusion over blocks of tokens, left-to-right across blocks.
- **BPE**: byte-pair encoding tokenizer.
- **Bradley–Terry**: model of pairwise preference `σ(r(A) − r(B))`.

**C**
- **Causal mask**: prevents attending to future positions.
- **CFG (classifier-free guidance)**: extrapolate between conditional and unconditional predictions.
- **Chinchilla-optimal**: ~20 tokens per parameter at fixed training compute.
- **Circuit**: a sparse subgraph of components implementing a behaviour.
- **CLIP**: contrastive image-text embedding model.
- **CoT (chain of thought)**: generated intermediate reasoning steps.
- **Cold start (R1)**: a small SFT phase before RL to establish format.
- **Constitutional AI**: alignment using a written set of principles + AI feedback.
- **Constitutional Classifiers**: input/output classifiers trained on constitution-derived data.
- **Context engineering**: managing what occupies the context window.
- **Context parallelism (CP)**: splitting the sequence across devices (ring attention).
- **Context rot**: degradation as context fills, independent of retrieval success.
- **Contamination**: benchmark items present in training data.
- **Cross-encoder**: model scoring a (query, document) pair jointly.
- **Cross-entropy**: `−Σ p log q`; the pretraining loss.

**D**
- **Data wall**: exhaustion of high-quality human training text.
- **Decode**: autoregressive generation phase, memory-bandwidth-bound.
- **Delta rule**: memory update that erases the old value at a key before writing.
- **DeltaNet / Gated DeltaNet**: linear-attention layers using the delta rule (+gating).
- **Dense model**: all parameters active for every token.
- **Diffusion model**: generative model that learns to denoise.
- **DiT**: Diffusion Transformer.
- **Distillation**: training a student to match a teacher.
- **dLLM**: diffusion language model.
- **DPO**: direct preference optimisation; supervised alignment without a reward model.
- **DSA (DeepSeek Sparse Attention)**: top-k token selection via a lightweight indexer.
- **Duty cycle / active parameters**: parameters actually used per token.

**E**
- **Effective context**: the length at which the model still works, vs. advertised.
- **Embedding**: learned vector representation.
- **Emergent ability**: capability that appears only past a scale threshold (contested).
- **EP (expert parallelism)**: distributing MoE experts across devices.
- **Evaluation differential**: behaving differently when a model detects it's being tested.
- **Expert**: one FFN in an MoE layer.

**F**
- **FID**: Fréchet Inception Distance; image-generation metric.
- **Fine-tuning**: further training on a specific dataset.
- **FlashAttention**: IO-aware exact attention that never materialises the n×n matrix.
- **Flow matching**: learn a velocity field transporting noise to data.
- **FLOPs**: floating-point operations; compute unit.
- **FP8 / FP4 / NVFP4**: low-precision float formats for training/inference.
- **FSDP**: fully sharded data parallel (ZeRO-3 style).

**G**
- **GQA**: grouped-query attention; fewer KV heads than query heads.
- **GRPO**: group relative policy optimisation; PPO without a critic.
- **Grokking**: delayed generalisation long after training loss converges.
- **Guided decoding**: constraining generation with a grammar or schema.

**H**
- **Hallucination**: confident generation of false content.
- **Head (attention)**: see attention head.
- **HLE**: Humanity's Last Exam benchmark.
- **Hybrid model**: mixes attention with linear/SSM layers.

**I**
- **In-context learning (ICL)**: learning a task from examples in the prompt, without weight updates.
- **Induction head**: circuit that completes previously-seen patterns in context.
- **Inference-time scaling**: improving accuracy by spending more compute at generation.
- **Instruction tuning**: SFT on instruction-response pairs.
- **IPO / KTO / ORPO / SimPO**: DPO variants.

**J–K**
- **Jailbreak**: input that bypasses safety training.
- **KDA**: Kimi Delta Attention; channel-wise-decay linear attention.
- **KV cache**: stored keys/values for past tokens during generation.
- **KV cache compression**: quantizing/evicting cached entries.

**L**
- **Latent diffusion**: diffusion in a compressed latent space.
- **Lean**: formal proof assistant used as a verifier.
- **Linear attention**: attention-like mixing with linear cost via a fixed-size state.
- **LoRA**: low-rank adapter fine-tuning.
- **Logit**: pre-softmax score.
- **LongRoPE / YaRN / PI**: RoPE context-extension methods.
- **Lost in the middle**: U-shaped performance by position in long contexts.

**M**
- **Mamba / Mamba-2**: selective state-space models; SSD duality.
- **Masked diffusion**: denoising by unmasking.
- **MCP**: Model Context Protocol for tool integration.
- **MCTS**: Monte Carlo tree search (used in AlphaProof).
- **Memory (agent)**: persistent write/read store beyond the context window.
- **METR time horizon**: task length at 50% agent success.
- **MFU**: model FLOPs utilisation.
- **MLA**: multi-head latent attention; low-rank KV cache.
- **MoE**: mixture of experts.
- **MTP**: multi-token prediction.
- **muP**: maximal update parametrisation; hyperparameter transfer.
- **Muon**: optimiser with orthogonalised momentum.

**N–O**
- **NIAH**: needle-in-a-haystack retrieval test.
- **NoPE**: no positional embedding in a layer.
- **Normalisation (RMSNorm/LayerNorm)**: rescaling activations.
- **ORM / PRM**: outcome / process reward model.
- **Over-refusal**: refusing benign requests.

**P**
- **PagedAttention**: paged KV-cache memory management (vLLM).
- **Parallelism (DP/TP/PP/EP/CP)**: the six splitting axes.
- **pass@k / pass^k**: probability at least one of k succeeds / all k succeed.
- **Perplexity**: exponentiated cross-entropy.
- **Pipeline parallel (PP)**: splitting layers into stages.
- **Post-training**: everything after pretraining.
- **PPO**: proximal policy optimisation.
- **Prefill**: parallel processing of the prompt, compute-bound.
- **Prefix caching**: reusing the KV cache of a shared prefix.
- **Pretraining**: next-token training on a large corpus.
- **Prompt injection**: instructions hidden in untrusted input.
- **Probe**: linear classifier on activations.

**Q–R**
- **QAT**: quantization-aware training.
- **Quantization**: lower-precision weights/activations.
- **ReAct**: interleaved reasoning and tool actions.
- **Rectified flow**: flow matching with reflow to straighten paths.
- **Refusal**: declining a request.
- **Rejection sampling**: keeping only verified-good samples for SFT.
- **Reliability horizon**: task length at which *all* attempts succeed.
- **Residual stream**: the running activation every layer reads/writes.
- **Reward hacking**: optimising the proxy instead of the goal.
- **RLAIF**: RL from AI feedback.
- **RLHF**: RL from human feedback.
- **RLVR**: RL with verifiable rewards.
- **RoPE**: rotary position embedding.
- **RSP**: responsible scaling policy.

**S**
- **SAE**: sparse autoencoder.
- **Sandbagging**: deliberately underperforming on evals.
- **Scaffold / harness**: the loop and tools around a model.
- **Scalable oversight**: supervising systems smarter than the supervisor.
- **Scaling law**: power-law relationship between loss and scale.
- **Scheming**: covert pursuit of misaligned goals.
- **Self-consistency**: majority vote over sampled reasoning paths.
- **SFT**: supervised fine-tuning.
- **SigLIP**: sigmoid-loss CLIP variant.
- **Sparse attention**: attending to a subset of positions.
- **Speculative decoding**: draft then verify for faster generation.
- **SSM**: state-space model.
- **Steering vector**: direction added to activations to change behaviour.
- **Superposition**: representing more features than dimensions.
- **Sycophancy**: telling users what they want to hear.

**T**
- **Temperature**: sampling randomness.
- **Tensor parallel (TP)**: splitting matrices across devices.
- **Test-time compute**: inference-time compute budget.
- **Test-time training (TTT)**: updating weights (or fast weights) at inference.
- **Token**: the atomic text unit.
- **Tokenizer**: text↔token mapping.
- **Tool use / function calling**: structured external calls.
- **Top-k / top-p**: sampling truncation strategies.
- **Transcoder**: sparse model of a sublayer's input→output map.
- **Transformer**: attention + MLP blocks with residuals.

**U–Z**
- **Unembedding**: final projection to vocabulary logits.
- **Verifier**: program/model that checks correctness.
- **VLA**: vision-language-action model.
- **VLM**: vision-language model.
- **Vocab size**: number of tokens.
- **Warmup**: learning-rate ramp at the start of training.
- **Weak-to-strong**: strong model trained from weak supervision.
- **World model**: model predicting environment dynamics under actions.
- **WSD**: warmup–stable–decay schedule.
- **ZeRO**: sharded optimizer/gradient/parameter states.
- **Zero-shot / few-shot**: no examples / a few examples in the prompt.

---

## Part D — Ten formulas worth memorising

1. `Attention = softmax(QKᵀ/√d_k)V`
2. `RMSNorm(x) = γ·x/√(mean(x²)+ε)`
3. `SwiGLU(x) = W₂(Swish(W_g x) ⊙ (W_u x))`
4. `L = −(1/T)Σ log p(x_t | x_<t)`
5. `C ≈ 6ND`
6. `KV bytes = 2·L·h_kv·d_head·n·bytes`
7. `DPO loss = −log σ(β[log(π_θ/π_ref)(y_w) − log(π_θ/π_ref)(y_l)])`
8. `GRPO advantage = (r_i − mean(r))/std(r)`
9. `DDPM loss = E‖ε − ε_θ(x_t,t)‖²`
10. `CFG = ε_uncond + s(ε_cond − ε_uncond)`

**Next:** [`19-reading-list.md`](19-reading-list.md)
