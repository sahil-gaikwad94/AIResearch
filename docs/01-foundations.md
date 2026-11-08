# 01 — Foundations: How a Transformer Actually Works

*Read time: 60–75 minutes. Prerequisites: §0.5 of the previous chapter (vectors, matmul, softmax, gradients).*

---

## 1.1 The whole idea in one sentence

A large language model is a function that takes a sequence of tokens and returns, for every position, a probability distribution over the next token — and it is trained by showing it enormous amounts of text and nudging its parameters so the probability it assigned to the *actual* next token goes up.

Everything else — chat, code, reasoning, agents, vision — is built by (a) making that function enormous, (b) feeding it the right data, and (c) reshaping its behaviour afterwards.

---

## 1.2 Step 1: Text → tokens

Neural networks need numbers. **Tokenization** is the mapping from text to integers.

### Byte-Pair Encoding (BPE), intuitively

Start with the alphabet as your vocabulary. Find the most frequent adjacent pair of symbols in your corpus ("e" + "r" perhaps, or "th"). Merge that pair into a new symbol. Repeat 50,000–200,000 times. What you get is a vocabulary of frequent *fragments*: `" the"`, `"ing"`, `" token"`, `"ization"`, common words, and for rarer words, sub-word pieces.

Why this design?

- **It's a compromise between characters and words.** Characters → sequences too long (attention cost grows with length); words → huge vocabulary and no way to represent a new word.
- **It handles anything.** With a **byte-level** fallback (GPT-2 onwards, using the 256 byte values as the base alphabet), any Unicode string is representable, including emoji and typos.
- **Vocabulary size is a real tradeoff.** Bigger vocab = fewer tokens per document (cheaper attention, faster generation) but a bigger embedding matrix and softer gradient signal per token. 32k was standard in 2023 (Llama 2); 128k–256k is standard in 2025–26 (Llama 3 = 128k, Qwen3.5 ≈ 155k, some 2026 models higher).

### Why tokenization keeps causing bugs

Tokenization is *the* most under-appreciated source of weird model behaviour, and it's worth knowing the classic pathologies:

1. **Arithmetic.** "1234" may be split as `123` + `4`, so the model has no clean digit-level view. This is a real contributor to arithmetic errors.
2. **String operations.** "How many r's in strawberry?" — the model never sees letters, it sees token IDs. Early models genuinely *could not* count characters; later models learn heuristics or are trained on such tasks.
3. **Multilingual inequity.** A Hindi or Thai sentence can cost 3–5× more tokens than its English translation. That directly multiplies cost and reduces effective context for those users. This was documented repeatedly (e.g. Petrov et al. 2023, "Language Model Tokenizers Introduce Unfairness Between Languages").
4. **Trailing spaces / leading spaces.** `"hello"` and `" hello"` are different tokens, which is why prompts are sometimes sensitive to whitespace in ways that feel absurd.
5. **Glitch tokens.** Certain token IDs (famously `SolidGoldMagikarp`) appeared in the vocabulary but were nearly absent in training data, producing bizarre behaviour when invoked. A 2023 curiosity; a good reminder that the vocabulary is a *learned artefact*, not a specification.

**2026 note:** byte-level models and learned tokenizers (e.g. architecture-level alternatives like MegaByte, or character-level SSMs) exist but have not displaced BPE at the frontier, because BPE's shorter sequences save more compute than it costs.

---

## 1.3 Step 2: Tokens → vectors (embeddings)

Each token ID indexes a row of an **embedding matrix** `E ∈ R^{V×d}` where V = vocab size, d = model width (e.g. 4096). So token 12345 becomes a 4096-dimensional vector.

Two properties matter:

- **These vectors are learned, not hand-designed.** The famous "king − man + woman ≈ queen" analogy works because the model organises the space geometrically to make prediction easier — not because anyone encoded semantics.
- **The residual stream.** In a Transformer, the embedding is added to a running vector that every layer reads from and writes to. Anthropic's interpretability work calls this the **residual stream** and treats it as a shared "bus" or working memory (`12`). This metaphor is unusually productive: layers don't each compute a fresh representation, they *edit* a common one.

---

## 1.4 Step 3: Attention — the heart of the Transformer

### The intuition

Imagine reading the sentence *"The trophy didn't fit in the suitcase because it was too big."* To resolve "it", you look back at earlier words and weight them by relevance: "trophy" strongly, "suitcase" weakly. **Attention** is that operation, done with dot products, in parallel, for every position at once.

### The mechanism, precisely

For each token position, we compute three vectors by linear projection:

- **Query** `q = x W_Q` — "what am I looking for?"
- **Key** `k = x W_K` — "what do I offer?"
- **Value** `v = x W_V` — "what do I actually pass along if selected?"

Then, for a single attention head:

```
scores_ij = (q_i · k_j) / sqrt(d_head)      # relevance of j to i
α_ij      = softmax_j(scores_ij)            # normalise across j
output_i  = Σ_j α_ij v_j                    # weighted sum of values
```

In matrix form for the whole sequence:

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
```

Read that formula as three steps:
1. **`QKᵀ`** — every query dotted with every key. An `n×n` matrix of "how much does position i care about position j". This is the O(n²) term.
2. **`/ √d_k`** — scaling. Without it, dot products of high-dimensional vectors have large variance, softmax saturates into a one-hot distribution, and gradients vanish. (This is the whole content of the original "Attention Is All You Need" footnote that people skip.)
3. **`softmax(...) V`** — turn scores into weights, then take the weighted average of values.

### Why "multi-head"?

One head can only express one kind of relationship. **Multi-head attention** runs h independent (Q,K,V) projections in parallel (e.g. h=32, each of dimension 128, concatenated and projected back to d). Different heads empirically specialise: some track syntactic dependencies, some attend to the previous token, some to matching brackets, some to semantically related content, and in bigger models, some implement surprisingly abstract circuits (induction heads, `12`).

Cost: `h × d_head = d_model`, so multi-head attention at fixed d_model costs the same as single-head — the parallelism is nearly free.

### Causal masking

For a language model, position i must not see position j > i (that would be cheating — predicting the next token while looking at it). So we add `−∞` to the upper triangle of the score matrix before softmax, making those weights exactly zero. This is **causal (autoregressive) attention**. Encoders (BERT-style) skip the mask; decoders (GPT-style) use it.

### Cross-attention

In encoder–decoder models (T5, Whisper, and the original Transformer), the decoder's queries come from the decoder's own stream while keys/values come from the encoder's output. That's how translation conditions on the source sentence. Modern decoder-only LLMs mostly don't use it — multimodal models instead project image/audio features into the token stream (`11`).

---

## 1.5 Step 4: The MLP / feed-forward block

After attention mixes information *across positions*, each position passes through a small MLP independently:

```
FFN(x) = W₂ · activation(W₁ x)
```

with an inner dimension typically 3–4× the model width. (Modern models use **SwiGLU** with a gating branch and a ratio closer to 8/3 of `d_model` to keep parameter count constant — see `02`.)

**Where does knowledge live?** Largely here. The dominant view from interpretability and from "knowledge editing" research is that the FFN layers act as key–value memories: the first matrix detects a pattern, the second retrieves associated content. Attention moves information; MLPs transform and store it. That's a simplification, but a useful one.

---

## 1.6 Step 5: Residual connections, normalisation, and depth

Each sublayer is wrapped:

```
x ← x + Attention(Norm(x))
x ← x + FFN(Norm(x))
```

**Residual connection** (`x + f(x)`): the output of each block is *added* to its input rather than replacing it. This is essential for training deep networks — it gives gradients a direct path backwards (the "gradient highway") and makes each layer a small *edit* rather than a total rewrite.

**Normalisation**: keeps activation magnitudes stable. The original Transformer used **LayerNorm** after each sublayer ("post-norm"); modern LLMs almost universally use **RMSNorm** *before* each sublayer ("pre-norm"), because pre-norm trains far more stably at depth. RMSNorm drops the mean-centering and just divides by the root-mean-square:

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ
```

**Depth** is what makes composition possible: layer 1 might detect "this is a date", layer 10 might combine that with "this is a contract", layer 40 might decide "so the deadline applies". Frontier models in 2025–26 have roughly 60–120 layers (Llama 3 405B: 126; DeepSeek-V3: 61). Depth is expensive because it serialises computation — you can't parallelise across layers the way you can across tokens.

---

## 1.7 Step 6: The training objective

At the final layer, a linear **unembedding** (`W_U`, sometimes tied to the embedding matrix) maps the residual stream to vocabulary logits, softmax gives probabilities, and the loss is **cross-entropy**:

```
L = − (1/T) Σ_t log p_θ(x_t | x_<t)
```

That's it. **Next-token prediction.** No labels, no human annotation, no explicit notion of truth or helpfulness. The entire capability of modern LLMs — translation, code, arithmetic, theory of mind, planning — is a *side effect* of getting very good at this one objective.

This is the deepest and strangest fact in the field, and it deserves the awe it gets. The standard explanation: to predict the next token well across all of text, you must internally model grammar, facts, style, arithmetic, social conventions, and the causal structure of the world that generated the text. Prediction is a *compression* objective, and compression requires understanding.

**Optimisation:** parameters are updated by gradient descent, in practice **AdamW** (adaptive per-parameter learning rates + decoupled weight decay), or in 2025–26 increasingly **Muon** (matrix-orthogonalised momentum, ~1.4× speedup at matched scale per 2025 work) — see `04`.

---

## 1.8 Scaling laws: the most important empirical result in the field

### What was measured

Kaplan et al. (2020) trained many models of varying size on varying amounts of data and found loss follows smooth power laws:

```
L(N) ≈ (N_c / N)^α_N          # as parameters N grow
L(D) ≈ (D_c / D)^α_D          # as data D grows
L(C) ≈ (C_c / C)^α_C          # as compute C grows
```

with exponents around α_N ≈ 0.076, α_D ≈ 0.095 in the original fit. **Power laws mean no wall**: 10× more compute reliably buys a fixed fractional loss reduction. This is why labs were willing to spend billions — the return was, for a while, predictable.

The combined form (Chinchilla):

```
L(N, D) = E + A/N^α + B/D^β
```

`E` is the irreducible loss (entropy of language itself), and the two terms are "not enough parameters" and "not enough data".

### The Chinchilla correction

Kaplan's fit implied you should grow parameters much faster than data. Hoffmann et al. (2022) re-ran the study more carefully (including learning-rate schedules matched to the token budget — a methodological subtlety that had biased the original) and found the compute-optimal allocation is roughly **equal scaling: N ∝ C^0.5, D ∝ C^0.5**, i.e.

```
D* ≈ 20 × N
```

So a 70B model should be trained on ~1.4T tokens. At the time, GPT-3 (175B) had been trained on only 300B tokens — dramatically undertrained. Chinchilla (70B, 1.4T tokens) beat GPT-3, GPT-3.5 and Jurassic-1 on most benchmarks despite being 2.5× smaller.

### The part everyone forgets

"20 tokens per parameter" is the optimum for **minimising training loss at fixed training compute**. It is *not* the optimum for a deployed model, because:

1. **Inference dominates lifetime cost.** A model that is served billions of times should be *smaller and better-trained* than the training-optimal point. The right objective is total cost of ownership, which pushes you to overtrain small models (Llama 3 8B: ~15T tokens ≈ 1,900 tokens/param).
2. **Repeatable data has diminishing returns.** If you only have 1T unique tokens, "more epochs" replaces "more tokens" and the law changes (Muennighoff et al. 2023/2025, *Scaling Data-Constrained Language Models*): up to ~4 epochs is nearly as good as fresh data; beyond that, returns decay fast, and the correct response is to train *smaller* models for more epochs than Chinchilla would suggest.
3. **Laws fit the pretraining loss, not downstream tasks.** Loss and capability correlate smoothly, but specific skills (arithmetic, instruction following, long-context retrieval) appear at thresholds and don't follow the aggregate curve.

2025–26 refinement: **quality-aware scaling laws** add a data-quality term Q, and **data-constrained laws** (SoftQ, 2026) model the interaction between N and D under repeated data, because classical additive forms are misspecified in the data-rich-compute-poor regime we're now in (see `04` and `17`).

### The second scaling axis: inference time

Separately, 2024–25 work established **inference-time scaling laws**: accuracy on hard reasoning tasks rises approximately log-linearly with generated tokens or sampled candidates, and can beat scaling the model itself for a given dollar budget on some problems. See `06` — this is arguably the most consequential result of the whole period.

---

## 1.9 "Emergent abilities" — and the fight about whether they exist

**Claim (Wei et al. 2022, "Emergent Abilities of Large Language Models"):** certain capabilities (multi-step arithmetic, chain-of-thought reasoning, instruction following) are near-zero for small models and then appear abruptly past a scale threshold. If true, this implies you cannot predict capabilities from small-scale experiments, which is both exciting and terrifying.

**Counter-claim (Schaeffer et al. 2023, "Are Emergent Abilities a Mirage?"):** the abruptness is largely an artefact of *discontinuous metrics*. If a task is scored all-or-nothing (exact match on a multi-digit answer), a model that gets 3 of 4 digits right scores 0 — so smooth underlying improvement looks like a cliff. Switch to a continuous metric (token-level accuracy, edit distance, log-probability of the correct answer), and the same runs show smooth improvement.

**Where the field landed (as of 2026):** both are partly right.

- For *many* specific metrics, emergence is a measurement artefact. This is well demonstrated.
- For *some* capabilities, there is genuine nonlinearity — most convincingly, **in-context learning** (which pure Mamba models struggled to develop at all, motivating hybrid architectures, §`03`) and **chain-of-thought**, which appears to require enough depth/width to sustain a multi-step computation in one forward pass.
- The practical lesson is boring and important: **always look at the metric before believing a discontinuity.** And always plot more than one metric.

Related, and worth knowing: **grokking** (a model suddenly generalises long after it has fit the training set — e.g. modular arithmetic) shows that "training loss" and "generalisation" are separate curves. And **double descent** shows that more parameters can *reduce* test error even past the point of perfect training fit. Both are reminders that the loss curve is not the capability curve.

---

## 1.10 A complete forward pass, in order

Putting §1.2–1.7 together, here is what happens to `"The capital of France is"` in a 32-layer model:

1. **Tokenize** → `["The", " capital", " of", " France", " is"]` → `[464, 3136, 286, 4881, 318]`
2. **Embed** → 5 vectors of dimension 4096; add **positional information** (RoPE, applied inside attention — `02`)
3. **For each of 32 blocks:**
   a. RMSNorm
   b. Multi-head causal attention: each position gathers context from earlier positions (at " is", heavy weight on "France" and "capital")
   c. Residual add
   d. RMSNorm
   e. SwiGLU MLP: per-position transformation (retrieves "Paris")
   f. Residual add
4. **Final norm + unembed** → logits over 128k tokens
5. **Softmax** → `p(" Paris") ≈ 0.91`, `p(" Lyon") ≈ 0.02`, …
6. **Sample** (temperature/top-p) or take argmax → append token, repeat.

---

## 1.11 Common misconceptions

- **"Attention is all the model does."** Attention is ~1/3 of parameters and a minority of FLOPs at long-ish sequence lengths in modern MoE models; the MLPs (and experts) hold most of the parameters.
- **"Bigger context = better."** Attention cost and cache size grow; quality often degrades in the middle (`08`).
- **"The model looks up facts from a database."** Facts are distributed across parameters, not stored as records. This is why hallucination is structural, not a bug you can patch out.
- **"Temperature changes what the model knows."** It only changes sampling from a fixed distribution.
- **"Scaling laws are laws of nature."** They're empirical fits to a specific regime, architecture family, and data distribution. Change the data distribution and the constants change; change the architecture enough and the exponents do.

---

## 1.12 If you remember five things

1. Next-token prediction on a huge corpus is the only pretraining objective that matters; everything else is post-training.
2. Attention = softmax(QKᵀ/√d)V: relevance-weighted averaging across positions, with causal masking.
3. Residual stream + pre-norm RMSNorm is why deep Transformers train at all.
4. Chinchilla: scale parameters and data together (~20 tokens/param) for training-optimal; overtrain small models for serving-optimal.
5. "Emergence" is often a metric artefact — check the metric before believing the cliff.

---

## 1.13 Self-check

1. Derive why the `1/√d_k` factor prevents softmax saturation.
2. A 1B model is trained on 20B tokens. Is it Chinchilla-optimal? Is it a good model to serve? Why the difference?
3. Give a task where exact-match scoring would manufacture "emergence" and a continuous metric would not.
4. Why do residual connections help gradients, mechanically?
5. If a tokenizer merges `"123"` into one token, what kinds of errors does that induce?
6. Explain in one sentence why MoE breaks the usual "parameters = compute" assumption.

**Next:** [`02-modern-llm-architecture.md`](02-modern-llm-architecture.md) — what a 2026 model actually contains.
