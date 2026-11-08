# 4 — The Tokeniser Is an Architecture Decision

> A prose-trained BPE wastes 15–25% of its tokens on code. Every wasted token is paid 40 times — once per layer of attention and cache. Fixing this is worth more than 30% more parameters, and it costs one afternoon of training.

---

## 4.1 Why this file exists

Sequence length is the tax base of the entire design:

- training FLOPs scale with tokens, and at 64 k context attention is 3.9× the parameter FLOPs;
- KV cache scales with tokens, and the cache is the binding constraint on concurrency;
- prefill latency scales with tokens, and prefill is 89–99% of compute;
- rollout cost scales with tokens, and rollouts are what RL spends money on.

A tokeniser that emits 18% fewer tokens cuts **every one of those by 18%**. At the full budget of ~45,700 GPU-hours that is ~7,000 GPU-hours, roughly $15k — for a change that costs about 200 GPU-hours to validate.

The standard tokeniser, trained mostly on web text, is badly matched to code:

| Code pattern | Typical BPE behaviour | Waste |
|---|---|---|
| Leading indentation | 4–8 spaces as separate tokens, or a merged `"    "` that only helps one depth | 10–15% of tokens in Python |
| `snake_case_name` | split at `_` or mid-word arbitrarily | 2–3 tokens where 1 could serve |
| `camelCaseName` | same | 2 tokens |
| Long numbers | merged into opaque tokens | hurts arithmetic, costs tokens |
| `==`, `->`, `!=`, `:=`, `**` | often split into characters | 2–3× |
| JSON/config punctuation | each brace and quote separate | large in tool output |
| Tool-call scaffolding | not present in the vocabulary at all | worst case: 5+ tokens per tag |

An agent session is dominated by exactly these patterns, because the *inputs* are files and tool output, not prose.

---

## 4.2 Design

**Vocabulary: 131,072 tokens.** Trained with a byte-level BPE on a mixture matched to the deployment distribution, not to The Pile.

### Training mixture

| Share | Content |
|---|---|
| 45 % | Permissively licensed code, stratified by language, deduplicated |
| 20 % | Code-adjacent text: READMEs, docstrings, man pages, API references, changelogs, issue threads, test names |
| 15 % | Natural language (web, books, technical writing) — needed for instruction following |
| 10 % | Structured data: JSON, YAML, TOML, CSV, SQL, diffs, logs |
| 7 % | Mathematics and LaTeX |
| 3 % | Multilingual (top 20 languages) |

Language weighting inside the code share: Python 22%, TypeScript/JavaScript 18%, C/C++ 12%, Java 9%, Go 8%, Rust 7%, SQL 5%, shell/PowerShell 5%, Ruby/PHP/C#/Kotlin/Swift 9%, everything else 5%.

### Explicitly reserved tokens

The tokeniser ships with fixed, single-token encodings for structures that appear thousands of times per session:

- **Indentation runs**: `"  "`, `"   "`, `"    "`, `"      "`, `"        "`, `"\t"`, `"\t\t"` — one token each, so depth 3 in Python costs one token, not three.
- **Blank-line and structural**: `"\n\n"`, `"\n\n\n"`, `"    \n"` (whitespace-only line).
- **Operators and syntax**: `" -> "`, `"::"`, `" != "`, `" == "`, `" <= "`, `" >= "`, `" => "`, `" := "`, `" ** "`, `"=> {"`, `"})"`, `"}, {"`, `");"`, `"::<"`.
- **Diff markers**: `"@@ "`, `"+++ "`, `"--- "`, `"-" ` at line start, `"+" ` at line start. Diffs are the preferred output format (§4.4), so this is directly on the hot path.
- **Tool and structure specials**: `<|tool_call|>`, `<|tool_result|>`, `<|file|>`, `<|patch|>`, `<|end|>`, `<|think|>`, `<|/think|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`.

Those last three matter more than they look: fill-in-the-middle training is how a coding model learns to insert code at a cursor rather than rewrite a file, and it needs dedicated sentinels.

### Digits

**Every digit is its own token.** `12345` costs 5 tokens instead of 1.

This looks like a regression in sequence length and is not, for three reasons: numbers are a small fraction of code tokens; merged digit tokens are the single largest source of arithmetic and off-by-one failures; and line numbers, array indices and hex constants are common enough that reliable digit-level handling pays for itself. GPT-4-class models all split digits for this reason.

### Identifiers

**Split on case and underscore boundaries, then let BPE merge back the frequent ones.** `getUserById` pre-tokenises to `get|User|By|Id`, and BPE will learn `getUser` as a single token because it is frequent, while `ById` may stay split.

This is the standard "regex pre-tokeniser" approach and it is where the code gains come from: it lets the vocabulary be compositional (morphemes) rather than memorising whole identifiers. A vocabulary that memorises `getUserById` needs a new entry for `getUserByName`; one that knows `get`, `User`, `By` does not.

**Risk, stated plainly:** over-splitting rare identifiers inflates sequence length on unusual codebases, and it makes the model slightly worse at reproducing long identifiers exactly. Mitigation: train the pre-tokeniser regex, then measure — §4.5.

### Whitespace fidelity

The tokeniser must round-trip **exactly**. Python indentation, Go's tab formatting, and YAML's significant whitespace all break if a normalisation step collapses runs. The encoder is byte-level with no normalisation beyond the pre-tokeniser split, and the test suite asserts `decode(encode(s)) == s` for a corpus of 10,000 files.

This is not a theoretical concern: a tokeniser that silently converts a tab to four spaces produces a model that emits code failing `gofmt`, `black --check`, and every linter in CI — and the failure looks like a model quality problem, not a tokeniser bug.

---

## 4.3 Expected gain, and how to know if it is real

Two measurements, both cheap:

1. **Tokens per file** on a held-out corpus: 2,000 repositories, post-cutoff, with a language mix matching the training mixture. Compare against a standard 128 k BPE trained on prose-heavy data.
2. **Bits per byte** at equal training compute. This is the honest comparison, because it normalises for sequence length: a tokeniser that halves tokens but doubles per-token entropy has gained nothing.

Predicted: **18% ± 6% fewer tokens on code**, 25–30% fewer on indented Python and JSON-heavy tool output, and a 2–4% improvement in bits-per-byte at fixed FLOPs.

The falsifier is in ablation **A1**: if the gain is under 10%, shrink the vocabulary to 65 k and spend the parameters on the FFN instead.

---

## 4.4 Output format as a tokeniser decision

The model should emit **unified diffs**, not whole files. This is a tokeniser-adjacent decision with large consequences:

| | Whole-file rewrite | Unified diff |
|---|---|---|
| Output tokens for a 20-line change in a 400-line file | ~4,000 | ~120 |
| Prefill to re-read the file afterwards | full file | unchanged prefix, cache hit |
| Reviewability | low | high |
| Failure blast radius | whole file | patch fails to apply |

A 33× reduction in generated tokens for the same edit. Since decode is what the user waits for and generated tokens are what RL must sample, this single formatting choice is worth more than most architecture decisions. It is also why diff markers are reserved tokens.

Fill-in-the-middle (`<|fim_*|>`) is the finer-grained version: insert at a cursor, generate only the missing span.

---

## 4.5 Implementation notes

- **Train the tokeniser on 40 GB** of the mixture with the same dedup as the corpus. A tokeniser trained on duplicates over-weights whatever was crawled most.
- **Freeze it before pretraining starts.** Changing the vocabulary mid-run invalidates every checkpoint.
- **Ship the pre-tokeniser regex with the model**, so downstream tooling reproduces token counts exactly. Token-count mismatches between training and serving are a classic silent quality bug: the model sees different boundaries than it was trained on, and the degradation is small enough to miss and large enough to matter.
- **Version it.** `tokenizer.json` hash goes in the run manifest; a mismatched hash should abort startup, not warn.

---

## 4.6 What would change my mind

- **Gain under 10%** → 65 k vocabulary, more FFN, see A1.
- **Byte-level modelling** (BLT-style, patching raw bytes with a learned entropy model) becomes competitive at 4 B scale → the tokeniser disappears entirely and the whole 18% argument is superseded. Not expected before 2028, but it is the most likely way this chapter becomes obsolete.
- **A tool ecosystem that standardises on structured output** (JSON patches over diffs) would change the reserved-token set, not the philosophy.

**Next:** [5 — Training](05-training.md).
