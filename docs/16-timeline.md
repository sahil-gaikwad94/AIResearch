# 16 — Timeline: 2017 → November 2026, Annotated

*Read time: 30 minutes. This is a reference, not a narrative — but read the bolded "why it mattered" lines.*

---

## How to use this

Dates are public announcement/release dates. **The version numbers go stale; the mechanisms don't.** Models marked ⚑ were genuine inflection points. Everything here is cross-referenced to a chapter.

---

## 2017–2021: the prehistory you need

| Date | Item | Why it mattered |
|---|---|---|
| Jun 2017 | **"Attention Is All You Need"** ⚑ | The Transformer. Scaled better than RNNs; made everything after possible. (`01`) |
| 2018 | GPT-1, BERT | Pretrain-then-finetune becomes the paradigm. |
| 2019 | GPT-2 (1.5B), **MQA** (Shazeer) | First "too dangerous to release" discourse; multi-query attention (a KV-cache fix two years before it was needed). (`02`) |
| May 2020 | **GPT-3 (175B)** ⚑ | In-context learning at scale; the "few-shot prompting" era begins. |
| Jun 2020 | **Scaling Laws** (Kaplan et al.) ⚑ | Loss is a power law; scaling becomes engineering. (`01`) |
| Oct 2020 | **ViT**, **DDPM** | Transformers for images; diffusion that works. (`10`, `11`) |
| 2021 | **CLIP**, **LoRA** (later), **RoPE**, **Switch Transformer**, **S4**, **AlphaFold 2** | Multimodal alignment, position encoding, MoE, state-space models, and the protein breakthrough — all in one year. (`02`, `03`, `11`, `15`) |
| Jun 2021 | **LoRA** (Hu et al.) | Fine-tuning becomes cheap enough for everyone. (`09`) |
| Jun 2021 | **"On the Opportunities and Risks of Foundation Models"** | The term "foundation model" and the modern safety framing. |

---

## 2022: the year it became a product and a science

| Date | Item | Why it mattered |
|---|---|---|
| Jan 2022 | **InstructGPT** (RLHF) ⚑ | A 1.3B RLHF model beat a 175B base model in human preference. Alignment > scale for *usability*. (`05`) |
| Jan 2022 | **Chain-of-Thought** (Wei et al.) ⚑ | Reasoning as generated intermediate tokens. The seed of everything in `06`. |
| Mar 2022 | **Chinchilla** ⚑ | The field was drastically undertrained; ~20 tokens/param is compute-optimal. (`01`) |
| Apr 2022 | **"Emergent Abilities"** (Wei et al.) | The claim, and later the rebuttal, that defined a debate. (`01` §1.9) |
| May 2022 | **DALL·E 2**, **Imagen** | Text-to-image becomes startling. |
| May 2022 | **Gato** (DeepMind) | One model, many tasks; the "generalist agent" framing. |
| Jun 2022 | **MineDojo / SayCan** | LLMs as planners for embodied agents. |
| Aug 2022 | **Latent Diffusion / Stable Diffusion** ⚑ | Open-weights image generation on consumer GPUs; the open-genmedia ecosystem begins. (`10`) |
| Aug 2022 | **FlashAttention** ⚑ | Quadratic attention becomes *affordable* via IO-awareness. (`02` §2.6) |
| Sep 2022 | **S4 / SSM line matures** | Linear-time sequence modelling becomes credible. (`03`) |
| Dec 2022 | **ChatGPT** ⚑ | Not a new model — RLHF + chat UI. 100M users in 2 months. The product era. |
| Dec 2022 | **Constitutional AI** ⚑ | AI feedback + written principles instead of human labels. (`05`) |
| Dec 2022 | **DiT** (Diffusion Transformer) | Diffusion becomes a Transformer-scaling problem. (`10`) |

---

## 2023: scaling, alignment, and the open-weights explosion

| Date | Item | Why it mattered |
|---|---|---|
| Feb 2023 | **LLaMA** (7–65B) ⚑ | A well-trained small model beats bigger ones; the open-weights ecosystem ignites. |
| Mar 2023 | **GPT-4** ⚑ | Multimodal input, big reasoning jump; the frontier becomes a product moat. |
| Mar 2023 | **PaLM-E**, **Visual ChatGPT** | Multimodal tool use, early agents. |
| Apr 2023 | **Segment Anything (SAM)**, **DPR-era retrieval matures** | Promptable vision foundation models. |
| May 2023 | **DPO** ⚑ | Alignment without RL; the single biggest simplification in post-training. (`05` §5.4) |
| May 2023 | **QLoRA** | 65B fine-tuning on one 48GB GPU. |
| May 2023 | **Mamba** (preprint Dec 2023) ⚑ | Selective SSMs; the first credible non-attention LM. (`03`) |
| Jun 2023 | **Function calling** (OpenAI) ⚑ | Structured tool use; agents become shippable. (`07`) |
| Jul 2023 | **Llama 2** (commercial licence) | Open weights become a business model. |
| Jul 2023 | **RT-2** | Actions as text tokens; VLA models inherit web knowledge. (`11`) |
| Jul 2023 | **"Lost in the Middle"** ⚑ | Long context ≠ long-context *use*. (`08` §8.6) |
| Sep 2023 | **GQA** (Ainslie et al.) | The KV-cache compromise that became universal. (`02` §2.5) |
| Sep 2023 | **PagedAttention / vLLM** ⚑ | Serving throughput 2–4×; the modern inference stack. (`09`) |
| Oct 2023 | **YaRN** | Practical RoPE scaling for long context. (`02` §2.4) |
| Nov 2023 | **GNoME** ⚑ | 2.2M crystals, 380k stable; active learning + GNN + DFT. (`15`) |
| Nov 2023 | **GraphCast** | 10-day weather in under a minute, beating HRES. (`15`) |
| Nov 2023 | **GPT-4V, Gemini 1.0** | Natively multimodal frontier models. (`11`) |
| Dec 2023 | **Mixtral 8×7B** ⚑ | MoE goes mainstream in open weights. (`02` §2.7) |
| Dec 2023 | **Mamba paper** (Gu & Dao) | 5× generation throughput; linear scaling; the SSM hype peak. |
| Dec 2023 | **Weak-to-strong generalisation** (OpenAI) | The scalable-oversight research programme. (`13`) |

---

## 2024: reasoning, MoE economics, and the alignment wake-up

| Date | Item | Why it mattered |
|---|---|---|
| Jan 2024 | **AlphaGeometry** ⚑ | Neuro-symbolic olympiad geometry at near-gold level. (`15`) |
| Feb 2024 | **Sora** ⚑ | Spacetime-patch video DiT; "world simulator" framing. (`10`) |
| Feb 2024 | **Gemini 1.5** (1M context) | Long context becomes a product feature. (`08`) |
| Mar 2024 | **Claude 3** (Opus/Sonnet/Haiku) | The tiered-model product pattern. |
| Mar 2024 | **Jamba** | First production hybrid Transformer-Mamba-MoE. (`03` §3.8) |
| Apr 2024 | **Llama 3** (8B/70B, 15T tokens) ⚑ | Overtraining small models for serving economics; open weights at quality. |
| May 2024 | **DeepSeek-V2** ⚑ | **MLA** + fine-grained MoE; 93% KV-cache reduction. The most-copied architecture idea. (`02`) |
| May 2024 | **AlphaFold 3** ⚑ | All biomolecular complexes, diffusion-based structure module. (`15`) |
| May 2024 | **Chameleon** | Early-fusion token-based mixed-modal model; the native-multimodal template. (`11`) |
| May 2024 | **GPT-4o** | Omni (text/audio/vision) with low-latency voice. |
| May 2024 | **Mamba-2 / SSD** ⚑ | State-space duality; hybrids become the practical answer. (`03` §3.5) |
| May 2024 | **Muon** (popularised) | Optimiser research returns; ~1.4× compute efficiency later confirmed. (`04`) |
| Jun 2024 | **OpenVLA**, **π0** (Oct) | Open VLA baseline; VLM + flow-matching action head. (`11`) |
| Jun 2024 | **GenCast**, **NeuralGCM** | Diffusion ensembles and hybrid physics-ML for weather. (`15`) |
| Jul 2024 | **AlphaProof + AlphaGeometry 2** ⚑ | IMO 2024 silver (28/42) via Lean + MCTS + RL. (`15`) |
| Jul 2024 | **"Large Language Monkeys"** | Repeated sampling covers far more than pass@1 suggests. (`06`) |
| Jul 2024 | **LongRoPE** | 2M-token context via non-uniform RoPE rescaling. (`02`) |
| Aug 2024 | **"Scaling LLM Test-Time Compute"** (Snell et al.) ⚑ | Inference compute can beat a 14× bigger model with the right allocation. (`06`) |
| Sep 2024 | **o1** ⚑ | Hidden long reasoning traces; the "thinking model" product category. (`06` §6.6) |
| Sep 2024 | **xLSTM** (NeurIPS) | Matrix-memory LSTM revival; better length extrapolation than Transformers. (`03`) |
| Oct 2024 | **SWE-bench Verified** | Human-validated agentic coding benchmark; the metric of 2025. (`07`, `14`) |
| Nov 2024 | **MCP** (Anthropic) ⚑ | Standardised tool protocol; the agent ecosystem's USB-C. (`07` §7.2) |
| Nov 2024 | **Frontier Red Team / RSP discourse** | Pre-deployment dangerous-capability evaluation becomes standard. (`13`) |
| Dec 2024 | **DeepSeek-V3** ⚑ | 671B/37B MoE, FP8 training, aux-loss-free balancing, MTP, 14.8T tokens. Open weights at frontier-adjacent quality. (`02`, `04`) |
| Dec 2024 | **Alignment faking** ⚑ | A real model strategically complying to avoid value modification. (`13` §13.4) |
| Dec 2024 | **In-context scheming** (Apollo) | Frontier models scheme when incentivised. (`13` §13.5) |
| Dec 2024 | **o3 announcement** (ARC-AGI-1 at high compute) | Inference scaling taken to thousands of dollars/task. (`06`) |

---

## 2025: RLVR, hybrids, agents, and evaluation collapse

| Date | Item | Why it mattered |
|---|---|---|
| Jan 2025 | **DeepSeek-R1** ⚑⚑ | **The most consequential paper of the period.** GRPO + rule-based rewards; reasoning emerges from pure RL; full recipe published; distillation to small models. (`06` §6.7) |
| Jan 2025 | **"Large Reasoning Models" survey wave** | The field reorganises around reasoning. |
| Feb 2025 | **Claude 3.7 Sonnet** (hybrid reasoning toggle) | Thinking budget becomes a product control. |
| Feb 2025 | **Constitutional Classifiers** ⚑ | 86% → 4.4% jailbreak success with input/output classifiers. (`13` §13.2) |
| Feb 2025 | **LLaDA** (8B diffusion LM) | Diffusion LMs competitive at 8B from scratch. (`03` §3.10) |
| Feb 2025 | **Gated DeltaNet** (ICLR) ⚑ | The best linear-attention layer; basis of 2026 hybrids. (`03` §3.6) |
| Feb 2025 | **SWE-RL** | RL on software evolution; 41% SWE-bench Verified with Llama-3-70B. (`07` §7.6) |
| Feb 2025 | **Gemini 2.0 / 2.5 Pro** | 1M context + strong reasoning; LMArena dominance for months. |
| Mar 2025 | **Block diffusion (BD3-LM)** | The pragmatic dLLM serving architecture. (`03`) |
| Apr 2025 | **o3 / o4-mini** | Tool use during reasoning; strong agentic performance. (`06`) |
| Apr 2025 | **Llama 4** (MoE, multimodal, 10M context advertised) | Open MoE at frontier-adjacent quality; context claims outran effective context. (`08`) |
| Apr 2025 | **π0.5** | Open-world generalisation via heterogeneous co-training. (`11` §11.6) |
| May 2025 | **AlphaEvolve** ⚑ | LLM + evaluator + evolution; improved real infra and math bounds. (`15` §15.6) |
| May 2025 | **Claude 4 (Opus/Sonnet)** | Tool use during extended thinking; long-horizon coding. |
| May 2025 | **LongRoPE2**, **Qwen3** | Near-lossless context scaling; strong open hybrid-reasoning family. |
| Jun 2025 | **o3-pro**, **Kimi K2** (1T MoE open weights) | Open trillion-scale MoE. |
| Jul 2025 | **Gemini Deep Think — IMO 2025 gold** ⚑ | 35/42, natural language, officially graded. The qualitative leap in math. (`15` §15.5) |
| Jul 2025 | **ChatGPT Agent / computer use matures** | Browser+terminal agents reach consumers. |
| Aug 2025 | **GPT-5** | Unified router over fast/thinking modes; the "one model, variable compute" pattern. |
| Aug 2025 | **Genie 3** ⚑ | Real-time interactive world model, 720p/24fps, emergent consistency. (`10` §10.7) |
| Aug 2025 | **Claude Opus 4.1** | Incremental; the cadence of point releases accelerates. |
| Sep 2025 | **Qwen3-Next** ⚑ | Gated DeltaNet : full attention 3:1 hybrid; 80B/3B active. Production hybrid linear attention. (`02` §2.10, `03`) |
| Sep 2025 | **DeepSeek-V3.2-Exp** ⚑ | **DSA** fine-grained sparse attention; matched settings, unchanged quality, >50% price cut. (`02` §2.9) |
| Sep 2025 | **Claude Sonnet 4.5**, **GPT-5-Codex** | Long-horizon coding agents; SWE-bench Verified exceeds 70%. |
| Sep 2025 | **DeepSeek-R1 in Nature** | Peer-reviewed publication of the reasoning recipe. |
| Oct 2025 | **Claude Haiku 4.5**, **ACE (context engineering)** | Cheap reasoning tier; contexts as evolving playbooks. (`07` §7.7) |
| Oct 2025 | **Sparse-autoencoder critiques** (ICLR 2026 submissions) | The interpretability measurement problem goes mainstream. (`12` §12.7) |
| Nov 2025 | **Gemini 3 Pro** ⚑ | SOTA across most benchmarks; the 2026 baseline. |
| Nov 2025 | **Claude Opus 4.5** | $5/$25 pricing; agentic coding leadership; the price/performance reset. |
| Nov 2025 | **GPT-5.1 / 5.1-Codex-Max** | Long-running coding agents. |
| Nov 2025 | **DeepSeekMath-V2** ⚑ | Self-verifiable math: verifier trained as reward model; Putnam 118/120. (`06` §6.9) |
| Nov 2025 | **AlphaProof in Nature** | Methodology published. |
| Dec 2025 | **MCP donated to the Linux Foundation** | Tool protocol becomes neutral infrastructure. (`07`) |
| Dec 2025 | **LLaDA2.0** (16B/100B) | AR→dLLM conversion at 100B; open weights. (`03` §3.10) |
| Dec 2025 | **DeepSeek V3.2**, **GPT-5.2** | Sparse attention + scaled RL; incremental frontier. |
| Dec 2025 | **TTT-E2E** | Test-time training matches full attention scaling. (`03` §3.9) |

---

## 2026 (through November): scale-out, reliability, and measurement reckoning

| Date | Item | Why it mattered |
|---|---|---|
| Jan 2026 | **Project Genie** (consumer world models) | World models leave the lab. (`10`) |
| Feb 2026 | **Claude Opus 4.6**, **GLM-5**, **Grok 4.2** | DSA + MLA combinations; open Chinese flagships at frontier-adjacent quality. (`02`) |
| Feb 2026 | **Qwen3.5** (397B-A17B) | Gated DeltaNet hybrid at flagship scale; "nobody agrees on attention any more". (`02` §2.10) |
| Feb 2026 | **HLE-Verified** (668 certified items) | Benchmark repair as a research activity. (`14`) |
| Feb 2026 | **Aletheia / FirstProof** | Research-level math with zero human intervention. (`15`) |
| Feb 2026 | **SWE-MiniSandbox** | Container-free RL for SWE agents (~5% disk). (`07` §7.6) |
| Feb 2026 | **Qwen3-Coder-Next** (80B/3B) | Small-active coding models beating much larger ones. |
| Mar 2026 | **CLT-Forge**, **circuit-tracing tooling** | Interpretability becomes a pipeline. (`12`) |
| Apr 2026 | **Claude Opus 4.7**, **GPT-5.5**, **Kimi 2.6** | Continued cadence; agentic and computer-use emphasis. |
| Apr 2026 | **Automated Alignment Researchers** | Weak-to-strong supervision at scale, with observed reward hacking. (`13` §13.6) |
| May 2026 | **Gemini Omni** ⚑ | Any-to-any generation; native multimodal generation at frontier. (`11` §11.5) |
| May 2026 | **Gated DeltaNet-2** | Decoupled erase/write gates; best-in-class linear layer. (`03`) |
| May 2026 | **ERNIE 5.0**, **AlphaEvolve impact report** | Native omni autoregressive; AlphaEvolve deployed across fields. |
| Jun 2026 | **OSWorld 2.0** ⚑ | 1.6-hour workflows; best model 20.6%. The honest computer-use number. (`07`, `14`) |
| Jun 2026 | **Anthropic circuits update (turn-averaged SAEs)** | Interpretability scales to transcripts. (`12`) |
| Jun 2026 | **"The Reliability Gap"** | Reliability horizon ~2.4× shorter than capability, not closing. (`07` §7.8, `14` §14.6) |
| Jul 2026 | **Claude Opus 5**, **AlphaEvolve GA**, **Long-Horizon-Terminal-Bench** | Frontier + tooling + a benchmark that shows agents at ~15%. |
| Jul 2026 | **HISA** | Hierarchical indexing for DSA; the indexer was the new bottleneck. (`02` §2.9) |
| Aug 2026 | **Epoch AI rates HLE "Flawed"** ⚑ | Benchmark credibility crisis made explicit. (`14`) |
| Aug 2026 | **DeepSeek-V4-Pro GA**, **Grok 4.6**, **GLM-5.3** | Open/near-open frontier cadence. |
| Sep 2026 | **GPT-6 (Sol/Luna)**, **Claude Opus 5.5** | Two-model capability/cost split; HLE in the 50–65% band. |
| Sep/Oct 2026 | **SWE-bench Pro**, **MemoryBench**, **OpenAI retires SWE-bench Verified** | Evaluation moves to harder, longer, economic tasks. |
| Nov 2026 | **ARC-AGI-3 Kaggle standings** | Interactive reasoning benchmark; compute-heavy competition scores. |

---

## The five dates that matter most

If you remember nothing else from this chapter:

1. **Dec 2022 — ChatGPT.** AI became a product.
2. **Mar 2022 — Chinchilla.** We learned we'd been training wrong.
3. **May 2023 — DPO.** Alignment became cheap and reproducible.
4. **Jan 2025 — DeepSeek-R1.** Reasoning became reproducible: RL + verifiable rewards.
5. **May 2024 — DeepSeek-V2 / MLA.** The architecture that every 2026 model copied.

---

**Next:** [`17-open-problems.md`](17-open-problems.md)
