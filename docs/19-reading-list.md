# 19 — Annotated Reading List

*~90 items, grouped. ★ = read first. All links are canonical where possible.*

---

## How to use this

Don't read chronologically. Pick a chapter you just read, then read its 2–3 ★ papers. Reading a paper *after* understanding the idea is 10× more efficient than reading it cold.

---

## Foundations & scaling

- ★ **Attention Is All You Need** (Vaswani et al., 2017) — arXiv:1706.03762. The original; still worth reading the actual text once.
- ★ **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) — arXiv:2001.08361.
- ★ **Training Compute-Optimal LLMs** (Hoffmann et al., 2022, Chinchilla) — arXiv:2203.15556.
- **Scaling Data-Constrained Language Models** (Muennighoff et al., 2023; JMLR 2025) — arXiv:2305.16264. Repeated epochs, the real data wall.
- **Are Emergent Abilities a Mirage?** (Schaeffer et al., 2023) — arXiv:2304.15004.
- **Emergent Abilities of Large Language Models** (Wei et al., 2022) — arXiv:2206.07682.
- **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2019) — nucleus sampling.
- **Scaling Laws Revisited: data quality** (Subramanyam et al., ICLR 2026) — quality-aware laws.
- **Data-Constrained Pretraining: Regularization and Scaling Laws** (2026) — arXiv:2606.06888.

## Architecture

- ★ **RoFormer / RoPE** (Su et al., 2021) — arXiv:2104.09864.
- ★ **FlashAttention** (Dao et al., 2022) — arXiv:2205.14135; **v2** arXiv:2307.08691; **v3** (2024) arXiv:2407.08608.
- ★ **GQA** (Ainslie et al., 2023) — arXiv:2305.13245.
- ★ **DeepSeek-V2** (2024) — arXiv:2405.04434. MLA + fine-grained MoE.
- ★ **DeepSeek-V3** (2024) — arXiv:2412.19437. FP8, aux-loss-free balancing, MTP; the best single "how to build a modern model" paper.
- **Switch Transformers** (Fedus et al., 2021) — arXiv:2101.03961.
- **DeepSeekMoE** (2024) — arXiv:2401.06066. Fine-grained + shared experts.
- **DeepSeek Sparse Attention (V3.2-Exp)** (2025) — github.com/deepseek-ai/DeepSeek-V3.2-Exp.
- **GQA/MQA, KV cache, and PagedAttention** — see vLLM blog, blog.vllm.ai/2023/06/20/vllm.html.
- **YaRN** (Peng et al., 2023) — arXiv:2309.00071.
- **LongRoPE** (Ding et al., 2024) — arXiv:2402.13753. **LongRoPE2** (2025) — arXiv:2502.20082.
- **GLU Variants Improve Transformer** (Shazeer, 2020) — arXiv:2002.05202.

## Beyond Transformers

- ★ **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, 2023) — arXiv:2312.00752.
- ★ **Transformers are SSMs (Mamba-2 / SSD)** (Dao & Gu, 2024) — arXiv:2405.21060.
- ★ **Gated Delta Networks** (Yang et al., ICLR 2025) — arXiv:2412.06464.
- **Gated DeltaNet-2** (2026) — arXiv:2605.22791.
- **S4** (Gu et al., 2021) — arXiv:2110.13985.
- **Jamba: Hybrid Transformer-Mamba** (ICLR 2025) — arXiv:2403.19887.
- **xLSTM** (Beck et al., NeurIPS 2024) — arXiv:2405.04517.
- **RWKV-7 "Goose"** (2025) — arXiv:2503.14456.
- **Samba** (Microsoft, 2024) — arXiv:2411.02393.
- **Test-Time Training Done Right (LaCT)** (ICLR 2026).
- **End-to-End Test-Time Training for Long Context (TTT-E2E)** (2025) — arXiv:2512.23675.
- **LLaDA** (Nie et al., 2025) — arXiv:2502.09992. **LLaDA2.0 to 100B** — arXiv:2512.15745.
- **Block Diffusion (BD3-LM)** (Arriola et al., 2025) — arXiv:2503.09516.
- **Survey on Diffusion Language Models** (2025) — arXiv:2508.10875.

## Training & systems

- ★ **ZeRO** (Rajbhandari et al., 2020) — arXiv:1910.02054.
- ★ **Tensor Programs V (µP)** (Yang et al., 2021) — arXiv:2203.03466.
- **Megatron-LM** (Shoeybi et al., 2019) — arXiv:1909.08053.
- **Muon / matrix-preconditioned optimisers at scale** (NeurIPS 2025) — scaling rules for Muon/SOAP/Shampoo.
- **Llama 3 Herd of Models** (2024) — arXiv:2407.21783. Unusually candid about infra, data, and failures (419 interruptions).
- **Demystifying Synthetic Data in LLM Pre-training** (EMNLP 2025) — arXiv:2510.01631.
- **BitNet b1.58 2B4T** — huggingface.co/microsoft/bitnet-b1.58-2B-4T.
- **Context Parallelism for Million-Token Inference** (2025) — MLSys.
- **Ring Attention** (Liu et al., 2023) — arXiv:2310.01889.

## Post-training & alignment (methods)

- ★ **InstructGPT** (Ouyang et al., 2022) — arXiv:2203.02155.
- ★ **DPO** (Rafailov et al., 2023) — arXiv:2305.18290.
- ★ **Constitutional AI** (Bai et al., 2022) — arXiv:2212.08073.
- **Anthropic RLHF paper** (Bai et al., 2022) — arXiv:2204.05862.
- **IPO** (Azar et al., 2023) — arXiv:2310.12036. **KTO** (Ethayarajh et al., 2024) — arXiv:2402.01306. **ORPO** (Hong et al., 2024) — arXiv:2403.07691. **SimPO** (Meng et al., 2024) — arXiv:2405.14734.
- **Nathan Lambert, "RLHF and Post-Training" book** — rlhfbook.com. The best single resource on this chapter.

## Reasoning

- ★ **Chain-of-Thought Prompting** (Wei et al., 2022) — arXiv:2201.11903.
- ★ **DeepSeek-R1** (2025; *Nature* 2025) — arXiv:2501.12948. Read §2 and the appendix on GRPO vs PPO.
- ★ **DeepSeekMath / GRPO** (Shao et al., 2024) — arXiv:2402.03300.
- ★ **Scaling LLM Test-Time Compute Optimally** (Snell et al., 2024) — arXiv:2408.03314.
- **Self-Consistency** (Wang et al., 2022) — arXiv:2203.11171.
- **Tree of Thoughts** (Yao et al., 2023) — arXiv:2305.10601.
- **ReAct** (Yao et al., 2022) — arXiv:2210.03629.
- **Large Language Monkeys** (Brown et al., 2024) — arXiv:2407.21787.
- **s1: Simple test-time scaling** (2025) — arXiv:2501.19393.
- **DeepSeekMath-V2: self-verifiable reasoning** (2025) — github.com/deepseek-ai/DeepSeek-Math-V2.
- **Survey of Process Reward Models** (2025) — arXiv:2510.08049.
- **Does RL Really Incentivize Reasoning Beyond the Base Model?** (2025) — the sharpening critique.

## Agents

- ★ **SWE-bench** (Jimenez et al., 2023) — arXiv:2310.06770; **SWE-bench Verified** (OpenAI, 2024).
- ★ **METR time horizons** — metr.org; and "The Reliability Gap" (2026), arXiv/report.
- ★ **SWE-Gym** (Pan et al., ICML 2025) and **SWE-RL** (arXiv:2502.18449).
- **Toolformer** (Schick et al., 2023) — arXiv:2302.04761.
- **MCP specification** — modelcontextprotocol.io.
- **OSWorld** (2024) — arXiv:2404.07972; **OSWorld 2.0** (2026).
- **τ-bench** (Yao et al., 2024) — arXiv:2406.12045.
- **Agentic Context Engineering (ACE)** (2025) — arXiv:2510.04618.
- **Memory for Autonomous LLM Agents** (survey, 2026) — arXiv:2603.07670.
- **Self-play SWE-RL** (2025–26) — arXiv:2512.18552.
- **VerlTool** (2025) — arXiv:2509.01055.

## Retrieval & long context

- ★ **Lost in the Middle** (Liu et al., 2023) — arXiv:2307.03172.
- **RAG** (Lewis et al., 2020) — arXiv:2005.11401.
- **DPR** (Karpukhin et al., 2020) — arXiv:2004.04906.
- **ColBERT** (Khattab & Zaharia, 2020) — arXiv:2004.12832.
- **RULER** (2024) — arXiv:2404.06654. Effective context vs. advertised.

## Efficiency & serving

- ★ **PagedAttention / vLLM** (Kwon et al., 2023) — arXiv:2309.06180.
- ★ **GPTQ** (Frantar et al., 2022) — arXiv:2210.17323. **AWQ** (Lin et al., 2023) — arXiv:2306.00978.
- ★ **LoRA** (Hu et al., 2021) — arXiv:2106.09685. **QLoRA** (Dettmers et al., 2023) — arXiv:2305.14314.
- **Medusa** (Cai et al., 2024) — arXiv:2401.10774. **EAGLE** (Li et al., 2024) — arXiv:2401.15077. **EAGLE-3** (2025).
- **SGLang** — arXiv:2312.07104.
- **Blackwell Ultra / GB300 NVL72** — NVIDIA technical blog (Mar 2025).
- **vLLM docs** — docs.vllm.ai (the feature list is a good map of the field).

## Generative media

- ★ **DDPM** (Ho et al., 2020) — arXiv:2006.11239.
- ★ **Latent Diffusion** (Rombach et al., 2022) — arXiv:2112.10752.
- ★ **DiT: Scalable Diffusion Models with Transformers** (Peebles & Xie, 2023) — arXiv:2212.09748.
- **Classifier-Free Guidance** (Ho & Salimans, 2022) — arXiv:2207.12598.
- **Flow Matching** (Lipman et al., 2022) — arXiv:2210.02747. **Rectified Flow** (Liu et al., 2022) — arXiv:2209.03003.
- **"Diffusion Meets Flow Matching"** — diffusionflow.github.io. The clearest explanation of the equivalence.
- **Sora technical report** (OpenAI, 2024).
- **Genie 3** (DeepMind, Aug 2025) — deepmind.google/blog/genie-3-a-new-frontier-for-world-models.
- **Moshi** (Kyutai, 2024) — arXiv:2410.00037.

## Vision & multimodal

- ★ **ViT** (Dosovitskiy et al., 2020) — arXiv:2010.11929.
- ★ **CLIP** (Radford et al., 2021) — arXiv:2103.00020. **SigLIP** (Zhai et al., 2023) — arXiv:2303.15343.
- ★ **LLaVA** (Liu et al., 2023) — arXiv:2304.08485.
- **Flamingo** (Alayrac et al., 2022) — arXiv:2204.14198.
- **DINOv2** (Oquab et al., 2023) — arXiv:2304.07193.
- **SAM** (Kirillov et al., 2023) — arXiv:2304.02643. **SAM 2** (2024) — arXiv:2408.00714.
- **Chameleon** (Meta, 2024) — arXiv:2405.09818.
- **RT-2** (2023) — arXiv:2307.15818. **OpenVLA** — arXiv:2406.09246. **π0** — arXiv:2410.24164. **π0.5** — arXiv:2504.16054.
- **Survey: VLA action tokenization** (2025) — arXiv:2507.01925.

## Interpretability

- ★ **Toy Models of Superposition** (Elhage et al., 2022) — transformer-circuits.pub.
- ★ **Towards Monosemanticity / Scaling Monosemanticity** (Anthropic, 2023/2024).
- ★ **In-context Learning and Induction Heads** (Olsson et al., 2022) — transformer-circuits.pub.
- **Circuit Tracing / Attribution Graphs** (Anthropic, 2025) — transformer-circuits.pub.
- **Circuits Updates — June 2026** — turn-averaged SAEs.
- **Transcoders Find Interpretable LLM Feature Circuits** (Dunefsky et al., NeurIPS 2024).
- **Automated Interpretability Metrics Do Not…** (ICLR 2026) — the critique.
- **Representation Engineering** (Zou et al., 2023) — arXiv:2310.01405.

## Safety

- ★ **Alignment Faking in LLMs** (Greenblatt et al., 2024) — arXiv:2412.14093.
- ★ **Frontier Models are Capable of In-context Scheming** (Meinke et al., 2024) — arXiv:2412.04984.
- ★ **Constitutional Classifiers** (Sharma et al., 2025) — anthropic.com/research/constitutional-classifiers.
- **Stress Testing Deliberative Alignment for Anti-Scheming** (2025) — arXiv:2509.15541.
- **Weak-to-Strong Generalization** (Burns et al., 2023) — arXiv:2312.09390.
- **The Evaluation Differential** (2026) — arXiv:2605.11496.
- **Anthropic system cards / RSP** — anthropic.com (read one to see what real safety evaluation looks like).
- **Many-shot Jailbreaking** (Anthropic, 2024).

## Evals

- **MMLU** (2020) — arXiv:2009.03300. **GPQA** (2023) — arXiv:2311.12022. **HLE** (2025) — arXiv:2501.14249.
- **ARC-AGI** — arcprize.org. **FrontierMath** — epoch.ai.
- **GDPval** (OpenAI, 2025). **Remote Labor Index** (Scale).
- **"Why we no longer evaluate SWE-bench Verified"** (OpenAI, 2026) — the audit.
- **Epoch AI benchmark reviews** — epoch.ai/benchmarks. Excellent, skeptical.
- **SWE-bench Pro** — labs.scale.com.

## Science

- ★ **Highly accurate protein structure prediction with AlphaFold** (Jumper et al., *Nature* 2021).
- ★ **Accurate structure prediction of biomolecular interactions with AlphaFold 3** (*Nature* 2024).
- **Scaling deep learning for materials discovery (GNoME)** (*Nature* 2023).
- **Learning skillful medium-range global weather forecasting (GraphCast)** (*Nature* 2023).
- **AlphaGeometry** (*Nature* 2024). **AlphaProof methodology** (*Nature*, Nov 2025).
- **AlphaEvolve** white paper (2025) — arXiv:2506.13131.
- **IMO 2025 Gemini Deep Think** — deepmind.google (Jul 2025).

## Books, courses, and living resources

- **Nathan Lambert, RLHF and Post-Training** — rlhfbook.com (free, current, excellent).
- **Sebastian Raschka, "The Big LLM Architecture Comparison"** — magazine.sebastianraschka.com. Best architecture walkthroughs of specific 2025 models.
- **Hugging Face LLM Course** — huggingface.co/learn.
- **Lilian Weng's blog** — lilianweng.github.io. Foundational explainers on diffusion, agents, RL.
- **Transformer Circuits Thread** — transformer-circuits.pub.
- **Epoch AI** — epoch.ai. Compute/scaling/benchmark data, unusually rigorous.
- **Stanford AI Index** (annual) — hai.stanford.edu. The best single statistical overview.

---

## A 4-week self-study plan

**Week 1 — Foundations.** `01`, `02` + Chinchilla + DeepSeek-V3 papers. Build: a 2-layer Transformer from scratch (nanoGPT-style) and train it on a small corpus.

**Week 2 — Training and alignment.** `04`, `05` + InstructGPT + DPO. Build: LoRA fine-tune a 1–3B model on 1k instructions; run DPO on a small preference set.

**Week 3 — Reasoning and agents.** `06`, `07` + DeepSeek-R1 + SWE-bench. Build: implement GRPO on a verifiable arithmetic task; wire up a tool-using agent with an MCP server.

**Week 4 — Everything else.** `08`–`15` + the one paper in each that interests you. Build: a RAG system, then measure retrieval recall separately from answer accuracy — that exercise teaches more than any paper.

---

**Back to the [index](../README.md).**
