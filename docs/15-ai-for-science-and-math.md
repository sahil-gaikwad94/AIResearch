# 15 — AI for Science and Mathematics

*Read time: 55 minutes. Prerequisites: `06`, `07`, `14`.*

---

## 15.1 Why science was the first "real" application

Science is the ideal domain for AI for a structural reason: **it has verifiers**. A protein structure can be checked against crystallography. A proof can be checked by Lean. A material's stability can be checked by DFT. A forecast can be checked against what happened. When correctness is checkable, you can train with RLVR, filter synthetic data, and measure progress honestly — which is exactly the machinery developed in `05`–`06`.

This is why AI-for-science results tend to be *real* in a way that chatbot benchmarks are not, and why the failure modes are different (the bottleneck is usually experimental validation, not model capability).

---

## 15.2 AlphaFold: the canonical story

**AlphaFold 1** (2018, CASP13): won the protein-folding competition by a large margin, using a CNN over distance/angle predictions plus gradient-descent structure refinement.

**AlphaFold 2** (2020, CASP14): the breakthrough. Median accuracy competitive with experimental structures on most targets. Key architectural ideas:

- **Evoformer**: a transformer-like block that jointly updates a **multiple sequence alignment (MSA)** representation and a **pairwise residue–residue representation**, with attention that lets information flow between the two (triangle attention/updates enforce geometric consistency).
- **Structure module**: SE(3)-equivariant attention (IPA) that predicts 3D coordinates directly, iteratively refining.
- **End-to-end training** on the final 3D loss, rather than on intermediate distance predictions.

**Impact:** the AlphaFold Protein Structure Database released **200M+ predicted structures** (essentially every catalogued protein), used by **3M+ researchers in 190+ countries**. This is the clearest case of an AI result becoming scientific infrastructure.

**AlphaFold 3** (May 2024): extends from proteins to **all of life's molecules** — proteins, DNA, RNA, ligands, ions, and their interactions — using a diffusion module for structure generation on top of a Pairformer (a simplified Evoformer). Predicts complexes, not just single chains. Free via the AlphaFold Server for non-commercial research. Reported to substantially outperform docking tools on protein–ligand complexes.

**What it didn't do:** solve dynamics (proteins move), solve intrinsically disordered regions, or replace experiments. It predicts a static structure. The gap between "predicted structure" and "functional understanding" is where drug discovery actually lives.

**Adjacent:** **AlphaProteo** (2024) designs novel protein binders, trained on 100M+ AI-generated structures; **ESMFold** (Meta) is a faster, smaller open alternative; **RFdiffusion** and **ProteinMPNN** are the design-side workhorses.

---

## 15.3 GNoME and materials discovery

**GNoME** (Graph Networks for Materials Exploration, DeepMind, Nov 2023, *Nature*): a **graph neural network** predicting the total energy of a crystal structure (i.e. its stability).

The pipeline is the interesting part — it's a **data flywheel / active learning** loop:

1. Generate candidate structures (by substitution and by structure search).
2. GNoME predicts stability (energy above the convex hull).
3. The most promising candidates are **validated with DFT** (density functional theory).
4. Validated results are added back to the training data → the model improves → next round.

**Results:** 2.2M new crystal structures predicted stable; **381,000** on the updated convex hull (i.e. genuinely new, stable materials) — described as ~800 years of prior knowledge. Discovery rate improved from <10% to >80%.

**Why a GNN and not an LLM:** crystal structures have permutation/rotation symmetries that graph networks encode naturally; the data is small (a few hundred thousand DFT calculations), so inductive bias beats scale. This is an important counterpoint to the "transformers everywhere" narrative.

**The honest caveat:** computational stability ≠ synthesizable ≠ useful. The follow-on work (Energy-GNoME, 2025: 38,500 candidate energy materials) and the experimental validation bottleneck are the real story. Very few of these materials have been made.

---

## 15.4 Weather and climate

**GraphCast** (DeepMind, 2023, *Science*): a GNN that predicts 10-day global weather at 0.25° resolution in **under a minute** on a TPU, beating the operational gold standard (HRES) on 90% of tested variables. It's autoregressive (predict 6h ahead, feed back in), trained on ERA5 reanalysis.

**GenCast** (2024): an **ensemble** diffusion model — generates 50+ plausible trajectories, giving calibrated uncertainty (crucial for extreme-event forecasting), beating ENS on most variables.

**NeuralGCM** (2024): a hybrid — differentiable physics + learned components — that simulates 70,000 days of atmosphere in the time a physics model takes for 19 days, while remaining stable over long rollouts.

**FourCastNet 3** (2025): 60-day global forecast at 0.25° in under 4 minutes.

**Why this area succeeded fast:** abundant high-quality data (ERA5), a crisp metric (RMSE vs. observations), and a task where a fast approximate model is genuinely useful even if imperfect (ensembles mitigate error).

---

## 15.5 Mathematics: the IMO timeline

This is the cleanest public scoreboard for "reasoning", because the problems are hard, novel, and human-gradable.

| Date | System | Result | Mechanism |
|---|---|---|---|
| **Jan 2024** | **AlphaGeometry** | 25/30 on IMO-AG-30 (2000–2022 geometry), near gold-medalist average | **Neuro-symbolic**: a language model proposes auxiliary constructions, a symbolic deduction engine (DD+AR) closes the proof. Trained on 100M synthetic proofs. |
| **Jul 2024** | **AlphaProof + AlphaGeometry 2** | **28/42** at IMO 2024 = **silver** (gold threshold 29) | AlphaProof: formal proofs in **Lean**, trained by RL with a policy + value network and **Monte Carlo tree search** (AlphaZero-style), with problems auto-formalised from natural language. Method published in *Nature*, Nov 2025. |
| **Jul 2025** | **Gemini Deep Think** (advanced version) | **35/42**, five of six problems perfect = **gold**, officially graded by IMO coordinators under student rules | **Natural language**, no formalisation step, no symbolic engine. End-to-end model reasoning with a long internal trace. This is the qualitative leap. |
| **Nov 2025** | **DeepSeekMath-V2** | Gold-level on IMO 2025 and CMO 2024; **118/120** on Putnam 2024 with scaled test-time compute | **Self-verifiable reasoning**: a trained verifier acts as the reward model for a proof generator; verification compute is scaled to auto-label new hard proofs, keeping the verifier ahead. |
| **Feb 2026** | **Aletheia** (Gemini 3 Deep Think) | 6/10 on **FirstProof** with zero human intervention (5 unanimous, 1 at 5/7) | Research-level problems, never published online, designed to test independent research ability. |

**What changed between 2024 and 2025** is the most instructive part: AlphaProof needed humans to translate problems into Lean (a formal language) — a huge practical bottleneck. Gemini Deep Think worked in **natural language**, using the model's own reasoning as the search space. That's the difference between "a system that can do olympiad math if you formalise it for it" and "a system that can do olympiad math".

**What did NOT happen:** AI did not produce novel *mathematical research* at IMO level. FirstProof and similar efforts (FrontierMath, research-level benchmarks) are the attempt to measure that, and the numbers are much lower.

---

## 15.6 AlphaEvolve: evolutionary program search

**AlphaEvolve** (DeepMind, May 2025) is the template for "AI doing science" in the 2025–26 sense: **an LLM proposes code mutations, an automatic evaluator scores them, an evolutionary loop keeps what works.**

Architecture:
- An **ensemble** of Gemini models (Flash for breadth, Pro for depth) proposes edits to a program.
- **Automated evaluators** score each candidate (correctness + speed, or a mathematical objective).
- An **evolutionary database** of programs, with prompt construction that includes past attempts and their scores.

**Results (verified, deployed):**
- **Data-centre scheduling**: a heuristic that continuously recovers ~**0.7% of Google's global compute** — a huge absolute number.
- **Hardware**: a functionally equivalent simplification in an accelerator circuit design.
- **AI training**: a 23% speedup of a critical kernel → ~1% reduction in Gemini's training time; later, optimizing a large transformer to double training speed while improving quality.
- **Mathematics**: improved a known bound on **4×4 complex matrix multiplication** (the first improvement over Strassen's 1969 algorithm in that setting), and new constructions on 50+ open problems in analysis, geometry, combinatorics, number theory.
- **Science applications**: 30% reduction in variant-detection errors in DeepConsensus (genomics); improved disaster prediction; power-grid stabilisation simulations; interpretable neuroscience models; new bounds in microeconomics.
- **Commercial**: GA on Google Cloud (July 2026); TSP routing improvements of ~10.4% over a heavily optimised baseline.

**Why it works:** it's RLVR with the reward being a *program's measured performance*, and the "policy" being an LLM's code-editing ability. Evolution supplies the exploration; the verifier supplies the truth.

**The follow-on** (2026): open-world multi-agent "research stations" that run the same loop across many problems and cross-pollinate; one reported new infinite families of finite-field Kakeya sets and exact 604-point kissing configurations in dimension 11. Whether these are "AI discoveries" or "AI-assisted discoveries" is a fair question — the humans wrote the evaluators and chose the problems.

---

## 15.7 The pattern behind all of it

Strip away the domain, and every success above has the same shape:

```
generator (LLM or domain model) → candidate → VERIFIER → keep/filter → iterate
```

- AlphaFold: verifier = experimental structures (for training) + geometric consistency.
- GNoME: verifier = DFT.
- AlphaProof: verifier = Lean kernel.
- AlphaEvolve: verifier = program runs and scores.
- DeepSeekMath-V2: verifier = a trained proof checker.
- GraphCast: verifier = observations.

**The corollary is the limitation:** AI-for-science advances fastest where verification is cheap and objective, and stalls where it's expensive ("does this drug work in humans?"). That's why biology has spectacular structure prediction and slow drug discovery, and why social science has almost nothing.

---

## 15.8 What's genuinely hard, and honest

- **Experimental validation is the bottleneck.** 380k predicted stable materials, very few synthesised. Predictions are cheap; lab time isn't.
- **Distribution shift is lethal.** A model trained on known proteins extrapolates poorly to novel folds; a model trained on ERA5 struggles with a changed climate regime.
- **Interpretability matters for trust.** A scientist won't use a prediction they can't interrogate — which is why AlphaFold shipped with per-residue confidence (pLDDT) and PAE matrices, and why "AI found X" papers often fail replication when the mechanism isn't understood.
- **Novel hypothesis generation is still weak.** These systems are superb at *search within a specified space* and poor at *choosing which space to search*. The humans still pick the problem.
- **Publication and credit norms are unresolved** (can an AI be an author? who checks the proof?).

---

## 15.9 Common misconceptions

- **"AlphaFold solved biology."** It solved static structure prediction from sequence for many (not all) proteins. Function, dynamics, and interactions are separate problems.
- **"AI does research now."** AI does *search and verification* extremely well; problem selection, framing, and mechanism-level insight remain human.
- **"IMO gold means human-level math research."** Olympiad problems have known answers and short solutions; research has neither.
- **"GNoME's materials are usable."** Most are computationally stable and never synthesised.
- **"These are all LLM achievements."** AlphaFold is not an LLM; GNoME is a GNN; GraphCast is a GNN. Domain inductive bias still wins where data is scarce.

---

## 15.10 If you remember five things

1. Science advanced where verification was cheap — that's the whole selection effect.
2. AlphaFold 2's Evoformer + structure module, and AlphaFold 3's extension to all biomolecular complexes, are the canonical architecture lessons.
3. AlphaProof (Lean + MCTS + RL) → Gemini Deep Think (natural language) was the qualitative leap from "formalised" to "general" math.
4. AlphaEvolve = LLM + evaluator + evolution; it improved real infrastructure and real mathematical bounds.
5. The verifier is the ceiling, and experimental validation is the bottleneck.

---

## 15.11 Self-check

1. Why is a graph network the right choice for crystal stability prediction?
2. What did AlphaProof require that Gemini Deep Think did not, and why did that matter?
3. Describe the active-learning loop in GNoME and why DFT is essential to it.
4. What does DeepSeekMath-V2 do about the "generation–verification gap"?
5. Give a scientific field where AI should progress fast and one where it shouldn't, and justify with the verifier criterion.

**Next:** [`16-timeline.md`](16-timeline.md)
