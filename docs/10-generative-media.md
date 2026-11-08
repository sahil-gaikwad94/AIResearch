# 10 — Generative Media: Diffusion, Flow Matching, Video, and World Models

*Read time: 60 minutes. Prerequisites: `01` §1.8, `03` §3.10.*

---

## 10.1 The problem: generating a whole image at once

A language model generates one token at a time, with each token conditioned on the previous ones. Images don't have a natural left-to-right order — pixel (0,0) doesn't come "before" pixel (300,200) in any meaningful sense. Autoregressive image generation exists (and works), but it's slow and awkward.

**Diffusion models** solve this: start from pure noise and iteratively denoise, getting a little closer to a real sample at each step. Generation is a *refinement process*, not a left-to-right decode.

The 2020–2026 arc:
1. **DDPM** (2020) — the formulation that worked.
2. **DDIM / samplers** (2021) — 1000 steps → 20–50 steps.
3. **Classifier-free guidance** (2022) — the trick that made text conditioning actually work.
4. **Latent diffusion / Stable Diffusion** (2022) — do diffusion in a compressed latent space, not pixels → 50× cheaper.
5. **DiT** (2022–23) — replace the U-Net with a Transformer; scales predictably like an LLM (this is the Sora/DALL·E 3/Flux lineage).
6. **Flow matching / rectified flow** (2023–24) — a cleaner objective with straighter paths → fewer steps.
7. **Distillation** (2024–26) — 1–4 step generation (consistency models, adversarial distillation, DMD).
8. **Video + world models** (2024–26) — Sora, Veo 3, Genie 3/4.

---

## 10.2 DDPM: the maths, explained

**Forward process (fixed, no learning):** take a real image `x₀`. Repeatedly add a tiny amount of Gaussian noise over `T` steps (e.g. 1000), until `x_T` is indistinguishable from pure noise.

A useful closed form: at step `t`, `x_t = √(ᾱ_t)·x₀ + √(1−ᾱ_t)·ε`, where `ᾱ_t` is a schedule that decays from 1 to 0.

**Reverse process (learned):** train a network `ε_θ(x_t, t)` to predict *the noise that was added*. The loss is beautifully simple:

```
L = E_{x₀, ε, t} [ ‖ ε − ε_θ( √ᾱ_t x₀ + √(1−ᾱ_t) ε , t ) ‖² ]
```

In words: **take a clean sample, add known noise, and ask the model to guess what noise you added.** That's it. No adversarial game, no likelihood computation.

**Sampling:** start from noise, and iteratively subtract the predicted noise (scaled appropriately), stepping `T → 0`. With `T=1000` this is very slow; DDIM (2021) reformulated it as a deterministic ODE that works in 20–50 steps.

**Why it works (intuitively):** each denoising step is a small, easy prediction. The composition of many small, local corrections produces a sample from a complex global distribution. It's a kind of learned iterative refinement — conceptually close to a fixed-point iteration.

---

## 10.3 Classifier-free guidance (CFG): the single most important trick

**Problem:** you want the output to match the prompt strongly, not just "be a plausible image that could be captioned this way". Naively, prompt adherence is weak.

**Solution** (Ho & Salimans 2022): train the model both with and without the condition (drop the text 10% of the time). At sampling, run the model twice — once conditioned, once unconditioned — and extrapolate:

```
ε̂ = ε_uncond + s · (ε_cond − ε_uncond)
```

`s > 1` amplifies the conditioning direction. This dramatically improves prompt adherence, at the cost of diversity and (at high `s`) saturated, over-contrasted images.

CFG is why "guidance scale" is a slider in every image tool, and why high guidance produces the characteristic "AI look". It is also a 2× inference cost (two forward passes) — hence **distillation** techniques that fold the guidance into a single pass in 2025–26.

---

## 10.4 Latent diffusion and DiT

**Latent diffusion** (Rombach et al. 2022, the Stable Diffusion paper): run diffusion in the latent space of a pretrained autoencoder (e.g. 512×512×3 → 64×64×4), then decode once at the end. This reduces compute by ~50–64× with minimal quality loss, making high-resolution generation affordable on consumer GPUs. This is the single biggest efficiency idea in image generation.

**DiT — Diffusion Transformer** (Peebles & Xie, 2022–23): replace the U-Net backbone with a Transformer operating on latent patches, with the timestep and class/text conditioning injected via adaptive layer norm. Key result: **DiTs scale like LLMs** — FID improves predictably with depth/width and compute (they demonstrated this with the "scaling laws of diffusion transformers"). Every modern flagship image/video model is a DiT variant (Sora, DALL·E 3, Flux, SD3, Veo, Imagen).

**Patchification** is the bridge to video: an image is a sequence of patches; a video is a sequence of *spacetime* patches. This made video generation a sequence-modelling problem, which is exactly the field's home turf.

---

## 10.5 Flow matching and rectified flow: the current default

**Diffusion** learns to reverse a noise process; the sampling path is curved, so you need many steps.

**Flow matching** (Lipman et al. 2022; and rectified flow, Liu et al. 2022) learns a **velocity field** that transports a simple distribution (noise) to the data distribution. Train by regressing onto the straight-line velocity between a noise sample and a data sample:

```
x_t = (1−t)·x₀ + t·x₁                 # linear interpolation, t ∈ [0,1]
target velocity = x₁ − x₀
L = E ‖ v_θ(x_t, t) − (x₁ − x₀) ‖²
```

**Why it's better:** the target paths are straight, so sampling needs fewer integration steps (as few as 4–8, or 1 after distillation). The objective is simpler, and it's mathematically equivalent to diffusion with a different weighting and schedule — the widely-cited "Diffusion Meets Flow Matching" tutorial shows that Gaussian flow matching and diffusion are the *same* framework with different specifications.

**Rectified flow** adds a **reflow** step: use the trained model to pair noise with the samples it actually produces, then retrain on those pairs — straightening the trajectories further, eventually enabling one-step generation (InstaFlow).

**Consistency models / distillation** (2023–26) take a different route: train a model that maps any point on a trajectory directly to the endpoint, so one step suffices. Combined with adversarial losses (DMD2, 2024), you get 1–4 step models with quality close to the many-step original. This is why 2026 image generators feel instant.

---

## 10.6 Video generation

Video is images plus **temporal consistency**: the character's face must stay the same across frames, physics must be plausible, and motion must be coherent.

**Sora** (OpenAI, Feb 2024) established the framing: spacetime patches + a DiT trained on video at varying durations/resolutions, with a re-captioning pipeline to improve text-video alignment. Its headline demos (long shots, camera motion, consistent characters) were a genuine step change, and the "world simulator" framing (that a video model must implicitly learn 3D structure and physics) became a research programme.

**Veo 2/3** (Google DeepMind, 2024–25) added strong physics intuition and, in Veo 3, **synchronised audio**.

**The 2026 state:** high-quality short clips with audio, strong prompt adherence, limited long-horizon consistency; cost still high (video is expensive: 24 fps × 720p × N seconds is a lot of latent tokens). Evaluation is immature — FVD-style metrics correlate poorly with human judgment, and human preference arenas are the real benchmark.

---

## 10.7 World models: interactive generative environments

A **world model** learns to predict how an environment evolves in response to actions, so an agent can plan or be trained inside it.

- **Genie 1/2** (2023–24): generate playable 2D environments from images/video.
- **Genie 3** (Aug 2025): a general-purpose world model generating **interactive, navigable** 3D environments in real time at 720p/24fps from a text prompt, with **promptable world events** (change weather, add objects) and memory of the world's state. Notably, consistency is *emergent* from autoregressive generation over frames rather than enforced by an explicit 3D representation (no NeRF/Gaussian splatting).
- **Genie 4** (2026): built on Gemini family models, with Gemini Omni anchoring appearance/behaviour and other Gemini models providing reasoning for coherence; natural-language live world editing.
- **Project Genie** (Jan 2026): consumer prototype for creating/exploring/remixing worlds.

**Why it matters:** an unlimited curriculum of interactive environments is the missing ingredient for training embodied agents (`11`) and for evaluating long-horizon planning. It's also the most direct test of whether a model has learned *physics* rather than *appearance*.

**The honest caveat:** "simulates physics" claims are hard to verify and often mean "produces plausible-looking motion". Genie-style models still drift, forget, and violate object permanence over long horizons.

---

## 10.8 Speech and audio

- **Whisper** (2022): 680k hours of weakly-supervised multilingual ASR; robust, open, still a baseline. Encoder–decoder Transformer on mel spectrograms.
- **Neural codecs** (SoundStream, EnCodec, 2021–23): compress audio into discrete tokens at low bitrate — the enabling technology for audio language models.
- **AudioLM / MusicLM** (2022–23): treat audio as a token sequence, generate autoregressively at multiple levels of abstraction (semantic tokens + acoustic tokens).
- **TTS**: diffusion- and flow-matching-based vocoders plus LLM-style token prediction; 2025–26 systems produce near-indistinguishable speech with voice cloning from seconds of reference audio.
- **Full-duplex speech-to-speech**: **Moshi** (Kyutai, 2024) runs a single model that listens and speaks simultaneously with ~200ms latency; GPT-4o's voice mode is the commercial landmark. The architecture trick is two parallel token streams (user + assistant) in one Transformer.
- **Audio understanding** is now folded into omni models (`11`).

---

## 10.9 Evaluation: FID and its discontents

- **FID** (Fréchet Inception Distance): compares the distribution of Inception-v3 features of real vs generated images. Lower is better. Widely reported, widely gamed, and **poorly correlated with perceptual quality** at the frontier — it rewards matching dataset statistics, not prompt adherence.
- **CLIPScore**: image-text alignment via CLIP similarity. Misses fine detail and compositional structure.
- **Human preference / arenas**: the real signal for prompt adherence and aesthetics.
- **Video**: FVD (same idea as FID for video), plus subjective coherence/physics judgments. No good automated metric exists.
- **The 2026 consensus:** use arenas + targeted probes (counting, spatial relations, text rendering) + failure taxonomies. Aggregate metrics hide exactly the failures users care about.

---

## 10.10 Common misconceptions

- **"Diffusion is a different kind of model from a Transformer."** Modern diffusion models *are* Transformers (DiT). The difference is the *objective* (denoising/velocity regression) and the *sampling procedure*, not the architecture family.
- **"Flow matching replaced diffusion."** They're the same framework with different schedules/weightings; "flow matching" is the currently fashionable specification because it trains better and samples in fewer steps.
- **"Higher guidance = better."** Beyond a point it destroys diversity and produces over-saturated artefacts.
- **"Video models understand physics."** They produce plausible motion; object permanence, counting, and causal interventions fail under systematic testing.
- **"World models are just video generators."** The defining property is *action-conditioned, real-time, consistent* interaction — much harder than generating a clip.

---

## 10.11 If you remember five things

1. Diffusion = learn to denoise; the loss is "predict the noise you added".
2. CFG is the reason prompts work and the reason "AI look" exists.
3. Latent diffusion + DiT scaling is why image/video quality exploded in 2023–25.
4. Flow matching/rectified flow + distillation is why 2026 generators are fast.
5. World models are video models plus action-conditioning, real-time interaction, and consistency — and that's a big "plus".

---

## 10.12 Self-check

1. Write the DDPM training loss and explain why it needs no adversarial discriminator.
2. What does the guidance scale `s` trade off?
3. Why does latent diffusion save so much compute, and what does it cost?
4. Explain in one sentence why flow matching needs fewer sampling steps.
5. Why is FID a bad primary metric for a text-to-image product?

**Next:** [`11-vision-and-multimodal.md`](11-vision-and-multimodal.md)
