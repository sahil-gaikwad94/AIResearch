# 11 — Vision and Multimodal Models: Seeing, and Acting

*Read time: 60 minutes. Prerequisites: `01`, `02`, `10`.*

---

## 11.1 The road from CNNs to ViT

**CNNs** (AlexNet 2012 → ResNet 2015 → EfficientNet 2019) dominated vision for a decade. Their inductive bias — local receptive fields, weight sharing, translation equivariance — is exactly right for images, and it made them sample-efficient.

**ViT — Vision Transformer** (Dosovitskiy et al. 2020) threw that away. An image is cut into 16×16 **patches**, each patch is linearly projected to a vector (like a token), position embeddings are added, and a standard Transformer encoder processes the sequence.

The surprising result: **ViTs underperform CNNs when trained on ImageNet alone, but outperform them when trained on enough data** (JFT-300M). With enough data, the architecture's inductive bias stops mattering and the Transformer's flexibility wins. This is the vision analogue of the scaling-law story, and it's the reason "just add data" became the field's reflex.

Practical notes:
- **Patch size** trades sequence length for resolution. A 224×224 image at patch 16 → 196 tokens. At 1024×1024 → 4,096 tokens (expensive).
- **DeiT** (2021) showed ViTs train with distillation and strong augmentation on ImageNet alone.
- **Hybrids (ConvNeXt, 2022)** reintroduced convolutional structure and matched ViTs — evidence that the pure-architecture debate was less important than the training recipe.

---

## 11.2 CLIP: the idea that unlocked multimodal AI

**CLIP** (Radford et al. 2021) trained two encoders — one for images, one for text — on 400M image–text pairs scraped from the web, with a **contrastive objective**: for a batch of N pairs, maximise the cosine similarity of the N correct pairs and minimise it for the N²−N incorrect ones.

```
L = symmetric cross-entropy over the N×N similarity matrix
```

**What you get:** a shared embedding space where "a photo of a dog" and an actual photo of a dog are close. Consequences:

- **Zero-shot classification**: embed the class names as text, pick the nearest image embedding. No task-specific training.
- **The bridge for VLMs**: a vision encoder whose output is *already aligned to language* can be plugged into an LLM.
- **The reward model for diffusion**: CLIPScore, and CLIP-guided generation.

**SigLIP** (Zhai et al. 2023) replaced the softmax contrastive loss with a **sigmoid** loss applied pairwise, removing the need for a global normalisation across the batch. This made training far more efficient at small batch sizes and became the standard vision encoder in 2024–26 VLMs (PaliGemma, Idefics, Qwen-VL, Llama 4's vision tower, and most open recipes).

**Other self-supervised approaches worth knowing:**
- **DINO / DINOv2** (2021/2023): self-distillation with no labels — a student matches a teacher's output across augmented views. Produces features that are excellent for dense tasks (segmentation, depth) and remarkably semantic. DINOv2 features are still a strong default for "I need good visual features without labels".
- **MAE** (2021): masked autoencoding — mask 75% of patches, reconstruct. Cheap, scalable pretraining.
- **SimCLR / MoCo**: earlier contrastive methods; historically important.

---

## 11.3 SAM: promptable segmentation

**Segment Anything (SAM)** (Kirillov et al. 2023): train a model to segment *anything* given a prompt (a point, a box, a mask, or text), on a 1.1B-mask dataset generated with a model-in-the-loop annotation pipeline. The result is a task-agnostic, zero-shot segmentation model — a "foundation model" for masks.

**SAM 2** (2024) extended it to **video**: a memory attention module carries object identity across frames, enabling promptable segmentation and tracking in video, at real-time speeds.

Why it matters beyond segmentation: it's the clearest example of **promptable foundation models** outside language, and it's the backbone of a lot of 2025–26 work in medical imaging, robotics perception, and data annotation.

---

## 11.4 How a VLM is built (the three recipes)

A **VLM (vision-language model)** takes image + text and produces text. There are three architectures, in increasing order of integration:

**(1) Late fusion / adapters (the workhorse).** A pretrained vision encoder (SigLIP, DINOv2) → a **projector** (MLP, or a Q-Former that compresses to a fixed number of query tokens) → prepend those tokens to the LLM's input. The LLM is (partly) frozen at first, then jointly fine-tuned.

- **LLaVA** (2023) is the canonical recipe: CLIP ViT + a 2-layer MLP + Vicuna, trained on 595k distilled image-caption pairs then 158k instruction-following multimodal data. Astonishingly cheap, and it established the open-source VLM playbook.
- **Flamingo** (2022) used cross-attention layers interleaved in a frozen LLM plus a Perceiver resampler — elegant, and the ancestor of the "gated cross-attention" design.
- **Qwen-VL, InternVL, PaliGemma, Idefics** all follow this family.

**(2) Early fusion (native).** Images are tokenized (by a VQ tokenizer or a continuous encoder) and interleaved with text tokens in a *single* sequence, trained from the start on mixed data. **Chameleon** (Meta, 2024) is the landmark: a token-based mixed-modal model that understands *and generates* images and text in arbitrary order. Its key engineering finding is worth remembering: early-fusion training is unstable and prone to learning degenerate unconditional priors when modality mixtures are imbalanced — it required careful architecture (QK-norm), normalisation, and data balancing.

**(3) Hybrid.** An LLM backbone plus a separate diffusion decoder for image generation (e.g. Transfusion-style, or GPT-4o's architecture as publicly described). Understanding is autoregressive; generation is diffusion.

---

## 11.5 Native omni models (2025–26)

The 2026 trend is **any-to-any**: one model that takes text, image, audio, video in and produces text, image, audio, video out.

- **Gemini** was "natively multimodal from the start" (2023–24), and **Gemini Omni** (May 2026) extends this to generation — combining image, audio, video and text inputs to generate and conversationally edit video.
- **Qwen2.5-Omni / Qwen3-Omni**, **Ming-Omni**, **MiniCPM-o**, **GLM-4-Voice**, **Moshi** (speech), **ERNIE 5.0** (2026): unified autoregressive frameworks with modality-specific tokenizers (visual tokenizer, audio codec tokenizer) feeding a shared backbone, with next-frame-and-scale prediction for vision and next-codec prediction for audio.
- **Architectural taxonomy** (2026 roadmap papers): **late fusion** (separate encoders + frozen LLM) → **mid fusion** (features injected into a joint backbone) → **early fusion / "born-native"** (everything is tokens from the start, with separate de-tokenizers at the output).

**Why native matters:** separate encoders create information bottlenecks (a caption-like summary of the image) and can't represent things language can't describe. Early fusion can, in principle, model the joint distribution. The costs are training instability, data mixture sensitivity, and expensive tokenizers.

---

## 11.6 Vision-language-action (VLA) models: robots

The most consequential multimodal application, and the one with the most honest gap between demo and deployment.

**The recipe:** take a pretrained VLM, add an **action output head**, fine-tune on robot episodes `(camera frames, proprioception, instruction, actions, outcome)`.

**Lineage:**
- **RT-1** (2022): Transformer over images + text → discretized actions, trained on 130k episodes.
- **RT-2** (2023): the key idea — **express actions as text tokens** ("1 128 91 241 …") so a web-scale VLM can be co-fine-tuned on robot data and inherit internet-scale semantics. This established the VLM-backbone standard.
- **RT-X / Open X-Embodiment** (2023): cross-robot datasets; training on many embodiments generalises better.
- **OpenVLA** (2024): 7B, Llama-2 + fused DINOv2/SigLIP vision, 970k real robot demos; beat the 55B RT-2-X by 16.5 points with 7× fewer parameters. Open weights → the research baseline.
- **π0** (Physical Intelligence, Oct 2024): VLM backbone + **flow-matching action head** for continuous, high-frequency control; trained across 7 robot configurations + Open X-Embodiment + DROID; laundry folding, table cleaning, box assembly. This "VLM + flow head" design is now the standard.
- **π0.5** (Apr 2025): **co-training on heterogeneous data** — other robots, web data, high-level semantic subtask prediction — enabling open-world generalisation (cleaning a kitchen it has never seen). The finding: knowledge transfer from web data and semantic subtasks is *essential*, not optional.
- **GR00T N1** (NVIDIA, 2025), **Gemini Robotics 1.5 / ER 1.5** (2025), **Helix** (Figure), **SmolVLA** (open, small): the current fleet.

**Action tokenization** is the central design question (there's a whole 2025 survey on it): language description, code, affordance, trajectory, goal state, latent, raw action, or reasoning. Discrete tokens are simple but lossy for smooth control; flow-matching heads are smooth but need a continuous decoder.

**Why robots are still hard:**
- **Data**: no internet-scale action data. Teleoperation is expensive; sim-to-real has a gap.
- **Reliability**: 95% success per grasp is a demo; 99.9% is a product.
- **Long-horizon composition**: a 20-minute task is a chain of hundreds of decisions (`07` §7.8 on compounding).
- **Safety**: a wrong action has physical consequences.

---

## 11.7 Document, chart, and video understanding

Worth calling out because these are where multimodal models actually earn money:

- **Documents**: OCR-free VLMs now beat OCR-pipeline approaches on many benchmarks, but still struggle with dense tables, handwriting, and multi-page reasoning. Resolution matters enormously — tiling a page into multiple crops and feeding them as separate images is the standard trick.
- **Charts**: requires reading axes and comparing values, not just describing. A classic compositional-reasoning failure case.
- **Video**: frame sampling + temporal aggregation is the pragmatic approach; native video tokens (`10`) are better but expensive. Long-video QA is far from solved.

---

## 11.8 Common misconceptions

- **"VLMs see like we do."** They see a grid of patches through a language-aligned encoder; spatial precision, counting, and fine text are known weak spots.
- **"Early fusion is strictly better."** It's more elegant and more expressive in principle, but harder to train and more data-hungry. Late fusion still wins on many benchmarks per dollar.
- **"Robotics is solved by scaling VLAs."** Data, reliability, and safety are the bottlenecks, not model capacity.
- **"A vision encoder is just a feature extractor."** Its training objective (contrastive vs self-distillation vs captioning) strongly determines which downstream tasks it's good at — SigLIP features are great for retrieval, DINOv2 for dense prediction.

---

## 11.9 If you remember five things

1. ViTs won once data got big; inductive bias matters less at scale.
2. CLIP's contrastive image-text space is the bridge that made VLMs possible; SigLIP is the modern default.
3. The standard VLM is encoder → projector → LLM tokens; LLaVA is the canonical recipe.
4. Native omni models are the 2025–26 direction: one backbone, modality tokenizers, any-to-any.
5. VLAs are VLM + action head (flow matching is the current best); the bottleneck is data and reliability, not architecture.

---

## 11.10 Self-check

1. Why did ViTs need JFT-300M-scale data to beat CNNs?
2. Write the CLIP contrastive loss conceptually and explain what a "negative" is.
3. What does a projector do in a VLM, and why does a Q-Former-style resampler help?
4. Why did RT-2's "actions as text" trick matter?
5. What does a flow-matching action head give you that discrete action tokens don't?

**Next:** [`12-interpretability.md`](12-interpretability.md)
