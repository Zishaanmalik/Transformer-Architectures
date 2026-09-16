# Transformer & Diffusion Architectures From Scratch

This repository contains complete, ground-up implementations of Transformer (and one diffusion) architectures, built directly in Python using raw PyTorch tensor operations. Every computational stage — embeddings, attention projections, scaling, masking, normalization, residual pathways, convolutional sub-modules, feed-forward transformations, stacking logic, and output heads — is implemented explicitly, with full control over tensor dimensions and parameter allocation.

**No external transformer/diffusion libraries** (e.g. HuggingFace `transformers`, `diffusers`) are used for any model internals — only `torch`, `torch.nn`, and `torch.nn.functional` primitives. Each notebook is fully self-contained: data pipeline, architecture definition, a forward-pass shape sanity check, a training loop, and (where applicable) inference/generation code.

---

## Repository Structure

Each notebook lives in its own folder, with its own dedicated README (in that folder) describing its architecture, data pipeline, and training configuration down to the exact `nn.Module` and tensor-shape level.

| Folder | Notebook | Architecture Type | Task |
|---|---|---|---|
| `BERT/` | `BERT.ipynb` | Encoder-only | Masked Language Modeling |
| `GPT-2/` | `GPT-2.ipynb` | Decoder-only | Causal Language Modeling |
| `Conformer/` | `Confomer.ipynb` | Encoder-only (Conformer) | Image Classification (Chest X-Ray) |
| `OCR-VIT/` | `OCR-VIT.ipynb` | Encoder-Decoder (ViT + Decoder) | Optical Character Recognition |
| `ASR-Transformer/` | `ASR-Transfomer.ipynb` | Encoder-Decoder (Conformer + Decoder) | Automatic Speech Recognition |
| `BSR-VIT/` | `BSR-VIT.ipynb` | Encoder-Decoder (Conformer + Decoder) | Brain Signal Recognition (EEG → Text) |
| `Diffusion/` | `Deffision.ipynb` | Transformer Text Encoder + UNet (Cross-Attention Diffusion) | Text-Conditioned Image Generation |

> Each subfolder's README documents that notebook's exact architecture — every module explained individually with its real code, its math, and its exact input/output tensor shapes — derived directly from the implementation, not summarized at a high level.

---

## Implementation Characteristics

- All modules implemented using `torch.nn.Module`
- No external transformer/diffusion libraries used — attention, masking, normalization, convolutional sub-blocks, and feed-forward layers are all hand-written
- Explicit tensor-level control over Q/K/V projections, multi-head reshaping, scaling factors, and masks
- Modular encoder / decoder / UNet stacks that can be instantiated, extended, or analyzed independently
- Custom training loops, loss functions, noise schedulers, and accuracy metrics per notebook
- Every subfolder README documents real design choices and quirks found in the code as written (including non-standard or experimental implementation details), not an idealized or simplified version of the architecture

---

## Architecture Family Overview

### Encoder-Only Models
- **BERT** — bidirectional self-attention encoder trained with Masked Language Modeling on the Complete Works of Shakespeare. Notably uses **per-head independent attention weights** (each of the 8 heads is its own separate `SelfAttention` module) rather than a single shared projection split across heads.
- **Conformer (Image Classifier)** — a Conformer encoder (macaron feed-forward + multi-head self-attention + convolution module) repurposed as a per-patch image classifier over chest X-ray scans, benchmarked in the same notebook against from-scratch FFN and CNN baselines.

### Decoder-Only Models
- **GPT-2** — causal, masked self-attention decoder trained with next-token prediction on Human ↔ AI conversational data. Uses an **attention-pooling output head** that collapses the whole sequence into one vector before the final vocabulary projection, rather than a standard per-position LM head.

### Encoder-Decoder Models
- **OCR-VIT** — a genuine Vision Transformer (ViT) image encoder (plain self-attention, no convolution module) + causal text decoder with cross-attention, trained to transcribe text from images.
- **ASR-Transformer** — Conformer audio encoder (mel-spectrogram patchified via strided convolutions) + causal text decoder with cross-attention, trained on LibriSpeech for speech-to-text.
- **BSR-VIT** — Conformer encoder over 16-channel EEG/brain-signal spectrogram-like input + causal text decoder with cross-attention, trained to map brain signals to character-level text.

### Diffusion Model
- **Diffusion-Transformer** — a Transformer text encoder (8-layer bidirectional self-attention over captions) conditions a convolutional **UNet** (ResBlocks + cross-attention + skip connections, time-conditioned via sinusoidal timestep embeddings) trained to predict added Gaussian noise (DDPM objective) on the Flickr30k dataset, generating images from text through iterative denoising at inference time.

---

## Common Building Blocks

Across the notebooks, the following components are reimplemented from scratch and reused with task-specific configuration. Design differs subtly between notebooks — see each subfolder README for the exact code and callouts of where a notebook deviates from the others.

| Component | Purpose | Notable variants across notebooks |
|---|---|---|
| `embpos` / `TextEmbedding` / `TextPositionalEmbedding` | Token embedding + positional encoding | BERT/GPT-2 don't scale by `sqrt(embdim)`; ASR/BSR do |
| `PatchEmbedding` | Conv2d-based patchification of images / spectrograms / signals | Input channels vary: 1 (audio), 3 (image/OCR), 16 (EEG) |
| `SelfAttention` / `MultiHeadAttention` | Scaled dot-product self-attention | BERT: per-head independent weights. GPT-2/ASR/Conformer: fused QKV. OCR-VIT/Diffusion text encoder: separate Q/K/V projections |
| `MaskedSelfAttention` / `MaskedMultiHeadAttention` | Causal (triangular-masked) self-attention for decoders | GPT-2, ASR, BSR, OCR-VIT decoders |
| `CrossAttention` | Query from one modality, key/value from another | Text↔Audio (ASR), Text↔Signal (BSR), Text↔Image (OCR-VIT), Image-pixels↔Text (Diffusion UNet, via `nn.MultiheadAttention`) |
| `ConformerConvModule` | Depthwise convolution + GLU + BatchNorm + SiLU, Conformer-style | ASR, BSR, Conformer notebooks |
| `ResBlock` | Time-conditioned convolutional residual block | Diffusion UNet only |
| `FeedForward` / `FeedForwardModule` | Position-wise MLP block | Activation varies: ReLU (BERT), GELU (everywhere else) |
| `AddResidual_LayerNorm` | Pre-norm residual wrapper | BERT, GPT-2 |
| `EncoderBlock` / `DecoderBlock` / `ConformerBlock` / `DownBlock` / `UpBlock` / `MidBlock` | Stacked architecture-specific layers | — |
| `VocabHead` / `MLMHead` / `LMHead` / `GPT2OutputHead` | Task-specific output projection | Per-token (most notebooks) vs. sequence-pooled (GPT-2) vs. per-class (Conformer classifier) |
| `SimpleNoiseScheduler` | DDPM linear beta schedule + closed-form forward diffusion | Diffusion notebook only |

---

## Requirements

- Python 3.x
- PyTorch
- `datasets` (HuggingFace, for dataset loading only — not model code)
- `librosa` (ASR notebook — audio feature extraction)
- `torchvision` (Diffusion notebook — image transforms)
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `Pillow`

---

## Model Storage Notice

Due to file size constraints, trained model weights (`.pth` files) are **not stored inside this repository**. Each subfolder README lists the expected saved file names and, where recorded, their sizes. Weights can be regenerated by running the corresponding notebook end-to-end.

---

## Notes

- All hyperparameters, tensor shape transitions, masking strategies, and parameter totals documented in each subfolder README are derived directly from the corresponding notebook's actual code — including non-obvious or non-standard design choices (e.g. BERT's per-head attention weights, GPT-2's pooled output head, unused/dead-code layers left in place) — so that every architectural decision is fully transparent and traceable to the source.
- Notebooks are exploratory/research artifacts: they include data preprocessing, architecture definition, a working forward-pass sanity check, and a training loop, in that order, with some cells showing iterative fixes (e.g. two versions of a class or function kept side by side) that are called out explicitly in the relevant subfolder README rather than silently omitted.
