# Conformer — Encoder-Only Transformer (Image Classification)

## Overview

This implementation builds a Conformer encoder from first principles and applies it to **image classification** on the **Chest X-Ray Pneumonia** dataset (Kaggle: `paultimothymooney/chest-xray-pneumonia`), classifying scans into `NORMAL`, `BACTERIA`, and `VIRUS`.

- Image Patch Embedding (Conv2d stack) + Learned Positional Embedding
- Conformer Encoder (macaron feed-forward + multi-head self-attention + convolution module)
- MLP Classification Head
- Two baseline models (`FFN`, `CNN`) implemented alongside the Conformer-Transformer for comparison

```
Image → PatchEmbed → Pos → Conformer (Encoder) → image_tokens (B, T, 512)
Output → Linear → class logits
```

---

## Data Pipeline

- **Dataset:** Chest X-Ray Pneumonia (3-class: NORMAL, BACTERIA, VIRUS)
- **DataLoader:** `train_loader` / `test_loader`, `batch_size=31`
- Images are RGB, `(3, 224, 224)`

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Model Dimension (`d_model`) | 512 |
| Number of Attention Heads | 8 |
| Encoder Layers (Conformer) | 8 |
| Max Sequence Length | 1,100 |
| Number of Classes | 3 |
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Batch Size | 31 |
| Epochs | 1 |

---

## Architecture Flow

### 1. Patch Embedding

**Class:** `PatchEmbedding(d_model=512)`

4-layer `Conv2d` + `GELU` stack, stride 2 each, on RGB input:

```
(B, 3, H, W) → Conv2d(3→64) → Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→512)
```

Followed by `LayerNorm(512)`.

### 2. Positional Embedding

**Class:** `ImagePositionalEmbedding(max_len=1200, d_model=512)`

Learned positional parameter added to patch embeddings, with dropout.

### 3. Conformer Encoder

**Classes:** `ConformerConvModule`, `MultiHeadAttention`, `FeedForwardModule`, `ConformerBlock`, `ConformerEncoder`

Each `ConformerBlock` (macaron structure):

```
x = x + 0.5 * ff1(x)
x = x + mha(x)
x = x + conv(x)
x = x + 0.5 * ff2(x)
x = norm(x)
```

`ConformerEncoder(num_layers=8, d_model=512, num_heads=8)` stacks the encoder blocks used in the final model.

### 4. Vocabulary / Classification Head

**Class:** `VocabHead(embdim, vocab_size)`

```
Linear(512 → 512) → GELU → Dropout
Linear(512 → 512) → GELU → Dropout
Linear(512 → 256) → GELU
Linear(256 → num_classes)
```

### 5. Full Model

**Class:** `Transfomer(d_model=512, num_layers=8, num_heads=8, vocab_size=3, max_len=1100)`

```
x = patch(x)      # (B, T, 512)
x = pos(x)         # (B, T, 512)
x = encoder(x)      # (B, T, 512)
logits = head(x)    # (B, T, num_classes)
```

Per-token class logits are produced; the training loop expands the single image label across the token dimension and computes loss over all tokens.

---

## Baseline Comparison Models

### Feed Forward Neural Network

**Class:** `FFN(input_size=3*224*224, num_classes=3)`

A flattened, fully-connected MLP: `150528 → 4096 → 2048 → 1024 → 512 → 256 → 128 → 64 → 3`, `GELU` activations throughout.

### Convolutional Neural Network

**Class:** `CNN(num_classes=3)`

A standard 3-block Conv+BatchNorm+GELU+MaxPool stack (`64 → 128 → 256` channels) feeding into a classification head.

Both baselines are trained with `Adam (lr=3e-4)` and `CrossEntropyLoss`, and their Accuracy / Precision / Recall / F1 metrics are compared against the Conformer-Transformer using bar-chart and radar-chart plotting utilities (`plot_model_metrics1`, `plot_model_metrics2`).

---

## Training Pipeline

**Loop:** `train_model(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)`

- **Loss Function:** `CrossEntropyLoss()`
- **Optimizer:** Adam, Learning Rate = 3e-4
- **Execution:** Epochs = 1, Batch Size = 31

---

## Comparison Plots

The notebook includes seaborn/matplotlib utilities to compare Train/Test Accuracy, Precision, Recall, and F1 across the Conformer-Transformer, FFN, and CNN models via grouped bar charts and polar/radar plots.

---

## Model Saving

Weights are saved under a `models/` directory (exact filenames as configured in the final notebook cell). Not stored in the repository due to size — regenerate by running the notebook end-to-end.
