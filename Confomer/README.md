# Conformer — Encoder-Only Transformer (Image Classification)

## Overview

A from-scratch **Conformer encoder** (the same macaron-style block used in the ASR and BSR notebooks) repurposed as an **image classifier**, applied to the **Chest X-Ray Pneumonia** dataset (Kaggle: `paultimothymooney/chest-xray-pneumonia`), classifying scans into 3 classes: `NORMAL`, `BACTERIA`, `VIRUS`. Two additional baseline models — a plain fully-connected network (`FFN`) and a convolutional network (`CNN`) — are also implemented from scratch in the same notebook purely for comparison against the Conformer-Transformer.

```
Image → PatchEmbed → Pos → Conformer (Encoder) → image_tokens (B, T, 512)
Output → MLP Head → per-token class logits (B, T, 3)
```

---

## Data Pipeline

- **Dataset:** Chest X-Ray Pneumonia — RGB images, `(3, 224, 224)`, 3 classes (`NORMAL`, `BACTERIA`, `VIRUS`).
- **DataLoader:** `train_loader` / `test_loader`, `batch_size=31`.

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

## Architecture — Component by Component

### 1. `PatchEmbedding` — RGB Image Patchification

```python
class PatchEmbedding(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.proj = nn.Sequential(                       # input: (B, 3, H, W)
            nn.Conv2d(3,   64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(256,512, kernel_size=3, stride=2, padding=1),
        )
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        x = self.proj(x)                    # 4 stride-2 convs: 3→64→128→256→512 channels
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # (B, C, H, W) → (B, H*W, C): tokenize the spatial grid
        return self.norm(x)
```

Same convolutional "patchify" strategy used across every image/audio/signal notebook in this repository: a stack of 4 stride-2 `Conv2d + GELU` layers converts the raw RGB image into a flattened sequence of 512-dimensional tokens, one per remaining spatial location.

### 2. `ImagePositionalEmbedding` — Learned Positional Parameter

`nn.Parameter(torch.randn(1, max_len=1200, d_model=512))`, added to the patch tokens (learned, not sinusoidal).

### 3. Conformer Encoder — `ConformerConvModule`, `MultiHeadAttention`, `FeedForwardModule`, `ConformerBlock`, `ConformerEncoder`

Identical implementation to the ASR / BSR notebooks (see the **ASR-Transformer** README for a full line-by-line walkthrough of the convolution module's `pointwise conv → GLU → depthwise conv (kernel=31) → BatchNorm → SiLU → pointwise conv` pipeline and the macaron block's `0.5·FFN → MHSA → Conv → 0.5·FFN` structure). Here it is used purely as a feature extractor over image-patch tokens rather than audio/signal tokens; there is no decoder in this notebook — classification is done directly on the encoder output.

`ConformerEncoder(num_layers=8, d_model=512, num_heads=8)` — the encoder used in the final model is **8 layers deep** (one layer deeper than the ASR/BSR/OCR encoders in this repository).

### 4. `VocabHead` — Classification MLP

```python
class VocabHead(nn.Module):
    def __init__(self, embdim, vocab_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embdim, embdim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embdim, embdim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embdim, embdim // 2), nn.GELU(),
            nn.Linear(embdim // 2, vocab_size)
        )
    def forward(self, x):
        return self.net(x)     # (B, T, 512) → (B, T, 3)
```

Despite the name `vocab_size`, here it is the **number of image classes (3)**. The head is a 4-layer MLP (`512→512→512→256→3`) with GELU activations and dropout, applied to **every token position independently** — meaning the model outputs a full grid of per-patch class predictions rather than one label per image.

### 5. `Transfomer` — Full Classifier Assembly

```python
class Transfomer(nn.Module):
    def __init__(self, d_model=512, num_layers=8, num_heads=8, vocab_size=3, max_len=1100):
        super().__init__()
        self.patch = PatchEmbedding(d_model=d_model)
        self.pos   = ImagePositionalEmbedding(max_len=max_len, d_model=d_model)
        self.encoder = ConformerEncoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads)
        self.head = VocabHead(embdim=d_model, vocab_size=vocab_size)

    def forward(self, x):
        x = self.patch(x)        # (B, 3, 224, 224) → (B, T, 512)
        x = self.pos(x)            # + learned positional embedding
        x = self.encoder(x)          # 8× ConformerBlock
        x = self.head(x)               # → (B, T, 3): per-token class logits
        return x
```

**Design note (per-token classification):** because `VocabHead` is applied at every sequence position, the raw model output is `(B, T, num_classes)` rather than a single `(B, num_classes)` image-level prediction. The training loop expands each image's single ground-truth label across the `T` token positions and computes the classification loss over all of them jointly (i.e. every patch token is trained to predict the whole image's class) — this is a from-scratch design choice distinct from the usual `[CLS]`-token or global-average-pooling approaches to Transformer image classification.

---

## Baseline Comparison Models

Built in the same notebook to benchmark the Conformer-Transformer against non-attention architectures.

### `FFN` — Fully-Connected Baseline

```python
class FFN(nn.Module):
    def __init__(self, input_size=3*224*224, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 4096), nn.GELU(),
            nn.Linear(4096, 2048), nn.GELU(),
            nn.Linear(2048, 1024), nn.GELU(),
            nn.Linear(1024, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten the whole image: (B, 3, 224, 224) → (B, 150528)
        return self.net(x)           # 8-layer MLP funnel down to 3 classes
```

A deep, purely fully-connected network with no convolutional or attention inductive bias — every pixel is treated as an independent input feature.

### `CNN` — Convolutional Baseline

```python
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Conv(3→64)+BN+GELU, Conv(64→64)+GELU, MaxPool  → 224→112
            # Block 2: Conv(64→128)+BN+GELU, Conv(128→128)+GELU, MaxPool → 112→56
            # Block 3: Conv(128→256)+BN+GELU, Conv(256→256)+GELU, MaxPool → 56→28
            # Block 4: Conv(256→512)+BN+GELU, Conv(512→512)+GELU, MaxPool → 28→14
            # Block 5: Conv(512→512)+GELU, Conv(512→512)+GELU, MaxPool → 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # (B, 512*7*7)
            nn.Linear(512*7*7, 1024), nn.GELU(),
            nn.Linear(1024, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)   # (B, 3)
```

A 5-block VGG-style CNN: each block doubles the channel count (up to 512) via two `3×3` convolutions and halves the spatial resolution via `MaxPool2d(2)`, ending at a `7×7×512` feature map before a fully-connected classifier head.

---

## Training Pipeline

**Loop:** `train_model(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)`

- **Loss:** `CrossEntropyLoss()`.
- **Optimizer:** Adam, learning rate `3e-4` (used for all three models — Conformer-Transformer, FFN, and CNN — for a fair comparison).
- **Execution:** 1 epoch, batch size 31, for each of the three models.

---

## Comparison Plots

The notebook includes plotting utilities (`plot_model_metrics1`, `plot_model_metrics2`) built with seaborn/matplotlib to compare **Train/Test Accuracy, Precision, Recall, and F1** across the Conformer-Transformer, FFN, and CNN, rendered as grouped bar charts and polar/radar plots — giving a direct empirical comparison of the attention-based architecture against classical MLP and CNN baselines on the same task and data split.

---

## Model Saving

Weights are saved under a `models/` directory. Not stored in this repository due to size — regenerate by running the notebook end-to-end.
