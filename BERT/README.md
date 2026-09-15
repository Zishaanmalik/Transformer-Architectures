# BERT — Bidirectional Encoder Transformer

## Overview

This implementation builds BERT from first principles using modular components:

- Token + Positional Embedding
- Self-Attention
- Multi-Head Attention
- Residual + Layer Normalization
- Feed Forward Network
- Stacked Encoder Blocks
- Masked Language Modeling Head

Training is performed using **Masked Language Modeling (MLM)** on the **Complete Works of Shakespeare**.

---

## Model Storage Notice

Due to file size constraints, model weights are not stored inside the repository.

| File | Size |
|---|---|
| `Bert.pth` | 1.65 GB |
| `mask.pth` | 115 MB |
| `mybert.pth` | 1.76 GB |

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Vocabulary Size | 27,964 |
| Embedding Dimension | 512 |
| Number of Attention Heads | 8 |
| Head Dimension | 64 (512 / 8) |
| Feed Forward Hidden Dimension | 2048 |
| Number of Encoder Blocks | 6 |
| MLM Hidden Dimension | 1024 |
| MLM Probability | 0.15 |
| Mask Token ID | 27963 |
| Special Token IDs | [0, 2, 3] |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Batch Size | 100 |
| Epochs | 5 |

---

## Parameter Statistics

### Base BERT (Encoder Stack Only)

| Metric | Value |
|---|---|
| Total Parameters | 444,416,000 |
| Trainable Parameters | 444,416,000 |
| Parameter Tensors | 385 |
| All Parameters Trainable | True |

### Full Model (BERT + MLM Head)

| Metric | Value |
|---|---|
| Total Parameters | 474,654,012 |
| Trainable Parameters | 474,654,012 |
| Parameter Tensors | 391 |
| All Parameters Trainable | True |

---

## Training Data

- Dataset: Complete Works of Shakespeare
- Objective: Masked Language Modeling
- Masking Probability: 15%
- Special tokens excluded from masking: `[0, 2, 3]`
- Mask token id: `27963`

---

## Architecture Flow

### 1. Embedding + Positional Encoding

**Module:** `embpos(vocab_size, embdim)`

- Vocabulary size: 27,964
- Embedding dimension: 512
- Input: `(B, T)` → Output: `(B, T, 512)`

Token embeddings are generated and sinusoidal positional encoding is added before entering the encoder stack.

### 2. Self-Attention

**Class:** `SelfAttention(embdim)`

- `Wq`, `Wk`, `Wv`: `Linear(512 → 512)`
- Scale: `sqrt(embdim)`

```
scores = QK^T / sqrt(D)
weights = softmax(scores)
output = weights @ V
```

Input/Output: `(B, T, 64)` per head.

### 3. Multi-Head Attention

**Class:** `MultiHeadAttention(embdim=512, num_heads=8)`

- Heads: 8, Head dimension: 64
- Output projection: `Linear(512 → 512)`
- Input reshaped to `(B, 8, T, 64)`, `SelfAttention` applied per head, outputs concatenated and projected.
- Input/Output: `(B, T, 512)`

### 4. Residual + Layer Normalization

**Class:** `AddResidual_LayerNorm(512)`

```
x + sublayer(LayerNorm(x))
```

Stabilizes training, preserves gradient flow, normalizes feature distribution.

### 5. Feed Forward Network

**Class:** `FeedForward(512, 2048)`

```
Linear(512 → 2048) → ReLU → Linear(2048 → 512)
```

### 6. Encoder Block

**Class:** `EncoderBlock(embdim=512, num_heads=8, ff_hidden_dim=2048)`

```
x = attn_norm(x, mha)
x = ff_norm(x, ff)
```

### 7. Stacked Encoder — BERT Core

**Class:** `BERT(vocab_size, embdim=512, num_heads=8, ff_hidden_dim=2048, num_layers=6)`

```
x = embedding(x)
for layer in layers:
    x = layer(x)
return x
```

Input: `(B, T)` → Output: `(B, T, 512)`

### 8. Masked Language Modeling Head

**Class:** `MLMHead(512, vocab_size, hidden_dim=1024)`

```
Linear(512 → 1024) → ReLU → Linear(1024 → 1024) → ReLU → Linear(1024 → 27964)
```

Input: `(B, T, 512)` → Output: `(B, T, 27964)` (logits for masked token prediction)

### 9. Final Model Wrapper

**Class:** `MY_BERT`

```
x = BERT_(x)
x = MLM_head(x)
return x
```

---

## Masking Strategy

**Dataset Class:** `custom_dataset`

- `mlm_prob = 0.15`
- `mask_token_id = 27963`
- `special_ids = [0, 2, 3]`

15% of eligible tokens are masked; special tokens are excluded; labels for non-masked tokens are set to `-100`; loss is computed only on masked tokens.

---

## Training Pipeline

- **Loss Function:** `CrossEntropyLoss(ignore_index=-100)`
- **Accuracy Metric:** `mlm_accuracy()`
- **Optimizer:** Adam, Learning Rate = 1e-5
- **Execution:** Epochs = 5, Batch Size = 100

---

## Model Saving

Saved under `models/` directory:

- `Bert.pth`
- `mask.pth`
- `mybert.pth`
