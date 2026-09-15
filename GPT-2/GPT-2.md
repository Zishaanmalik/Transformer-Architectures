# GPT-2 — Generative Pretrained Transformer (Decoder-Only)

## Overview

This implementation builds GPT-2 from first principles using modular components:

- Token + Positional Embedding
- Masked Self-Attention (Causal Attention)
- Multi-Head Attention
- Residual + Layer Normalization
- Feed Forward Network
- Stacked Decoder Blocks
- Attention-Pooled Output Head

Training is performed using **Autoregressive Next-Token Prediction** on a conversational dataset of Human ↔ AI prompt-response pairs (`Dahoas/instruct-human-assistant-prompt`, loaded via `pandas.read_parquet` from the HuggingFace hub). Dataset download code is present in the notebook.

---

## Model Storage Notice

Due to file size constraints, model weights are not stored inside the repository.

| File | Size |
|---|---|
| `GPT.pth` | 368 MB |
| `GPT_outlayer.pth` | 114 MB |
| `MY_GPT.pth` | 482 MB |

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Vocabulary Size | 38,987 |
| Embedding Dimension | 768 |
| Number of Attention Heads | 12 |
| Head Dimension | 64 (768 / 12) |
| Feed Forward Hidden Dimension | 3,072 |
| Number of Decoder Blocks | 12 |
| Max Sequence Length | 128 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning Rate | 6e-4 |
| Training Objective | Causal Language Modeling |

---

## Parameter Statistics

### Base GPT (Decoder Stack Only)

| Metric | Value |
|---|---|
| Total Parameters | 96,109,824 |
| Trainable Parameters | 96,109,824 |
| Parameter Tensors | 145 |
| All Parameters Trainable | True |

### Full Model (GPT + Output Head)

| Metric | Value |
|---|---|
| Total Parameters | 126,091,596 |
| Trainable Parameters | 126,091,596 |
| Parameter Tensors | 149 |
| All Parameters Trainable | True |

---

## Training Data

**Dataset Type:** Human ↔ AI conversational prompt data

**Preprocessing Pipeline:**

1. Sentence normalization
2. Tokenization
3. Conversion to integer token IDs
4. Fixed-length sequence chunking (context window)
5. Creation of input-target shifted pairs

Example transformation:

```
Input : [t1, t2, t3, t4]              Target : [t5]
Input : [t1, t2, t3, t4, t5]          Target : [t6]
Input : [t1, t2, t3, t4, t5, t6]      Target : [t7]
```

The model predicts the next token at every position. Dataset is wrapped by `CustomDataset(x, y)` and loaded via `DataLoader(batch_size=4, shuffle=True)`.

---

## Architecture Flow

### 1. Embedding + Positional Encoding

**Module:** `embpos(vocab_size, embdim)`

- Embedding dimension: 768, positional embedding: sinusoidal (precomputed)
- Input: `(B, T)` → Output: `(B, T, 768)`

### 2. Masked Self-Attention (Causal Attention)

**Class:** `MaskedSelfAttention(embdim)`

- `Wq`, `Wk`, `Wv`: `Linear(768 → 768)`, scale = `sqrt(64)`
- Causal mask via `torch.tril`, masked positions set to `-inf` before softmax.

```
scores = QK^T / sqrt(D)
scores = scores.masked_fill(mask == 0, -inf)
weights = softmax(scores)
output = weights @ V
```

### 3. Multi-Head Attention

**Class:** `MultiHeadAttention(embdim=768, num_heads=12)`

- Combined `qkv = Linear(768 → 3*768)`, split into 12 heads of dim 64
- Causal mask applied per head, outputs concatenated, `out = Linear(768 → 768)`
- Input/Output: `(B, T, 768)`

### 4. Residual + Layer Normalization

**Class:** `AddResidual_LayerNorm(768)`

```
x + sublayer(LayerNorm(x))
```

### 5. Feed Forward Network

**Class:** `FeedForward(768, 3072)`

```
Linear(768 → 3072) → GELU → Linear(3072 → 768)
```

### 6. Decoder Block

**Class:** `DecoderBlock(embdim=768, num_heads=12, ff_hidden_dim=3072)`

```
x = attn_norm(x, mha)
x = ff_norm(x, ff)
```

### 7. Stacked Decoder — GPT-2 Core

**Class:** `GPT2(vocab_size, embdim=768, num_heads=12, ff_hidden_dim=2048, num_layers=12)`

```
x = embedding(x)
for layer in layers:
    x = layer(x)
return x
```

Input: `(B, T)` → Output: `(B, T, 768)`

### 8. Output Head

**Class:** `AttentionPooling(embdim)` + `GPT2OutputHead(embdim, vocab_size)`

- `AttentionPooling`: learns a per-token score, softmax-weights and sums token representations into a single pooled vector `(B, D)`.
- `GPT2OutputHead`: `fc = Linear(768 → vocab_size)` applied to the pooled vector.

```
pooled = AttentionPooling(hidden)   # (B, 768)
logits = fc(pooled)                 # (B, vocab_size)
```

### 9. Final Model Wrapper

**Class:** `MY_GPT`

```
x = model(x)        # GPT2 decoder stack
x = out_layer(x)     # AttentionPooling + Linear head
return x
```

---

## Training Pipeline

- **Loss Function:** `CrossEntropyLoss()`
- **Optimizer:** Adam, Learning Rate = 6e-4
- **Execution:** Epochs = 1, Batch Size = 4

---

## Model Saving

Saved under `models/` directory:

- `GPT.pth`
- `GPT_outlayer.pth`
- `MY_GPT.pth`
