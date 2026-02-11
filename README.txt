# Transformer Architectures From Scratch

This work presents a complete, ground-up implementation of Transformer architectures developed directly in Python using PyTorch tensor operations. The encoder-only and decoder-only systems are constructed by explicitly defining every computational stage attention projections, scaling, masking, normalization, residual pathways, feed-forward transformations, stacking logic, and output layers with full control over tensor dimensions and parameter allocation.

The implementations include:

- **BERT - Encoder-Only Transformer**
- **GPT-2 - Decoder-Only Transformer**

Both architectures are written as structured class-based modules where each layer corresponds to a clearly defined mathematical operation. The attention mechanism is implemented through explicit query, key, and value projections; scaling factors are applied manually; masks are constructed at tensor level; and multi-head aggregation is performed through controlled reshaping and concatenation. Feed-forward layers, normalization blocks, and residual connections are integrated in the exact order defined within the notebooks.

The implementation maintains transparency over:

- Embedding dimensions and projection sizes  
- Head configuration and attention scaling  
- Encoder and decoder depth  
- Parameter counts and tensor statistics  
- Training objectives and loss computation  

Each component operates as an independent module that can be instantiated, extended, or analyzed without altering the structural integrity of the full model. This organization preserves architectural clarity while allowing direct inspection of how Transformer computations propagate from input tokens to final logits.

All hyperparameters, tensor transitions, masking strategies, and parameter totals are defined within the notebooks themselves, ensuring that the documented work reflects the precise structure and behavior of the implemented models.


# Implementation Characteristics

- All modules implemented using `torch.nn.Module`
- No external transformer libraries used
- All parameters trainable
- Explicit tensor-level control
- Modular encoder stack
- Fully custom MLM pipeline

---

# BERT - Bidirectional Encoder Transformer

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

The training setup, masking logic, architecture, and parameter statistics are derived directly from the implementation.

---

# Model Storage Notice

Due to file size constraints:

- Model weights are not stored inside the repository.

Saved files:
- `Bert.pth`    model size: **1.65 GB**
- `mask.pth`    model size: **115 MB**
- `mybert.pth`  model size: **1.76 GB** 

---

# BERT - Architectural Configuration

## Core Hyperparameters

| Parameter | Value |
|------------|--------|
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

# Parameter Statistics

## Base BERT (Encoder Stack Only)

| Metric | Value |
|--------|--------|
| Total Parameters | 444,416,000 |
| Trainable Parameters | 444,416,000 |
| Parameter Tensors | 385 |
| All Parameters Trainable | True |

---

## Full Model (BERT + MLM Head)

| Metric | Value |
|--------|--------|
| Total Parameters | 474,654,012 |
| Trainable Parameters | 474,654,012 |
| Parameter Tensors | 391 |
| All Parameters Trainable | True |

---

# Training Data

- Dataset: **Complete Works of Shakespeare**
- Objective: **Masked Language Modeling**
- Masking Probability: **15%**
- Special tokens excluded from masking: `[0, 2, 3]`
- Mask token id: `27963`

---

# Architecture Flow

---

## 1. Embedding + Positional Encoding

### Module
`embpos(vocab_size, embdim)`

### Configuration
- Vocabulary size: 27,964
- Embedding dimension: 512

### Input
(B, T)


### Output
(B, T, 512)


Token embeddings are generated and positional encoding is added before entering the encoder stack.

---

## 2. Self-Attention

### Class
`SelfAttention(embdim)`

### Internal Projections
- Wq: Linear(512 → 512)
- Wk: Linear(512 → 512)
- Wv: Linear(512 → 512)

### Scaling
scale = sqrt(embdim)


### Computation
scores = QK^T / sqrt(D)
weights = softmax(scores)
output = weightsV


### Input
(B, T, 64)


### Output
(B, T, 64)


---

## 3. Multi-Head Attention

### Class
`MultiHeadAttention(embdim=512, num_heads=8)`

### Configuration
- Heads: 8
- Head dimension: 64
- Output projection: Linear(512 → 512)

### Process
- Input reshaped to:
(B, 8, T, 64)

- SelfAttention applied independently per head
- Outputs concatenated
- Final linear projection applied

### Input
(B, T, 512)


### Output
(B, T, 512)


---

## 4. Residual + Layer Normalization

### Class
`AddResidual_LayerNorm(512)`

### Operation
x + sublayer(LayerNorm(x))


### Purpose
- Stabilizes training
- Preserves gradient flow
- Normalizes feature distribution

### Input / Output
(B, T, 512) → (B, T, 512)


---

## 5. Feed Forward Network

### Class
`FeedForward(512, 2048)`

### Architecture
Linear(512 → 2048)
ReLU
Linear(2048 → 512)


### Input
(B, T, 512)


### Output
(B, T, 512)


---

## 6. Encoder Block

### Class
`EncoderBlock(embdim=512, num_heads=8, ff_hidden_dim=2048)`

### Structure
- Multi-Head Attention
- Residual + LayerNorm
- Feed Forward
- Residual + LayerNorm

### Flow
x = attn_norm(x, mha)
x = ff_norm(x, ff)


### Input
(B, T, 512)


### Output
(B, T, 512)


---

## 7. Stacked Encoder - BERT Core

### Class
`BERT(vocab_size, embdim=512, num_heads=8, ff_hidden_dim=2048, num_layers=6)`

### Encoder Depth
- 6 Encoder Blocks

### Forward Pass
x = embedding(x)
for layer in layers:
x = layer(x)
return x


### Input
(B, T)


### Output
(B, T, 512)


---

## 8. Masked Language Modeling Head

### Class
`MLMHead(512, vocab_size, hidden_dim=1024)`

### Architecture
Linear(512 → 1024)
ReLU
Linear(1024 → 1024)
ReLU
Linear(1024 → 27964)


### Input
(B, T, 512)


### Output
(B, T, 27964)


Produces logits for masked token prediction.

---

## 9. Final Model Wrapper

### Class
`MY_BERT`

### Structure
BERT Encoder
+
MLM Head


### Forward
x = BERT_(x)
x = MLM_head(x)
return x


---

# Masking Strategy

### Dataset Class
`custom_dataset`

### Configuration
- mlm_prob = 0.15
- mask_token_id = 27963
- special_ids = [0, 2, 3]

### Behavior
- 15% of eligible tokens masked
- Special tokens excluded
- Labels for non-masked tokens set to -100
- Loss computed only on masked tokens

---

# Training Pipeline

### Loss Function
CrossEntropyLoss(ignore_index=-100)


### Accuracy Metric
mlm_accuracy()


### Optimizer
Adam
Learning Rate = 1e-5


### Execution
Epochs = 5
Batch Size = 100


---

# Model Saving

Saved under `models/` directory:

- `Bert.pth`
- `mask.pth`
- `mybert.pth`

---

# GPT - Generative Pretrained Transformer (Decoder-Only)

## Overview

This implementation builds GPT from first principles using modular components:

- Token + Positional Embedding
- Masked Self-Attention (Causal Attention)
- Multi-Head Attention
- Residual + Layer Normalization
- Feed Forward Network
- Stacked Decoder Blocks
- Language Modeling Head

Training is performed using **Autoregressive Next-Token Prediction** on a conversational dataset consisting of:
  
- Human ↔ AI prompt-response pairs
- dataset download code is present in file `GPT-2.ipynb`

The training pipeline, causal masking logic, architecture, and parameter statistics are derived directly from the implementation.

---

# Model Storage Notice

Due to file size constraints:

- Model weights are not stored inside the repository.
- 
Saved files:
- `GPT.pth`             model size: **368 MB**
- `GPT_outlayer.pth`    model size: **114 MB**
- `MY_GPT.pth`          model size: **482 MB**

---

# GPT - Generative Pretrained Transformer (Decoder-Only)

## Overview

This implementation builds GPT from first principles using modular components:

- Token + Positional Embedding
- Masked Self-Attention (Causal Attention)
- Multi-Head Attention
- Residual + Layer Normalization
- Feed Forward Network
- Stacked Decoder Blocks
- Language Modeling Head

Training is performed using **Autoregressive Next-Token Prediction** on a conversational dataset consisting of:
  
- Human ↔ AI prompt-response pairs
- dataset download code is present in file `GPT-2.ipynb`

The training pipeline, causal masking logic, architecture, and parameter statistics are derived directly from the implementation.

---

# Model Storage Notice

Due to file size constraints:

- Model weights are not stored inside the repository.
- 
Saved files:
- `GPT.pth`             model size: **368 MB**
- `GPT_outlayer.pth`    model size: **114 MB**
- `MY_GPT.pth`          model size: **482 MB**

---

# GPT - Architectural Configuration

## Core Hyperparameters

| Parameter | Value |
|------------|--------|
| Vocabulary Size | 38987 |
| Embedding Dimension | 768 |
| Number of Attention Heads | 12 |
| Head Dimension | 64 (768 / 12) |
| Feed Forward Hidden Dimension | 3072 |
| Number of Decoder Blocks | 12 |
| Context Length (Block Size) | Defined in notebook |
| Optimizer | AdamW |
| Learning Rate | Defined in notebook |
| Training Objective | Causal Language Modeling |

---

# Parameter Statistics

## Base GPT (Decoder Stack Only)

| Metric | Value |
|--------|--------|
| Total Parameters | 96,109,824 |
| Trainable Parameters | 96,109,824 |
| Parameter Tensors | 145 |
| All Parameters Trainable | True |

---

## Full Model (GPT + Output Head)

| Metric | Value |
|--------|--------|
| Total Parameters | 126,091,596 |
| Trainable Parameters | 126,091,596 |
| Parameter Tensors | 149 |
| All Parameters Trainable | True |

---

# Training Data

## Dataset Type

- Human ↔ Human dialogue
- Human ↔ AI conversational prompt data

## Preprocessing Pipeline

Raw conversational text is transformed as follows:

1. Sentence normalization
2. Tokenization
3. Conversion to integer token IDs
4. Fixed-length sequence chunking (context window)
5. Creation of input-target shifted pairs

Example transformation:

Input : [t1, t2, t3, t4]
Target : [t5]
Input : [t1, t2, t3, t4,t5]
Target : [t6]
Input : [t1, t2, t3, t4,t5,t6]
Target : [t7]


The model predicts the next token at every position.

Dataset download code is provided inside the notebook.

---

# Architecture Flow

---

## 1. Embedding + Positional Encoding

### Module
`Embedding(vocab_size, embdim)`

### Configuration
- Embedding dimension: 768
- Positional embedding: Learned

### Input
(B, T)


### Output
(B, T, 768)


Token embeddings and learned positional embeddings are added before entering the decoder stack.

---

## 2. Masked Self-Attention (Causal Attention)

### Class
`SelfAttention(embdim)`

### Internal Projections
- Wq: Linear(768 → 768)
- Wk: Linear(768 → 768)
- Wv: Linear(768 → 768)

### Scaling
scale = sqrt(64)


### Causal Mask

Upper triangular mask applied to prevent attention to future tokens.

mask = torch.triu(torch.ones(T, T), diagonal=1)


Masked positions set to `-inf` before softmax.

### Computation
scores = QK^T / sqrt(D)
scores = scores.masked_fill(mask == 1, -inf)
weights = softmax(scores)
output = weightsV


### Input
(B, T, 64)


### Output
(B, T, 64)


---

## 3. Multi-Head Attention

### Class
`MultiHeadAttention(embdim=768, num_heads=12)`

### Configuration
- Heads: 12
- Head dimension: 64
- Output projection: Linear(768 → 768)

### Process
- Input reshaped to:
(B, 12, T, 64)

- Causal SelfAttention applied independently per head
- Outputs concatenated
- Final linear projection applied

### Input
(B, T, 768)


### Output
(B, T, 768)


---

## 4. Residual + Layer Normalization

### Class
`AddResidual_LayerNorm(768)`

### Operation
x + sublayer(LayerNorm(x))


### Purpose
- Stabilizes training
- Maintains gradient flow
- Ensures autoregressive consistency

### Input / Output
(B, T, 768) → (B, T, 768)


---

## 5. Feed Forward Network

### Class
`FeedForward(768, 3072)`

### Architecture
Linear(768 → 3072)
ReLU
Linear(3072 → 768)


### Input
(B, T, 768)


### Output
(B, T, 768)


---

## 6. Decoder Block

### Class
`DecoderBlock(embdim=768, num_heads=12, ff_hidden_dim=3072)`

### Structure
- Masked Multi-Head Attention
- Residual + LayerNorm
- Feed Forward
- Residual + LayerNorm

### Flow
x = attn_norm(x, mha)
x = ff_norm(x, ff)


### Input
(B, T, 768)


### Output
(B, T, 768)


---

## 7. Stacked Decoder - GPT Core

### Class
`GPT(vocab_size, embdim=768, num_heads=12, ff_hidden_dim=3072, num_layers=12)`

### Decoder Depth
- 12 Decoder Blocks

### Forward Pass
x = embedding(x)
for layer in layers:
x = layer(x)
return x


### Input
(B, T)


### Output
(B, T, 768)


---

## 8. Language Modeling Head

### Class
`LMHead(768, vocab_size)`

### Architecture
Linear(768 → vocab_size)


### Input
(B, T, 768)


### Output
(B, T, vocab_size)


Produces logits for next-token prediction.

---

## 9. Final Model Wrapper

### Structure
GPT Decoder
+
LM Head


### Forward
x = GPT(x)
logits = LM_head(x)
return logits


---

# Loss & Training Objective

### Loss Function
CrossEntropyLoss()


---

### Execution
Epochs = 1 
Batch Size = 4 
Applied across all time steps with target tokens shifted by one position. 

---

# Model Saving

Saved under `models/` directory:

- `GPT.pth`
- `GPT_outlayer.pth`
- `MY_GPT.pth`



