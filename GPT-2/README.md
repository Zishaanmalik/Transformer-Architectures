# GPT-2 — Generative Pretrained Transformer (Decoder-Only)

## Overview

A from-scratch, decoder-only Transformer following the GPT-2 recipe: causal (masked) self-attention stacked 12 deep, trained with next-token prediction on Human ↔ AI conversational data. Every sub-layer is hand-written `nn.Module` code; no `transformers` library is used for the model itself.

**Dataset:** `Dahoas/instruct-human-assistant-prompt` (loaded via `pandas.read_parquet` directly from the HuggingFace hub, saved locally as `human_prompt.csv`).

---

## Data Pipeline

1. **Sentence normalization** of the raw prompt/response text.
2. **Tokenization** into integer IDs (vocabulary size 38,987).
3. **Fixed-length sequence chunking** — text is cut into a context window.
4. **Input–target shifted pairs** are created for autoregressive training:

```
Input : [t1, t2, t3, t4]               Target : [t5]
Input : [t1, t2, t3, t4, t5]           Target : [t6]
Input : [t1, t2, t3, t4, t5, t6]       Target : [t7]
```

This produces a `CustomDataset(x, y)` where each `x[idx]` is a prefix and `y[idx]` is the token that should follow it, loaded through `DataLoader(batch_size=4, shuffle=True)`.

> Note: this chunking scheme differs from the typical GPT training setup of predicting *every* position in parallel across a fixed window — here each example is a single (prefix → next-token) pair. This is consistent with how the model's output head works (see `GPT2OutputHead` below), which pools the whole sequence into one vector and predicts a single next token per forward pass rather than one prediction per position.

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Vocabulary Size | 38,987 |
| Embedding Dimension (`embdim`) | 768 |
| Number of Attention Heads | 12 |
| Head Dimension | 64 (768 / 12) |
| Feed Forward Hidden Dimension | 3,072 (per the documented hyperparameter table; the `GPT2` class default argument is 2048 unless overridden) |
| Number of Decoder Blocks | 12 |
| Max Sequence Length | 128 |
| Dropout | 0.1 (documented; not wired into the attention/FFN modules themselves — see below) |
| Optimizer | Adam |
| Learning Rate | 6e-4 |
| Batch Size | 4 |
| Epochs | 1 |

---

## Parameter Statistics

**Base GPT (decoder stack only):** 96,109,824 total/trainable parameters, 145 parameter tensors, 100% trainable.

**Full model (GPT + output head):** 126,091,596 total/trainable parameters, 149 parameter tensors, 100% trainable.

---

## Architecture — Component by Component

### 1. `embpos` — Token Embedding + Sinusoidal Positional Encoding

```python
class embpos(nn.Module):
    def __init__(self, vocab_size, embdim, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embdim)
        pe = torch.zeros(max_len, embdim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embdim, 2) * (-math.log(10000.0) / embdim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        B, T = x.size()
        emb = self.embedding(x)
        emb = emb + self.pe[:, :T, :].to(emb.device)
        return emb
```

Identical construction to the BERT notebook's `embpos`: learned token embedding table + fixed (non-trainable) sinusoidal positional table, added together. `(B, T)` → `(B, T, 768)`.

### 2. `MaskedSelfAttention` — Single-Head Causal Attention (utility/reference block)

```python
class MaskedSelfAttention(nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.Wq = nn.Linear(embdim, embdim)
        self.Wk = nn.Linear(embdim, embdim)
        self.Wv = nn.Linear(embdim, embdim)
        self.scale = math.sqrt(embdim)

    def forward(self, x):
        B, T, D = x.size()
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device))   # lower-triangular = causal
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
```

Defined as a standalone single-head reference implementation of causal attention; the model itself uses the fused multi-head version below.

### 3. `MultiHeadAttention` — Causal Multi-Head Attention (fused QKV)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embdim, num_heads):
        super().__init__()
        assert embdim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embdim // num_heads
        self.qkv = nn.Linear(embdim, 3 * embdim)   # single fused projection
        self.out = nn.Linear(embdim, embdim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)                          # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)
```

Unlike BERT's per-head-independent-weights design, GPT-2's multi-head attention here uses the more conventional approach: **one shared `Linear(768 → 3*768)` projection** produces Q, K, V together, which are then split into 12 heads of dimension 64. A `torch.tril` lower-triangular mask enforces that position `i` can only attend to positions `≤ i`, giving the model its autoregressive (causal) property.

### 4. `AddResidual_LayerNorm` / `FeedForward` / `DecoderBlock`

Structurally identical pattern to BERT's encoder block, but with `MultiHeadAttention` being causally masked:

```python
class FeedForward(nn.Module):
    def __init__(self, embdim, hidden_dim):
        super().__init__()
        self.output_linear = nn.Sequential(nn.Linear(embdim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embdim))
    def forward(self, x):
        return self.output_linear(x)

class DecoderBlock(nn.Module):
    def __init__(self, embdim, num_heads, ff_hidden_dim):
        super().__init__()
        self.mha = MultiHeadAttention(embdim, num_heads)
        self.ff = FeedForward(embdim, ff_hidden_dim)
        self.attn_norm = AddResidual_LayerNorm(embdim)
        self.ff_norm = AddResidual_LayerNorm(embdim)

    def forward(self, x):
        x = self.attn_norm(x, self.mha)   # x + CausalMHA(LayerNorm(x))
        x = self.ff_norm(x, self.ff)      # x + FFN(LayerNorm(x))
        return x
```

(Note: `FeedForward` here uses `GELU` activation, whereas BERT's `FeedForward` used `ReLU` — a small but real difference between the two notebooks.)

### 5. `GPT2` — Stacked Decoder Core

```python
class GPT2(nn.Module):
    def __init__(self, vocab_size, embdim=768, num_heads=12, ff_hidden_dim=2048, num_layers=12):
        super().__init__()
        self.emb = embpos(vocab_size, embdim)
        self.layers = nn.ModuleList([DecoderBlock(embdim, num_heads, ff_hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return x   # (B, T, 768)
```

12 stacked causal decoder blocks — the same structural pattern GPT-family models use, reimplemented from scratch.

### 6. `AttentionPooling` + `GPT2OutputHead` — Sequence-Pooled Output Head

```python
class AttentionPooling(nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.score = nn.Linear(embdim, 1)

    def forward(self, x):
        weights = self.score(x)                 # (B, T, 1) — a learned scalar score per token
        weights = torch.softmax(weights, dim=1)   # softmax over the T (sequence) dimension
        pooled = (weights * x).sum(dim=1)         # (B, D) — weighted sum of all token vectors
        return pooled

class GPT2OutputHead(nn.Module):
    def __init__(self, embdim, vocab_size):
        super().__init__()
        self.pool = AttentionPooling(embdim)
        self.fc = nn.Linear(embdim, vocab_size)

    def forward(self, hidden):
        pooled = self.pool(hidden)   # (B, 768) — the ENTIRE sequence collapsed to one vector
        logits = self.fc(pooled)     # (B, vocab_size)
        return logits
```

**Architectural note:** this is a deliberate departure from the standard GPT language-modeling head (which normally applies `Linear(embdim → vocab_size)` at *every* position to get `(B, T, vocab_size)` per-token next-token logits). Here, an attention-pooling layer first learns a scalar importance score for every token in the sequence, softmaxes those scores across the time dimension, and computes a single weighted-average vector representing the whole sequence. The vocabulary projection is then applied **once** to that pooled vector, producing a single `(B, vocab_size)` prediction per input sequence — i.e. the model predicts one "next token" for the entire prefix, matching the (prefix → next-token) pair structure of the dataset described above, rather than a full per-position language-modeling head.

### 7. `MY_GPT` — Full Wrapper

```python
class MY_GPT(nn.Module):
    def __init__(self, embdim, vocab_size):
        super().__init__()
        self.out_layer = GPT2OutputHead(768, vocab_size)
        self.model = GPT2(vocab_size=vocab_size)

    def forward(self, x):
        x = self.model(x)      # (B, T) → (B, T, 768)
        x = self.out_layer(x)   # (B, T, 768) → (B, vocab_size)
        return x
```

---

## Training Pipeline

- **Loss:** `CrossEntropyLoss()` comparing the `(B, vocab_size)` pooled prediction against the single next-token target `y`.
- **Optimizer:** Adam, learning rate `6e-4`.
- **Accuracy:** custom `accuracy_fn()` comparing `argmax(logits)` to the target token.
- **Execution:** 1 epoch, batch size 4.

---

## Model Saving

Saved under `models/` directory:

| File | Size |
|---|---|
| `GPT.pth` | 368 MB |
| `GPT_outlayer.pth` | 114 MB |
| `MY_GPT.pth` | 482 MB |

Not stored in this repository due to size — regenerate by running the notebook end-to-end.
