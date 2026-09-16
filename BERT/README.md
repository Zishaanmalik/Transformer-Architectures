# BERT — Bidirectional Encoder Transformer (Masked Language Modeling)

## Overview

A from-scratch, encoder-only Transformer built exactly to the original BERT recipe: bidirectional self-attention (no causal mask) trained with a **Masked Language Modeling (MLM)** objective. Every sub-layer — embeddings, attention, normalization, feed-forward, and the MLM head — is a hand-written `nn.Module`; no `transformers` library is used anywhere.

Trained on the **Complete Works of Shakespeare** (plain text corpus, tokenized with a custom word-level tokenizer).

---

## Data Pipeline (exact preprocessing order)

1. **Lowercase** the raw text.
2. **Sentence split** into a list of sentences.
3. **Remove punctuation.**
4. **Remove unwanted whitespace.**
5. **Remove empty/null entries** produced by the previous steps.
6. **Add special tokens** to the vocabulary (`[0, 2, 3]` are reserved special IDs; a dedicated `[mask]` token is also added).
7. **Tokenize** every sentence into integer IDs using a word-level `Tokenizer` (`sen_tokenizer`), with `num_words = 27964`.
8. **Pad** every sequence to a fixed length.
9. **Mask** 15% of eligible tokens for the MLM objective (see below).

Final vocabulary size: **27,964**. Mask token ID: **27963**. Special (never-masked) token IDs: **[0, 2, 3]**.

### Masking function — `simple_mask`

```python
def simple_mask(input_ids, vocab_size, mask_token_id=27963, special_ids=[0,2,3], mlm_prob=0.15):
    rand = torch.rand_like(input_ids.float())
    maskable = ~torch.isin(input_ids, torch.tensor(special_ids))
    mask = (rand < mlm_prob) & maskable          # positions selected for masking

    mlm_labels = input_ids.clone()
    mlm_labels[~mask] = -100                       # ignored by CrossEntropyLoss

    masked_x = input_ids.clone()
    masked_x[mask] = mask_token_id                 # replace with [MASK]
    return masked_x.long(), mlm_labels.long()
```

- A uniform random number is drawn per token; if it is below `mlm_prob` **and** the token is not a special token, that position is selected.
- Selected positions in the **input** are overwritten with the mask token ID.
- The **label** tensor keeps the ground-truth token only at masked positions; everywhere else it is set to `-100`, PyTorch's `CrossEntropyLoss(ignore_index=-100)` convention for "don't compute loss here".
- This means, unlike the original BERT paper's 80/10/10 (mask/random/keep) split, this implementation always replaces the selected token with `[MASK]` — a simplified single-strategy masking scheme.

Wrapped by `custom_dataset(sequences, vocab_size, mask_token_id, mlm_prob, special_ids)`, whose `__getitem__` calls `simple_mask` fresh for every sample retrieval (so masking is re-sampled every epoch).

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Vocabulary Size | 27,964 |
| Embedding Dimension (`embdim`) | 512 |
| Number of Attention Heads | 8 |
| Head Dimension | 64 (512 / 8) |
| Feed Forward Hidden Dimension | 2048 |
| Number of Encoder Blocks | 6 |
| MLM Head Hidden Dimension | 1024 |
| MLM Masking Probability | 0.15 |
| Mask Token ID | 27963 |
| Special Token IDs (never masked) | [0, 2, 3] |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Batch Size | 100 |
| Epochs | 5 |

---

## Parameter Statistics

**Base BERT (encoder stack only):** 444,416,000 total / trainable parameters, across 385 parameter tensors, 100% trainable.

**Full model (BERT + MLM head):** 474,654,012 total / trainable parameters, across 391 parameter tensors, 100% trainable.

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
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, T, D), not trainable

    def forward(self, x):
        B, T = x.size()
        emb = self.embedding(x)                    # (B, T, D) — learned token vectors
        emb = emb + self.pe[:, :T, :].to(emb.device) # add fixed sinusoidal signal
        return emb
```

- The positional table is the classic Transformer sinusoid: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`.
- It is computed once at construction time and stored as a non-trainable `buffer`, then simply added to the (trainable) token embedding — it is **not** learned, and there is no scaling of the embedding by `sqrt(embdim)` in this notebook's `embpos` (unlike some of the other notebooks in this repo, which do scale).
- Input `(B, T)` integer token IDs → Output `(B, T, 512)` float embeddings.

### 2. `SelfAttention` — Single-Head Scaled Dot-Product Attention

```python
class SelfAttention(nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.Wq = nn.Linear(embdim, embdim)
        self.Wk = nn.Linear(embdim, embdim)
        self.Wv = nn.Linear(embdim, embdim)
        self.scale = math.sqrt(embdim)

    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
```

This is the building block used **per head** (see below) — it is instantiated with `embdim = head_dim = 64`, i.e. each head gets its own independent `Wq/Wk/Wv` of shape `64 × 64`, not a shared, sliced projection.

### 3. `MultiHeadAttention` — Per-Head Independent Weights (non-standard design)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embdim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embdim // num_heads
        self.heads = nn.ModuleList([SelfAttention(self.head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embdim, embdim)
        # NOTE: this deeper projection is defined but never called in forward() — dead code, kept as-is for transparency
        self.deep_output_linear = nn.Sequential(
            nn.Linear(embdim * num_heads, embdim * num_heads * 2), nn.ReLU(),
            nn.Linear(embdim * num_heads * 2, embdim * num_heads), nn.ReLU(),
            nn.Linear(embdim * num_heads, embdim)
        )

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)

        head_outputs = []
        for i, head in enumerate(self.heads):
            head_x = x[:, i, :, :]           # (B, T, head_dim) slice for this head
            head_outputs.append(head(head_x))  # each head runs its OWN SelfAttention module

        concatenated = torch.cat(head_outputs, dim=-1)   # (B, T, D)
        return self.output_linear(concatenated)
```

**Important architectural detail:** unlike most Transformer implementations (including the other notebooks in this repository), this `MultiHeadAttention` does **not** use one shared `Linear(embdim → embdim)` projection that is then split into heads. Instead, `embdim` is first split into `num_heads` slices of size `head_dim`, and **each slice is routed through its own independently-parameterized `SelfAttention` instance** (its own `Wq/Wk/Wv` of size `64×64`). The outputs are concatenated back to `512` and passed through one final `output_linear`. A second, deeper projection head (`deep_output_linear`) is defined in `__init__` but is not used in `forward` — it is present in the notebook as an unused/experimental layer and is documented here for full transparency.

### 4. `AddResidual_LayerNorm` — Pre-Norm Residual Wrapper

```python
class AddResidual_LayerNorm(nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.norm = nn.LayerNorm(embdim)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
```

Generic wrapper: normalizes the input, runs it through whatever sub-layer function is passed in (attention or feed-forward), and adds the result back to the un-normalized residual stream (pre-norm residual pattern).

### 5. `FeedForward` — Position-wise MLP

```python
class FeedForward(nn.Module):
    def __init__(self, embdim, hidden_dim):
        super().__init__()
        self.output_linear = nn.Sequential(
            nn.Linear(embdim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embdim)
        )
    def forward(self, x):
        return self.output_linear(x)
```

`Linear(512 → 2048) → ReLU → Linear(2048 → 512)`, applied independently at every token position.

### 6. `EncoderBlock`

```python
class EncoderBlock(nn.Module):
    def __init__(self, embdim, num_heads, ff_hidden_dim):
        super().__init__()
        self.mha = MultiHeadAttention(embdim, num_heads)
        self.ff = FeedForward(embdim, ff_hidden_dim)
        self.attn_norm = AddResidual_LayerNorm(embdim)
        self.ff_norm = AddResidual_LayerNorm(embdim)

    def forward(self, x):
        x = self.attn_norm(x, self.mha)   # x + MHA(LayerNorm(x))
        x = self.ff_norm(x, self.ff)      # x + FFN(LayerNorm(x))
        return x
```

Standard pre-norm Transformer encoder block: self-attention sub-layer, then feed-forward sub-layer, each wrapped in the residual+norm module above. No causal mask is applied anywhere — every token attends to every other token, which is what makes this an **encoder** (bidirectional) rather than a decoder.

### 7. `BERT` — Stacked Encoder Core

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, embdim=512, num_heads=8, ff_hidden_dim=2048, num_layers=6):
        super().__init__()
        self.emb = embpos(vocab_size, embdim)
        self.layers = nn.ModuleList([EncoderBlock(embdim, num_heads, ff_hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        x = self.emb(x)               # (B, T) → (B, T, 512)
        for layer in self.layers:
            x = layer(x)               # 6× EncoderBlock
        return x                       # (B, T, 512) contextualized representations
```

### 8. `MLMHead` — Masked Language Modeling Output

```python
class MLMHead(nn.Module):
    def __init__(self, embdim, vocab_size, hidden_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embdim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
    def forward(self, x):
        return self.mlp(x)   # (B, T, 512) → (B, T, 27964)
```

A 3-layer MLP (with a hidden bottleneck of 1024) projects every token's contextual representation to a distribution over the full vocabulary — the model's guess for what the (possibly masked) token at that position actually is.

### 9. `MY_BERT` — Full Wrapper

```python
class MY_BERT(nn.Module):
    def __init__(self, vocab_size, embdim=512):
        super().__init__()
        self.BERT_ = BERT(vocab_size=vocab_size)
        self.MY_mlm_head = MLMHead(embdim, vocab_size)

    def forward(self, x):
        x = self.BERT_(x)
        x = self.MY_mlm_head(x)
        return x
```

`(B, T)` masked token IDs in → `(B, T, 27964)` per-token vocabulary logits out.

---

## Training Pipeline

- **Loss:** `CrossEntropyLoss(ignore_index=-100)` — loss is only computed at masked positions (label `-100` elsewhere is skipped by PyTorch).
- **Accuracy:** custom `mlm_accuracy()` metric, computed only on masked positions.
- **Optimizer:** Adam, learning rate `1e-5`.
- **Loop:** for each batch, `masked_x` is fed through the model to get `(B, T, vocab_size)` logits, loss is computed against `mlm_labels` flattened to `(B*T,)`, backpropagated, and the optimizer steps.
- **Execution:** 5 epochs, batch size 100.
- The notebook trains **two variants**: `model` (`BERT` + a separately-instantiated `mlm_head`, trained via `train()`) and `my_model` (the fused `MY_BERT` wrapper, trained via `train2()`) — both use the identical architecture, just different Python-level composition.

---

## Model Saving

Saved under `models/` directory:

| File | Size |
|---|---|
| `Bert.pth` | 1.65 GB |
| `mask.pth` | 115 MB |
| `mybert.pth` | 1.76 GB |

Not stored in this repository due to size — regenerate by running the notebook end-to-end.
