# ASR-Transformer — Encoder-Decoder Transformer (Audio Speech Recognition)

## Overview

A from-scratch encoder-decoder speech-to-text model: a **Conformer** encoder consumes mel-spectrogram patches, and a causal Transformer decoder with cross-attention produces the transcription, token by token. Every module — patch embedding, positional embeddings, the Conformer block's convolution module, masked self-attention, cross-attention, and the output head — is hand-written.

**Dataset:** LibriSpeech ASR (`openslr/librispeech_asr`, `clean` config, `train.100[:30000]` split, via HuggingFace `datasets`).

```
Audio → PatchEmbed → Pos → Conformer (Encoder) → audio_tokens (B, 64, 768)
Text  → Embedding → Pos → text_tokens (B, 127, 768)
Decoder: Masked Self-Attn (text) → Cross-Attn (text ↔ audio) → FF
Output → Linear → vocab
```

---

## Data Pipeline

### Audio feature extraction (waveform → mel-spectrogram)

| Step | Operation | Shape |
|---|---|---|
| Load | `librosa.load(path, sr=16000)` | `(L,)` waveform |
| Mel spectrogram | `librosa.feature.melspectrogram(...)` (Framing → FFT → Power → Mel filterbank) | `(L,) → (M, T)` |
| Log scale | `librosa.power_to_db(mel)` | `(M, T)` |
| Transpose | `log_mel.T` | `(M, T) → (T, M)` |
| Pad | `np.pad(...)` to fixed length | `(T, M) → (T_max, M)`, `M=80` |

### Text pipeline

- Lowercased transcripts, custom word-level vocabulary built directly from the training captions.
- `encode_caption`-style encoding to integer IDs with padding/truncation.

### `ASRDataset`

```python
class ASRDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.data[idx]
        audio = torch.tensor(sample["audio_features"], dtype=torch.float32)          # (T, 80)
        decoder_input = torch.tensor(sample["decoder_input_ids"], dtype=torch.long)   # (L)
        labels = torch.tensor(sample["labels"], dtype=torch.long)                     # (L)
        return {"audio_features": audio, "decoder_input_ids": decoder_input, "labels": labels}
```

`DataLoader(batch_size=5, shuffle=True)`.

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Model Dimension (`d_model`) | 512 |
| Number of Attention Heads | 8 |
| Head Dimension | 64 (512 / 8) |
| Encoder Layers (Conformer) | 6 |
| Decoder Layers | 6 |
| Feed Forward Hidden Dimension | 2048 |
| Conformer FF Expansion Factor | 4 |
| Conformer Conv Expansion Factor | 2 |
| Conformer Conv Kernel Size | 31 |
| Max Audio Length | 1,200 |
| Max Text Length | 518 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Batch Size | 5 |
| Loss ignore index | 0 (PAD) |

---

## Architecture — Component by Component

### 1. `PatchEmbedding` — Audio "Patchification" via Strided Convolutions

```python
class PatchEmbedding(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.proj = nn.Sequential(               # input: (B, 1, 3516, 80)
            nn.Conv2d(1,   64, kernel_size=3, stride=2, padding=1), nn.GELU(),   # → (B,64,1758,40)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GELU(),   # → (B,128,879,20)
            nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1), nn.GELU(),   # → (B,256,440,10)
            nn.Conv2d(256,d_model,kernel_size=3, stride=2, padding=1),           # → (B,512,220,5)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                        # x: (B, 3516, 80) log-mel spectrogram
        x = x.unsqueeze(1)                        # → (B, 1, 3516, 80): treat spectrogram as a 1-channel image
        x = self.proj(x)                          # → (B, 512, 220, 5): 4 strided Conv2d layers, doubling channels, halving H/W each time
        B, C, H, W = x.shape
        x = x.flatten(2)                          # → (B, 512, 1100): flatten spatial dims into a token sequence
        x = x.transpose(1, 2)                     # → (B, 1100, 512): (batch, tokens, channels)
        x = self.norm(x)
        return x
```

This treats the log-mel spectrogram as a single-channel 2D "image" and progressively downsamples it with four stride-2 `Conv2d` layers (`1→64→128→256→512` channels), each followed by GELU. The final `(H, W)` grid is flattened into a sequence of "audio patch tokens" — conceptually equivalent to a Vision Transformer's patch embedding, but built from overlapping convolutional receptive fields rather than a single non-overlapping patch-conv.

### 2. `AudioPositionalEmbedding` — Learned Positional Parameter

```python
class AudioPositionalEmbedding(nn.Module):
    def __init__(self, max_len=1200, d_model=512, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))  # fully learned, not sinusoidal
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pos_embed[:, :T, :]
        return self.dropout(x)
```

Unlike the text side (sinusoidal, fixed), the audio side uses a **learned** positional embedding table, randomly initialized and updated by gradient descent.

### 3. `TextPositionalEmbedding` — Scaled Token Embedding + Sinusoidal Position

```python
class TextPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embdim, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embdim, padding_idx=0)
        # ... sinusoidal pe table built exactly as in BERT/GPT-2 ...
        self.scale = math.sqrt(embdim)

    def forward(self, x):
        emb = self.embedding(x) * self.scale     # scaled embedding (as in "Attention Is All You Need")
        emb = emb + self.pe[:, :T, :]
        return self.dropout(emb)
```

`padding_idx=0` means the embedding for the PAD token is fixed at zero and receives no gradient. The token embedding is scaled by `sqrt(embdim)` before adding the positional signal (a detail present here but absent in the BERT/GPT-2 `embpos` modules).

### 4. Conformer Encoder — `ConformerConvModule`, `MultiHeadAttention`, `FeedForwardModule`, `ConformerBlock`, `ConformerEncoder`

**`ConformerConvModule`** — the convolutional sub-layer that distinguishes a Conformer block from a plain Transformer block:

```python
class ConformerConvModule(nn.Module):
    def __init__(self, d_model, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)                                    # gated linear unit — halves channel count back to inner_dim
        self.depthwise_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=31, padding=15, groups=inner_dim)  # one filter per channel
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.slu = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(inner_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                       # x: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)                    # → (B, D, T) — Conv1d expects channels-first
        x = self.pointwise_conv1(x)               # → (B, 2*inner, T): 1×1 conv expanding channels
        x = self.glu(x)                            # → (B, inner, T): gated linear unit, halves channels back down, applies a learned gate
        x = self.depthwise_conv(x)                  # → (B, inner, T): depthwise (grouped) conv, kernel=31 — captures LOCAL temporal patterns per channel
        x = self.batch_norm(x)
        x = self.slu(x)                              # SiLU (Swish) activation
        x = self.pointwise_conv2(x)                   # → (B, D, T): 1×1 conv projecting back to model dimension
        x = x.transpose(1, 2)                          # → (B, T, D)
        return self.dropout(x)
```

Structure: `LayerNorm → Pointwise Conv (expand ×2·expansion) → GLU (gate + halve) → Depthwise Conv (kernel=31, per-channel) → BatchNorm → SiLU → Pointwise Conv (project back to d_model) → Dropout`. This is the exact Conformer convolution-module recipe: pointwise convolutions handle channel mixing, the depthwise convolution (kernel size 31, `groups=inner_dim` so every channel gets its own independent 1D filter) captures local temporal correlations that self-attention alone tends to under-model, and the GLU gate lets the network learn to suppress/emphasize channels dynamically.

**`MultiHeadAttention`** (Conformer's self-attention, fused QKV):

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)
```

Standard bidirectional (no mask) fused-QKV multi-head self-attention — full attention over the entire audio sequence.

**`FeedForwardModule`** (the Conformer's "macaron" half-step FFN):

```python
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, inner_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(inner_dim, d_model), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
```

**`ConformerBlock`** — assembles the full "macaron" structure:

```python
class ConformerBlock(nn.Module):
    def forward(self, x):
        x = x + 0.5 * self.ff1(x)   # first HALF-STEP feed-forward (weighted 0.5, macaron style)
        x = x + self.mha(x)          # full multi-head self-attention
        x = x + self.conv(x)          # convolution module (local patterns)
        x = x + 0.5 * self.ff2(x)      # second HALF-STEP feed-forward
        x = self.norm(x)                # final layer norm
        return x
```

The defining Conformer trait: **two half-weighted feed-forward sub-layers sandwiching the attention and convolution modules** (the "macaron-net" pattern from the original Conformer paper), rather than a single full-weight FFN as in vanilla Transformers.

**`ConformerEncoder`** stacks 6 `ConformerBlock`s sequentially.

### 5. Decoder — `MaskedMultiHeadAttention`, `CrossAttention`, `FeedForward`, `DecoderBlock`, `Decoder`

**`MaskedMultiHeadAttention`** — causal self-attention over the text being generated:

```python
class MaskedMultiHeadAttention(nn.Module):
    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)   # separate projections (not fused)
        scores = (Q @ K.transpose(-2,-1)) / sqrt(head_dim)
        mask = torch.tril(torch.ones(T, T)).bool()
        scores = scores.masked_fill(~mask, -inf)
        weights = softmax(scores, dim=-1)
        out = weights @ V
        return self.out(out)
```

**`CrossAttention`** — text tokens attend to encoder (audio) output:

```python
class CrossAttention(nn.Module):
    def forward(self, query, context):   # query: text (B,T,D), context: audio (B,N,D)
        Q = self.Wq(query); K = self.Wk(context); V = self.Wv(context)
        scores = (Q @ K.transpose(-2,-1)) / sqrt(head_dim)   # (B, heads, T, N)
        weights = softmax(scores, dim=-1)
        out = weights @ V
        return self.out(out)
```

Q comes from the decoder's text representation; K and V come from the Conformer encoder's audio output — this is the mechanism by which the decoder "listens" to the audio while generating text.

**`DecoderBlock`**:

```python
def forward(self, text_tokens, audio_tokens):
    x = text_tokens + dropout(self_attn(norm1(text_tokens)))          # 1. causal self-attention
    x = x + dropout(cross_attn(norm2(x), audio_tokens))                 # 2. cross-attention to audio
    x = x + dropout(ff(norm3(x)))                                        # 3. feed-forward
    return x
```

Three residual sub-layers per block, each pre-normed: masked self-attention → cross-attention → feed-forward. `Decoder` stacks 6 of these.

### 6. `VocabHead` — Output Projection

`Linear(512 → vocab_size)`, applied at every decoder position: `(B, T, 512) → (B, T, vocab_size)`.

### 7. `ASRTransformer` — Full Model Assembly

```python
def forward(self, audio, text):
    audio_tokens = self.patch_embed(audio)          # (B, 3516, 80) → (B, 1100, 512)
    audio_tokens = self.audio_pos(audio_tokens)
    enc_out = self.encoder(audio_tokens)              # 6× ConformerBlock → (B, 1100, 512)

    text_tokens = self.text_pos(text)                  # (B, T_text) → (B, T_text, 512)
    dec_out = self.decoder(text_tokens, enc_out)         # 6× DecoderBlock, cross-attending to enc_out

    logits = self.vocab_head(dec_out)                     # (B, T_text, vocab_size)
    return logits
```

---

## Training Pipeline

**Loop:** `train_asrtransformer(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)`

- **Teacher forcing:** decoder input = `text[:, :-1]`, target = `labels[:, 1:]` (predict each next token given all previous ground-truth tokens).
- **Loss:** `CrossEntropyLoss(ignore_index=0)` (PAD tokens excluded).
- **Optimizer:** Adam, learning rate `3e-4`.
- **Execution:** epochs configurable (loop run with `epochs=1` in the notebook's final cell), batch size 5.

---

## Model Saving

Weights are saved under a `models/` directory. Not stored in this repository due to size — regenerate by running the notebook end-to-end.
