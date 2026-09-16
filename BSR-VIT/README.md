# BSR-VIT — Encoder-Decoder Transformer (Brain Signal Recognition)

## Overview

A from-scratch encoder-decoder model that maps multi-channel brain-signal (EEG-style) spectrogram data to character-level text, using the **same Conformer-encoder / causal-decoder architecture family** as the ASR-Transformer notebook, re-parameterized for a 16-channel signal input and a much shorter output sequence.

> **Naming note:** despite the "VIT" suffix in the filename, the encoder implemented here is a **Conformer** encoder — identical block-for-block structure (macaron feed-forward + multi-head self-attention + convolution module) to the ASR and Conformer notebooks in this repository, not a plain Vision Transformer. This README documents the architecture exactly as implemented.

```
Signal → PatchEmbed → Pos → Conformer (Encoder) → signal_tokens (B, 6, 360)
Text   → Embedding → Pos → text_tokens (B, 15, 360)
Decoder: Masked Self-Attn (text) → Cross-Attn (text ↔ signal) → FF
Output  → Linear → vocab
```

---

## Data Pipeline

- **Signal input:** loaded from `CSV1.csv` / `CSV2.csv`, concatenated into one combined dataframe (`pd.concat`). Each sample is reshaped into a multi-channel tensor of shape `(16, 64, 256)` — 16 channels (e.g. EEG electrode/frequency-band channels), analogous to a 16-channel "image".
- **Text:** character-level vocabulary built from `string.ascii_letters + string.digits + string.punctuation + " "`, plus `<pad>`, `<eos>`, `<bos>` special tokens.
- **`BSRDataset`:**

```python
class BSRDataset(Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = torch.tensor(row["signal"], dtype=torch.float32)     # (16, 64, 256)
        tokens = encode_text(row["text"], self.max_len)
        return {
            "image": signal,
            "input_ids": torch.tensor(tokens[:-1]),   # decoder input (shifted right)
            "labels": torch.tensor(tokens[1:])          # target (shifted left)
        }
```

- `DataLoader(batch_size=8, shuffle=True)`.

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Model Dimension (`d_model`) | 360 |
| Number of Attention Heads | 6 |
| Head Dimension | 60 (360 / 6) |
| Encoder Layers (Conformer) | 6 |
| Decoder Layers | 6 |
| Feed Forward Hidden Dimension | 2048 |
| Conformer FF Expansion Factor | 3 |
| Conformer Conv Expansion Factor | 2 |
| Conformer Conv Kernel Size | 31 |
| Max Text Length | 16 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning Rate | 3e-5 |
| Batch Size | 8 |
| Epochs | 100 |
| Loss ignore index | 0 (PAD) |

---

## Architecture — Component by Component

The building blocks are structurally identical to the ASR-Transformer notebook (same class names, same math), reconfigured for `d_model=360`, `num_heads=6`, and a 16-channel input. Full detail is given once here; refer to the ASR-Transformer README for line-by-line explanation of shared internals (Conformer conv module, macaron block, masked/cross-attention).

### 1. `PatchEmbedding` — 16-Channel Signal Patchification

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=16, d_model=360):
        super().__init__()
        self.proj = nn.Sequential(                       # input: (B, 16, F, T)
            nn.Conv2d(16,  64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(256,360, kernel_size=3, stride=2, padding=1),
        )
    def forward(self, x):
        x = self.proj(x)                                   # → (B, 360, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                    # → (B, H'*W', 360)
        return x
```

Four stride-2 `Conv2d` layers take the raw 16-channel signal grid straight from 16 input channels up to `d_model=360`, halving the spatial resolution at each layer (no single-channel `unsqueeze` step is needed here since the signal is already multi-channel, unlike the single-channel spectrogram in the ASR notebook).

### 2. `ImagePositionalEmbedding` — Learned Positional Parameter

```python
class ImagePositionalEmbedding(nn.Module):
    def __init__(self, num_patches=150, d_model=360):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]
```

A learned (not sinusoidal) positional table, added directly to the flattened patch tokens.

### 3. `TextPositionalEmbedding`

Same sinusoidal-plus-embedding construction as the ASR notebook (`nn.Embedding(vocab_size, 360, padding_idx=0)` + precomputed sin/cos table).

### 4. Conformer Encoder Stack

Same `ConformerConvModule → MultiHeadAttention → FeedForwardModule → ConformerBlock (macaron) → ConformerEncoder` chain as the ASR notebook, instantiated with `d_model=360, num_heads=6, num_layers=6`. The macaron block computes:

```
x = x + 0.5 * ff1(x)      # half-step FFN #1
x = x + mha(x)             # full self-attention over signal tokens
x = x + conv(x)              # depthwise-conv module (kernel=31) capturing local temporal/spatial structure in the signal
x = x + 0.5 * ff2(x)          # half-step FFN #2
x = norm(x)
```

### 5. Decoder — `MaskedMultiHeadAttention`, `CrossAttention`, `FeedForward`, `DecoderBlock`, `Decoder`

Identical structure to ASR-Transformer's decoder, with `embdim=360, num_heads=6`:

```
x = text + self_attn(norm1(text))                # causal self-attention over generated characters
x = x + cross_attn(norm2(x), signal_tokens)         # cross-attention: characters attend to encoded brain-signal tokens
x = x + ff(norm3(x))                                  # feed-forward
```

`Decoder(depth=6, embdim=360, num_heads=6, ff_hidden_dim=2048)`.

### 6. `VocabHead` — `Linear(360 → vocab_size)`

Applied per position over the character vocabulary (letters, digits, punctuation, space, plus `<pad>/<eos>/<bos>`).

### 7. `BSRTransformer` — Full Model Assembly

```python
class BSRTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=360, encoder_layers=6, decoder_layers=6,
                 num_heads=6, ff_expansion=3, conv_expansion=2, ff_hidden_dim=2048,
                 max_text_len=16, dropout=0.1):
        ...
    def forward(self, signal, text):
        signal_tokens = self.encoder(self.signal_pos(self.patch_embed(signal)))   # (B, 16,64,256) → (B, N, 360)
        text_tokens = self.text_pos(text)                                          # (B, T) → (B, T, 360)
        decoder_out = self.decoder(text_tokens, signal_tokens)                       # cross-attends to signal
        logits = self.vocab_head(decoder_out)                                          # (B, T, vocab_size)
        return logits
```

---

## Training Pipeline

**Loop:** `train_asrtransformer(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)` — the same training-loop function name is reused from the ASR notebook, adapted to the `(signal, text)` input pair.

- **Teacher forcing:** decoder input/target are the pre-shifted `input_ids` / `labels` produced by `BSRDataset`.
- **Loss:** `CrossEntropyLoss(ignore_index=0)`.
- **Optimizer:** Adam, learning rate `3e-5`.
- **Execution:** 100 epochs, batch size 8 — a much longer training run than the other notebooks, appropriate for the very small (16-token) max sequence length and small dataset.

---

## Model Saving

Weights are saved under a `models/` directory. Not stored in this repository due to size — regenerate by running the notebook end-to-end.
