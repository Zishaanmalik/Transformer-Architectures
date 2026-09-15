# BSR-VIT — Encoder-Decoder Transformer (Brain Signal Recognition)

## Overview

This implementation builds a Conformer-encoder / Transformer-decoder model from first principles to map brain-signal (EEG-style) spectrogram data to text sequences:

- Signal Patch Embedding (Conv2d stack) + Learned Positional Embedding
- Conformer Encoder (macaron feed-forward + multi-head self-attention + convolution module)
- Character-level Text Embedding + Sinusoidal Positional Embedding
- Masked Self-Attention, Cross-Attention, and Feed Forward Decoder Blocks
- Vocabulary Projection Head

> Despite the "VIT" suffix in the filename, the encoder implemented in this notebook is a **Conformer** encoder (identical block structure to the ASR and Conformer notebooks), not a plain Vision Transformer.

```
Signal → PatchEmbed → Pos → Conformer (Encoder) → signal_tokens (B, 6, 360)
Text   → Embedding → Pos → text_tokens (B, 127, 360)
Decoder: Masked Self-Attn (text) → Cross-Attn (text ↔ signal) → FF
Output  → Linear → vocab
```

---

## Data Pipeline

- **Signal input:** loaded from `CSV1.csv` / `CSV2.csv`, column-stacked into a combined dataframe. Samples are shaped `(16, 64, 256)`-style multi-channel spectrogram-like tensors (16 channels, e.g. EEG electrodes).
- **Text:** character-level vocabulary built from `string.ascii_letters + digits + punctuation + " "`, with `<pad>`, `<eos>`, `<bos>` tokens added.
- **`BSRDataset`** returns `image (16, 64, 256)`, `input_ids`, `labels` (next-character shifted).
- **DataLoader:** `batch_size=8, shuffle=True`.

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Model Dimension (`d_model`) | 360 |
| Number of Attention Heads | 6 |
| Encoder Layers (Conformer) | 6 |
| Decoder Layers | 6 |
| Feed Forward Hidden Dimension | 2048 |
| FF Expansion Factor (Conformer) | 3 |
| Conv Expansion Factor (Conformer) | 2 |
| Conv Kernel Size | 31 |
| Max Text Length | 16 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning Rate | 3e-5 |
| Batch Size | 8 |
| Epochs | 100 |

---

## Architecture Flow

### 1. Signal Patch Embedding

**Class:** `PatchEmbedding(in_channels=16, d_model=360)`

4-layer `Conv2d` + `GELU` stack, stride 2 each:

```
(B, 16, F, T) → Conv2d(16→64) → Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→360)
```

### 2. Signal Positional Embedding

**Class:** `ImagePositionalEmbedding(num_patches=150, d_model=360)`

Learned positional parameter added directly to patch embeddings.

### 3. Text Embedding + Positional Encoding

**Class:** `TextPositionalEmbedding(vocab_size, embdim, max_len=512)`

`nn.Embedding` (padding_idx=0) + sinusoidal positional encoding.

### 4. Conformer Encoder

**Classes:** `ConformerConvModule`, `MultiHeadAttention`, `FeedForwardModule`, `ConformerBlock`, `ConformerEncoder`

Same macaron structure as the ASR-Transformer notebook:

```
x = x + 0.5 * ff1(x)
x = x + mha(x)
x = x + conv(x)
x = x + 0.5 * ff2(x)
x = norm(x)
```

`ConformerEncoder(num_layers=6, d_model=360, num_heads=6)` stacks 6 blocks.

### 5. Decoder

**Classes:** `MaskedMultiHeadAttention`, `CrossAttention`, `FeedForward`, `DecoderBlock`, `Decoder`

```
x = text_tokens + self_attn(norm1(text_tokens))
x = x + cross_attn(norm2(x), signal_tokens)
x = x + ff(norm3(x))
```

`Decoder(depth=6, embdim=360, num_heads=6, ff_hidden_dim=2048)` stacks 6 blocks.

### 6. Vocabulary Head

**Class:** `VocabHead(embdim, vocab_size)` — `Linear(360 → vocab_size)`

### 7. Full Model

**Class:** `BSRTransformer(vocab_size, d_model=360, encoder_layers=6, decoder_layers=6, num_heads=6, ff_expansion=3, conv_expansion=2, ff_hidden_dim=2048, max_text_len=16, dropout=0.1)`

```
signal_tokens = encoder(signal_pos(patch_embed(signal)))
text_tokens   = text_pos(text)
decoder_out   = decoder(text_tokens, signal_tokens)
logits        = vocab_head(decoder_out)
```

---

## Training Pipeline

**Loop:** `train_asrtransformer(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)` (shared naming with the ASR notebook; adapted for signal→text inputs)

- **Loss Function:** `CrossEntropyLoss(ignore_index=0)`
- **Optimizer:** Adam, Learning Rate = 3e-5
- **Execution:** Epochs = 100, Batch Size = 8

---

## Model Saving

Weights are saved under a `models/` directory (exact filenames as configured in the final notebook cell). Not stored in the repository due to size — regenerate by running the notebook end-to-end.
