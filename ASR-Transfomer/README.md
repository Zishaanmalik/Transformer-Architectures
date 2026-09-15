# ASR-Transformer — Encoder-Decoder Transformer (Audio Speech Recognition)

## Overview

This implementation builds a Conformer-encoder / Transformer-decoder speech-to-text model from first principles:

- Audio Patch Embedding (Conv2d stack over mel-spectrograms) + Learned Positional Embedding
- Conformer Encoder (macaron feed-forward + multi-head self-attention + convolution module)
- Text Embedding + Sinusoidal Positional Embedding
- Masked Self-Attention, Cross-Attention, and Feed Forward Decoder Blocks
- Vocabulary Projection Head

Training is performed using **teacher-forced sequence-to-sequence learning** on the **LibriSpeech ASR** dataset (`openslr/librispeech_asr`, `clean` config, `train.100[:30000]` split).

```
Audio → PatchEmbed → Pos → Conformer (Encoder) → audio_tokens (B, 64, 768)
Text  → Embedding → Pos → text_tokens (B, 127, 768)
Decoder: Masked Self-Attn (text) → Cross-Attn (text ↔ audio) → FF
Output → Linear → vocab
```

---

## Data Pipeline

- **Audio:** loaded with `librosa.load(path, sr=16000)` → `librosa.feature.melspectrogram` → `librosa.power_to_db` → transpose → zero-padded to fixed length. Output shape per sample: `(T_max, 80)`.
- **Text:** lowercased, custom word-level vocabulary built from the transcripts, encoded to integer IDs, and padded.
- **`ASRDataset`** returns `audio_features (T, 80)`, `decoder_input_ids (L)`, `labels (L)`.
- **DataLoader:** `batch_size=5, shuffle=True`.

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Model Dimension (`d_model`) | 512 |
| Number of Attention Heads | 8 |
| Encoder Layers (Conformer) | 6 |
| Decoder Layers | 6 |
| Feed Forward Hidden Dimension | 2048 |
| FF Expansion Factor (Conformer) | 4 |
| Conv Expansion Factor (Conformer) | 2 |
| Conv Kernel Size | 31 |
| Max Audio Length | 1,200 |
| Max Text Length | 518 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Batch Size | 5 |
| Epochs | 1 (loop supports configurable epochs) |

---

## Architecture Flow

### 1. Audio Patch Embedding

**Class:** `PatchEmbedding(d_model=512)`

4-layer `Conv2d` + `GELU` stack, stride 2 each, progressively downsampling the mel-spectrogram:

```
(B, 1, 3516, 80) → Conv2d(1→64) → Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→512)
```

Followed by `LayerNorm(512)`.

### 2. Audio Positional Embedding

**Class:** `AudioPositionalEmbedding(max_len=1200, d_model=512)`

Learned positional parameter added to patch embeddings, with dropout.

### 3. Text Embedding + Positional Encoding

**Class:** `TextPositionalEmbedding(vocab_size, embdim, max_len=512)`

`nn.Embedding` (padding_idx=0) + precomputed sinusoidal positional encoding, scaled by `sqrt(embdim)`.

### 4. Conformer Encoder

**Classes:** `ConformerConvModule`, `MultiHeadAttention`, `FeedForwardModule`, `ConformerBlock`, `ConformerEncoder`

Each `ConformerBlock` (macaron structure):

```
x = x + 0.5 * ff1(x)      # first half-step FFN
x = x + mha(x)             # multi-head self-attention
x = x + conv(x)             # depthwise conv module (GLU + BatchNorm + SiLU)
x = x + 0.5 * ff2(x)      # second half-step FFN
x = norm(x)
```

`ConformerEncoder(num_layers=6, d_model=512, num_heads=8)` stacks 6 such blocks.

### 5. Decoder

**Classes:** `MaskedMultiHeadAttention`, `CrossAttention`, `FeedForward`, `DecoderBlock`, `Decoder`

```
x = text_tokens + self_attn(norm1(text_tokens))          # causal self-attention
x = x + cross_attn(norm2(x), audio_tokens)                 # cross-attention to encoder output
x = x + ff(norm3(x))                                         # feed forward
```

`Decoder(depth=6, embdim=512, num_heads=8, ff_hidden_dim=2048)` stacks 6 such blocks.

### 6. Vocabulary Head

**Class:** `VocabHead(embdim, vocab_size)` — `Linear(512 → vocab_size)`

### 7. Full Model

**Class:** `ASRTransformer(vocab_size, d_model=512, encoder_layers=6, decoder_layers=6, num_heads=8, ff_expansion=4, conv_expansion=2, ff_hidden_dim=2048, max_audio_len=1200, max_text_len=518, dropout=0.1)`

```
audio_tokens = encoder(audio_pos(patch_embed(audio)))
text_tokens  = text_pos(text)
decoder_out  = decoder(text_tokens, audio_tokens)
logits       = vocab_head(decoder_out)
```

---

## Training Pipeline

**Loop:** `train_asrtransformer(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)`

- Teacher forcing: decoder input = `text[:, :-1]`, target = `labels[:, 1:]`
- **Loss Function:** `CrossEntropyLoss(ignore_index=0)`
- **Optimizer:** Adam, Learning Rate = 3e-4
- **Execution:** Epochs = 1, Batch Size = 5

---

## Model Saving

Weights are saved under a `models/` directory (exact filenames as configured in the final notebook cell). Not stored in the repository due to size — regenerate by running the notebook end-to-end.
