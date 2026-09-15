# OCR-VIT — Encoder-Decoder Transformer (Optical Character Recognition)

## Overview

This implementation builds a Vision Transformer (ViT) encoder / Transformer decoder model from first principles to transcribe text from images:

- Image Patch Embedding (Conv2d stack) + Learned Positional Embedding
- ViT Encoder (standard multi-head self-attention + feed forward, pre-norm)
- Character-level Text Embedding + Sinusoidal Positional Embedding
- Masked Self-Attention, Cross-Attention, and Feed Forward Decoder Blocks
- Vocabulary Projection Head

```
Image → PatchEmbed → Pos → ViT → image_tokens (B, 64, 768)
Text  → Embedding → Pos → text_tokens (B, 127, 768)
Decoder: Masked Self-Attn (text) → Cross-Attn (text ↔ image) → FF
Output → Linear → vocab
```

---

## Data Pipeline

- **Image:** image-text pair dataset (loaded via a HuggingFace-style dataset object), image tensor shape `(3, 64, 256)`.
- **Text:** character-level vocabulary built from `string.ascii_letters + digits + punctuation + " "`, with `<pad>`, `<eos>`, `<bos>` tokens added.
- **`OCRDataset`** returns `image (3, 64, 256)`, `input_ids`, `labels` (next-character shifted), `max_len=128`.
- **DataLoader:** `train_loader` / `test_loader`, `batch_size=8, shuffle=True`.

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Model Dimension (`embdim`) | 768 |
| Number of Attention Heads | 12 |
| Image Encoder (ViT) Depth | 6 |
| Text Decoder Depth | 6 |
| Feed Forward Hidden Dimension | 3,072 |
| Number of Image Patches | 64 |
| Max Text Length | 128 |
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Batch Size | 8 |
| Epochs | 1 |

---

## Architecture Flow

### 1. Image Patch Embedding

**Class:** `PatchEmbedding(in_channels=3, d_model=768)`

4-layer `Conv2d` + `GELU` stack, stride 2 each:

```
(B, 3, 64, 256) → Conv2d(3→64) → Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→768)
```

### 2. Image Positional Embedding

**Class:** `ImagePositionalEmbedding(num_patches=64, d_model=768)`

Learned positional parameter added directly to patch embeddings.

### 3. Text Embedding + Positional Encoding

**Class:** `TextPositionalEmbedding(vocab_size, embdim, max_len=512)`

`nn.Embedding` (padding_idx=0) + sinusoidal positional encoding.

### 4. ViT Encoder

**Classes:** `MultiHeadAttention`, `FeedForward`, `EncoderBlock`, `ViTEncoder`

Standard pre-norm transformer encoder block:

```
x = x + mha(norm1(x))
x = x + ff(norm2(x))
```

`ViTEncoder(depth=6, embdim=768, num_heads=12, ff_hidden_dim=3072)` stacks 6 blocks.

### 5. Decoder

**Classes:** `MaskedMultiHeadAttention`, `CrossAttention`, `FeedForward`, `DecoderBlock`, `Decoder`

```
x = text_tokens + self_attn(norm1(text_tokens))
x = x + cross_attn(norm2(x), image_tokens)
x = x + ff(norm3(x))
```

`Decoder(depth=6, embdim=768, num_heads=12, ff_hidden_dim=3072)` stacks 6 blocks.

### 6. Vocabulary Head

**Class:** `VocabHead(embdim, vocab_size)` — `Linear(768 → vocab_size)`

### 7. Full Model

**Class:** `OCRTransformer(vocab_size, embdim=768, img_depth=6, txt_depth=6, num_heads=12, ff_hidden_dim=3072, max_len=128)`

```
image_tokens = vit(img_pos(patch_embed(image)))
text_tokens  = text_emb(text)
decoder_out  = decoder(text_tokens, image_tokens)
logits       = vocab_head(decoder_out)
```

---

## Training Pipeline

**Loop:** `train_model(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)`

- Teacher forcing: decoder input = `labels[:, :-1]`, target = `labels[:, 1:]`
- **Loss Function:** `CrossEntropyLoss(ignore_index=0)`
- **Optimizer:** Adam, Learning Rate = 3e-4
- **Execution:** Epochs = 1, Batch Size = 8

---

## Model Saving

Weights are saved under a `models/` directory (exact filenames as configured in the final notebook cell). Not stored in the repository due to size — regenerate by running the notebook end-to-end.
