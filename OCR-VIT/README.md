# OCR-VIT — Encoder-Decoder Transformer (Optical Character Recognition)

## Overview

A from-scratch encoder-decoder model that transcribes text from images: a plain **Vision Transformer (ViT)** encoder (standard multi-head self-attention, no convolution module — unlike the Conformer used in the ASR/BSR/Conformer notebooks) processes image patches, and a causal Transformer decoder with cross-attention generates the output text character by character.

```
Image → PatchEmbed → Pos → ViT (Encoder) → image_tokens (B, 64, 768)
Text  → Embedding → Pos → text_tokens (B, T, 768)
Decoder: Masked Self-Attn (text) → Cross-Attn (text ↔ image) → FF
Output → Linear → vocab
```

---

## Data Pipeline

- **Source:** an image–text pair dataset (loaded as a HuggingFace-style dataset object with `"image"` and `"text"` fields).
- **Image:** `(3, 64, 256)` RGB tensor after transform.
- **Text:** character-level vocabulary from `string.ascii_letters + digits + punctuation + " "` plus `<pad>`, `<eos>`, `<bos>`.
- **`OCRDataset`:**

```python
class OCRDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, max_len=128):
        self.data = hf_dataset
        self.transform = transform
        self.max_len = max_len

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self.transform(sample["image"]) if self.transform else sample["image"]
        tokens = encode_text(sample["text"], self.max_len)
        return {
            "image": image,                          # (3, 64, 256)
            "input_ids": torch.tensor(tokens[:-1]),    # decoder input (shifted right)
            "labels": torch.tensor(tokens[1:])           # target (shifted left)
        }
```

---

## Architectural Configuration

| Parameter | Value |
|---|---|
| Model Dimension (`embdim`) | 768 |
| Number of Attention Heads | 12 |
| Head Dimension | 64 (768 / 12) |
| Image Encoder (ViT) Depth | 6 |
| Text Decoder Depth | 6 |
| Feed Forward Hidden Dimension | 3,072 |
| Number of Image Patches | 64 |
| Max Text Length | 128 |
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Batch Size | 8 |
| Loss ignore index | 0 (PAD) |

---

## Architecture — Component by Component

### 1. `PatchEmbedding` — Convolutional Patchification

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, d_model=768):
        super().__init__()
        self.proj = nn.Sequential(                        # input: (B, 3, 64, 256)
            nn.Conv2d(3,   64, kernel_size=3, stride=2, padding=1), nn.GELU(),   # → (B,64,32,128)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GELU(),   # → (B,128,16,64)
            nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1), nn.GELU(),   # → (B,256,8,32)
            nn.Conv2d(256,768, kernel_size=3, stride=2, padding=1),               # → (B,768,4,16)
        )
    def forward(self, x):
        x = self.proj(x)                       # (B, 768, 4, 16)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)         # → (B, 64, 768): 4×16=64 patch tokens
        return x
```

Rather than a single non-overlapping-patch `Conv2d(patch_size, stride=patch_size)` (the classic ViT patchify), this implementation uses the same 4-layer strided-convolution "patchify" pipeline seen throughout this repository: `3 → 64 → 128 → 256 → 768` channels, each layer halving H and W, ending at a `4×16 = 64`-token grid — matching the fixed `num_patches=64` used by the positional embedding below.

### 2. `ImagePositionalEmbedding` — Learned Positional Parameter

```python
class ImagePositionalEmbedding(nn.Module):
    def __init__(self, num_patches=64, d_model=768):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))
    def forward(self, x):
        return x + self.pos_embed     # (B, 64, 768) + (1, 64, 768)
```

### 3. `TextPositionalEmbedding` — Sinusoidal Position + Padding-Aware Embedding

`nn.Embedding(vocab_size, 768, padding_idx=0)` plus the standard precomputed sin/cos positional table, added together (no `sqrt(embdim)` scaling in this notebook's version, unlike the ASR/BSR text embedding).

### 4. ViT Encoder — `MultiHeadAttention`, `FeedForward`, `EncoderBlock`, `ViTEncoder`

**`MultiHeadAttention`** (standard, bidirectional, separate Q/K/V projections):

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        Q = Q.view(B,T,H,hd).transpose(1,2); K = ...; V = ...
        scores = (Q @ K.transpose(-2,-1)) / sqrt(head_dim)
        weights = softmax(scores, dim=-1)     # NO mask — full bidirectional attention over image patches
        out = (weights @ V).transpose(1,2).reshape(B,T,D)
        return self.out(out)
```

**`FeedForward`:** `Linear(768→3072) → GELU → Linear(3072→768)`.

**`EncoderBlock`** — a standard **pre-norm** Transformer encoder block:

```python
def forward(self, x):
    x = x + self.mha(self.norm1(x))
    x = x + self.ff(self.norm2(x))
    return x
```

**`ViTEncoder`** stacks 6 of these `EncoderBlock`s — a genuine, unmodified Vision Transformer encoder (no convolution module, no macaron FFN, unlike the Conformer encoders used elsewhere in this repository). This is the one notebook where the "encoder" really is a vanilla ViT.

### 5. Decoder — `MaskedMultiHeadAttention`, `CrossAttention`, `DecoderBlock`, `Decoder`

**`MaskedMultiHeadAttention`** — causal self-attention over generated characters (separate Wq/Wk/Wv, `torch.tril` boolean mask, `-inf` fill before softmax) — identical pattern to the ASR/BSR decoders.

**`CrossAttention`** — text queries attend to the 64 ViT image tokens:

```python
def forward(self, query, context):    # query: text (B,T,768), context: image (B,64,768)
    Q = self.Wq(query); K = self.Wk(context); V = self.Wv(context)
    scores = (Q @ K.transpose(-2,-1)) / sqrt(head_dim)   # (B, heads, T, 64)
    weights = softmax(scores, dim=-1)
    out = (weights @ V).transpose(1,2).reshape(B,T,D)
    return self.out(out)
```

**`DecoderBlock`**:

```python
def forward(self, text_tokens, image_tokens):
    x = text_tokens + self.self_attn(self.norm1(text_tokens))     # 1. causal self-attention over characters so far
    x = x + self.cross_attn(self.norm2(x), image_tokens)            # 2. cross-attention: "look at" the image
    x = x + self.ff(self.norm3(x))                                    # 3. feed-forward
    return x
```

`Decoder(depth=6, embdim=768, num_heads=12, ff_hidden_dim=3072)` stacks 6 such blocks.

### 6. `VocabHead` — `Linear(768 → vocab_size)`, applied per character position.

### 7. `OCRTransformer` — Full Model Assembly

```python
class OCRTransformer(nn.Module):
    def __init__(self, vocab_size, embdim=768, img_depth=6, txt_depth=6,
                 num_heads=12, ff_hidden_dim=3072, max_len=128):
        self.patch_embed = PatchEmbedding(in_channels=3, d_model=embdim)
        self.img_pos = ImagePositionalEmbedding(num_patches=64, d_model=embdim)
        self.vit = ViTEncoder(img_depth, embdim, num_heads, ff_hidden_dim)
        self.text_emb = TextPositionalEmbedding(vocab_size, embdim, max_len)
        self.decoder = Decoder(txt_depth, embdim, num_heads, ff_hidden_dim)
        self.vocab_head = VocabHead(embdim, vocab_size)

    def forward(self, image, text):
        img_tokens = self.vit(self.img_pos(self.patch_embed(image)))   # (B,3,64,256) → (B,64,768)
        text_tokens = self.text_emb(text)                                # (B,T) → (B,T,768)
        decoded = self.decoder(text_tokens, img_tokens)                    # cross-attends to img_tokens
        logits = self.vocab_head(decoded)                                    # (B,T,vocab_size)
        return logits
```

---

## Training Pipeline

**Loop:** `train_model(model, dataloader, optimizer, loss_fn, device, epochs, vocab_size)`

- **Teacher forcing:** decoder input = `labels[:, :-1]`, target = `labels[:, 1:]`.
- **Loss:** `CrossEntropyLoss(ignore_index=0)`.
- **Optimizer:** Adam, learning rate `3e-4`.
- **Execution:** 1 epoch, batch size 8.

---

## Model Saving

Weights are saved under a `models/` directory. Not stored in this repository due to size — regenerate by running the notebook end-to-end.
