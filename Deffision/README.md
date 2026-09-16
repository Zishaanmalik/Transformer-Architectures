# Diffusion-Transformer — Text-Conditioned Image Generation (UNet + Cross-Attention)

## Overview

A from-scratch **text-conditioned image diffusion model**: a Transformer text encoder produces contextual embeddings from a caption, and a convolutional **UNet with cross-attention** learns to predict the Gaussian noise added to an image at a random diffusion timestep, conditioned on those text embeddings. This is a DDPM-style (Denoising Diffusion Probabilistic Model) system, built entirely from primitive `nn.Module` and `nn.functional` operations — no `diffusers`, no pretrained UNet, no pretrained text encoder.

**Dataset:** Flickr30k (Kaggle: `srinivasac/flickr30k-dataset`) — images paired with captions.

```
Text:   Tokenize → Embedding → Pos → Transformer Encoder → text_tokens (B, L, D)
Image:  Resize → Normalize → image (B, 3, H, W) → Sample timestep t → Add Noise → noisy_image x_t

Time:   t → Time Embedding (sinusoidal + MLP) → time_tokens (B, D)

UNet:   Conv In (x_t)
        → Down Blocks (Encoder: ResBlock + Cross-Attn + Downsample, saving skip connections)
        → Mid Block   (ResBlock + Cross-Attn + ResBlock)
        → Up Blocks   (Decoder: Upsample + skip-concat + ResBlock + Cross-Attn)
        → Conv Out

Output:    predicted_noise ε̂ (B, 3, H, W)
Training:  Loss = MSE(ε̂, ε)
Inference: x_T ~ N(0, I) → iterative denoising (t → 0) → generated image x_0
```

---

## Data Pipeline

### Caption preprocessing

1. Read `captions.txt` (CSV: `image_name, caption`) with Python's `csv` module.
2. Build a **manual word-frequency vocabulary** (no external tokenizer library): every unique whitespace-split word becomes a vocabulary entry.
3. Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>` (indices 0–3).
4. `encode_caption(caption, word2idx, max_len)`: lowercases, splits on whitespace, prepends `<SOS>`, appends `<EOS>`, maps unknown words to `<UNK>`, pads/truncates to `max_len` (25 in the notebook).
5. `create_mask`: a binary attention mask — `0` at `<PAD>` positions, `1` elsewhere.

### Image preprocessing

```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),                                   # → [0, 1]
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # → [-1, 1] (required for diffusion)
])
```

Images are resized to a fixed `128×128` and rescaled to `[-1, 1]`, the standard input range for diffusion models (so that Gaussian noise added at `x_1` and the clean image at `x_0` live on a comparable scale).

### `FlickrDataset`

```python
class FlickrDataset(Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(Image.open(row["image_path"]).convert("RGB"))
        caption = torch.tensor(row["encoded_caption"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.long) if "attention_mask" in row else (caption != 0).long()
        return {"image": image, "caption": caption, "mask": mask}
```

`DataLoader(batch_size=1, shuffle=True, pin_memory=True)`.

---

## Architectural Configuration

| Parameter | Value (as instantiated for training) |
|---|---|
| Text Embedding Dimension | 1,024 |
| Text Encoder Layers | 8 |
| Text Encoder Attention Heads | 8 |
| Text Vocabulary Size | `len(word2idx)` (built from the Flickr30k captions) |
| Max Caption Length | 25 |
| UNet Base Channel Width | 256 (also tested at 512 and 128 in sanity-check cells) |
| Time Embedding Dimension | 1,024 |
| UNet Cross-Attention Heads | 8 |
| UNet Depth | 2 down-blocks / 1 mid-block / 2 up-blocks |
| Diffusion Timesteps (`num_timesteps`) | 1,000 |
| Beta Schedule | Linear, `1e-4 → 0.02` |
| Image Size | 128 × 128 (also tested at 256×256 in shape-check cells) |
| Optimizer | Adam |
| Learning Rate | 5e-6 |
| Batch Size | 1 |
| Epochs | up to 200 (run in multiple stages: 100, then +200) |
| Loss Function | MSE (between predicted and true noise) |

---

## Architecture — Text Side (Transformer Encoder)

### 1. `TextEmbedding` — Token Embedding + Sinusoidal Positional Encoding

```python
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))

    def forward(self, x):
        emb = self.token_embedding(x)                              # (B, T, D)
        emb = emb + self.pos_encoding[:, :x.size(1), :].to(emb.device)
        return emb
```

Same sinusoidal-positional-encoding pattern used throughout this repository.

### 2. `MultiHeadSelfAttention` — Standard Bidirectional Multi-Head Attention

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.head_dim = emb_dim // num_heads
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask=None):
        Q = self.q_proj(x).view(B,T,H,hd).transpose(1,2)
        K = self.k_proj(x).view(B,T,H,hd).transpose(1,2)
        V = self.v_proj(x).view(B,T,H,hd).transpose(1,2)
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.head_dim)   # (B,H,T,T)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)         # (B,T) → (B,1,1,T): broadcast over heads & query positions
            scores = scores.masked_fill(mask == 0, float('-inf'))    # block attention TO padding tokens

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1,2).contiguous().view(B,T,D)
        return self.out_proj(out)
```

Separate Q/K/V linear projections (not fused), with an optional **padding mask** (not a causal mask — this encoder is bidirectional): positions marked `0` in the attention mask are excluded as *keys*, so real tokens never attend to `<PAD>` positions, but there is no restriction on attending to future tokens.

### 3. `FeedForward` — Position-wise MLP

`Linear(emb_dim → hidden_dim) → GELU → Linear(hidden_dim → hidden_dim) → GELU → Linear(hidden_dim → emb_dim)` — a **3-linear-layer** FFN (deeper than the single-hidden-layer FFNs used in the other notebooks in this repository).

### 4. `EncoderBlock` — Pre-Norm Transformer Block

```python
def forward(self, x, mask=None):
    x = x + self.attn(self.norm1(x), mask)   # self-attention sub-layer (padding-masked)
    x = x + self.ffn(self.norm2(x))           # feed-forward sub-layer
    return x
```

8 of these are stacked (`enc_layers=8` in the final `DiffusionTransformer` instantiation) to form the text encoder. The output, `text_tokens (B, L, D)`, is the conditioning signal fed into every cross-attention layer of the UNet.

---

## Architecture — Image/Diffusion Side (UNet)

### 5. `TimeEmbedding` — Sinusoidal Timestep Embedding

```python
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim*4), nn.GELU(), nn.Linear(emb_dim*4, emb_dim))
        self.emb_dim = emb_dim

    def forward(self, t):                                        # t: (B,) integer timesteps
        half_dim = self.emb_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=t.device) / half_dim)
        args = t[:, None].float() * freqs[None]                    # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, emb_dim): sinusoidal timestep embedding
        return self.mlp(emb)                                          # (B, emb_dim): projected through a small MLP
```

This is the classic **transformer-style sinusoidal timestep embedding** used in diffusion models (same functional form as positional encodings, but applied to the integer diffusion step `t` rather than a sequence position), passed through a 2-layer MLP (`GELU` in between) to give the network a learnable, non-linear way to condition every block on "how noisy is this image right now".

### 6. `ResBlock` — Time-Conditioned Convolutional Residual Block

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()
        self.time_mlp = nn.Linear(time_dim, out_channels)              # projects time embedding to per-channel bias
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))                        # Conv → GroupNorm → GELU
        t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)             # (B, out_ch) → (B, out_ch, 1, 1)
        h = h + t                                                          # inject timestep info as a per-channel additive bias, broadcast over H,W
        h = self.act(self.norm2(self.conv2(h)))                             # Conv → GroupNorm → GELU
        return h + self.res_conv(x)                                          # residual connection (1×1 conv if channel count changes)
```

This is the fundamental building block of the UNet: two `3×3` convolutions with `GroupNorm` (8 groups) and `GELU`, with the diffusion timestep embedding injected as an **additive bias broadcast across every spatial location** between the two convolutions — this is how the network knows how much noise to expect at the current step. A residual (skip) connection wraps the whole block, using a `1×1` convolution to match channel counts when `in_channels != out_channels`.

### 7. `CrossAttention` — Image Features Attend to Text Tokens

```python
class CrossAttention(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, img_dim)                    # project text tokens into the image's channel space
        self.attn = nn.MultiheadAttention(embed_dim=img_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, text_tokens):                                     # x: (B, C, H, W) feature map
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)                          # (B, H*W, C): flatten spatial map into a token sequence
        text_tokens = self.text_proj(text_tokens)                             # (B, L, img_dim)
        out, _ = self.attn(query=x_flat, key=text_tokens, value=text_tokens)   # PyTorch's built-in nn.MultiheadAttention
        out = out.permute(0, 2, 1).view(B, C, H, W)                             # reshape back to a feature map
        return out + x                                                            # residual connection
```

This is the mechanism that makes the model **text-conditioned**: at every resolution level of the UNet, the spatial feature map is flattened into a sequence of `H×W` "pixel tokens", and each one performs cross-attention against the 25 text tokens produced by the Transformer text encoder (query = image pixels, key/value = text). Unlike the from-scratch attention implementations elsewhere in this repository, this cross-attention module uses PyTorch's built-in `nn.MultiheadAttention` directly.

### 8. `DownBlock` — Encoder Stage (Resolution Halving)

```python
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_heads, text_dim=512):
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.attn = CrossAttention(img_dim=out_ch, text_dim=text_dim, num_heads=num_heads)
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb, text_tokens):
        x = self.res(x, t_emb)          # time-conditioned residual conv block
        x = self.attn(x, text_tokens)     # text cross-attention
        skip = x                            # SAVE for the matching UpBlock's skip connection
        x = self.downsample(x)                # stride-2 conv: halve H and W
        return x, skip
```

Each down-block does: ResBlock → CrossAttention → save skip connection → downsample (stride-2 conv, halving spatial resolution). The UNet has **2 down-blocks** (`base_ch → base_ch*2 → base_ch*4`).

### 9. `MidBlock` — Bottleneck

```python
class MidBlock(nn.Module):
    def forward(self, x, t_emb, text_tokens):
        x = self.res1(x, t_emb)         # ResBlock
        x = self.attn(x, text_tokens)     # CrossAttention
        x = self.res2(x, t_emb)             # ResBlock
        return x
```

At the UNet's lowest resolution / highest channel count (`base_ch * 4`), a `ResBlock → CrossAttention → ResBlock` sandwich processes the most compressed, most semantically-rich representation of the image.

### 10. `UpBlock` — Decoder Stage (Resolution Doubling with Skip Connections)

```python
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_heads, text_dim=512):
        self.upsample = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.attn = CrossAttention(img_dim=out_ch, text_dim=text_dim, num_heads=num_heads)

    def forward(self, x, skip, t_emb, text_tokens):
        x = self.upsample(x)                                                      # transposed conv: double H and W
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:])                           # safety net for odd input sizes
        x = torch.cat([x, skip], dim=1)                                              # concatenate with the matching DownBlock's saved feature map (classic UNet skip connection)
        x = self.res(x, t_emb)                                                         # ResBlock over the concatenated features
        x = self.attn(x, text_tokens)                                                    # CrossAttention
        return x
```

Each up-block: transposed-convolution upsampling (doubling spatial resolution) → concatenate with the corresponding down-block's saved skip feature map along the channel dimension → `ResBlock` (which absorbs the now-doubled channel count) → `CrossAttention`. The skip connections are what let the UNet recover fine spatial detail that was lost during downsampling — the decoder doesn't have to reconstruct everything from the compressed bottleneck alone.

### 11. `UNet` — Full Assembly

```python
class UNet(nn.Module):
    def __init__(self, base_ch=128, time_dim=512, num_heads=8, text_dim=512):
        self.time_emb = TimeEmbedding(time_dim)
        self.conv_in = nn.Conv2d(3, base_ch, 3, padding=1)

        self.down1 = DownBlock(base_ch,     base_ch * 2, time_dim, num_heads, text_dim)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim, num_heads, text_dim)

        self.mid = MidBlock(base_ch * 4, time_dim, num_heads, text_dim)

        self.up1 = UpBlock(base_ch * 4 + base_ch * 4, base_ch * 2, time_dim, num_heads, text_dim)
        self.up2 = UpBlock(base_ch * 2 + base_ch * 2, base_ch,     time_dim, num_heads, text_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.GELU()
        self.conv_out = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x, t, text_tokens):
        t_emb = self.time_emb(t)                                  # (B,) → (B, time_dim)
        x = self.conv_in(x)                                          # (B, 3, H, W) → (B, base_ch, H, W)

        x, skip1 = self.down1(x, t_emb, text_tokens)                   # → (B, base_ch*2, H/2, W/2), skip1 saved
        x, skip2 = self.down2(x, t_emb, text_tokens)                     # → (B, base_ch*4, H/4, W/4), skip2 saved

        x = self.mid(x, t_emb, text_tokens)                                # bottleneck, same shape

        x = self.up1(x, skip2, t_emb, text_tokens)                           # → (B, base_ch*2, H/2, W/2), concat with skip2
        x = self.up2(x, skip1, t_emb, text_tokens)                             # → (B, base_ch, H, W), concat with skip1

        x = self.conv_out(self.out_act(self.out_norm(x)))                        # → (B, 3, H, W): predicted noise ε̂
        return x
```

**Full data flow through the UNet, channel-by-channel (base_ch=128 example):**

```
Input image x_t:  (B, 3,      H,   W)
conv_in:           (B, 128,    H,   W)
down1 (res+attn):   (B, 256,    H,   W)   → downsample → (B, 256,  H/2, W/2)   [skip1 saved at 256ch, H,W]
down2 (res+attn):     (B, 512,   H/2, W/2)   → downsample → (B, 512,  H/4, W/4)   [skip2 saved at 512ch, H/2,W/2]
mid (res+attn+res):     (B, 512,   H/4, W/4)
up1: upsample→(B,256,H/2,W/2), concat skip2→(B,1024,H/2,W/2), res→(B,256,H/2,W/2)
up2: upsample→(B,128,H,W),     concat skip1→(B,256,H,W),     res→(B,128,H,W)
conv_out:                (B, 3,      H,   W)  ← same shape as the input noisy image
```

This is a genuine **UNet with skip connections**: a symmetric encoder-decoder convolutional architecture where every down-sampling stage's pre-downsample feature map is concatenated back into the corresponding up-sampling stage, and every stage is additionally conditioned on (a) the diffusion timestep via additive bias in each `ResBlock`, and (b) the input text caption via cross-attention in each block.

### 12. `DiffusionTransformer` — Full Text-to-Image Model

```python
class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size, text_emb_dim=512, enc_layers=4, enc_heads=8,
                 unet_base=128, time_dim=512, unet_heads=8):
        self.text_embed = TextEmbedding(vocab_size, text_emb_dim)
        self.encoder_stack = nn.ModuleList([EncoderBlock(text_emb_dim, enc_heads, text_emb_dim*4) for _ in range(enc_layers)])
        self.unet = UNet(base_ch=unet_base, time_dim=time_dim, num_heads=unet_heads, text_dim=text_emb_dim)

    def encode_text(self, text, mask=None):
        x = self.text_embed(text)
        for layer in self.encoder_stack:
            x = layer(x, mask)
        return x                              # (B, L, D)

    def forward(self, image, t, text, mask=None):
        text_tokens = self.encode_text(text, mask)     # Transformer text encoder
        noise_pred = self.unet(image, t, text_tokens)    # UNet predicts the noise in `image`, conditioned on text
        return noise_pred
```

The model trained in the notebook is instantiated as:

```python
my_model = DiffusionTransformer(
    vocab_size=vocab_size, text_emb_dim=1024, enc_layers=8, enc_heads=8,
    unet_base=256, time_dim=1024, unet_heads=8
)
```

---

## Noise Scheduler — `SimpleNoiseScheduler` (DDPM Forward Process)

```python
class SimpleNoiseScheduler:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps)        # linear noise schedule
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)          # ᾱ_t = ∏ (1 - β_i) for i ≤ t

    def add_noise(self, x0, noise, t):
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise   # closed-form q(x_t | x_0)
```

This implements the standard DDPM closed-form forward diffusion process: instead of iteratively adding noise `t` times, the noisy image at any timestep `t` can be computed in one step as `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε`, where `ᾱ_t` is the cumulative product of `(1 - β_i)` up to step `t`, and `β` follows a **linear schedule from 1e-4 to 0.02** across 1,000 timesteps.

---

## Training Pipeline

**Loop:** `train_diffusion(model, dataloader, optimizer, loss_fn, noise_scheduler, device, epochs)`

```python
for batch in dataloader:
    image, text, mask = batch["image"], batch["caption"], batch["mask"]
    t = torch.randint(0, noise_scheduler.num_timesteps, (B,), device=device)   # random timestep per sample
    noise = torch.randn_like(image)                                             # sample Gaussian noise
    x_t = noise_scheduler.add_noise(image, noise, t)                              # forward-diffuse the image to step t

    noise_pred = model(x_t, t, text, mask)                                          # UNet predicts the noise
    loss = loss_fn(noise_pred, noise)                                                 # MSE between predicted and true noise

    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

- **Loss:** `nn.MSELoss()` — the standard DDPM training objective (`ε`-prediction).
- **Optimizer:** Adam, learning rate `5e-6` (notably small, consistent with the numerical sensitivity of diffusion training).
- **Execution:** trained in stages — 100 epochs, then continued for another 200 epochs — batch size 1.

---

## Inference — Iterative Denoising

```python
@torch.no_grad()
def generate_image(model, image_size, text, mask, noise_scheduler, device, steps=30):
    text_tokens = model.encode_text(text, mask)             # encode the caption once
    x = torch.randn(1, 3, image_size, image_size).to(device)  # start from pure Gaussian noise x_T

    timesteps = torch.linspace(noise_scheduler.num_timesteps - 1, 0, steps, dtype=torch.long)

    for t in timesteps:
        noise_pred = model.unet(x, t.unsqueeze(0), text_tokens)
        alpha_t = noise_scheduler.alphas_cumprod[t]
        x = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()   # one-step denoising estimate of x_0 from x_t
    return x
```

Generation starts from pure noise `x_T ~ N(0, I)` and walks backward through a subsampled set of `steps` timesteps (e.g. 30, 100, or 5000 — several values are tried in the notebook), each time using the UNet's noise prediction to compute a cleaner estimate of the image, conditioned throughout on the fixed encoded caption. `show_image()` denormalizes the result from `[-1,1]`-ish back to a displayable `[0,1]` range and plots it with matplotlib.

> The notebook includes two versions of `generate_image` — one that re-derives `x_0` via the closed-form scheduler formula above (used as the primary implementation), and an experimental (`#EXP`) variant that simply lets the model's raw output overwrite `x` at each step. Both are present in the notebook and are documented here for transparency; the closed-form version is the mathematically standard DDPM-style update.

---

## Model Saving

```python
from pathlib import Path
root = Path('models'); root.mkdir(exist_ok=True)
torch.save(my_model, root / "Deffision_Image.pth")
```

Saved under `models/Deffision_Image.pth`. Not stored in this repository due to size — regenerate by running the notebook end-to-end.
