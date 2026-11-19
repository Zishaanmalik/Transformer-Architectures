# BSR Signals to Symbol — EEG Character Classification with a Conformer-Style Transformer

## Overview

A from-scratch **Conformer-style Transformer** that reads a raw multi-channel EEG recording of a person *imagining a letter* and classifies which letter it was. Unlike the encoder-decoder `BSR-VIT` folder (EEG → character-sequence text), this folder solves the simpler, cleaner problem of **signal → single symbol**: 5 classes (`A`, `K`, `N`, `P`, `Z`).

The folder contains **four notebooks**:

| Notebook | Role |
|---|---|
| `data_char.ipynb` | Builds the model-ready dataset from raw OpenBCI EEG recordings |
| `BSR-Char.ipynb` | Main model — **8 heads, 4 blocks** (baseline configuration) |
| `BSR-Char_head12.ipynb` | Identical code, **12 heads, 4 blocks** |
| `BSR-Char_stack8.ipynb` | Identical code, **8 heads, 8 blocks** |

The three model notebooks are **line-for-line identical** except for a single hyperparameter each (`NUM_HEADS` or `NUM_BLOCKS`), so the architecture is documented once below and the variants are compared at the end.

```
Raw EEG (15 channels, 125 Hz)
  → 156 windows × 8 samples × 15 channels
  → flatten each window to 120 features           X: (B, 156, 120)
  → InputBlock                                     (B, 156, 240)
  → N × TransformerBlock [FFN → CNN → MHSA → CNN → FFN]   (B, 156, 240)
  → mean over the 156 windows                       (B, 240)
  → MLP head                                         (B, 5)
```

---

## Dataset

**Source:** *EEG Alphabet Imagery Dataset* (Mendeley Data, `cp586sfzrt`, v3) — participants imagine writing a letter while EEG is recorded with an OpenBCI board.

| Property | Value |
|---|---|
| Participants | 20 |
| Classes (letters) | 5 — `A`, `K`, `N`, `P`, `Z` |
| Trials | 2,500 (20 participants × 5 letters × 25 trials) |
| Trials per class | 500 (perfectly balanced) |
| Raw recordings | 100 OpenBCI `.txt` files |
| Sampling rate | 125 Hz |
| Windows per trial | 156 |
| Samples per window | 8 (= 0.064 s) |
| Total windows (metadata rows) | 390,000 (2,500 × 156) |
| EEG channels used | 15 (`EXG Channel 0–15` excluding channel 13) |
| Features per window | 8 samples × 15 channels = **120** |

> Channel 13 (T8) is deliberately excluded in the code (`EEG_COLUMNS = [f"EXG Channel {i}" for i in range(16) if i != 13]`), leaving 15 channels.

---

## Part 1 — Data Creation (`data_char.ipynb`)

### 1.1 Metadata and path repair

The dataset ships a preprocessed `metadata.csv` (390,000 rows × 20 columns: `participant`, `session_label`, `label`, `class_index`, `trial`, `trial_uid`, `window_id`, `window_uid`, `fs`, `task_start_used`, `task_end_used`, `bad_channels`, `raw_path`, `sample_rejected_by_preproc`, and several `sample_*` quality statistics). Its `raw_path` column points at the original author's Linux machine, so a regex rewrites it to the local copy of the raw data:

```python
def fix_path(p):
    p = re.sub(
        r"^/home/naveed/Desktop/c3_picard_50HzNotch_FDF_epochs/extracted_data/TEST_SUBJECT_(\d+)/TEST_SUBJECT_\1/",
        lambda m: "F:\\eeg\\EEG_Alphabet_Imagery_Dataset\\raw data\\TEST_SUBJECT_" + m.group(1) + "\\",
        p
    )
    return p.replace("/", "\\")

df["raw_path2"] = df["raw_path"].apply(fix_path)
```

The notebook then asserts that required columns exist and prints sanity checks: 20 participants, 2,500 trials, 5 labels, **exactly 156 windows for every one of the 2,500 trials**, and 0 missing paths.

> Only the **timing** information (`task_start_used`) and the label come from the preprocessed metadata. The actual signal values are read from the **raw** recordings, not from the preprocessed epochs.

### 1.2 Timing investigation (debug cells)

Cells 12–19 are exploratory checks that justified the final extraction logic:

- Expected sample interval is `1/125 s`; the notebook inspects the actual timestamp intervals (min / max / mean / median) and how many fall outside a 6–10 ms band.
- It inspects the OpenBCI `Sample Index` counter (which rolls over modulo 256) for unexpected jumps.
- It checks that, within every trial, consecutive windows start **exactly 8 raw rows apart** (`all_differences_8`), and that using the nominal 125 Hz timeline (`round(task_start_used * 125)`) is consistent.

The outcome of this investigation is the final design: **ignore noisy per-row timestamps and index the raw rows with the nominal 125 Hz timeline**, advancing exactly 8 rows per window.

### 1.3 Reading raw EEG — `read_raw_eeg`

Two versions exist in the notebook; the **second one overrides the first** and is the one actually used.

| | Version 1 (superseded) | Version 2 (final) |
|---|---|---|
| Reads columns | 15 EEG columns + `Sample Index` | 15 EEG columns only |
| Counter check | Requires `diff(Sample Index) % 256 == 2` everywhere, else raises | Removed (investigation showed the counter check was too strict for these files) |
| Non-numeric / NaN | Raises | Raises (rows are never silently dropped, because that would shift row indices) |

```python
def read_raw_eeg(path):
    raw = pd.read_csv(path, skiprows=4, low_memory=False)   # OpenBCI header = 4 lines
    raw.columns = raw.columns.str.strip()                    # fix leading spaces in column names
    missing = [c for c in EEG_COLUMNS if c not in raw.columns]
    if missing: raise ValueError(...)
    for col in EEG_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if raw[EEG_COLUMNS].isna().any().any():                  # never drop rows silently
        raise ValueError(...)
    return raw[EEG_COLUMNS].to_numpy(dtype=np.float32)       # (n_rows, 15)
```

### 1.4 Windowing — `extract_all_trials`

For each raw file → for each trial in that file:

1. Sort by `window_id` and **assert** there are exactly 156 windows with IDs `1…156`.
2. **Assert** consecutive start times differ by exactly `8 / 125 = 0.064 s` (`np.allclose(..., atol=1e-6)`).
3. Compute the first raw row **once**: `first_idx = int(round(starts[0] * 125))`.
4. For each of the 156 windows take `eeg[idx : idx+8]` with `idx = first_idx + window_number * 8` (always advancing exactly 8 rows), assert shape `(8, 15)` and that all values are finite.
5. Flatten each `(8, 15)` window to a vector of **120** values (`window.reshape(120)`).

The windows of a trial are therefore **non-overlapping, back-to-back**, covering `156 × 8 = 1,248` samples ≈ **9.98 seconds** of EEG per trial.

### 1.5 Output files

| File | Shape | Layout |
|---|---|---|
| `training_data1.csv` | `(2500, 159)` | `participant`, `trial_uid`, `W001 … W156`, `label` — one row per **full trial**, each `Wxxx` cell is a JSON list of 120 floats |
| `training_data2.csv` | `(5000, 82)` | `participant`, `trial_uid`, `segment`, `W001 … W078`, `label` — each trial **split into two halves** of 78 windows (segment 1 = windows 1–78, segment 2 = 79–156) |

Both are verified after saving: every window has 120 values and the labels are balanced (500 per class in file 1, 1,000 per class in file 2).

> **Which file the models use:** all three model notebooks load **`training_data1.csv`** (the full 156-window trials). The `training_data2.csv` read cell exists in the model notebooks but is a *raw (non-executed) cell*, so it is not part of the executed pipeline.

> **No amplitude normalization** is applied anywhere. Values are raw OpenBCI counts (the visualized sample ranges from about −7,764 to 93,910).

---

## Part 2 — Model Notebooks

### 2.1 Data pipeline

```python
X = np.array([
    [json.loads(window) for window in row]
    for row in df.iloc[:, 2:-1].values        # columns W001…W156
], dtype=np.float32)                            # → (2500, 156, 120)
Y = df["label"].values                           # letters as strings
```

- **Split:** `train_test_split(test_size=0.2, random_state=42, stratify=Y)` → **2,000 train / 500 test** samples.
- The split is a stratified random split over trials, **not** a participant-wise split — trials from the same participant can appear in both train and test.
- **Label map** (inside `CustomDataset`): `A→0, K→1, N→2, P→3, Z→4`.
- **DataLoaders:** batch size 16; 125 training batches, 32 test batches; training set shuffled.
- **Seed:** `torch.manual_seed(43)` (+ CUDA seeds).
- **Input tensor:** each sample is a `(156, 120)` matrix: 156 time windows (the "sequence") × 120 EEG features (the "embedding input"). The notebook visualizes it as a heatmap (`Analyze`) and a 3D surface (`Analyze3D`).

### 2.2 Hyperparameters

| Parameter | `BSR-Char` (base) | `BSR-Char_head12` | `BSR-Char_stack8` |
|---|---|---|---|
| `INPUT_DIM` | 120 | 120 | 120 |
| `EMBED_DIM` | 240 | 240 | 240 |
| **`NUM_HEADS`** | **8** | **12** | **8** |
| **`NUM_BLOCKS`** | **4** | **4** | **8** |
| `OUTPUT_DIM` | 5 | 5 | 5 |
| Head dimension (`240 / heads`) | 30 | 20 | 30 |
| Dropout | 0.1 | 0.1 | 0.1 |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1e-5 | 1e-5 | 1e-5 |
| Loss | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| Batch size | 16 | 16 | 16 |
| Epochs | 100 | 100 | 100 |

---

## Architecture — Component by Component

### 1. `InputBlock` — Feature Embedding

```python
class InputBlock(nn.Module):
    def __init__(self, INPUT_DIM, EMBED_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, EMBED_DIM), nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU()
        )
    def forward(self, x):
        return self.layers(x)
```

A 2-layer MLP applied independently to every time window: `(B, 156, 120) → Linear(120→240) → GELU → Linear(240→240) → GELU → (B, 156, 240)`. This replaces both the token-embedding table and the patch-embedding convolution used in the other notebooks — the "token" here is one 8-sample × 15-channel window of EEG.

> **No positional encoding is added anywhere in this model.** Temporal order is only available to the network through the convolution layers (which are local and order-aware); the self-attention layer itself is order-agnostic.

### 2. `FFN` — Residual Feed-Forward Module

```python
class FFN(nn.Module):
    def __init__(self, EMBED_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(EMBED_DIM*2, EMBED_DIM*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(EMBED_DIM*2, EMBED_DIM)
        )
        self.norm = nn.LayerNorm(EMBED_DIM)
    def forward(self, x):
        return self.norm(x + self.layers(x))      # residual, then LayerNorm (post-norm)
```

`240 → 480 → 480 → 240` — a **3-linear-layer** MLP (one more hidden layer than a standard Transformer FFN), with GELU and dropout after each hidden layer. The result is added to the input and **then** normalized (post-norm residual). Note that, unlike the original Conformer paper's "half-step" macaron FFN, the residual here is full-weight (no ×0.5).

### 3. `CNN` — Residual 1-D Convolution Module (over time)

```python
class CNN(nn.Module):
    def __init__(self, EMBED_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(EMBED_DIM,   EMBED_DIM*2, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(EMBED_DIM*2, EMBED_DIM*2, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(EMBED_DIM*2, EMBED_DIM,   kernel_size=3, padding=1)
        )
        self.norm = nn.LayerNorm(EMBED_DIM)
    def forward(self, x):                      # x: (B, 156, 240)
        residual = x
        x = x.transpose(1, 2)                   # → (B, 240, 156): Conv1d wants channels first
        x = self.layers(x)                       # convolve ALONG THE TIME AXIS
        x = x.transpose(1, 2)                    # → (B, 156, 240)
        return self.norm(residual + x)           # residual + LayerNorm
```

Three stacked `Conv1d` layers (`240 → 480 → 480 → 240` channels, **kernel 3, padding 1**, so the sequence length stays 156). Each output time step therefore mixes information from its immediate neighbours, and the three stacked layers give a receptive field of **7 consecutive windows** (≈ 0.45 s). This is the **local temporal pattern extractor** of the block.

**How this differs from the `ConformerConvModule` in the ASR / BSR-VIT / Conformer folders:** that module uses pointwise conv → GLU → *depthwise* conv (kernel 31) → BatchNorm → SiLU → pointwise conv. The `CNN` module here is simpler and denser: ordinary (full-channel-mixing, non-depthwise) convolutions with GELU, no GLU, no BatchNorm, and a small kernel of 3.

### 4. `MHSA` — Multi-Head Self-Attention (PyTorch built-in)

```python
class MHSA(nn.Module):
    def __init__(self, EMBED_DIM, NUM_HEADS):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(EMBED_DIM)
    def forward(self, x):
        attention_output, _ = self.attention(x, x, x, need_weights=False)
        return self.norm(x + attention_output)
```

Bidirectional (unmasked) self-attention across all 156 windows: every window can attend to every other window, capturing **long-range** temporal dependencies across the ~10-second trial. This module uses PyTorch's built-in `nn.MultiheadAttention` (not a hand-written attention like the other folders). The head count is the only thing that changes between `BSR-Char` (8 heads → 30 dims each) and `BSR-Char_head12` (12 heads → 20 dims each); the number of parameters is identical because the projection matrices are the same size regardless of the head split.

### 5. `TransformerBlock` — The Conformer-Style Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, EMBED_DIM, NUM_HEADS):
        super().__init__()
        self.ffn1 = FFN(EMBED_DIM)
        self.cnn1 = CNN(EMBED_DIM)
        self.mhsa = MHSA(EMBED_DIM, NUM_HEADS)
        self.cnn2 = CNN(EMBED_DIM)
        self.ffn2 = FFN(EMBED_DIM)
    def forward(self, x):
        x = self.ffn1(x)
        x = self.cnn1(x)
        x = self.mhsa(x)
        x = self.cnn2(x)
        x = self.ffn2(x)
        return x
```

Five sub-modules in sequence — **FFN → CNN → MHSA → CNN → FFN** — a "sandwich" in which the attention layer sits in the middle, wrapped on both sides by a convolution and a feed-forward module. Every sub-module has its own residual connection and its own LayerNorm, so a block contains **5 residual connections and 5 LayerNorms**. This is a Conformer-inspired layout (feed-forward modules bracketing attention and convolution) with two convolution modules instead of one, and different internals as described above.

### 6. `OutputHead` — Pool and Classify

```python
class OutputHead(nn.Module):
    def __init__(self, EMBED_DIM, OUTPUT_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(EMBED_DIM, OUTPUT_DIM),
            nn.Softmax(dim=0)
        )
    def forward(self, x):
        x = x.mean(dim=1)              # (B, 156, 240) → (B, 240): average over time windows
        return self.layers(x)           # (B, 240) → (B, 5)
```

The 156 window representations are collapsed into one vector by **mean pooling over time**, then passed through a 2-layer MLP (`240 → 240 → 5`).

> **Implementation detail worth knowing:** the final `nn.Softmax(dim=0)` normalizes **across the batch dimension** (`dim=0`), not across the 5 classes (`dim=1`). Its output is then fed into `CrossEntropyLoss`, which itself applies a log-softmax over classes. This is how the code is written in all three notebooks. Consequences: (1) the loss starts near `ln(5) ≈ 1.609` (epoch-1 loss is 1.5974) and decreases slowly; (2) a prediction depends on the other samples in the same batch; (3) with a batch of exactly one sample (as in the `Predict()` helper, which calls `unsqueeze(0)`), the softmax over a size-1 batch is 1.0 for every class, so `argmax` returns class `0` regardless of the input.

### 7. `EEGTransformer` — Full Model

```python
class EEGTransformer(nn.Module):
    def __init__(self, INPUT_DIM=120, EMBED_DIM=240, NUM_HEADS=NUM_HEADS,
                 NUM_BLOCKS=NUM_BLOCKS, OUTPUT_DIM=5):
        super().__init__()
        self.input_block = InputBlock(INPUT_DIM, EMBED_DIM)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(EMBED_DIM, NUM_HEADS) for _ in range(NUM_BLOCKS)])
        self.output_head = OutputHead(EMBED_DIM, OUTPUT_DIM)
    def forward(self, x):
        x = self.input_block(x)          # (B, 156, 120) → (B, 156, 240)
        x = self.transformer_blocks(x)    # NUM_BLOCKS × TransformerBlock, shape preserved
        x = self.output_head(x)            # (B, 156, 240) → (B, 5)
        return x
```

---

## Parameter Statistics

Counts from the notebooks' own output (`sum(p.numel() for p in model.parameters())`):

| Component | Parameters |
|---|---|
| `InputBlock` | 86,880 |
| One `FFN` | 462,480 |
| One `CNN` | 1,384,080 |
| One `MHSA` | 231,840 |
| **One `TransformerBlock`** (2 FFN + 2 CNN + 1 MHSA) | **3,924,960** |
| `OutputHead` | 59,045 |

| Variant | Blocks | Total / Trainable Params | Parameter Tensors |
|---|---|---|---|
| `BSR-Char` (8 heads, 4 blocks) | 4 | **15,845,765** | 160 |
| `BSR-Char_head12` (12 heads, 4 blocks) | 4 | **15,845,765** | 160 |
| `BSR-Char_stack8` (8 heads, 8 blocks) | 8 | **31,545,605** | 312 |

All parameters are trainable. The head count does not change the parameter count; doubling the blocks adds `4 × 3,924,960 = 15,699,840` parameters. The convolution modules (2 per block) hold the largest share of each block's parameters.

---

## Training Pipeline

```python
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
```

- `Train(epochs, train_loader)`: standard loop (`zero_grad → forward → loss → backward → step`), tracking per-epoch loss and accuracy and printing progress every 25 batches. Run with **100 epochs**, GPU (`cuda`).
- `TrainGraph`: plots training accuracy and loss per epoch.
- **Evaluation** (`Test`): accuracy, macro precision, macro recall and macro F1 (scikit-learn), on both train and test loaders, followed by `PlotMetrics` (grouped bar chart) and `ConfusionMatrix` (labels `A K N P Z`).
- `Predict(data)`: classifies a single `(156, 120)` sample and prints the class index and character.

---

## Results (from the notebooks' saved outputs)

Final evaluation, macro-averaged, in eval mode (no dropout):

| Variant | Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **`BSR-Char`** (8 heads, 4 blocks) | Train | 97.05% | 97.08% | 97.05% | 97.03% |
| | **Test** | **97.60%** | 97.64% | 97.60% | 97.58% |
| **`BSR-Char_head12`** (12 heads, 4 blocks) | Train | 91.95% | 91.98% | 91.95% | 91.90% |
| | **Test** | **91.80%** | 92.06% | 91.80% | 91.71% |
| **`BSR-Char_stack8`** (8 heads, 8 blocks) | Train | 97.75% | 97.79% | 97.75% | 97.75% |
| | **Test** | **97.40%** | 97.51% | 97.40% | 97.40% |

Running training accuracy during the last (100th) epoch, with dropout active: base 96.70%, head12 90.75%, stack8 96.45%. The base model's training accuracy after epoch 1 was 33.35% (chance is 20%).

**Reading the comparison:**
- Going from 8 to **12 heads** (smaller 20-dim heads) at the same parameter count gave a clearly lower score (≈ 91.8% vs 97.6% test) under this training budget.
- Doubling depth to **8 blocks** (2× the parameters) did not improve the test score over the 4-block base (97.40% vs 97.60%).
- The 4-block, 8-head configuration is the best accuracy-per-parameter of the three.

---

## Model Saving

Each notebook ends with:

```python
root = Path('models'); root.mkdir(exist_ok=True)
path1 = root / "BSRT_char.pth"
# torch.save(model, path1)          # (raw cell in the notebook)
model = torch.load(path1, map_location=device, weights_only=False)
```

The whole model object is saved (not just the `state_dict`), hence `weights_only=False` at load time. All three notebooks use the same filename `BSRT_char.pth`, so rename the file per variant (e.g. `BSRT_char_head12.pth`, `BSRT_char_stack8.pth`) to avoid overwriting. Weights are not stored in this repository — regenerate them by running the notebooks.

---

## Reproduction Order

1. Download the dataset from Mendeley Data (`cp586sfzrt`, v3) and update the local paths at the top of `data_char.ipynb`.
2. Run `data_char.ipynb` → produces `transformer_dataset/training_data1.csv` and `training_data2.csv`.
3. Run any of `BSR-Char.ipynb`, `BSR-Char_head12.ipynb`, `BSR-Char_stack8.ipynb` (they read `transformer_dataset/training_data1.csv`).
