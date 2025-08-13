# Music Emotion Regression with MambaVision

<img width="1339" height="312" alt="image" src="https://github.com/user-attachments/assets/98b829bd-f84f-4590-b70b-5710681e9515" />

## Table of Contents
- [Overview](#Overview)
- [Training Configuration](#Training-Configuration)
- [Project Structure](#Project-Structure)
- [Setup](#Setup)
- [Folder structure](#Folder-structure)
- [Data & Annotations](#Data--Annotations)
- [Precompute Mel-Spectrogram Chunks](#Precompute-Mel-Spectrogram-Chunks)
- [Dataset Wrapper](#Dataset-Wrapper)
- [Hyperparameters](#Hyperparameters)
- [Results](#Results)


---

## Overview

Predict continuous **valence** and **arousal** from audio using a **MambaVision** backbone and vision-style training on log-mel spectrogram “images”. This project includes a reproducible preprocessing pipeline, dataset wrapper, dual-head regressor, cyclic learning rate scheduling, and correlation/variance regularization.

---

## Training Configuration
- **Input**: ~45 s audio clips → 6 evenly spaced **5 s** chunks.
- **Features**: Log-mel spectrogram, normalized per sample, resized to **3 × 224 × 224** (RGB-like) for vision backbones.
- **Backbone**: `mamba_vision_S` (pretrained).
- **Heads**: Two separate regression heads (valence, arousal).
- **Loss**: MSE + variance penalty (promote spread) + correlation penalty (reduce valence–arousal coupling).
- **LR Schedule**: `CyclicLR` (triangular2).
- **Reproducibility**: Global seeding and deterministic splits.

---

## Project Structure (key components)

- **Mel cache creation**: precomputes per-song spectrogram chunks to `mel_cache/{song_id}.pt`.
- **`WaveToVisionWrapper`**: wraps a waveform dataset; loads cached chunks and normalized labels.
- **Custom collate**: flattens song-level chunks into a batch while tracking song IDs.
- **Model**: `MambaEmotionRegressor` with MambaVision backbone and two heads.
- **Train/Eval**: epoch loop with best model checkpointing; evaluation aggregates predictions to the **song level**.

---

## Setup

make sure you have data_utils file. Once `data_utils.py` is in the folder, you can run the notebook—no further setup is required.

### Install dependencies

```bash
#install requirements for mambavision
# a. Get system build deps
!apt-get update && \
 apt-get install -y build-essential cmake libomp-dev

# b. Pre-install ninja (so the wheel build uses it)
!pip install --upgrade ninja

# c. Build & install mamba-ssm against your existing torch/CUDA
!pip install --no-build-isolation mamba-ssm==2.2.4

# d. Now install mambavision
!pip install mambavision==1.1.0
```

> **Note:** The notebook already installs this dependecies

---

## Folder structure
```text
MambaVisionModel.ipynb
data_utils.py
mel_cache/
   ├─ 1.pt
   ├─ 2.pt
   └─ ... 
```
---

## Data & Annotations

Uses minimal DEAM access helper (download via KaggleHub and serve raw audio waveforms):

**`data_utils.py`**
  - `DEAMHandler`: downloads DEAM, locates audio, loads static annotations (labels).
  - `get_waveform(song_id, ...)`: loads `<song_id>.mp3`.
---

## Precompute Mel-Spectrogram Chunks

This step creates the mel cache used during training and evaluation.

Key parameters (must match your dataset wrapper for consistency):
- `sr=16000`, `duration=45`, `n_fft=2048`, `hop_length=512`, `win_length=2048`, `n_mels=256`
- 6 chunks × 5 s each, resized to **224×224** and **3 channels**.


---

## Dataset Wrapper

`WaveToVisionWrapper` loads cached tensors and returns `(chunks, labels, sid)` for each song:
- `chunks`: `[num_chunks, 3, 224, 224]`
- `labels`: `[num_chunks, 2]` (same normalized label replicated per chunk)
- `sid`: song ID

Normalization is computed **only on train+val** and applied consistently to all splits.

---


## Hyperparameters

```python
BATCH_SIZE      = 16
max_LR          = 1e-3
min_LR          = 1e-5
LR              = 1e-4
EPOCHS          = 50
weight_decay    = 1e-4
var_lambda = 0.05
corr_lambda = 0.05
train_augmentations = transforms.Compose([
    transforms.RandomApply([T.TimeMasking(time_mask_param=30)], p=0.5),
    transforms.RandomApply([T.FrequencyMasking(freq_mask_param=15)], p=0.5)
])
```

---

## Note

- You can switch to a different MambaVision variant by adjusting `create_model('mamba_vision_S', ...)`.

---

## Results

**MSE vs. Epochs:**  

<img width="751" height="452" alt="image" src="https://github.com/user-attachments/assets/2305ad94-2e59-46a2-8328-75f551aed9c8" />

**Valence Arousal Scatter:**

<img width="726" height="593" alt="image" src="https://github.com/user-attachments/assets/2ac0f42e-7772-4cc7-86dc-2fb9df9ebd79" />

**Confusion Matrix:**

<img width="662" height="576" alt="image" src="https://github.com/user-attachments/assets/2e0b3a0f-7b6d-4e74-811e-54a6370e59be" />

