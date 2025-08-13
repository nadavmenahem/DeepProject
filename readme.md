# DEAM Emotion Regression Comparasion – Wav2Vec2 & Mamba Vision

<img width="1412" height="794" alt="image" src="https://github.com/user-attachments/assets/5f2430b8-9c72-4ad3-9020-b9f00d06bb24" />

## Table of Contents
- [Overview](#Overview)
- [Helpers](#helpers)
- [Dataset](#dataset)
- [Project Layout](#project-layout)
- [Goals](#goals)


## Overview

This project compares two deep learning approaches for **continuous emotion prediction** (valence & arousal) from music using the **[DEAM dataset](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music)**:

1. **Wav2Vec 2.0** — operates on raw audio waveforms with pre-trained self-supervised speech features.  
2. **Mamba Vision** — operates on spectrogram representations using a vision-oriented selective state-space model.

Both models share common dataset helpers but have independent training pipelines.

---

## Helpers

- **`data_utils.py`** — Downloads DEAM via KaggleHub, loads annotations, and serves audio waveforms.  
- **`data_utils_feats.py`** — Handles pre-extracted features (Wav2Vec embeddings) and provides padded PyTorch DataLoaders.

The DEAM dataset provides **static per-song valence/arousal** as labels.

---

## Dataset

- Consists 1744 tracks from various genres with labels.
- Labels: Valence & Arousal constracting 2D emotion space.

<img width="505" height="413" alt="image" src="https://github.com/user-attachments/assets/a2cdd32f-2a17-406e-b9bf-b773cf488e28" />

---

## Project Layout

```
.
├── wav2vec/
   ├── wav2vec_model.ipynb        # Wav2Vec2 training & evaluation
   ├── data_utils.py              # Common DEAM loader for raw audio
   ├── data_utils_feats.py        # Pre-computed feature loader
   └── readme_wav2vec.md          # Model-specific Wav2Vec2 README
├── MambaVision/
   ├── MambaVisionModel.ipynb     # Mamba Vision training & evaluation
   ├── data_utils.py              # Attached again for user comfort
   └── readme.md      # Model-specific Mamba Vision README
```

---


## Goals

- Benchmark **Wav2Vec2** vs **Mamba Vision** on the same dataset and labels.  
- Explore how **raw-waveform** vs **spectrogram** inputs impact emotion regression performance.  
- Provide reproducible pipelines with reusable dataset helpers.
