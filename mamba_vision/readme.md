# Mamba Vision on DEAM

This repository contains a Jupyter notebook that trains a **Mamba Vision** model to predict **valence** and **arousal** from music using the **DEAM** dataset. Audio is fetched via a small helper (`data_utils.py`), converted to model-ready inputs in the notebook (e.g., spectrograms), and optimized to regress continuous emotion targets.

> **Files**
>
> - `MambaVisionModel.ipynb` — end‑to‑end training & evaluation workflow
> - `data_utils.py` — compact utility for downloading DEAM with KaggleHub and serving waveforms / labels

---

## 1) Setup

### Install dependencies

```bash
# PyTorch: choose the command that matches your OS/CUDA from https://pytorch.org/get-started/locally/
# Example (CUDA 12.1 wheels):
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Audio / utilities
pip install librosa kagglehub pandas numpy matplotlib scikit-learn tqdm

# Mamba model dependencies (adjust if your notebook uses a specific implementation)
pip install mamba-ssm timm
```

> **Note:** The notebook may define the Mamba Vision backbone inline. If it imports a specific package instead, install that package accordingly.

---

## 2) Data: DEAM (Database for Emotion Analysis using Music)

You **do not need to download DEAM manually**. The helper will attempt to fetch it via **KaggleHub** the first time you run the notebook and cache it locally. You may be prompted to sign in to Kaggle and accept the dataset license/terms in your environment.

### What the helper does
- Downloads the dataset `imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music` via KaggleHub
- Locates the audio directory (MP3s)
- Loads the official annotation CSVs (static per‑song means, dynamic per‑second arousal/valence)
- Serves waveforms at a target sample rate (default **16 kHz**, **mono**)

### Labels
For training, the provided `WaveDataset` returns **[valence_mean, arousal_mean]** from the static per‑song annotations so you can regress both targets jointly.

---

## 3) Quickstart (minimal code)

```python
from data_utils import DEAMHandler, WaveDataset
from torch.utils.data import DataLoader

# 1) Download & index DEAM
handler = DEAMHandler()

# 2) Build a tiny dataset of song IDs
train_ids = [23, 24, 25, 26, 27]  # example IDs

# 3) Waveform dataset: returns (wave_tensor[C,T], label_tensor[2])
ds = WaveDataset(handler, train_ids, target_sr=16_000, mono=True)
dl = DataLoader(ds, batch_size=4, shuffle=True)

# 4) Iterate
for wav, y in dl:
    # wav: [B, 1, T], y: [B, 2] => (valence, arousal)
    # convert wav -> spectrograms / features -> feed to your Mamba Vision model
    break
```

Then open the notebook and run the cells top‑to‑bottom:

```bash
jupyter lab MambaVisionModel.ipynb
# or
jupyter notebook MambaVisionModel.ipynb
```

> The notebook handles feature extraction (e.g., spectrograms), model definition (Mamba Vision), training loop, evaluation, and plots.

---

## 4) Training Notes & Tips

- **Input Representation:** Common choices are log‑mel spectrograms (e.g., 64–128 mel bands, hop length ~10 ms). Convert waveforms to 2‑D “images” for the vision backbone.
- **Targets:** Normalize valence/arousal to `[0, 1]` or `[-1, 1]` consistently with your loss and evaluation.
- **Loss:** Start with MSE; optionally try Concordance Correlation Coefficient (CCC) loss.
- **Batching:** Long songs can be chunked into fixed‑length windows (e.g., 5–10 s) and pooled for song‑level predictions.
- **Metrics:** Report MSE/RMSE and Pearson/CCC for both valence and arousal.
- **Reproducibility:** Set RNG seeds (`torch`, `numpy`, `random`) and log hyperparameters.

---

## 5) Project Structure (suggested)

```
.
├── MambaVisionModel.ipynb
├── data_utils.py
├── README.md
├── env.yml / requirements.txt      # (optional) pin your environment
├── outputs/                        # (optional) checkpoints, logs, figures
└── src/                            # (optional) reusable modules
```

---

## 6) Troubleshooting

- **Kaggle/KaggleHub auth:** Ensure you can access the dataset with your Kaggle account and that `kagglehub` can authenticate in your environment.
- **Librosa backend:** If you see codec errors, install `ffmpeg` (system package) so MP3 decoding works.
- **CUDA:** Match your PyTorch build to the installed CUDA drivers. If you are on CPU‑only, install the CPU wheels from PyTorch.

---

## 7) Acknowledgments

- **DEAM:** Database for Emotion Analysis using Music and all original authors/uploader(s).
- **Mamba:** The Mamba/state‑space model authors and maintainers of open‑source Mamba implementations.
- **Libraries:** PyTorch, librosa, KaggleHub, and the broader Python ecosystem.

---

## 8) Citation

If you use this work or the dataset, please cite the appropriate sources (DEAM dataset and Mamba papers). Add your own paper/report here if applicable.

```bibtex
@misc{deam,
  title        = {Database for Emotion Analysis using Music (DEAM)},
  howpublished = {Dataset},
  year         = {2015}
}

@inproceedings{mamba2023,
  title        = {Mamba: Selective State Space Models},
  year         = {2023}
}
```

---

## 9) License

Specify the license for your code here (e.g., MIT). Note that **dataset license/terms are separate** and governed by the DEAM/Kaggle terms.
