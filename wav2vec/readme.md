# Wav2Vec2

This part of the project predicts **continuous valence & arousal** from music using **Wav2Vec 2.0** features.\
This part include the following files:

- **`wav2vec_model.ipynb`** — the end‑to‑end workflow: data loading, feature use, model training, and evaluation.
- **`data_utils.py`** — minimal DEAM access helper (download via KaggleHub and serve raw audio waveforms).
- **`data_utils_feats.py`** — thin wrapper for **pre‑extracted Wav2Vec features** + DataLoader utilities. Expects a `DEAM_feats/` folder with `annotations.csv` and `songs/<song_id>.pt`.

The notebook uses **pre-extracted Wav2Vec features from the first 10 layers**. 
Two separate regression heads are trained — one for **valence** prediction and one for **arousal** — using **MSE loss** and **AdamW** with configurable learning-rate schedules.


---

## Overview (Wav2Vec2)

- **Input**: ~45 s audio clips (or features of the audio extracted using wav2vec2's first 10 layers)
- **Features**: Wav2Vec 2.0 hidden states at **16 kHz**.
- **Backbone**: Pretrained **Wav2Vec 2.0** encoder (frozen or fine-tuned on the last layer or two last layers).
- **Heads**: Two separate regression heads (valence, arousal) on pooled sequence features.
- **Loss**: **MSE** + variance penalty (encourage prediction spread) + correlation penalty (discourage valence–arousal coupling).
- **LR Schedule**: `CyclicLR` with `mode="triangular2"` (optimizer: AdamW).
- **Reproducibility**: Global seeding (Python/NumPy/PyTorch) and deterministic, fixed-ID splits.

---

## Running the Pipline

Once `data_utils.py` and `data_utils_feats.py` is in the folder, you can run the notebook—no further setup is required.

The notebook has **two main parts**:

1. **Feature extraction** – Uses Wav2Vec to generate pre-extracted features from the first 10 layers for each audio track.\
resulted folder:
```text
DEAM_feats/
├─ annotations.csv
└─ songs/
   ├─ 1.pt
   ├─ 2.pt
   └─ ...  # each is FloatTensor[T, D]
```
   - You can run this step once and **save the extracted features** (in `DEAM_feats/`) for later use.  
   - This allows you to skip the expensive audio processing step in future runs — simply load the saved features and focus on training.  
   - Useful if you want to try different model hyperparameters without re-running the extraction every time.

2. **Model training** – Uses the features to train two separate regression heads: one for **valence** and one for **arousal**.  
   - fine-tune the last two Wav2Vec layers (configurable, we trained only the last layer).  
   - visualizations of the prediction against the true labels

---

## Helpers

- **`data_utils.py`**
  - `DEAMHandler`: downloads DEAM, locates audio, loads static annotations (labels).
  - `get_waveform(song_id, ...)`: loads `<song_id>.mp3`.

- **`data_utils_feats.py`**
  - `DEAMFeatHandler`: loads `annotations.csv`, verifies `songs/<id>.pt` exist, and returns PyTorch dataloaders.
  - `FeatDataset`: yields `(features[T, D], label[2])` per song.  

---

## Folder layout
```text
├─ wav2vec_model.ipynb              # main notebook
├─ data_utils.py                    # raw‑audio helper (KaggleHub + librosa)
├─ data_utils_feats.py              # pre‑computed feature helper + loaders
└─ DEAM_feats/                      # (optional) precomputed features
   ├─ annotations.csv
   └─ songs/*.pt
```

## Results
After mutiple experiments we got the best results for the following configuration:
```python
EPOCHS = 20
LR  = 3e-5
MAX_LR = 1e-3  # for cyclic scheduler
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
LAMBDA_CORR = 0.05  # correlation penalty
LAMBDA_VAR  = 0.05  # variance penalty
GRAD_CLIP = 1.0
WeightLoss = False  # weight the loss to treat imbalance - does not work well
FINETUNE = "last1"   # options: "frozen", "last1", "lastN", "all"
```

**MSE vs. Epochs:**  

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/008f4953-c8f0-4d45-b56e-a4e4ce188db2" />


**Valence Arousal Scatter:**

<img width="554" height="455" alt="image" src="https://github.com/user-attachments/assets/b1716862-d4be-46e5-b6dc-a8a37beb746b" />


**Confusion Matrix:**

<img width="581" height="503" alt="image" src="https://github.com/user-attachments/assets/52faaf6f-d9c1-4eb5-829f-10425fd51310" />

