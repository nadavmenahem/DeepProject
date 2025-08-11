"""data_utils.py - minimal DEAM helper, robust audio-folder discovery

Example
-------
    from data_utils import DEAMHandler
    handler = DEAMHandler()
    y, sr = handler.get_waveform(23)
"""

from __future__ import annotations

import os
import pandas as pd
import kagglehub
import librosa
import torch
from torch.utils.data import Dataset



__all__ = ["DEAMHandler"]


class DEAMHandler:
    """Download DEAM via KaggleHub, load annotations, locate audio, serve waveforms."""

    _KAGGLE_ID = "imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music"

    _STATIC_CSV = (
        "DEAM_Annotations/annotations/annotations averaged per song/"
        "song_level/static_annotations_averaged_songs_1_2000.csv"
    )
    _DYN_AROUSAL_CSV = (
        "DEAM_Annotations/annotations/annotations averaged per song/"
        "dynamic (per second annotations)/arousal.csv"
    )
    _DYN_VALENCE_CSV = (
        "DEAM_Annotations/annotations/annotations averaged per song/"
        "dynamic (per second annotations)/valence.csv"
    )

    # ---------------------------------------------------------------------
    def __init__(self):
        print("Downloading DEAM via kagglehub …")
        self.path = kagglehub.dataset_download(self._KAGGLE_ID)
        print("Dataset root:", self.path)

        # locate audio directory (.mp3 or .wav) -----------------------------
        self.audio_dir = os.path.join(self.path, 'DEAM_audio', 'MEMD_audio')
        print("Audio directory:", self.audio_dir)

        # load annotation CSVs ---------------------------------------------
        self.static_annotations = self._load_csv(self._STATIC_CSV, "static annotations")
        self.dynamic_arousal = self._load_csv(self._DYN_AROUSAL_CSV, "dynamic arousal")
        self.dynamic_valence = self._load_csv(self._DYN_VALENCE_CSV, "dynamic valence")

        for df in (self.static_annotations, self.dynamic_arousal, self.dynamic_valence):
            df.columns = df.columns.str.strip()

        self._print_preview()

    # ---------------------------------------------------------------- util
    def _load_csv(self, rel_path: str, label: str) -> pd.DataFrame:
        full = os.path.join(self.path, rel_path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"{label} CSV not found at {full}")
        return pd.read_csv(full)

    def _print_preview(self):
        print("\nDataset Preview:")
        print(self.static_annotations.head())
        print("\nDataset Info:")
        print(self.static_annotations.info())
        print("\nSummary Statistics:")
        print(self.static_annotations.describe())

    # ---------------------------------------------------------------- api
    def get_waveform(
        self,
        song_id: int,
        verbose: bool = False, 
        *,
        target_sr: int | None = 16_000,
        mono: bool = True,
    ):
        """Return (waveform, sr) for the given DEAM `song_id`."""
        audio_path = os.path.join(self.audio_dir, f"{song_id}.mp3")
        y, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
        if verbose:
            print(f"Audio loaded: {len(y)} samples at {sr} Hz")
        return y, sr
    # --------------------------------------------------------- inside DEAMHandler


class WaveDataset(Dataset):
    def __init__(self, handler: DEAMHandler, id_list, target_sr=16_000, mono=True):
        self.h = handler
        self.ids = id_list
        self.target_sr = target_sr
        self.mono = mono

    def __len__(self):             return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        wav_np, _ = self.h.get_waveform(sid, target_sr=self.target_sr, mono=self.mono)

        # pull both valence_mean and arousal_mean from the static_annotations
        row = self.h.static_annotations.loc[
                  self.h.static_annotations["song_id"] == sid
              ].iloc[0]
        valence = row["valence_mean"]
        arousal = row["arousal_mean"]

        # return a 2‐D float label
        label = torch.tensor([valence, arousal], dtype=torch.float)

        return torch.from_numpy(wav_np).unsqueeze(0), label
