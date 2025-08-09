# data_utils_feats.py
from __future__ import annotations

import os, torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

__all__ = ["DEAMFeatHandler", "FeatDataset", "feat_collate"]

# ──────────────────────── Dataset ────────────────────────────
class FeatDataset(Dataset):
    """
    Yields (feat, label) for a single song_id.
        feat   : FloatTensor, shape = [T, D]         (variable-length sequence)
        label  : FloatTensor, shape = [2]            (valence_mean, arousal_mean)
    """

    def __init__(self, handler: "DEAMFeatHandler", ids):
        self.h, self.ids = handler, ids
        # cache labels in a dict for speed
        df = handler.static_annotations.set_index("song_id")
        self._labels = {
            sid: (row.valence_mean, row.arousal_mean) for sid, row in df.iterrows()
        }

    def __len__(self):                       return len(self.ids)
    def __getitem__(self, idx):
        sid        = self.ids[idx]
        feats      = self.h.get_features(sid)        # [T, D]
        val, aro   = self._labels[sid]
        label      = torch.tensor([val, aro], dtype=torch.float32)
        return feats, label


# ──────────────────────── Handler ────────────────────────────
class DEAMFeatHandler:
    """
    Thin wrapper around your *DEAM_feats* folder:

        DEAM_feats/
            annotations.csv
            songs/
                <song_id>.pt      # torch.FloatTensor, shape [T, D]

    """

    def __init__(self, *, feats_root: str):
        if not os.path.isdir(feats_root):
            raise FileNotFoundError(f"folder not found: {feats_root!r}")

        self.feat_dir = os.path.join(feats_root, "songs")
        if not os.path.isdir(self.feat_dir):
            raise FileNotFoundError(f"'songs/' directory missing in {feats_root}")

        csv_path = os.path.join(feats_root, "annotations.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"annotations.csv not found inside {feats_root}"
            )

        # load annotations
        self.static_annotations = pd.read_csv(csv_path)
        self.static_annotations.columns = self.static_annotations.columns.str.strip()

        print("✔ static annotations :", csv_path)
        print("✔ feature root       :", self.feat_dir)

    # ----------------------------------------------------------
    def get_features(self, song_id: int, *, verbose: bool = False):
        """
        Return FloatTensor[T, D] for given song_id.
        """
        path = os.path.join(self.feat_dir, f"{song_id}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"feature tensor missing: {path}")
        feats = torch.load(path, map_location="cpu").float()
        if verbose:
            print(f" loaded {path}  {tuple(feats.shape)}")
        return feats  # [T, D]

    # ----------------------------------------------------------
    def build_dataloaders(
        self,
        *,
        batch_size: int = 8,
        val_split: float = 0.15,
        test_split: float = 0.15,
        shuffle: bool = True,
        num_workers: int = 2,
    ):
        """
        Returns standard (train, val, test) loaders with padding-aware collate_fn.
        """
        ids = self.static_annotations["song_id"].tolist()

        # keep only ids that really have a .pt feature file
        ids = [sid for sid in ids if os.path.isfile(os.path.join(self.feat_dir, f"{sid}.pt"))]

        # stratify on binary valence (≥5) so splits have similar mood mix
        labels = (self.static_annotations["valence_mean"] >= 5).astype(int).tolist()

        train_ids, tmp_ids, train_lbl, tmp_lbl = train_test_split(
            ids, labels, test_size=val_split + test_split,
            stratify=labels, random_state=42
        )
        rel_val = val_split / (val_split + test_split)
        val_ids, test_ids = train_test_split(
            tmp_ids, test_size=1 - rel_val,
            stratify=tmp_lbl, random_state=42
        )

        # handy maker
        def _mk(ids_, shuf):
            ds = FeatDataset(self, ids_)
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuf,
                num_workers=num_workers, pin_memory=True,
                collate_fn=feat_collate, drop_last=True
            )

        return _mk(train_ids, shuffle), _mk(val_ids, False), _mk(test_ids, False)


# ────────────────────── collate function ──────────────────────
def feat_collate(batch):
    """
    Pads variable-length sequences so they form a tensor [B, L_max, D].

    Returns
    -------
    feats    : FloatTensor[B, L_max, D]
    lengths  : LongTensor[B]                (number of valid frames per sample)
    labels   : FloatTensor[B, 2]
    """
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    feats   = pad_sequence(feats, batch_first=True)   # pad with zeros
    labels  = torch.stack(labels)
    return feats, lengths, labels
