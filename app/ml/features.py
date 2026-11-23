"""Feature engineering shared by training and inference.

Target: `class_group`, the functional category derived from the recorded Space Class.

Inputs are the physical, administrative and occupant profile of a room. Fields that are
*derived from* the Space Class in the export are deliberately excluded to prevent label
leakage: Seat Stat, SSA and Avble Seats (only workpoints have them), W/S Area (workstations
only) and spaceStandard (a finer label of the same concept).

The encoder is fitted on the training split only; categories unseen in training map to
<UNK>, numeric scaling uses training statistics.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import OCCUPANT_GROUPS

CATEGORICAL = ["building", "floor", "allocation_primary", "room_pattern", "room_suffix"]
NUMERIC = ["log_area", "area_missing", "allocation_count", "room_number_mod100",
           "log_occ_total", *[f"log_occ_{g}" for g in OCCUPANT_GROUPS]]
UNK, MISSING = "<UNK>", "<MISSING>"


def room_pattern(room: str) -> str:
    """Shape of the room identifier, e.g. '420-06' -> '999-99', 'W114' -> 'A999', '001A' -> '999A'."""
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", str(room)))[:8]


def derive_columns(spaces: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=spaces.index)
    df["building"] = spaces["building"].fillna(MISSING).astype(str)
    df["floor"] = spaces["floor"].fillna(MISSING).astype(str)
    df["allocation_primary"] = spaces["allocation_primary"].fillna(MISSING).astype(str)
    room = spaces["room"].fillna("").astype(str)
    df["room_pattern"] = room.map(room_pattern)
    df["room_suffix"] = room.str.extract(r"([A-Za-z]+)$")[0].fillna("none").str.upper().str[:2]
    area = pd.to_numeric(spaces["area"], errors="coerce")
    df["log_area"] = np.log1p(area.fillna(0).clip(lower=0))
    df["area_missing"] = area.isna().astype(float)
    df["allocation_count"] = spaces["allocation_count"].clip(lower=0, upper=5).astype(float)
    digits = room.str.extract(r"(\d+)")[0].fillna("0").astype(int)
    df["room_number_mod100"] = (digits % 100).astype(float)
    df["log_occ_total"] = np.log1p(spaces["occ_total"].astype(float))
    for g in OCCUPANT_GROUPS:
        df[f"log_occ_{g}"] = np.log1p(spaces[f"occ_{g}"].astype(float))
    return df


@dataclass
class FeatureEncoder:
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)

    def fit(self, derived: pd.DataFrame, labels: pd.Series, min_count: int = 5,
            cat_cols: list[str] | None = None, num_cols: list[str] | None = None) -> "FeatureEncoder":
        """Fit vocabularies and scaling on training rows. Column subsets are used by the ablation study."""
        for col in (CATEGORICAL if cat_cols is None else cat_cols):
            counts = derived[col].value_counts()
            keep = sorted(counts[counts >= min_count].index)
            self.vocab[col] = {UNK: 0, **{v: i + 1 for i, v in enumerate(keep)}}
        for col in (NUMERIC if num_cols is None else num_cols):
            self.mean[col] = float(derived[col].mean())
            self.std[col] = float(derived[col].std() or 1.0)
        self.classes = sorted(labels.unique())
        return self

    def transform(self, derived: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(derived)
        cat = np.stack([derived[c].map(self.vocab[c]).fillna(0).astype(int).to_numpy()
                        for c in self.vocab], axis=1) if self.vocab else np.zeros((n, 0), np.int64)
        num = np.stack([((derived[c] - self.mean[c]) / self.std[c]).to_numpy(dtype=np.float32)
                        for c in self.mean], axis=1) if self.mean else np.zeros((n, 0), np.float32)
        return cat.astype(np.int64), num.astype(np.float32)

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        index = {c: i for i, c in enumerate(self.classes)}
        return labels.map(index).to_numpy(dtype=np.int64)

    @property
    def cardinalities(self) -> list[int]:
        return [len(v) for v in self.vocab.values()]

    def to_json(self) -> str:
        return json.dumps({"vocab": self.vocab, "mean": self.mean, "std": self.std, "classes": self.classes})

    @classmethod
    def from_json(cls, text: str) -> "FeatureEncoder":
        d = json.loads(text)
        return cls(d["vocab"], d["mean"], d["std"], d["classes"])
