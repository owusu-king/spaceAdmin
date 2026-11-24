"""Model inference service used by the agents and the API.

Two prediction sources, deliberately kept apart:
* Out-of-fold (cross-fitted) probabilities for existing records. Every record is scored by
  a model that never saw its label, so a disagreement with the recorded class is a fair
  signal that the record may not reflect actual use.
* The deployed model (trained on the training split) for records not in the export, such
  as a hypothetical space description.

Outputs are MODEL PREDICTIONS (evidence class "inference"), never agent decisions.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from app.config import AGENTS, MODEL_DIR
from app.logging_setup import get_logger, timed
from app.ml.features import FeatureEncoder, derive_columns
from app.ml.model import SpaceFunctionNet

log = get_logger("ml.inference")
LAB_GROUPS = ("Research Lab", "Teaching Lab")


class ModelNotAvailable(RuntimeError):
    pass


class SpacePredictor:
    def __init__(self, spaces: pd.DataFrame) -> None:
        weights, enc_path, oof_path = (MODEL_DIR / "space_function_net.pt", MODEL_DIR / "encoder.json",
                                       MODEL_DIR / "oof_predictions.csv")
        for p in (weights, enc_path, oof_path):
            if not p.exists():
                raise ModelNotAvailable(f"Missing model artifact {p.name}. Run: python -m app.ml.train")
        with timed(log, "model loaded", artifact=weights.name):
            self.encoder = FeatureEncoder.from_json(enc_path.read_text(encoding="utf-8"))
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            self.model = SpaceFunctionNet(ckpt["cardinalities"], ckpt["n_numeric"], ckpt["n_classes"],
                                          tuple(ckpt["hidden"]), ckpt["dropout"], ckpt.get("use_embeddings", True))
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
            oof = pd.read_csv(oof_path).set_index("space_id")
        self.classes = self.encoder.classes
        self.table = self._summarize(spaces, oof)
        log.info(f"predictions indexed records={len(self.table)} "
                 f"flagged_for_verification={int(self.table['verify_flag'].sum())}")

    def _summarize(self, spaces: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
        proba = oof.reindex(spaces["space_id"])[self.classes]
        # Rooms added after training (records pipeline) have no out-of-fold prediction: use the deployed model.
        missing = proba.isna().any(axis=1).to_numpy()
        self.without_oof = int(missing.sum())
        if missing.any():
            fill = self.predict_frame(spaces.loc[missing])
            proba.iloc[missing.nonzero()[0]] = fill[self.classes].to_numpy()
        top = proba.to_numpy().argmax(1)
        t = pd.DataFrame(index=spaces["space_id"].values)
        t["recorded_group"] = spaces["class_group"].astype(str).values
        t["predicted_group"] = np.array(self.classes)[top]
        t["confidence"] = proba.to_numpy().max(1)
        t["p_recorded"] = [proba.iat[i, self.classes.index(g)] if g in self.classes else np.nan
                           for i, g in enumerate(t["recorded_group"])]
        t["p_lab"] = proba[list(LAB_GROUPS)].sum(axis=1).values
        t["verify_flag"] = (t["predicted_group"] != t["recorded_group"]) & (t["confidence"] >= AGENTS.mismatch_confidence)
        self._proba = proba
        return t

    # ---- queries used by agent tools -------------------------------------------------
    def assessment(self, space_id: str) -> dict:
        """Model view of one existing record (out-of-fold)."""
        if space_id not in self.table.index:
            return {"status": "not_found"}
        r = self.table.loc[space_id]
        probs = self._proba.loc[space_id].sort_values(ascending=False)
        return {"status": "success", "recorded_group": r.recorded_group, "predicted_group": r.predicted_group,
                "confidence": round(float(r.confidence), 3), "p_recorded": round(float(r.p_recorded), 3),
                "p_lab": round(float(r.p_lab), 3), "verify_flag": bool(r.verify_flag),
                "top3": [{"group": g, "p": round(float(p), 3)} for g, p in probs.head(3).items()]}

    def probability(self, space_ids: list[str], group: str) -> pd.Series:
        if group not in self.classes:
            return pd.Series(np.nan, index=space_ids)
        return self._proba.reindex(space_ids)[group]

    @torch.no_grad()
    def predict_frame(self, spaces: pd.DataFrame) -> pd.DataFrame:
        """Deployed-model probabilities for arbitrary rows (same schema as the space export)."""
        cat, num = self.encoder.transform(derive_columns(spaces))
        p = torch.softmax(self.model(torch.from_numpy(cat), torch.from_numpy(num)), 1).numpy()
        return pd.DataFrame(p, columns=self.classes, index=spaces.index)
