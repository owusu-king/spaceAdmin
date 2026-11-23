"""Model loading, inference, leakage controls and training reproducibility."""
import numpy as np
import pandas as pd

from app.config import TrainConfig
from app.ml.features import CATEGORICAL, NUMERIC, FeatureEncoder, derive_columns
from app.ml.train import predict_proba, split_indices, train_model


def test_artifacts_load_and_metrics_present(system):
    m = system.predictor.metrics
    assert m["test"]["macro_f1"] > m["baselines"]["majority_class"]["macro_f1"]
    assert len(system.predictor.classes) == 12


def test_inference_probabilities(system):
    rows = system.ds.spaces.sample(50, random_state=0)
    p = system.predictor.predict_frame(rows)
    assert p.shape == (50, 12)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_out_of_fold_assessment(system):
    sid = system.ds.spaces["space_id"].iloc[100]
    a = system.predictor.assessment(sid)
    assert a["status"] == "success" and 0 <= a["confidence"] <= 1 and len(a["top3"]) == 3
    assert system.predictor.assessment("nope")["status"] == "not_found"


def test_no_label_derived_features():
    banned = {"seat_status", "ssa", "available_seats", "ws_area", "space_standard", "space_class", "class_group"}
    assert not banned & set(CATEGORICAL + NUMERIC)


def test_unseen_categories_map_to_unk(system):
    spaces = system.ds.spaces.head(20).copy()
    d = derive_columns(spaces)
    enc = FeatureEncoder().fit(d, spaces["class_group"].astype(str))
    d.loc[d.index[0], "building"] = "Brand New Building"
    cat, num = enc.transform(d)
    assert cat[0, CATEGORICAL.index("building")] == 0 and not np.isnan(num).any()


def test_split_is_stratified_and_disjoint(system):
    labels = system.ds.spaces["class_group"].astype(str)
    tr, va, te = split_indices(labels)
    assert not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(labels)


def test_training_is_reproducible(system):
    spaces = system.ds.spaces.sample(3000, random_state=1).reset_index(drop=True)
    d, y = derive_columns(spaces), spaces["class_group"].astype(str)
    cfg = TrainConfig(max_epochs=3, patience=3)
    idx = np.arange(len(y))
    tr, va = idx[:2400], idx[2400:]
    m1, e1, _ = train_model(d, y, tr, va, cfg, verbose=False)
    m2, e2, _ = train_model(d, y, tr, va, cfg, verbose=False)
    assert np.allclose(predict_proba(m1, e1, d.iloc[va]), predict_proba(m2, e2, d.iloc[va]), atol=1e-6)
