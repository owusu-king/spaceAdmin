"""Ablation study for SpaceFunctionNet.  Run:  python -m app.ml.ablation

Each variant changes ONE design choice and is retrained on the identical stratified split,
once per training seed (default 3 seeds: different initialization and batch order).
We report mean and standard deviation of
  * validation macro-F1  -> the ONLY score used to choose a configuration, and
  * test macro-F1        -> the final unbiased report (never used for selection).
Choosing by test score would leak the test set into model selection.

Variants
  full               deployed configuration
  one_hot            categorical inputs as one-hot vectors instead of learned embeddings
  no_occupants       drop the people-record occupant features
  no_room_id         drop room-number pattern, suffix and number
  no_allocation      drop departmental allocation features
  no_class_weights   plain cross-entropy (no imbalance correction)
  shallow            one hidden layer (64)
  deeper             three hidden layers (256-128-64)
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np

from app.config import MODEL_DIR, TRAIN
from app.logging_setup import get_logger, kv, timed
from app.ml.features import CATEGORICAL, NUMERIC, derive_columns
from app.ml.train import evaluate, split_indices, train_model

log = get_logger("ml.ablation")
OCC = [c for c in NUMERIC if c.startswith("log_occ_")]
ORIG = {"use_embeddings": True, "class_weighting": True}   # feature and depth ablations are relative to the original design


def variants() -> dict[str, dict]:
    drop = lambda cols, out: [c for c in cols if c not in out]  # noqa: E731
    return {
        # Every variant states its encoding and weighting explicitly, so results do not depend on the config defaults.
        "full": {"label": "Original design: embeddings + class weights", "use_embeddings": True, "class_weighting": True},
        "one_hot": {"label": "One-hot instead of embeddings", "use_embeddings": False, "class_weighting": True},
        "no_occupants": {"label": "Without occupant features", "num_cols": drop(NUMERIC, OCC), **ORIG},
        "no_room_id": {"label": "Without room-number features", "cat_cols": drop(CATEGORICAL, ["room_pattern", "room_suffix"]),
                       "num_cols": drop(NUMERIC, ["room_number_mod100"]), **ORIG},
        "no_allocation": {"label": "Without departmental allocation", "cat_cols": drop(CATEGORICAL, ["allocation_primary"]),
                          "num_cols": drop(NUMERIC, ["allocation_count"]), **ORIG},
        "no_class_weights": {"label": "Without class weighting", "use_embeddings": True, "class_weighting": False},
        "one_hot_no_class_weights": {"label": "One-hot and no class weighting (selected, deployed)", "use_embeddings": False,
                                     "class_weighting": False},
        "shallow": {"label": "One hidden layer (64)", "cfg": replace(TRAIN, hidden=(64,)), **ORIG},
        "deeper": {"label": "Three hidden layers (256-128-64)", "cfg": replace(TRAIN, hidden=(256, 128, 64)), **ORIG},
    }


def run_ablation(spaces=None, names: list[str] | None = None, seeds: tuple[int, ...] = (42, 43, 44),
                 save: bool = True) -> dict:
    if spaces is None:
        from app.data.loader import load_dataset
        spaces = load_dataset().spaces
    X, y = derive_columns(spaces), spaces["class_group"].astype(str)
    tr, va, te = split_indices(y, TRAIN)
    rows = []
    for name, v in variants().items():
        if names and name not in names:
            continue
        base_cfg = v.get("cfg", TRAIN)
        val_f1, test_f1, test_acc, epochs = [], [], [], []
        with timed(log, "variant trained", variant=name, seeds=len(seeds)) as t:
            for sd in seeds:
                cfg = replace(base_cfg, seed=sd)
                model, enc, hist = train_model(X, y, tr, va, cfg, verbose=False, cat_cols=v.get("cat_cols"),
                                               num_cols=v.get("num_cols"), use_embeddings=v["use_embeddings"],
                                               class_weighting=v["class_weighting"])
                m = evaluate(model, enc, X, y, te)
                val_f1.append(max(h["val_macro_f1"] for h in hist))
                test_f1.append(m["macro_f1"]); test_acc.append(m["accuracy"]); epochs.append(len(hist))
        r4 = lambda a: round(float(a), 4)  # noqa: E731
        rows.append({"variant": name, "label": v["label"], "seeds": list(seeds),
                     "val_macro_f1": r4(np.mean(val_f1)), "val_std": r4(np.std(val_f1)),
                     "macro_f1": r4(np.mean(test_f1)), "test_std": r4(np.std(test_f1)), "accuracy": r4(np.mean(test_acc)),
                     "epochs": round(float(np.mean(epochs)), 1), "parameters": sum(p.numel() for p in model.parameters()),
                     "seconds": round(t["duration_ms"] / 1000, 1), "hidden": list(base_cfg.hidden)})
        log.info("ablation " + kv(variant=name, val_f1=rows[-1]["val_macro_f1"], test_f1=rows[-1]["macro_f1"]))
    full = next((r for r in rows if r["variant"] == "full"), None)
    for r in rows:
        r["delta_val"] = r4(r["val_macro_f1"] - full["val_macro_f1"]) if full else None
        r["delta_macro_f1"] = r4(r["macro_f1"] - full["macro_f1"]) if full else None
    selected = max(rows, key=lambda r: r["val_macro_f1"])["variant"] if rows else None
    out = {"split": {"train": len(tr), "val": len(va), "test": len(te)}, "seed": TRAIN.seed, "selected_by_validation": selected,
           "selection_rule": "highest mean validation macro-F1 across seeds; test scores are reported, never used to select",
           "config": asdict(TRAIN), "variants": rows}
    if save:
        (MODEL_DIR / "ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_ablation()
    for r in res["variants"]:
        print(f"{r['label']:<34} val {r['val_macro_f1']:.4f}±{r['val_std']:.4f} ({r['delta_val']:+.4f})  "
              f"test {r['macro_f1']:.4f}±{r['test_std']:.4f} ({r['delta_macro_f1']:+.4f})  params {r['parameters']:,}")
