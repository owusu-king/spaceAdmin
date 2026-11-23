"""Training and evaluation pipeline for SpaceFunctionNet.

Run:  python -m app.ml.train

Steps
1. Stratified 70/15/15 train/validation/test split (fixed seed).
2. Fit the feature encoder on the training split only.
3. Train with class-weighted cross-entropy, AdamW, early stopping on validation macro-F1.
4. Evaluate once on the untouched test split; compare with two baselines.
5. Robustness check: hold out entire buildings (GroupShuffleSplit) and retrain.
6. Cross-fitting: 5-fold out-of-fold predictions so every record the agents inspect is
   scored by a model that never saw that record's label.
7. Save weights, encoder, metrics and out-of-fold predictions to models/.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from torch import nn

from app.config import MODEL_DIR, TRAIN, TrainConfig
from app.logging_setup import get_logger, kv, timed
from app.ml.features import CATEGORICAL, FeatureEncoder, derive_columns
from app.ml.model import SpaceFunctionNet

log = get_logger("ml.train")
TARGET = "class_group"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(labels: pd.Series, cfg: TrainConfig = TRAIN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(labels))
    train_val, test = train_test_split(idx, test_size=cfg.test_size, stratify=labels, random_state=cfg.seed)
    rel_val = cfg.val_size / (1 - cfg.test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=labels.iloc[train_val],
                                  random_state=cfg.seed)
    return train, val, test


def _tensors(enc: FeatureEncoder, derived: pd.DataFrame, labels: pd.Series | None = None):
    cat, num = enc.transform(derived)
    out = [torch.from_numpy(cat), torch.from_numpy(num)]
    if labels is not None:
        out.append(torch.from_numpy(enc.encode_labels(labels).copy()))
    return out


@torch.no_grad()
def predict_proba(model: SpaceFunctionNet, enc: FeatureEncoder, derived: pd.DataFrame) -> np.ndarray:
    model.eval()
    cat, num = _tensors(enc, derived)
    return torch.softmax(model(cat, num), dim=1).numpy()


def train_model(derived: pd.DataFrame, labels: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray,
                cfg: TrainConfig = TRAIN, verbose: bool = True, cat_cols: list[str] | None = None,
                num_cols: list[str] | None = None, use_embeddings: bool | None = None,
                class_weighting: bool | None = None):
    """Train one model; returns (model, encoder, history). History holds per-epoch metrics.

    The keyword options exist for the ablation study (app/ml/ablation.py); None = use the config (deployed model)."""
    use_embeddings = cfg.use_embeddings if use_embeddings is None else use_embeddings
    class_weighting = cfg.class_weighting if class_weighting is None else class_weighting
    set_seed(cfg.seed)
    enc = FeatureEncoder().fit(derived.iloc[train_idx], labels.iloc[train_idx], cfg.min_category_count,
                               cat_cols, num_cols)
    tr = _tensors(enc, derived.iloc[train_idx], labels.iloc[train_idx])
    va = _tensors(enc, derived.iloc[val_idx], labels.iloc[val_idx])

    # Inverse-sqrt-frequency class weights soften the imbalance (Building Service is 33% of rows).
    counts = np.bincount(tr[2].numpy(), minlength=len(enc.classes)).astype(float)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    weights = weights / weights.mean() if class_weighting else torch.ones(len(enc.classes))

    model = SpaceFunctionNet(enc.cardinalities, tr[1].shape[1], len(enc.classes), cfg.hidden, cfg.dropout, use_embeddings)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tr), batch_size=cfg.batch_size,
                                         shuffle=True, generator=gen)

    history, best_f1, best_state, stale = [], -1.0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total = 0.0
        for cat, num, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(cat, num), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        model.eval()
        with torch.no_grad():
            logits = model(va[0], va[1])
            val_loss = loss_fn(logits, va[2]).item()
            pred = logits.argmax(1).numpy()
        val_f1 = f1_score(va[2].numpy(), pred, average="macro")
        val_acc = accuracy_score(va[2].numpy(), pred)
        history.append({"epoch": epoch, "train_loss": total / len(tr[2]), "val_loss": val_loss,
                        "val_accuracy": val_acc, "val_macro_f1": val_f1})
        sched.step(val_f1)
        if val_f1 > best_f1 + 1e-4:
            best_f1, stale = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if verbose and (epoch % 5 == 0 or epoch == 1):
            log.info("epoch " + kv(epoch=epoch, train_loss=f"{history[-1]['train_loss']:.4f}",
                                   val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}", val_f1=f"{val_f1:.4f}"))
        if stale >= cfg.patience:
            if verbose:
                log.info(f"early stopping epoch={epoch} best_val_macro_f1={best_f1:.4f}")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, enc, history


def evaluate(model, enc, derived: pd.DataFrame, labels: pd.Series, idx: np.ndarray) -> dict:
    proba = predict_proba(model, enc, derived.iloc[idx])
    y_true = enc.encode_labels(labels.iloc[idx])
    y_pred = proba.argmax(1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": classification_report(y_true, y_pred, labels=range(len(enc.classes)),
                                           target_names=enc.classes, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(enc.classes))).tolist(),
        "n": int(len(idx)),
    }


def baselines(derived: pd.DataFrame, labels: pd.Series, train_idx, test_idx, cfg: TrainConfig = TRAIN) -> dict:
    """Majority-class and multinomial logistic regression (one-hot + numerics) on the same split."""
    enc = FeatureEncoder().fit(derived.iloc[train_idx], labels.iloc[train_idx], cfg.min_category_count)
    cat, num = enc.transform(derived)
    onehot = np.concatenate([np.eye(c, dtype=np.float32)[cat[:, i]] for i, c in enumerate(enc.cardinalities)], 1)
    X = np.concatenate([onehot, num], 1)
    y = enc.encode_labels(labels)
    out = {}
    for name, clf in [("majority_class", DummyClassifier(strategy="most_frequent")),
                      ("logistic_regression", LogisticRegression(max_iter=2000, C=1.0))]:
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        out[name] = {"accuracy": float(accuracy_score(y[test_idx], pred)),
                     "macro_f1": float(f1_score(y[test_idx], pred, average="macro"))}
    return out


def building_holdout(derived, labels, buildings: pd.Series, cfg: TrainConfig = TRAIN) -> dict:
    """Generalization to buildings never seen in training (harder than the random split)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
    train_val, test = next(gss.split(derived, labels, groups=buildings))
    tr, va = train_test_split(train_val, test_size=0.15, random_state=cfg.seed)
    model, enc, _ = train_model(derived, labels, tr, va, cfg, verbose=False)
    m = evaluate(model, enc, derived, labels, test)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "n": m["n"],
            "held_out_buildings": int(buildings.iloc[test].nunique())}


def out_of_fold(derived, labels, cfg: TrainConfig = TRAIN) -> tuple[np.ndarray, list[str]]:
    oof = None
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    for k, (fit_idx, hold_idx) in enumerate(skf.split(derived, labels), 1):
        tr, va = train_test_split(fit_idx, test_size=0.12, stratify=labels.iloc[fit_idx], random_state=cfg.seed)
        model, enc, _ = train_model(derived, labels, tr, va, cfg, verbose=False)
        if oof is None:
            oof = np.zeros((len(labels), len(enc.classes)), dtype=np.float32)
        oof[hold_idx] = predict_proba(model, enc, derived.iloc[hold_idx])
        log.info(f"cross-fit fold={k}/{cfg.cv_folds} done")
    return oof, enc.classes


def save_artifacts(model, enc: FeatureEncoder, metrics: dict, cfg: TrainConfig = TRAIN) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cardinalities": enc.cardinalities,
                "n_numeric": len(enc.mean), "n_classes": len(enc.classes),
                "hidden": cfg.hidden, "dropout": cfg.dropout, "use_embeddings": model.use_embeddings},
               MODEL_DIR / "space_function_net.pt")
    (MODEL_DIR / "encoder.json").write_text(enc.to_json(), encoding="utf-8")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_training(cfg: TrainConfig = TRAIN, with_oof: bool = True) -> dict:
    from app.data.loader import load_dataset
    ds = load_dataset()
    spaces = ds.spaces
    derived, labels = derive_columns(spaces), spaces[TARGET].astype(str)
    train_idx, val_idx, test_idx = split_indices(labels, cfg)
    log.info("split " + kv(train=len(train_idx), val=len(val_idx), test=len(test_idx), classes=labels.nunique()))
    with timed(log, "training complete"):
        model, enc, history = train_model(derived, labels, train_idx, val_idx, cfg)
    test = evaluate(model, enc, derived, labels, test_idx)
    log.info("test " + kv(accuracy=f"{test['accuracy']:.4f}", macro_f1=f"{test['macro_f1']:.4f}"))
    base = baselines(derived, labels, train_idx, test_idx, cfg)
    log.info("baselines " + " ".join(f"{k}_macro_f1={v['macro_f1']:.4f}" for k, v in base.items()))
    holdout = building_holdout(derived, labels, spaces["building"].astype(str), cfg)
    log.info("building holdout " + kv(accuracy=f"{holdout['accuracy']:.4f}", macro_f1=f"{holdout['macro_f1']:.4f}"))
    metrics = {"config": asdict(cfg), "classes": enc.classes, "features": {"categorical": CATEGORICAL,
               "numeric": list(enc.mean)}, "split": {"train": len(train_idx), "val": len(val_idx),
               "test": len(test_idx)}, "history": history, "test": test, "baselines": base,
               "building_holdout": holdout, "parameters": sum(p.numel() for p in model.parameters())}
    save_artifacts(model, enc, metrics, cfg)
    if with_oof:
        with timed(log, "cross-fitting complete"):
            oof, classes = out_of_fold(derived, labels, cfg)
        frame = pd.DataFrame(oof, columns=classes)
        frame.insert(0, "space_id", spaces["space_id"].values)
        frame.to_csv(MODEL_DIR / "oof_predictions.csv", index=False)
    log.info(f"artifacts saved dir={MODEL_DIR}")
    return metrics


if __name__ == "__main__":
    run_training()
