"""Supporting experiments: cooperation and competition. Run:  python -m app.experiments

Cooperation — shared problem: long outages that make space unusable. For every recorded
outage request of >= 4 h affecting lab or occupancy-limiting systems, the Outage Agent runs
(a) alone and (b) with the Space Administrator Agent supplying temporary-space options.
Metrics: share of outages with options, constraint compliance (options outside affected
buildings, never "incompatible"), mean suitability/coverage, messages and added latency.

Competition — alternative strategies for one decision: which space records should be sent
for actual-use verification? We inject known class errors into a held-out sample (records
scored out-of-fold) and compare three strategies under identical conditions:
  A. Rule-based field consistency (workpoint class vs seating data)
  B. Model-based (SpaceFunctionNet disagreement with confidence >= threshold)
  C. Combined (A or B)
Metrics: precision, recall, F1, flags raised. Fixed seed, reproducible.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from app.agents.coordinator import Coordinator, build_system
from app.agents.outage_agent import LAB_SYSTEMS, RELOCATION_SYSTEMS
from app.agents.schemas import OutageRequest
from app.config import AGENTS, DATA_DIR, SEED
from app.logging_setup import get_logger, kv
from app.scenarios import message_text

log = get_logger("experiments")
WORKPOINT_GROUPS = {"Office", "Workstation"}


def cooperation_experiment(coord: Coordinator) -> dict:
    eligible = [h for h in coord.corpus.history
                if (h["duration_hours"] or 0) >= AGENTS.relocation_min_hours and h["buildings"]
                and set(h["systems"]) & (LAB_SYSTEMS | RELOCATION_SYSTEMS)]
    rows = []
    for h in eligible:
        msg = next((m for m in coord.corpus.messages if h["sr"] in m.subject and "Outage Request ID" in m.full_text), None)
        if not msg:
            continue
        req = OutageRequest(text=message_text(msg), source=h["sr"])
        t0 = time.perf_counter()
        solo = coord.run_outage(req, cooperate=False)
        t1 = time.perf_counter()
        coop = coord.run_outage(req, cooperate=True)
        t2 = time.perf_counter()
        affected = set(h["buildings"])
        support = coop["space_support"]
        cands = support["reply"]["payload"]["candidates"] if support else []
        rows.append({
            "sr": h["sr"], "buildings": ", ".join(h["buildings"]), "hours": h["duration_hours"],
            "ns_departments": len(coop["affected_departments"]["natural_sciences"]),
            "solo_options": 0, "coop_options": len(cands),
            "outside_affected": all(c["building"] not in affected for c in cands),
            "no_incompatible": all(c["status"] != "incompatible" for c in cands),
            "mean_suitability": round(float(np.mean([c["suitability"] for c in cands])), 3) if cands else None,
            "mean_coverage": round(float(np.mean([c["evidence_coverage"] for c in cands])), 3) if cands else None,
            "messages": len(coop["messages"]), "solo_ms": round((t1 - t0) * 1000, 1), "coop_ms": round((t2 - t1) * 1000, 1),
            "solo_actions": len(solo["recommended_actions"]), "coop_actions": len(coop["recommended_actions"]),
        })
    df = pd.DataFrame(rows)
    with_ns = df[df["ns_departments"] > 0] if len(df) else df
    summary = {
        "outages_evaluated": int(len(df)),
        "with_natural_sciences_departments": int(len(with_ns)),
        "solo_share_with_options": 0.0,
        "coop_share_with_options": round(float((with_ns["coop_options"] > 0).mean()), 3) if len(with_ns) else 0.0,
        "constraint_compliance": round(float((df["outside_affected"] & df["no_incompatible"]).mean()), 3) if len(df) else None,
        "mean_suitability": round(float(df["mean_suitability"].dropna().mean()), 3) if len(df) else None,
        "mean_coverage": round(float(df["mean_coverage"].dropna().mean()), 3) if len(df) else None,
        "mean_added_latency_ms": round(float((df["coop_ms"] - df["solo_ms"]).mean()), 1) if len(df) else None,
    }
    log.info("cooperation " + kv(**summary))
    return {"summary": summary, "rows": rows}


def _rule_flags(df: pd.DataFrame, recorded: pd.Series) -> pd.Series:
    """Field-consistency rule: a workpoint class must have seating data and vice versa."""
    seated = df["seat_status"].fillna("") != "Not Available for Seating"
    workpoint = recorded.isin(WORKPOINT_GROUPS)
    return workpoint != seated


def competition_experiment(coord: Coordinator, n: int = 1000, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    spaces = coord.ds.spaces.reset_index(drop=True)
    table = coord.predictor.table.reindex(spaces["space_id"]).reset_index(drop=True)
    groups = sorted(spaces["class_group"].astype(str).unique())
    idx = rng.choice(len(spaces), size=2 * n, replace=False)
    corrupt_idx, clean_idx = idx[:n], idx[n:]
    recorded = spaces["class_group"].astype(str).copy()
    for i in corrupt_idx:                      # inject a wrong recorded class (simulated record error)
        recorded.iat[i] = rng.choice([g for g in groups if g != recorded.iat[i]])
    sample = np.concatenate([corrupt_idx, clean_idx])
    truth = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    rec = recorded.iloc[sample].reset_index(drop=True)
    sub = spaces.iloc[sample].reset_index(drop=True)
    tab = table.iloc[sample].reset_index(drop=True)
    p_rec = np.array([coord.predictor._proba.loc[sid, g] for sid, g in zip(sub["space_id"], rec)])
    model_flag = (tab["predicted_group"].values != rec.values) & (tab["confidence"].values >= AGENTS.mismatch_confidence)
    rule_flag = _rule_flags(sub, rec).values
    strategies = {"A_rule_based": rule_flag, "B_model_based": model_flag, "C_combined": rule_flag | model_flag}
    results = {}
    for name, flag in strategies.items():
        tp = int((flag & truth).sum()); fp = int((flag & ~truth).sum()); fn = int((~flag & truth).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
        results[name] = {"precision": round(prec, 3), "recall": round(rec_, 3), "f1": round(f1, 3),
                         "flags": int(flag.sum()), "true_positives": tp, "false_positives": fp}
    winner = max(results, key=lambda k: results[k]["f1"])
    log.info("competition " + " ".join(f"{k}_f1={v['f1']}" for k, v in results.items()) + f" winner={winner}")
    return {"setup": {"injected_errors": n, "clean_records": n, "seed": seed,
                      "model_threshold": AGENTS.mismatch_confidence,
                      "mean_p_of_corrupted_label": round(float(p_rec[:n].mean()), 3)},
            "results": results, "winner": winner}


def run_all(coord: Coordinator | None = None, save: bool = True) -> dict:
    coord = coord or build_system()
    out = {"cooperation": cooperation_experiment(coord), "competition": competition_experiment(coord),
           "generated": pd.Timestamp.now().isoformat(timespec="seconds")}
    if save:
        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / "experiments.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run_all()
    print(json.dumps({"cooperation": r["cooperation"]["summary"], "competition": r["competition"]["results"]}, indent=2))
