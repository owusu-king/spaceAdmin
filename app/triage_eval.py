"""Evaluate email triage: keyword rule vs local model vs the triage agent (rules + model + human review).

Run:  python -m app.triage_eval            (about 5 minutes with qwen3:8b; results in data/triage_eval.json)

Data
* The supplied correspondence (381 messages in 168 threads), labelled per thread in data/triage_labels.json
  (provisional labels; "unsure" threads are left out of the scores).
* data/triage_synthetic.json: a few clearly marked synthetic emails, because the supplied mail comes from outage
  folders and has few ordinary inbox messages (meetings, IT, newsletters) or informal reports without keywords.

The costly mistake is a real outage that triage silently ignores, so "missed outages" is reported for each strategy.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from app.config import DATA_DIR
from app.integrations.email import InboundEmail, thread_key
from app.integrations.triage import ModelClassifier, build_graph, keyword_verdict, rule_verdict
from app.logging_setup import get_logger

log = get_logger("triage_eval")


def dataset() -> list[dict]:
    from app.data.notices import load_corpus
    labels = {x["thread_key"]: x["label"] for x in json.loads((DATA_DIR / "triage_labels.json").read_text(encoding="utf-8"))["labels"]}
    rows = []
    for m in load_corpus().messages:
        e = InboundEmail(f"corpus-{len(rows)}", m.sender, m.subject, m.date, m.full_text)
        rows.append({"email": e, "label": labels.get(thread_key(e), "unsure"), "source": "supplied"})
    syn = json.loads((DATA_DIR / "triage_synthetic.json").read_text(encoding="utf-8"))["emails"]
    for i, x in enumerate(syn):
        rows.append({"email": InboundEmail(f"synthetic-{i}", x["sender"], x["subject"], None, x["body"]),
                     "label": x["label"], "source": "synthetic"})
    return [r for r in rows if r["label"] != "unsure"]


def scores(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (truth, predicted) with labels outage / not_outage; 'outage' is the positive class."""
    tp = sum(t == "outage" and p == "outage" for t, p in pairs)
    fp = sum(t == "not_outage" and p == "outage" for t, p in pairs)
    fn = sum(t == "outage" and p != "outage" for t, p in pairs)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "accuracy": round(sum(t == p for t, p in pairs) / len(pairs), 3) if pairs else 0.0, "n": len(pairs)}


def run(model: str = "qwen3:8b", limit: int | None = None, save: bool = True) -> dict:
    from app.voice.llm import OllamaChat
    rows = dataset()[:limit] if limit else dataset()
    model_clf, memo = ModelClassifier(OllamaChat(model)), {}

    def clf(sender, subject, body):                     # one model call per email, shared by B and C
        key = (sender, subject, body[:500])
        if key not in memo:
            memo[key] = model_clf(sender, subject, body)
        return memo[key]
    graph = build_graph(lambda: clf)
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        e = r["email"]
        r["keywords"] = (rule_verdict(e.sender, e.subject, e.body) or keyword_verdict(e.subject))["label"]
        mv = clf(e.sender, e.subject, e.body)
        r["model"] = mv["label"] if mv else "error"
        r["model_conf"] = mv["confidence"] if mv else None
        v = graph.invoke({"email": e})["verdict"]
        r["agent"], r["agent_by"], r["agent_reason"] = v["label"], v["by"], v.get("reason", "")
        if (i + 1) % 25 == 0:
            log.info(f"triage eval {i + 1}/{len(rows)}")
    secs = time.perf_counter() - t0
    truth = [r["label"] for r in rows]
    auto = [r for r in rows if r["agent"] != "review"]
    held = [r for r in rows if r["agent"] == "review"]
    res = {
        "generated": datetime.now().isoformat(timespec="seconds"), "model": model, "emails": len(rows),
        "by_source": {s: sum(r["source"] == s for r in rows) for s in ("supplied", "synthetic")},
        "labels": {"outage": truth.count("outage"), "not_outage": truth.count("not_outage")},
        "strategies": {
            "A_keywords": {**scores([(r["label"], r["keywords"]) for r in rows]),
                           "missed_outages": sum(r["label"] == "outage" and r["keywords"] != "outage" for r in rows)},
            "B_model": {**scores([(r["label"], r["model"]) for r in rows]),
                        "missed_outages": sum(r["label"] == "outage" and r["model"] != "outage" for r in rows)},
            "C_agent": {
                "auto_decided": len(auto), "held_for_review": len(held),
                "held_share": round(len(held) / len(rows), 3),
                "on_auto_decided": scores([(r["label"], r["agent"]) for r in auto]),
                "after_review": scores([(r["label"], r["label"] if r["agent"] == "review" else r["agent"]) for r in rows]),
                "missed_outages": sum(r["label"] == "outage" and r["agent"] == "not_outage" for r in rows),
                "decided_by": {b: sum(r["agent_by"] == b for r in rows) for b in ("rule", "model", "keywords")},
            },
        },
        "seconds_per_email": round(secs / len(rows), 2),
        "examples": {
            "held": [{"subject": r["email"].subject[:100], "truth": r["label"], "reason": r["agent_reason"][:160]} for r in held[:12]],
            "agent_errors": [{"subject": r["email"].subject[:100], "truth": r["label"], "agent": r["agent"], "by": r["agent_by"],
                              "reason": r["agent_reason"][:160]} for r in auto if r["agent"] != r["label"]][:15],
            "keyword_misses_caught": [r["email"].subject[:100] for r in rows
                                      if r["label"] == "outage" and r["keywords"] != "outage" and r["agent"] == "outage"][:12],
        },
        "caveats": ["Labels are provisional (written by the development assistant, to be confirmed by James).",
                    "Most supplied emails come from outage threads; synthetic examples are few and hand-written.",
                    "Model confidence is the model's own estimate and is not calibrated."],
    }
    if save:
        (DATA_DIR / "triage_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = run(a.model, a.limit)
    print(json.dumps({k: out[k] for k in ("emails", "labels", "strategies", "seconds_per_email")}, indent=1))
