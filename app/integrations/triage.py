"""Email triage agent: which incoming emails are about building outages (LangGraph).

    START ─► rules ─┬─ decided (TRIRIGA notice, follow-up in a thread with a case, James's own decision) ─┐
                    └─► model ─► decide ─────────────────────────────────────────────────────────────────┤
                                                                                                         ▼
                                    ┌────────────── route by the verdict ──────────────┐
                                    ▼                         ▼                        ▼
                                  case                      ignore                    hold
                  (Outage Agent opens or updates     (only sender, subject     ("Needs review":
                   the outage case, as before)        and date are kept)        James decides)

* rules   certain cases need no model: TRIRIGA sender or an "Outage Request ID", a reply in a thread that already
          has a case, or a decision James made in the Email dialog.
* model   the local language model (Ollama, the model chosen in Settings) reads sender, subject and the start of
          the body and returns outage or not, a confidence and a one-line reason (JSON schema output).
          Without Ollama the keyword rule is used instead, so intake never stops.
* decide  auto-files only when the model is confident (>= 0.75) and does not contradict the keyword rule;
          anything else is held for James. A wrong "not an outage" is the costly mistake, so disagreement
          always goes to a person.

Human review is a re-entry: James's choice in the Email dialog runs the same graph again with his label,
which the rules node accepts as decided. Held emails keep their text until then; ignored ones never do.
"""
from __future__ import annotations

import re
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from app.logging_setup import get_logger

log = get_logger("integrations.triage")
AUTO_CONFIDENCE = 0.75
TRIRIGA_SENDER = re.compile(r"tririga", re.I)
OUTAGE_SUBJECT = re.compile(r"(SR-\d{7}|outage|shut ?down|elevator|steam|chilled water|hot water|exhaust|power|"
                            r"\bAC\b|HVAC|air handler|AHU|out of (order|service)|maintenance|leak)", re.I)

SCHEMA = {"type": "object",
          "properties": {"outage": {"type": "boolean"}, "confidence": {"type": "number"}, "reason": {"type": "string"}},
          "required": ["outage", "confidence", "reason"]}

PROMPT = """You sort email for a university facilities administrator. Decide if this email is about a BUILDING
OUTAGE: an interruption, failure, shutdown, reduction, test or restoration of a building service or system
(electricity, HVAC/air handlers, chilled or hot water, domestic water, steam, natural gas, lab air, vacuum, nitrogen,
exhaust/fume hoods, elevators, fire alarm systems, network), including planned maintenance that interrupts a service,
approvals and coordination of such work, and follow-ups about one.
NOT an outage: meetings, events, holiday closures, parking, surveys, newsletters, purchasing, HR, IT accounts,
space or key requests, construction or renovation that does not interrupt a service, fire drills.
Return JSON: outage (true/false), confidence from 0 to 1 (how sure you are), reason (at most 15 words)."""


class Verdict(TypedDict, total=False):
    label: str            # outage | not_outage | review
    by: str               # rule | model | keywords | james
    confidence: float | None
    reason: str


def rule_verdict(sender: str, subject: str, body: str) -> Verdict | None:
    if TRIRIGA_SENDER.search(sender) or "Outage Request ID" in body[:4000]:
        return {"label": "outage", "by": "rule", "confidence": 1.0, "reason": "TRIRIGA outage notice"}
    return None


def keyword_verdict(subject: str) -> Verdict:
    if OUTAGE_SUBJECT.search(subject):
        return {"label": "outage", "by": "keywords", "confidence": None, "reason": "outage keywords in the subject"}
    return {"label": "not_outage", "by": "keywords", "confidence": None, "reason": "no outage keywords in the subject"}


class ModelClassifier:
    """Asks the local LLM. Any failure returns None, so the caller falls back to the keyword rule."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, sender: str, subject: str, body: str) -> Verdict | None:
        text = f"From: {sender}\nSubject: {subject}\n\n{' '.join(body.split())[:1500]}"
        try:
            out = self.llm.chat_json([{"role": "system", "content": PROMPT}, {"role": "user", "content": text}], SCHEMA)
        except Exception as exc:
            log.warning(f"triage model unavailable error={type(exc).__name__}")
            return None
        conf = max(0.0, min(1.0, float(out.get("confidence") or 0)))
        return {"label": "outage" if out.get("outage") else "not_outage", "by": "model", "confidence": round(conf, 2),
                "reason": str(out.get("reason") or "")[:160]}


class TriageState(TypedDict, total=False):
    email: object                 # InboundEmail
    human: str | None             # outage | not_outage, when James decided in the Email dialog
    known_case: str | None        # a case already open for this email's thread
    model_verdict: Verdict | None
    verdict: Verdict
    outcome: dict                 # what the action did (case opened/updated, ignored, held)


def decide(model: Verdict | None, keywords: Verdict) -> Verdict:
    """Combine the model's view with the keyword rule. Confident agreement is filed; anything else goes to James."""
    if model is None:
        return keywords                                           # no model available: the rule decides, as before
    if (model["confidence"] or 0) < AUTO_CONFIDENCE:
        return {**model, "label": "review", "reason": f"model unsure ({model['confidence']:.0%}): {model['reason']}"}
    if model["label"] == "not_outage" and keywords["label"] == "outage":
        return {**model, "label": "review", "reason": f"model says not an outage, but the subject suggests one: {model['reason']}"}
    return model


def build_graph(classifier_factory: Callable[[], Callable | None], actions=None):
    """classifier_factory() returns a ModelClassifier or None. actions (an EmailIntake) carries out the verdict;
    without actions the graph only classifies (used by the evaluation)."""

    def rules(s: TriageState) -> dict:
        m = s["email"]
        if s.get("human"):
            return {"verdict": {"label": s["human"], "by": "james", "confidence": None, "reason": "decided by James"}}
        v = rule_verdict(m.sender, m.subject, m.body)
        if v:
            return {"verdict": v}
        if s.get("known_case"):
            return {"verdict": {"label": "outage", "by": "rule", "confidence": 1.0,
                                "reason": f"reply in a thread that already has case {s['known_case']}"}}
        return {}

    def model(s: TriageState) -> dict:
        m, clf = s["email"], classifier_factory()
        return {"model_verdict": clf(m.sender, m.subject, m.body) if clf else None}

    def decide_node(s: TriageState) -> dict:
        m = s["email"]
        return {"verdict": decide(s.get("model_verdict"), keyword_verdict(m.subject))}

    g = StateGraph(TriageState)
    g.add_node("rules", rules)
    g.add_node("model", model)
    g.add_node("decide", decide_node)
    g.add_edge(START, "rules")
    g.add_conditional_edges("rules", lambda s: "routed" if s.get("verdict") else "model",
                            {"routed": "route" if actions else END, "model": "model"})
    g.add_edge("model", "decide")
    if actions is None:
        g.add_edge("decide", END)
        return g.compile()

    g.add_node("route", lambda s: {})
    g.add_edge("decide", "route")
    g.add_node("case", lambda s: {"outcome": actions.open_or_update(s["email"], s["verdict"])})
    g.add_node("ignore", lambda s: {"outcome": actions.ignore(s["email"], s["verdict"])})
    g.add_node("hold", lambda s: {"outcome": actions.hold(s["email"], s["verdict"])})
    g.add_conditional_edges("route", lambda s: {"outage": "case", "not_outage": "ignore"}.get(s["verdict"]["label"], "hold"),
                            ["case", "ignore", "hold"])
    for n in ("case", "ignore", "hold"):
        g.add_edge(n, END)
    return g.compile()
