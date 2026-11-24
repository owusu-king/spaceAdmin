"""Email triage agent (LangGraph): rules, model verdicts, the review queue and James's decisions."""
from datetime import datetime

import pytest

from app.integrations.email import EmailIntake, InboundEmail
from app.integrations.triage import build_graph, decide, keyword_verdict


def _email(mid, subject, body="", sender="Someone <s@rice.edu>"):
    return InboundEmail(mid, sender, subject, datetime(2026, 11, 20, 9, 0), body)


def scripted(verdicts: dict):
    """A stand-in for the local model: subject keyword -> (label, confidence)."""
    calls = []

    def clf(sender, subject, body):
        calls.append(subject)
        for k, (label, conf) in verdicts.items():
            if k in subject:
                return {"label": label, "by": "model", "confidence": conf, "reason": f"scripted: {k}"}
        return {"label": "not_outage", "by": "model", "confidence": 0.95, "reason": "scripted default"}
    clf.calls = calls
    return clf


def test_decide_files_only_confident_agreement():
    out, no = keyword_verdict("Power outage Friday"), keyword_verdict("Lunch on Friday?")
    assert decide({"label": "outage", "by": "model", "confidence": 0.9, "reason": ""}, no)["label"] == "outage"
    assert decide({"label": "not_outage", "by": "model", "confidence": 0.95, "reason": ""}, no)["label"] == "not_outage"
    assert decide({"label": "not_outage", "by": "model", "confidence": 0.95, "reason": ""}, out)["label"] == "review"
    assert decide({"label": "outage", "by": "model", "confidence": 0.5, "reason": ""}, out)["label"] == "review"
    assert decide(None, out) == out                     # no model: the keyword rule decides, as before


def test_rules_decide_without_asking_the_model():
    clf = scripted({})
    g = build_graph(lambda: clf)
    v = g.invoke({"email": _email("1", "Notification", "Outage Request ID: SR-1234567")})["verdict"]
    assert v["label"] == "outage" and v["by"] == "rule"
    v = g.invoke({"email": _email("2", "Re: thanks"), "known_case": "OC-0007"})["verdict"]
    assert v["label"] == "outage" and "OC-0007" in v["reason"]
    v = g.invoke({"email": _email("3", "Lunch?"), "human": "outage"})["verdict"]
    assert v["by"] == "james"
    assert not clf.calls


@pytest.fixture
def intake(system, tmp_path):
    from app.services.cases import CaseStore

    class Source:
        mode = "replay"
        def __init__(self): self.queue = []
        def health_check(self): return {"status": "ok", "mode": "replay", "remaining": len(self.queue)}
        def fetch(self, limit): out, self.queue = self.queue[:limit], self.queue[limit:]; return out
    clf = scripted({"chiller": ("outage", 0.9), "Odd smell": ("outage", 0.55), "hot water": ("not_outage", 0.9)})
    it = EmailIntake(Source(), system, CaseStore(tmp_path / "t.db"), classifier_factory=lambda: clf)
    return it


def test_intake_files_ignores_and_holds(intake):
    intake.source.queue = [
        _email("a", "Lunch on Friday?", "Free after the committee meeting?"),
        _email("b", "Heads up: chiller work in Dell Butcher Saturday",
               "We will take the secondary chiller offline Saturday 6am to noon at Dell Butcher Hall."),
        _email("c", "Odd smell near the BRC loading dock", "Something smells like gas near the dock."),
        _email("d", "No hot water on the 3rd floor", "Since this morning no hot water at the sinks in Keck."),
    ]
    s = intake.poll(10)
    assert s["ignored"] == 1 and s["held"] == 2 and len(s["cases_opened"]) == 1
    rows = {r["message_id"]: r for r in intake.status()["items"]}
    assert rows["a"]["action"] == "ignored" and rows["a"]["decided_by"] == "model"
    assert intake.store.db.execute("SELECT body FROM inbox WHERE message_id='a'").fetchone()[0] is None   # privacy
    assert rows["c"]["action"] == "needs review"                       # model unsure
    assert rows["d"]["action"] == "needs review"                       # model "not outage" vs "hot water" keywords
    assert intake.status()["needs_review"] == 2
    assert intake.status()["items"][0]["action"] == "needs review"     # review queue first


def test_held_email_follows_a_case_opened_later_in_its_thread(intake):
    intake.source.queue = [_email("h", "Odd smell near the BRC loading dock", "Smells like gas near the BRC dock."),
                           _email("i", "Re: Odd smell near the BRC loading dock", "Still smells near the BRC dock this afternoon.")]
    intake.poll(10)
    assert intake.status()["needs_review"] == 2                        # both held: the model is unsure
    out = intake.decide("h", "outage", "James")                        # James files one: a case opens ...
    row = intake.store.db.execute("SELECT action, case_id, decided_by FROM inbox WHERE message_id='i'").fetchone()
    assert intake.status()["needs_review"] == 0                        # ... and the other follows it (thread rule)
    assert row["case_id"] == out["case_id"] and row["decided_by"] == "rule"


def test_james_decides_held_email(intake):
    intake.source.queue = [_email("c", "Odd smell near the BRC loading dock", "Something smells like gas near the BRC dock."),
                           _email("d", "No hot water on the 3rd floor", "No hot water at the sinks in Keck.")]
    intake.poll(10)
    out = intake.decide("d", "outage", "James")
    assert out["action"] == "opened" and out["case_id"].startswith("OC-")
    out = intake.decide("c", "not_outage", "James")
    assert out["action"] == "ignored"
    row = intake.store.db.execute("SELECT action, decided_by, body FROM inbox WHERE message_id='c'").fetchone()
    assert row["action"] == "ignored" and row["decided_by"] == "james" and row["body"] is None
    with pytest.raises(KeyError):
        intake.decide("c", "outage", "James")                         # no longer waiting for review
    with pytest.raises(ValueError):
        intake.decide("d", "maybe", "James")
