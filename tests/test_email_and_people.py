"""Email intake (replay + read-only IMAP), affected people, notification drafts and ablation plumbing."""
from datetime import datetime
from email.message import EmailMessage

import pytest
import torch

from app.agents.schemas import OutageRequest
from app.integrations.email import (EmailIntake, ExecutionNotPermitted, ImapSource, InboundEmail, ReplaySource,
                                    is_outage_related, parse_rfc822, thread_key)
from app.ml.model import SpaceFunctionNet
from app.scenarios import resolve_payload, scenarios
from app.services.cases import CaseStore


def _raw(subject="Elevator out of order", body="The GRB East elevator is out.", html=False, mid="<m1@rice.edu>"):
    m = EmailMessage()
    m["From"], m["Subject"], m["Message-ID"] = "Jose Presas <jap20@rice.edu>", subject, mid
    m["Date"] = "Tue, 20 Oct 2026 12:44:00 -0500"
    if html:
        m.set_content("<p>Steam <b>shutdown</b> at ABL</p><br>Autoclaves out")
        m.replace_header("Content-Type", "text/html; charset=utf-8")
    else:
        m.set_content(body)
    return m.as_bytes()


def test_parse_rfc822_plain_and_html():
    e = parse_rfc822(_raw())
    assert e.subject == "Elevator out of order" and e.date.day == 20 and "GRB East" in e.body
    h = parse_rfc822(_raw(html=True))
    assert "Steam shutdown at ABL" in h.body and "<" not in h.body


class FakeImap:
    instances = []

    def __init__(self, host):
        self.calls = []
        FakeImap.instances.append(self)

    def login(self, u, p): self.calls.append(("login", u))
    def select(self, folder, readonly=False):
        self.calls.append(("select", folder, readonly)); return "OK", [b"1"]
    def search(self, charset, *crit): return "OK", [b"1"]
    def fetch(self, i, spec):
        self.calls.append(("fetch", spec)); return "OK", [(b"1 (BODY[] {10}", _raw())]
    def close(self): self.calls.append(("close",))
    def logout(self): self.calls.append(("logout",))
    def store(self, *a): raise AssertionError("adapter must never modify flags")


def test_imap_is_read_only():
    env = {"FAI_IMAP_HOST": "imap.example.edu", "FAI_IMAP_USER": "u", "FAI_IMAP_PASSWORD": "p"}
    src = ImapSource(env, client_factory=FakeImap)
    msgs = src.fetch(5)
    calls = FakeImap.instances[-1].calls
    assert ("select", "INBOX", True) in calls                 # folder opened read-only
    assert ("fetch", "(BODY.PEEK[])") in calls                 # PEEK: does not mark as read
    assert calls[-1] == ("logout",) and msgs[0].subject == "Elevator out of order"
    with pytest.raises(ExecutionNotPermitted):
        src.execute("send")


def test_imap_not_configured():
    src = ImapSource({})
    assert src.health_check()["status"] == "not_configured" and src.fetch() == []


def test_triage_and_thread_keys():
    tri = InboundEmail("1", "TRIRIGA Notification <TRIRIGA-ServiceCenter@rice.edu>", "Notification", None, "")
    park = InboundEmail("2", "BRC Management", "[BRC Tenants] Busy BRC Parking Garage next week", None, "")
    assert is_outage_related(tri)[0] and not is_outage_related(park)[0]
    assert thread_key(InboundEmail("3", "", "ALERT! Unscheduled Outage: SR-1118242 - Emergency", None, "")) == "SR-1118242"
    assert thread_key(InboundEmail("4", "", "Re: FW: GRB East elevator", None, "")) == thread_key(
        InboundEmail("5", "", "GRB East elevator", None, ""))


def test_intake_opens_updates_and_skips(system, tmp_path):
    store = CaseStore(tmp_path / "i.db")
    intake = EmailIntake(ReplaySource(system.corpus, datetime(2026, 10, 19)), system, store)
    first = intake.poll(4)                                     # Space Science elevator thread (4 messages)
    assert len(first["cases_opened"]) == 1 and len(first["cases_updated"]) == 3
    case = store.get(first["cases_opened"][0])
    assert case["rev_letter"] == "D" and case["kind"] == "outage"
    intake.source._pos = 0                                     # re-deliver the same messages
    again = intake.poll(4)
    assert again["duplicates"] == 4 and not again["cases_opened"]


def test_non_outage_mail_keeps_no_body(system, tmp_path):
    store = CaseStore(tmp_path / "j.db")

    class OneMessage:
        mode = "test"
        def health_check(self): return {"status": "ok", "mode": "test"}
        def fetch(self, limit): return [InboundEmail("x1", "BRC Management", "Graduate Student Feedback Needed", datetime(2026, 5, 14), "private text")]
        def execute(self, a): raise ExecutionNotPermitted()

    intake = EmailIntake(OneMessage(), system, store)
    assert intake.poll()["ignored"] == 1
    row = store.db.execute("SELECT body, action FROM inbox WHERE message_id='x1'").fetchone()
    assert row[0] is None and row[1] == "ignored"


def _run(system, sid):
    sc = next(s for s in scenarios() if s.id == sid)
    return system.run_outage(OutageRequest(**resolve_payload(sc, system.corpus)))


def test_affected_people_lab_scope(system):
    r = _run(system, "OUT-OES-EXHAUST")                         # exhaust: lab systems only
    a = r["affected_people"]
    assert a["total"] > 0 and "laboratory" in a["scope_rule"]
    assert all(p["email"].endswith("@rice.edu") and ("laboratory" in p["reason"] or "lab-like" in p["reason"]) for p in a["people"])


def test_affected_people_building_wide_and_notice_draft(system):
    r = _run(system, "OUT-BRC-POWER")
    a = r["affected_people"]
    assert a["total"] > a["natural_sciences"] > 0
    kinds = [d["kind"] for d in r["communication_drafts"]]
    assert kinds == ["coordination", "notification"]
    note = r["communication_drafts"][1]
    assert note["status"].startswith("DRAFT") and note["recipients"]["affected_people"] == a["natural_sciences"]


def test_one_hot_variant_forward():
    net = SpaceFunctionNet([5, 3], 4, 6, (8,), 0.0, use_embeddings=False)
    out = net(torch.tensor([[1, 2], [0, 0]]), torch.zeros(2, 4))
    assert out.shape == (2, 6)
