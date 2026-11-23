"""Sending an approved draft: the checks, the recipients and the record of what was sent.

Rules (all must hold, checked in this order):
  1. The 'send_email' switch is on.
  2. The case is an outage case, Approved, and the request names its current revision.
  3. The draft exists and has not already been sent for this revision.
  4. Recipients are on record.
  5. 'Deliver to real recipients' is on, or a test address is configured; otherwise refuse.
"""
from __future__ import annotations

from datetime import datetime

from app.integrations.mailer import OutgoingEmail
from app.services.cases import rev_letter


class SendRefused(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _split_subject(body: str) -> tuple[str, str]:
    first, _, rest = body.partition("\n")
    if first.lower().startswith("subject:"):
        return first.split(":", 1)[1].strip(), rest.lstrip("\n")
    return "Facilities notice", body


def recipients(case: dict, kind: str) -> tuple[list[str], list[str]]:
    """(to, bcc) taken from the case analysis. Natural Sciences only, as in the drafts."""
    out = case["output"]
    if kind == "coordination":
        to = [c["email"] for d in out["stakeholders"]["departments"] if d["natural_sciences"]
              for c in d["coordination_contacts"] if c.get("email")]
        return list(dict.fromkeys(to)), []
    people = (out.get("affected_people") or {}).get("people", [])
    return [], list(dict.fromkeys(p["email"] for p in people if p["natural_sciences"] and p.get("email")))


class Outbox:
    def __init__(self, store) -> None:
        self.store, self.db = store, store.db
        self.db.execute("""CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, rev INTEGER,
            draft TEXT, intended INTEGER, delivered_to TEXT, redirected INTEGER, message_id TEXT, sent_by TEXT, ts TEXT)""")
        self.db.commit()

    def sent(self, case_id: str) -> list[dict]:
        rows = self.db.execute("SELECT * FROM outbox WHERE case_id=? ORDER BY id", (case_id,)).fetchall()
        return [dict(r) | {"rev_letter": rev_letter(r["rev"])} for r in rows]

    def prepare(self, case: dict | None, kind: str, rev: int, live: bool, mailer) -> tuple[OutgoingEmail, dict]:
        """Run checks 2-5 and build the message. Raises SendRefused."""
        if not case or case["kind"] != "outage":
            raise SendRefused(404, "Outage case not found")
        if case["status"] != "Approved":
            raise SendRefused(409, "Approve the case before sending its drafts")
        if rev != case["rev"]:
            raise SendRefused(409, f"Send targets Rev {rev_letter(rev)} but the current revision is {case['rev_letter']}")
        draft = next((d for d in case["output"].get("communication_drafts", []) if d["kind"] == kind), None)
        if not draft:
            raise SendRefused(404, f"No '{kind}' draft in this case")
        if self.db.execute("SELECT 1 FROM outbox WHERE case_id=? AND rev=? AND draft=?", (case["id"], rev, kind)).fetchone():
            raise SendRefused(409, f"This draft was already sent for Rev {rev_letter(rev)}")
        ok, why = mailer.readiness()
        if not ok:
            raise SendRefused(409, why)
        to, bcc = recipients(case, kind)
        if not to and not bcc:
            raise SendRefused(409, "No recipients on record for this draft")
        subject, body = _split_subject(draft["body"])
        intended = to + bcc
        if live:
            if not to:                          # people notice: everyone in BCC, a copy to our own mailbox
                to = [mailer.sender]
            mail = OutgoingEmail(subject, body, to, bcc)
        else:
            if not mailer.redirect_to:
                raise SendRefused(409, "Test mode: set FAI_EMAIL_REDIRECT_TO in .env, or turn on 'Deliver to real recipients'")
            listed = "\n".join(f"  {a}" for a in intended[:50]) + (f"\n  … +{len(intended) - 50} more" if len(intended) > 50 else "")
            head = (f"[TEST SEND · {case['id']} Rev {case['rev_letter']} · would have gone to {len(intended)} "
                    f"recipient(s){' (BCC)' if bcc else ''}]\n{listed}\n\n{'-' * 60}\n\n")
            mail = OutgoingEmail(f"[TEST] {subject}", head + body, [mailer.redirect_to])
        meta = {"intended": len(intended), "redirected": not live, "delivered_to": mail.to + mail.bcc}
        return mail, meta

    def record(self, case: dict, kind: str, meta: dict, message_id: str, actor: str) -> dict:
        shown = meta["delivered_to"][:5]
        with self.store._lock:
            cur = self.db.execute("INSERT INTO outbox (case_id, rev, draft, intended, delivered_to, redirected, message_id, sent_by, ts) "
                                  "VALUES (?,?,?,?,?,?,?,?,?)",
                                  (case["id"], case["rev"], kind, meta["intended"],
                                   ", ".join(shown) + (f" +{len(meta['delivered_to']) - 5}" if len(meta["delivered_to"]) > 5 else ""),
                                   int(meta["redirected"]), message_id, actor, datetime.now().isoformat(timespec="seconds")))
            self.db.commit()
        mode = "test address" if meta["redirected"] else f"{meta['intended']} recipient(s)"
        self.store.audit(actor, "email_sent", case["id"], f"{kind} draft · Rev {case['rev_letter']} · to {mode}")
        row = self.db.execute("SELECT * FROM outbox WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row) | {"rev_letter": rev_letter(row["rev"])}
