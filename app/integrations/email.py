"""Email intake for the Outage Agent (integration Level 2: read-only).

Sources
  ReplaySource  replays the supplied correspondence in chronological order (demo / offline)
  ImapSource    reads a mailbox over IMAPS, strictly read-only: the folder is opened with
                readonly=True and bodies are fetched with BODY.PEEK[] so nothing is marked read,
                moved or deleted. Credentials come only from the environment (the project's
                .env file, loaded by app.config, or variables set in the terminal).

Intake pipeline (EmailIntake.poll)
  fetch -> skip seen Message-IDs -> triage (outage-related or not) -> group into an incident
  thread (SR number, else normalized subject) -> new thread: open an outage case
  -> follow-up on a known thread: re-run that case with all its messages (new revision,
  earlier approvals invalidated).

Privacy: for non-outage mail only sender, subject and date are stored, never the body.
Nothing is ever sent; adapters reject execution.
"""
from __future__ import annotations

import email
import imaplib
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Protocol

from app.config import ENV_FILE  # noqa: F401  importing config loads .env before os.environ is read
from app.logging_setup import get_logger, kv

log = get_logger("integrations.email")
OUTAGE_SUBJECT = re.compile(r"(SR-\d{7}|outage|shut ?down|elevator|steam|chilled water|hot water|exhaust|power|"
                            r"\bAC\b|HVAC|air handler|AHU|out of (order|service)|maintenance|leak)", re.I)
TRIRIGA_SENDER = re.compile(r"tririga", re.I)
SUBJECT_NOISE = re.compile(r"^((re|fw|fwd|aw)\s*:\s*|\[[^\]]+\]\s*|(urgent|alert!?)\s*[:,-]?\s*)+", re.I)


class ExecutionNotPermitted(PermissionError):
    pass


@dataclass
class InboundEmail:
    message_id: str
    sender: str
    subject: str
    date: datetime | None
    body: str

    def as_notice_text(self) -> str:
        """Header block in the format the Outage Agent's parser expects."""
        d = self.date.strftime("%B %d, %Y at %I:%M:%S %p") if self.date else ""
        return f"From: {self.sender}\nSubject: {self.subject}\nDate: {d}\nTo: \n{self.body}"


class EmailSource(Protocol):
    mode: str
    def health_check(self) -> dict: ...
    def fetch(self, limit: int) -> list[InboundEmail]: ...
    def execute(self, action) -> None: ...


# ------------------------------------------------------------------------------ replay
class ReplaySource:
    """Delivers the supplied correspondence as if it were arriving, oldest first."""
    mode = "replay"

    def __init__(self, corpus, since: datetime | None = None) -> None:
        msgs = [m for m in corpus.messages if m.date and (since is None or m.date >= since)]
        self._queue = sorted(msgs, key=lambda m: m.date)
        self._pos = 0

    def health_check(self) -> dict:
        return {"status": "ok", "mode": self.mode, "remaining": len(self._queue) - self._pos, "total": len(self._queue)}

    def fetch(self, limit: int = 10) -> list[InboundEmail]:
        batch = self._queue[self._pos: self._pos + limit]
        self._pos += len(batch)
        return [InboundEmail(f"replay-{m.thread}-{m.index}-{m.date:%Y%m%d%H%M%S}", m.sender, m.subject, m.date, m.full_text)
                for m in batch]

    def execute(self, action) -> None:
        raise ExecutionNotPermitted("Email adapter is read-only")


# ------------------------------------------------------------------------------ IMAP
def _decode(value: str | None) -> str:
    return str(make_header(decode_header(value))) if value else ""


def _body_text(msg: email.message.Message) -> str:
    plain, html = [], []
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart" or part.get("Content-Disposition", "").startswith("attachment"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        (plain if part.get_content_type() == "text/plain" else html if part.get_content_type() == "text/html" else []).append(text)
    if plain:
        return "\n".join(plain)
    raw = "\n".join(html)
    raw = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", raw)
    return unescape(re.sub(r"<[^>]+>", "", raw))


def parse_rfc822(raw: bytes) -> InboundEmail:
    msg = email.message_from_bytes(raw)
    try:
        date = parsedate_to_datetime(msg.get("Date")).replace(tzinfo=None) if msg.get("Date") else None
    except (TypeError, ValueError):
        date = None
    mid = (msg.get("Message-ID") or "").strip() or f"no-id-{hash(raw)}"
    return InboundEmail(mid, _decode(msg.get("From")), _decode(msg.get("Subject")), date, _body_text(msg).strip())


class ImapSource:
    """Read-only IMAPS reader. Configure with FAI_IMAP_HOST, FAI_IMAP_USER, FAI_IMAP_PASSWORD,
    optional FAI_IMAP_FOLDER (INBOX) and FAI_IMAP_SINCE_DAYS (14)."""
    mode = "imap"

    def __init__(self, env: dict | None = None, client_factory=imaplib.IMAP4_SSL) -> None:
        e = env if env is not None else os.environ
        self.host, self.user, self._password = e.get("FAI_IMAP_HOST"), e.get("FAI_IMAP_USER"), e.get("FAI_IMAP_PASSWORD")
        self.folder = e.get("FAI_IMAP_FOLDER", "INBOX")
        self.since_days = int(e.get("FAI_IMAP_SINCE_DAYS", "14"))
        self._factory = client_factory

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user and self._password)

    def health_check(self) -> dict:
        if not self.configured:
            return {"status": "not_configured", "mode": self.mode,
                    "detail": "Set FAI_IMAP_HOST, FAI_IMAP_USER and FAI_IMAP_PASSWORD"}
        try:
            with self._connect():
                return {"status": "ok", "mode": self.mode, "host": self.host, "folder": self.folder}
        except Exception as exc:  # report, never crash the app
            return {"status": "error", "mode": self.mode, "detail": f"{type(exc).__name__}: {exc}"}

    def _connect(self):
        client = self._factory(self.host)
        client.login(self.user, self._password)
        typ, _ = client.select(self.folder, readonly=True)          # read-only: no flags change
        if typ != "OK":
            client.logout()
            raise ConnectionError(f"Cannot open folder {self.folder}")
        return client

    def fetch(self, limit: int = 25) -> list[InboundEmail]:
        if not self.configured:
            return []
        since = (datetime.now() - timedelta(days=self.since_days)).strftime("%d-%b-%Y")
        client = self._connect()
        try:
            typ, data = client.search(None, "SINCE", since)
            ids = data[0].split()[-limit:] if typ == "OK" and data and data[0] else []
            out = []
            for i in ids:
                typ, parts = client.fetch(i, "(BODY.PEEK[])")             # PEEK: does not set \\Seen
                raw = next((p[1] for p in parts if isinstance(p, tuple)), None)
                if typ == "OK" and raw:
                    out.append(parse_rfc822(raw))
            return out
        finally:
            try:
                client.close()
            finally:
                client.logout()

    def execute(self, action) -> None:
        raise ExecutionNotPermitted("Email adapter is read-only")


# ------------------------------------------------------------------------------ intake
def is_outage_related(m: InboundEmail) -> tuple[bool, str]:
    """Rule-based triage (a learned classifier is a planned replacement; see README)."""
    if TRIRIGA_SENDER.search(m.sender) or "Outage Request ID" in m.body[:4000]:
        return True, "TRIRIGA notice"
    if OUTAGE_SUBJECT.search(m.subject):
        return True, "outage keywords in subject"
    return False, "no outage indicators in sender or subject"


def thread_key(m: InboundEmail) -> str:
    sr = re.search(r"SR-\d{7}", m.subject + " " + m.body[:4000])
    if sr:
        return sr.group(0)
    return "subj:" + re.sub(r"\s+", " ", SUBJECT_NOISE.sub("", m.subject)).strip().lower()[:120]


class EmailIntake:
    def __init__(self, source: EmailSource, coord, store) -> None:
        self.source, self.coord, self.store = source, coord, store
        self._lock = threading.Lock()
        self.last_poll: str | None = None
        store.db.executescript("""
            CREATE TABLE IF NOT EXISTS inbox (message_id TEXT PRIMARY KEY, received TEXT, sent TEXT, sender TEXT,
                subject TEXT, relevant INTEGER, reason TEXT, thread_key TEXT, body TEXT, case_id TEXT, action TEXT);
        """)

    def _seen(self, mid: str) -> bool:
        return self.store.db.execute("SELECT 1 FROM inbox WHERE message_id=?", (mid,)).fetchone() is not None

    def _record(self, m: InboundEmail, relevant: bool, reason: str, key: str | None, case_id: str | None, action: str) -> None:
        with self.store._lock:
            self.store.db.execute("INSERT INTO inbox VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                m.message_id, datetime.now().isoformat(timespec="seconds"), m.date.isoformat() if m.date else None,
                m.sender, m.subject, int(relevant), reason, key, m.as_notice_text() if relevant else None, case_id, action))
            self.store.db.commit()

    def _open_case_for(self, key: str) -> str | None:
        row = self.store.db.execute("SELECT case_id FROM inbox WHERE thread_key=? AND case_id IS NOT NULL LIMIT 1", (key,)).fetchone()
        return row[0] if row else None

    def poll(self, limit: int = 10) -> dict:
        from app.agents.schemas import OutageRequest
        with self._lock:
            fetched = self.source.fetch(limit)
            summary = {"fetched": len(fetched), "duplicates": 0, "ignored": 0, "cases_opened": [], "cases_updated": []}
            for m in fetched:
                if self._seen(m.message_id):
                    summary["duplicates"] += 1
                    continue
                relevant, reason = is_outage_related(m)
                if not relevant:
                    self._record(m, False, reason, None, None, "ignored")
                    summary["ignored"] += 1
                    continue
                key = thread_key(m)
                existing = self._open_case_for(key)
                self._record(m, True, reason, key, existing, "pending")
                bodies = [r[0] for r in self.store.db.execute(
                    "SELECT body FROM inbox WHERE thread_key=? AND relevant=1 ORDER BY sent", (key,)).fetchall()]
                text = "\n".join(bodies)
                try:
                    out = self.coord.run_outage(OutageRequest(text=text, source=f"email · {m.subject}"[:200]))
                except Exception as exc:
                    log.error(f"intake failed message={m.message_id} error={exc!r}")
                    self._set_action(m.message_id, None, f"error: {type(exc).__name__}")
                    continue
                if not existing:
                    merged = self._related_informal_case(m, key, out)
                    if merged:                 # informal heads-up + later TRIRIGA notice = one incident
                        existing = merged
                        bodies = [r[0] for r in self.store.db.execute(
                            "SELECT body FROM inbox WHERE thread_key=? AND relevant=1 ORDER BY sent", (key,)).fetchall()]
                        text = "\n".join(bodies)
                        out = self.coord.run_outage(OutageRequest(text=text, source=f"email · {m.subject}"[:200]))
                inp = {"source": "email", "thread_key": key, "messages": len(bodies), "text": text if len(text) < 20000 else None}
                if existing:
                    case = self.store.rerun(existing, inp, out)
                    self.store.audit("email_intake", "case_updated_by_email", existing, f"{m.subject} · Rev {case['rev_letter']}")
                    self._set_action(m.message_id, existing, f"updated Rev {case['rev_letter']}")
                    summary["cases_updated"].append(existing)
                else:
                    case = self.store.create("outage", None, inp, out)
                    self.store.audit("email_intake", "case_opened_by_email", case["id"], m.subject)
                    self._set_action(m.message_id, case["id"], "opened case")
                    summary["cases_opened"].append(case["id"])
            self.last_poll = datetime.now().isoformat(timespec="seconds")
        log.info("email poll " + kv(mode=self.source.mode, fetched=summary["fetched"], ignored=summary["ignored"],
                                     opened=len(summary["cases_opened"]), updated=len(summary["cases_updated"])))
        return summary

    def _related_informal_case(self, m: InboundEmail, key: str, out: dict) -> str | None:
        """For a new SR thread, find a case opened by informal email in the previous 14 days that covers
        the same building and system; re-key its messages to the SR so they form one incident."""
        if not key.startswith("SR-") or not m.date:
            return None
        bldgs = {b["export_name"] for b in out["outage"]["buildings"]}
        systems = set(out["outage"]["affected_systems"])
        rows = self.store.db.execute(
            "SELECT DISTINCT case_id, thread_key FROM inbox WHERE case_id IS NOT NULL AND thread_key LIKE 'subj:%' AND sent >= ?",
            ((m.date - timedelta(days=14)).isoformat(),)).fetchall()
        for case_id, old_key in rows:
            case = self.store.get(case_id)
            o = case["output"]["outage"] if case else None
            if o and bldgs & {b["export_name"] for b in o["buildings"]} and systems & set(o["affected_systems"]):
                with self.store._lock:
                    self.store.db.execute("UPDATE inbox SET thread_key=? WHERE thread_key=?", (key, old_key))
                    self.store.db.commit()
                log.info(f"thread merged informal={old_key} into={key} case={case_id}")
                return case_id
        return None

    def _set_action(self, mid: str, case_id: str | None, action: str) -> None:
        with self.store._lock:
            self.store.db.execute("UPDATE inbox SET case_id=COALESCE(?, case_id), action=? WHERE message_id=?", (case_id, action, mid))
            self.store.db.commit()

    def status(self, limit: int = 40) -> dict:
        rows = self.store.db.execute("SELECT message_id, received, sent, sender, subject, relevant, reason, thread_key, case_id, action "
                                     "FROM inbox ORDER BY sent DESC LIMIT ?", (limit,)).fetchall()
        counts = self.store.db.execute("SELECT COUNT(*), SUM(relevant) FROM inbox").fetchone()
        return {"source": self.source.health_check(), "last_poll": self.last_poll,
                "processed": counts[0] or 0, "relevant": counts[1] or 0, "items": [dict(r) for r in rows]}


def build_source(corpus) -> EmailSource:
    """FAI_EMAIL_SOURCE=imap uses the live mailbox; anything else replays the supplied threads."""
    if os.environ.get("FAI_EMAIL_SOURCE", "replay").lower() == "imap":
        return ImapSource()
    since = os.environ.get("FAI_REPLAY_SINCE", "2026-10-01")
    return ReplaySource(corpus, datetime.fromisoformat(since) if since else None)
