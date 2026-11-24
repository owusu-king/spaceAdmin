"""Case management: persistent cases, revisions, human review and audit (SQLite).

Lifecycle:  ReadyForReview --approve--> Approved (execution locked)
                           --return---> NeedsRevision --rerun--> ReadyForReview (Rev +1)
                           --decline--> Declined
An approval is recorded against one revision letter; a rerun creates a new revision and
invalidates earlier approvals (architecture §15.2).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path

import numpy as np

from app.config import DB_PATH
from app.logging_setup import current_run_id, get_logger

log = get_logger("cases")
DECISIONS = {"approve": "Approved", "return": "NeedsRevision", "decline": "Declined"}
REVIEWER = "James Delgado (Facilities Administrator)"


def rev_letter(n: int) -> str:
    return chr(ord("A") + n - 1) if n <= 26 else f"R{n}"


def _default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "__dict__"):
        return vars(o)
    return str(o)


def dumps(obj) -> str:
    return json.dumps(obj, default=_default)


SPACE_LABELS = {"new_office": "New office", "change_assignment": "Reassignment", "vacancy_review": "Vacancy review"}


def _short(items: list[str], n: int) -> str:
    return ", ".join(items[:n]) + (f" +{len(items) - n}" if len(items) > n else "")


def case_title(kind: str, out: dict) -> str:
    """One format for every case: '<what> · <where or who>'. Derived from the analysis, so a rerun updates it."""
    if kind == "space":
        r = out.get("request", {})
        what = SPACE_LABELS.get(r.get("request_type"), r.get("function") or "Space request")
        where = r.get("department") or _short(r.get("preferred_buildings") or [], 2) or "Department not given"
        return f"{what} · {where}"
    o = out.get("outage", {})
    systems = _short(o.get("affected_systems") or [], 1) or "Outage"
    where = _short([b.get("code") or b.get("export_name") for b in o.get("buildings") or []], 3) or "Building not identified"
    return f"{systems} · {where}"


def case_reference(kind: str, out: dict) -> str:
    """Where the case came from: the TRIRIGA request number, or the source of the request."""
    if kind == "space":
        return "Space request"
    return (out.get("outage") or {}).get("request_reference") or "Email"


class CaseStore:
    def __init__(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.outage_lock = threading.RLock()       # held while checking for and opening an outage case (API and email)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, kind TEXT, title TEXT, status TEXT, rev INTEGER,
                created TEXT, updated TEXT, input TEXT, output TEXT, run_id TEXT);
            CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, rev INTEGER,
                decision TEXT, rationale TEXT, reviewer TEXT, ts TEXT, invalidated INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, actor TEXT,
                action TEXT, case_id TEXT, detail TEXT);
        """)

    # ------------------------------------------------------------------ audit
    def audit(self, actor: str, action: str, case_id: str | None = None, detail: str = "") -> None:
        with self._lock:
            self.db.execute("INSERT INTO audit (ts, run_id, actor, action, case_id, detail) VALUES (?,?,?,?,?,?)",
                            (datetime.now().isoformat(timespec="seconds"), current_run_id(), actor, action, case_id, detail))
            self.db.commit()

    def audit_events(self, limit: int = 200) -> list[dict]:
        rows = self.db.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ cases
    def create(self, kind: str, title: str | None, inp: dict, out: dict) -> dict:
        title = case_title(kind, out)
        prefix = "SA" if kind == "space" else "OC"
        with self._lock:
            cid = f"{prefix}-{self._next_number(prefix):04d}"
            now = datetime.now().isoformat(timespec="seconds")
            self.db.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (cid, kind, title, "ReadyForReview", 1, now, now, dumps(inp), dumps(out), out.get("run_id")))
            self.db.commit()
        self.audit(out.get("agent", "agent"), "case_created", cid, f"{title} · Rev A")
        log.info(f"case created id={cid} kind={kind}")
        return self.get(cid)

    def _next_number(self, prefix: str) -> int:
        """One more than the highest number ever used, including deleted cases (kept in the audit log),
        so a case number is never reused."""
        used = [r[0] for r in self.db.execute("SELECT id FROM cases WHERE id LIKE ?", (f"{prefix}-%",))]
        used += [r[0] for r in self.db.execute("SELECT DISTINCT case_id FROM audit WHERE case_id LIKE ?", (f"{prefix}-%",))]
        nums = [int(x.split("-")[1]) for x in used if x and x.split("-")[1].isdigit()]
        return max(nums, default=0) + 1

    def delete(self, cid: str, actor: str) -> dict:
        """Delete a case with its reviews and sent-email records. Emails that opened it stay in the inbox (unlinked).
        The audit log keeps a record of the deletion."""
        case = self.get(cid)
        if not case:
            raise KeyError(cid)
        with self._lock:
            self.db.execute("DELETE FROM reviews WHERE case_id=?", (cid,))
            for table, sql in (("outbox", "DELETE FROM outbox WHERE case_id=?"),
                               ("inbox", "UPDATE inbox SET case_id=NULL, action='case deleted' WHERE case_id=?")):
                if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                    self.db.execute(sql, (cid,))
            self.db.execute("DELETE FROM cases WHERE id=?", (cid,))
            self.db.commit()
        self.audit(actor, "case_deleted", cid, f"{case['title']} · Rev {case['rev_letter']} · {case['status']}")
        log.info(f"case deleted id={cid}")
        return {"deleted": cid, "title": case["title"]}

    def rerun(self, cid: str, inp: dict, out: dict) -> dict:
        with self._lock:                                   # read the revision and write the next one together
            case = self.get(cid)
            if not case:
                raise KeyError(cid)
            rev = case["rev"] + 1
            self.db.execute("UPDATE reviews SET invalidated=1 WHERE case_id=? AND rev<?", (cid, rev))
            self.db.execute("UPDATE cases SET rev=?, status='ReadyForReview', updated=?, input=?, output=?, run_id=?, title=? WHERE id=?",
                            (rev, datetime.now().isoformat(timespec="seconds"), dumps(inp), dumps(out), out.get("run_id"),
                             case_title(case["kind"], out), cid))
            self.db.commit()
        self.audit(out.get("agent", "agent"), "case_rerun", cid, f"Rev {rev_letter(rev)}; earlier approvals invalidated")
        return self.get(cid)

    def review(self, cid: str, decision: str, rationale: str, rev: int) -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        if decision in ("return", "decline") and not rationale.strip():
            raise ValueError("A written rationale is required to return or decline")
        with self._lock:                                   # checks and the decision together: two clicks record one
            case = self.get(cid)
            if not case:
                raise KeyError(cid)
            if rev != case["rev"]:
                raise ValueError(f"Review targets Rev {rev_letter(rev)} but the current revision is {case['rev_letter']}")
            if case["status"] in ("Approved", "Declined"):
                raise ValueError(f"Case is already {case['status']}")
            if any(r["rev"] == rev and r["decision"] == decision and not r["invalidated"] for r in case["reviews"]):
                raise ValueError(f"Rev {rev_letter(rev)} was already {decision}d" if decision.endswith("e") else f"Rev {rev_letter(rev)} was already {decision}ed")
            self.db.execute("INSERT INTO reviews (case_id, rev, decision, rationale, reviewer, ts) VALUES (?,?,?,?,?,?)",
                            (cid, rev, decision, rationale, REVIEWER, datetime.now().isoformat(timespec="seconds")))
            self.db.execute("UPDATE cases SET status=?, updated=? WHERE id=?",
                            (DECISIONS[decision], datetime.now().isoformat(timespec="seconds"), cid))
            self.db.commit()
        self.audit(REVIEWER, f"review_{decision}", cid, f"Rev {rev_letter(rev)}: {rationale}".strip())
        return self.get(cid)

    def find_outage(self, reference: str | None = None, source: dict | None = None) -> dict | None:
        """The existing case for an outage request: same TRIRIGA request number (SR-…), or the same source notice
        (message, thread file or scenario). One request has one case; later information becomes a new revision."""
        if not reference and not source:
            return None
        rows = self.db.execute("SELECT id, input, output FROM cases WHERE kind='outage' ORDER BY updated DESC").fetchall()
        for r in rows:
            inp, out = json.loads(r["input"]), json.loads(r["output"])
            if reference and reference in ((out.get("outage") or {}).get("request_reference"), inp.get("thread_key")):
                return self.get(r["id"])
            if source and all(k in inp and inp[k] == v for k, v in source.items()):
                return self.get(r["id"])
        return None

    def get(self, cid: str) -> dict | None:
        r = self.db.execute("SELECT * FROM cases WHERE id=?", (cid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["input"], d["output"] = json.loads(d["input"]), json.loads(d["output"])
        d["rev_letter"] = rev_letter(d["rev"])
        d["title"], d["reference"] = case_title(d["kind"], d["output"]), case_reference(d["kind"], d["output"])
        d["reviews"] = [dict(x) | {"rev_letter": rev_letter(x["rev"])} for x in
                        self.db.execute("SELECT * FROM reviews WHERE case_id=? ORDER BY id", (cid,)).fetchall()]
        return d

    def list(self) -> list[dict]:
        rows = self.db.execute("SELECT id, kind, title, status, rev, created, updated, output FROM cases ORDER BY updated DESC").fetchall()
        out = []
        for r in rows:
            o = json.loads(r["output"])
            out.append({"id": r["id"], "kind": r["kind"], "title": case_title(r["kind"], o), "status": r["status"],
                        "reference": case_reference(r["kind"], o),
                        "rev_letter": rev_letter(r["rev"]), "created": r["created"], "updated": r["updated"],
                        "summary": o.get("summary", ""),
                        "open_questions": len(o.get("verification_questions") or o.get("follow_up_questions") or [])})
        return out
