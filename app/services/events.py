"""Events calendar used for outage conflict detection.

Sources (each event keeps its source so the UI can show where it came from):
  correspondence  dated building activities found in the supplied emails (tenant notices,
                  closures, drills, recess dates mentioned in messages). Date-only.
  ics             imported iCalendar files, e.g. an export from Rice's public events calendar
  manual          events entered by James or the team
  blocked         times James has ruled out for outages ("we cannot have the outage then")

An event with no building applies campus-wide. The store never claims completeness: the
scheduling engine still reports class schedules and unlisted events as coverage gaps.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from app.logging_setup import get_logger

log = get_logger("events")

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
RX_NAMED = re.compile(_MONTH + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*[-–]\s*(\d{1,2})(?:st|nd|rd|th)?)?(?:,?\s+(\d{4}))?\b", re.I)
RX_NUMERIC = re.compile(r"(?<![\d/:])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/:])")
ACADEMIC = re.compile(r"((?:[A-Z][a-z]+ )?(?:recess|holiday|commencement|graduation|move-in|final exams?|finals|"
                      r"closed to public|spring break|fall break))", re.I)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}


def _year_for(month: int, day: int, ref: datetime) -> int:
    """Pick the year that puts the date nearest to the reference date (emails talk about the near future/past)."""
    best = min((ref.year - 1, ref.year, ref.year + 1),
               key=lambda y: abs((date(y, month, min(day, 28)) - ref.date()).days))
    return best


def extract_dates(text: str, ref: datetime) -> list[tuple[date, date]]:
    """Dates written in prose: 'Wednesday, November 25th', 'Feb 18-19, 2026', '5/15/26'. Returns (start, end) days."""
    out = []
    for m in RX_NAMED.finditer(text):
        month = MONTHS[m.group(1)[:3].lower()]
        d1 = int(m.group(2))
        d2 = int(m.group(3)) if m.group(3) else d1
        year = int(m.group(4)) if m.group(4) else _year_for(month, d1, ref)
        try:
            out.append((date(year, month, d1), date(year, month, d2)))
        except ValueError:
            continue
    for m in RX_NUMERIC.finditer(text):
        mo, d = int(m.group(1)), int(m.group(2))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            continue
        y = m.group(3)
        year = (2000 + int(y) if len(y) == 2 else int(y)) if y else _year_for(mo, d, ref)
        try:
            out.append((date(year, mo, d), date(year, mo, d)))
        except ValueError:
            continue
    return list(dict.fromkeys(out))


def extract_duration_hours(text: str) -> float | None:
    m = re.search(r"\b(\d{1,2}|" + "|".join(WORDS) + r")[- ]hours?\b", text, re.I)
    if not m:
        return None
    v = m.group(1).lower()
    return float(WORDS.get(v, v))


def parse_ics(text: str) -> list[dict]:
    """Minimal iCalendar reader: VEVENT with SUMMARY, DTSTART, DTEND, LOCATION (folded lines supported)."""
    text = re.sub(r"\r?\n[ \t]", "", text)              # unfold continuation lines
    events = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        fields = {}
        for line in block.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                fields[key.split(";")[0].upper()] = (val.strip(), "VALUE=DATE" in key.upper())
        if "DTSTART" not in fields:
            continue
        start, all_day = _ics_time(*fields["DTSTART"])
        end = _ics_time(*fields["DTEND"])[0] if "DTEND" in fields else start + (timedelta(days=1) if all_day else timedelta(hours=1))
        events.append({"title": fields.get("SUMMARY", ("Untitled event", False))[0].replace("\\,", ","),
                       "start": start, "end": end, "all_day": all_day,
                       "location": fields.get("LOCATION", ("", False))[0].replace("\\,", ",")})
    return events


def _ics_time(value: str, is_date: bool) -> tuple[datetime, bool]:
    v = value.rstrip("Z")
    if is_date or len(v) == 8:
        return datetime.strptime(v[:8], "%Y%m%d"), True
    return datetime.strptime(v[:15], "%Y%m%dT%H%M%S"), False


class EventStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, start TEXT,
            end TEXT, building TEXT, kind TEXT, source TEXT, note TEXT, all_day INTEGER, created TEXT)""")
        self.db.commit()

    def add(self, title: str, start: datetime, end: datetime, building: str | None = None, kind: str = "event",
            source: str = "manual", note: str = "", all_day: bool = False) -> dict:
        """Add an event. The same event twice (same title, time, building, kind and source, e.g. a double click or
        importing one calendar file again) is kept once: the existing event comes back with duplicate=True."""
        if end <= start:
            raise ValueError("Event end must be after its start")
        with self._lock:
            same = self.db.execute("SELECT id FROM events WHERE title=? AND start=? AND end=? AND building IS ? AND kind=? AND source=?",
                                   (title[:200], start.isoformat(), end.isoformat(), building, kind, source)).fetchone()
            if same:
                return {**self.get(same[0]), "duplicate": True}
            cur = self.db.execute("INSERT INTO events (title, start, end, building, kind, source, note, all_day, created) "
                                  "VALUES (?,?,?,?,?,?,?,?,?)", (title[:200], start.isoformat(), end.isoformat(), building,
                                                                 kind, source, note[:500], int(all_day),
                                                                 datetime.now().isoformat(timespec="seconds")))
            self.db.commit()
        return self.get(cur.lastrowid)

    def get(self, eid: int) -> dict | None:
        r = self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        return dict(r) if r else None

    def delete(self, eid: int) -> bool:
        with self._lock:
            n = self.db.execute("DELETE FROM events WHERE id=? AND source IN ('manual','blocked','ics')", (eid,)).rowcount
            self.db.commit()
        return n > 0

    def list(self, since: datetime | None = None, limit: int = 300) -> list[dict]:
        q, args = "SELECT * FROM events", []
        if since:
            q += " WHERE end >= ?"
            args.append(since.isoformat())
        rows = self.db.execute(q + " ORDER BY start LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) for r in rows]

    def overlapping(self, buildings: list[str], start: datetime, end: datetime) -> list[dict]:
        """Events in the window that apply to one of the buildings or campus-wide."""
        rows = self.db.execute("SELECT * FROM events WHERE start < ? AND end > ? ORDER BY start",
                               (end.isoformat(), start.isoformat())).fetchall()
        return [dict(r) for r in rows if r["building"] is None or r["building"] in buildings]

    def count(self) -> dict:
        return {r[0]: r[1] for r in self.db.execute("SELECT source, COUNT(*) FROM events GROUP BY source").fetchall()}

    # ---------------------------------------------------------------- importers
    def import_ics(self, text: str, resolver) -> dict:
        """Import a calendar file; events already on the calendar are skipped, so importing again is safe."""
        n = skipped = 0
        for e in parse_ics(text):
            hits = [m for m in resolver.find_in_text(e["location"] + " " + e["title"]) if m.status == "resolved"]
            note = f"Location: {e['location']}" if e["location"] else "No location given; treated as campus-wide"
            ev = self.add(e["title"], e["start"], e["end"], hits[0].export_name if hits else None, "event", "ics", note, e["all_day"])
            if ev.get("duplicate"):
                skipped += 1
            else:
                n += 1
        log.info(f"ics imported events={n} already_present={skipped}")
        return {"imported": n, "skipped": skipped}

    def seed_from_corpus(self, corpus, resolver) -> int:
        """Dated building activities from the correspondence (subjects, plus academic dates in bodies)."""
        if self.db.execute("SELECT COUNT(*) FROM events WHERE source='correspondence'").fetchone()[0]:
            return 0
        n, seen = 0, set()
        for m in corpus.messages:
            if not m.date or "SR-" in m.subject:
                continue                                    # TRIRIGA requests are tracked as outages
            texts = [(m.subject, None)]
            texts += [(s, "academic") for s in re.split(r"(?<=[.!?])\s+", m.own_text[:3000]) if ACADEMIC.search(s)]
            for text, kind in texts:
                for d1, d2 in extract_dates(text, m.date)[:2]:
                    title = re.sub(r"^((re|fw|fwd)\s*:\s*|\[[^\]]+\]\s*)+", "", m.subject, flags=re.I).strip()
                    if kind == "academic":
                        name = re.sub(r"^(the|this|upcoming|day|a|our)\s+", "", ACADEMIC.search(text).group(1).strip(), flags=re.I)
                        title = f"{name[0].upper() + name[1:]} (per email from {m.sender.split('<')[0].strip()}, {m.date:%b %d})"
                    key = (title.lower(), d1)
                    if key in seen:
                        continue
                    seen.add(key)
                    hits = [h for h in resolver.find_in_text(m.subject + " " + text) if h.status == "resolved"]
                    building = hits[0].export_name if hits and kind != "academic" else None
                    self.add(title, datetime.combine(d1, datetime.min.time()), datetime.combine(d2 + timedelta(days=1), datetime.min.time()),
                             building, kind or "building_activity", "correspondence",
                             f"From email of {m.date:%b %d, %Y}: {text[:160]}", all_day=True)
                    n += 1
        log.info(f"events seeded from correspondence events={n}")
        return n
