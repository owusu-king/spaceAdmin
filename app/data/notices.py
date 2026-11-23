"""Facilities correspondence: RTF threads -> messages -> structured outage records.

James's rules implemented here:
* Quoted replies are cut from each message body so repeated quotations are not counted
  as separate events.
* Building identity comes from notice *contents*, never from the uploaded filename.
* Planned, revised and confirmed timings, confirmed impacts, preliminary concerns and
  explicitly unaffected services are kept apart.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from striprtf.striprtf import rtf_to_text

from app.config import SOURCE_DIR
from app.data.buildings import BuildingResolver
from app.logging_setup import get_logger, timed

log = get_logger("data.notices")

SYSTEMS = ["Domestic Hot and Cold Water", "Domestic Hot Water only", "Domestic Hot Water", "Domestic Cold Water",
           "Heating Hot Water", "Fume Hood Exhaust", "Building Exhaust", "Outside Air", "Air Conditioning",
           "Chilled Water", "Control Air", "Compressed Air", "Emergency Power", "Electrical", "Fire Alarm",
           "Fire", "Elevator", "IT Related", "Clean Room", "Pumps", "Steam", "Natural Gas", "Nitrogen",
           "Lab Air", "Vacuum", "Sewer", "Other"]
# Informal-email vocabulary -> canonical system names.
SYSTEM_KEYWORDS = {
    "steam": "Steam", "autoclave": "Steam", "elevator": "Elevator", "chilled water": "Chilled Water",
    "hot water": "Heating Hot Water", "domestic water": "Domestic Hot and Cold Water", "water outage": "Domestic Hot and Cold Water",
    "exhaust": "Building Exhaust", "fume hood": "Fume Hood Exhaust", "power": "Electrical", "electrical": "Electrical",
    "hvac": "Air Conditioning", " ac ": "Air Conditioning", "air handler": "Air Conditioning", "ahu": "Air Conditioning",
    "compressed air": "Compressed Air", "nitrogen": "Nitrogen", "fire alarm": "Fire Alarm", "network": "IT Related",
}
QUOTE_MARKERS = re.compile(
    r"^(On .{5,120}wrote:|-----Original Message-----|From: .{3,200}(Sent|Date): |Begin forwarded message|"
    r"-{5,} Forwarded message)", re.M)
EXCLUSION = re.compile(r"(will not be (affected|impacted)|not be affected|will remain (operational|unaffected|on)|"
                       r"remain operational|is not a full|unaffected|will not affect|not affected)", re.I)
IMPACT = re.compile(r"(out of service|will be (affected|shut|off|interrupted)|will affect|interruption|reduction|"
                    r"reduced|fluctuation|shut (off|down)|shutdown|unavailable|loss of|expect|no autoclaves|outage)", re.I)
IMPACT_INFORMAL = re.compile(r"(out of service|out of order|shut ?(off|down)|shutdown|will be (affected|interrupted|off)|"
                             r"will affect|unavailable|loss of|interruption|\bis out\b|not (working|functioning)|offline|"
                             r"no (water|power|steam|ac)\b)", re.I)
HEDGE = re.compile(r"\b(could|may|might|potential(ly)?|possible|if (a|the) .{0,40}(is|are) (necessary|required)|"
                   r"thinking of|waiting for)\b", re.I)
PRELIMINARY = re.compile(r"(could lead|potential disruption|if (a|the) [\w\s-]{0,30}(shutdown|outage) (is|becomes) "
                         r"(necessary|required)|waiting for an update|will let you know if|may be needed|possible outage)", re.I)
RESTORED = re.compile(r"(back in service|service (has been )?restored|has been restored|is restored|"
                      r"returned to normal|back online|operating normally)", re.I)
FILLER = {"space", "pa", "", "links"}


@dataclass
class Message:
    thread: str
    index: int
    sender: str
    subject: str
    date: datetime | None
    to: str
    cc: str
    own_text: str          # body without quoted history
    full_text: str         # body including forwarded content (TRIRIGA fields often live here)
    quoted: bool = False   # recovered from quoted history (timeline only, never counted as an event)

    @property
    def sr_ids(self) -> list[str]:
        return sorted(set(re.findall(r"SR-\d{7}", self.subject + " " + self.full_text[:4000])))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat() if self.date else None
        d["sr_ids"] = self.sr_ids
        return d


@dataclass
class OutageNotice:
    """Structured fields of one TRIRIGA outage notice."""
    sr: str | None
    kind: str                     # approval_request | planned_notification | unscheduled_alert | informal
    priority: str | None          # Scheduled | Urgent | Emergency
    start: datetime | None
    end: datetime | None
    buildings_raw: str
    buildings: list[str]          # resolved export names
    buildings_partial: bool       # TRIRIGA truncated the list ("...More")
    systems: list[str]
    shops: str
    reason: str
    scope: str
    equipment: str
    sent: datetime | None = None
    sender: str = ""
    subject: str = ""
    exclusions: list[str] = field(default_factory=list)
    impact_statements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end", "sent"):
            d[k] = d[k].isoformat() if d[k] else None
        d["duration_hours"] = self.duration_hours
        return d

    @property
    def duration_hours(self) -> float | None:
        if self.start and self.end:
            return round((self.end - self.start).total_seconds() / 3600, 2)
        return None


# ---------------------------------------------------------------------------- parsing helpers
_DATE_FORMATS = ["%B %d, %Y at %I:%M:%S %p", "%a, %b %d, %Y at %I:%M %p", "%a, %b %d, %Y, %I:%M %p",
                 "%m/%d/%y %I:%M %p", "%A, %B %d, %Y %I:%M %p", "%m/%d/%Y %H:%M:%S", "%b %d, %Y, at %I:%M %p",
                 "%b %d, %Y at %I:%M %p", "%a, %b %d, %Y %I:%M %p"]


def parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    t = text.replace("\u202f", " ").replace("\xa0", " ").strip()
    t = re.sub(r"\s+(CDT|CST|PST|PDT|EST|EDT|UTC)\b.*$", "", t)
    t = re.sub(r"\s*\(GMT[^)]*\)", "", t).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            continue
    return None


def rtf_file_text(path: Path) -> str:
    raw = path.read_text(encoding="latin-1")
    text = rtf_to_text(raw, errors="ignore")
    text = text.encode("utf-8", "replace").decode("utf-8").replace("\xa0", " ").replace("\u202f", " ")
    text = re.sub(r'\("(https?|mailto|tel):[^)]*"\)', "", text)      # drop hyperlink targets
    return re.sub(r"[ \t]*\n\s*\n+", "\n", text).replace("\u2028", "\n")


_QUOTED_HEADERS = [
    re.compile(r"^On (?P<hdr>[^\n]{8,160}?) wrote:\s*$", re.M),
    re.compile(r"From: (?P<from>[^\n]+?)\s+Sent: (?P<date>[^\n]+?)\s+To: (?:(?!From:)[\s\S]){0,600}?Subject: (?P<subj>[^\n]+)"),
]


def quoted_messages(body: str, thread: str) -> list[Message]:
    """Recover quoted replies ('On ... wrote:' / Outlook headers) as timeline-only messages."""
    hits = []
    for rx in _QUOTED_HEADERS:
        for m in rx.finditer(body):
            if "hdr" in m.groupdict() and m.group("hdr"):
                hdr = m.group("hdr")
                who = re.search(r"(?<![\w.'-])((?!(?:AM|PM)\b)[A-Z][\w.'-]+(?: [A-Z][\w.'-]+)+)\s*<", hdr)
                date = parse_date(hdr[: who.start()].strip(" ,")) if who else None
                hits.append((m.start(), m.end(), who.group(1) if who else "", date, ""))
            else:
                hits.append((m.start(), m.end(), m.group("from"), parse_date(m.group("date")), m.group("subj").strip()))
    hits.sort()
    out = []
    for k, (s, e, who, date, subj) in enumerate(hits):
        nxt = hits[k + 1][0] if k + 1 < len(hits) else len(body)
        seg = body[e:nxt].strip()
        if seg and date:
            out.append(Message(thread, -1, who, subj, date, "", "", seg[:3000], seg[:3000], quoted=True))
    return out


def split_messages(text: str, thread: str = "pasted", include_quoted: bool = False) -> list[Message]:
    """Split a thread into top-level messages (From/Subject/Date header blocks).

    With include_quoted, quoted replies whose (sender, minute) is not already a top-level
    message are recovered for the timeline (e.g. the original report in a reply chain)."""
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    msgs = _split_top_level(text, thread)
    if include_quoted:
        seen = {(m.sender.split("<")[0].strip().lower(), m.date.replace(second=0) if m.date else None) for m in msgs}
        for m in list(msgs):
            for q in quoted_messages(m.full_text, thread):
                key = (q.sender.split("<")[0].strip().lower(), q.date.replace(second=0))
                if key not in seen:
                    seen.add(key)
                    msgs.append(q)
    return msgs


def _split_top_level(text: str, thread: str) -> list[Message]:
    lines = text.splitlines()
    starts = [i for i in range(len(lines) - 2)
              if lines[i].startswith("From:") and lines[i + 1].startswith("Subject:") and lines[i + 2].startswith("Date:")]
    if not starts:
        body = text.strip()
        own = QUOTE_MARKERS.split(body)[0] if QUOTE_MARKERS.search(body) else body
        return [Message(thread, 0, "", "", None, "", "", own.strip(), body)]
    out = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(lines)
        header, body_start = {}, s
        for j in range(s, min(s + 6, e)):
            m = re.match(r"^(From|Subject|Date|To|Cc|Reply-To):\s*(.*)$", lines[j])
            if not m:
                break
            header[m.group(1)] = m.group(2).strip()
            body_start = j + 1
        body = "\n".join(lines[body_start:e]).strip()
        q = QUOTE_MARKERS.search(body)
        own = body[: q.start()] if q else body
        out.append(Message(thread, n, header.get("From", ""), header.get("Subject", ""),
                           parse_date(header.get("Date")), header.get("To", ""), header.get("Cc", ""),
                           own.strip(), body))
    return out


def _section(text: str, label: str, next_labels: list[str]) -> str:
    m = re.search(re.escape(label) + r"\s*(.*?)(?=" + "|".join(re.escape(x) for x in next_labels) + r"|$)", text, re.S)
    if not m:
        return ""
    lines = [ln.strip(" \t•") for ln in m.group(1).splitlines()]
    return " ".join(ln for ln in lines if ln.lower() not in FILLER).strip()


def _systems(segment: str) -> list[str]:
    found, rest = [], " " + segment + " "
    for s in SYSTEMS:
        if f" {s} " in rest or f" {s}\n" in rest:
            found.append(s)
            rest = rest.replace(s, " ")
    return found


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s*•\s*", text)
    return [p.strip(" -\t") for p in parts if len(p.strip()) > 8]


def _kind(subject: str) -> str:
    s = subject.lower()
    if "approval is required" in s:
        return "approval_request"
    if "notification of planned outage" in s:
        return "planned_notification"
    if "unscheduled outage" in s:
        return "unscheduled_alert"
    return "informal"


def parse_notice(text: str, resolver: BuildingResolver, subject: str = "") -> OutageNotice:
    """Extract structured fields from one notice (TRIRIGA template or informal email)."""
    flat = text.replace("\n", " \n ")
    sr = re.search(r"Outage Request ID:\s*(SR-\d+)", text) or re.search(r"(SR-\d{7})", subject + " " + text)
    prio = re.search(r"SR-\d+\s*-\s*(Scheduled|Urgent|Emergency) Outage", subject + " " + text)
    win = re.search(r"duration will be from\s*(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d)\s*to\s*(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d)", text)
    b_seg = _section(flat, "following building(s):", ["Following Systems Impacted:", "Shops"])
    sys_seg = _section(flat, "Following Systems Impacted:", ["Shops", "Outage Request Summary"])
    labels = ["Reason for Outage:", "Scope of Work (What you can expect):", "Affected Lab or Other Equipment:",
              "Contractors Involved:", "Links", "Outage Request Summary"]
    reason = _section(text, "Reason for Outage:", labels[1:])
    scope = _section(text, "Scope of Work (What you can expect):", labels[2:])
    equipment = _section(text, "Affected Lab or Other Equipment:", labels[3:])
    shops = _section(flat, "Shops Involved:", ["Outage Request Summary"])
    is_tririga = bool(win or b_seg)
    if is_tririga:
        buildings = [m.export_name for m in resolver.find_in_text(b_seg) if m.status == "resolved"]
        systems = _systems(sys_seg)
        body = " ".join([scope, equipment])
    else:
        buildings = [m.export_name for m in resolver.find_in_text(subject + " " + text) if m.status == "resolved"]
        low = " " + (subject + " " + text).lower() + " "
        systems = sorted({v for k, v in SYSTEM_KEYWORDS.items() if k in low})
        body = text
    sents = sentences(body)
    return OutageNotice(
        sr=sr.group(1) if sr else None, kind=_kind(subject) if is_tririga else "informal",
        priority=prio.group(1) if prio else None,
        start=parse_date(win.group(1)) if win else None, end=parse_date(win.group(2)) if win else None,
        buildings_raw=b_seg, buildings=list(dict.fromkeys(buildings)), buildings_partial="...More" in b_seg,
        systems=systems, shops=shops, reason=reason, scope=scope, equipment=equipment, subject=subject,
        exclusions=[s for s in sents if EXCLUSION.search(s) and not re.search(r"\bensure\b", s, re.I)],
        impact_statements=[s for s in sents if (IMPACT if is_tririga else IMPACT_INFORMAL).search(s)
                           and not EXCLUSION.search(s)],
    )


# ---------------------------------------------------------------------------- corpus & history
@dataclass
class Corpus:
    messages: list[Message]
    notices: list[OutageNotice]        # one per TRIRIGA message (duplicates kept for timeline use)
    history: list[dict]                # one row per SR id (deduplicated)


def _history(notices: list[OutageNotice]) -> list[dict]:
    by_sr: dict[str, list[OutageNotice]] = {}
    for n in notices:
        if n.sr and n.kind != "informal":
            by_sr.setdefault(n.sr, []).append(n)
    rows = []
    for sr, group in by_sr.items():
        group.sort(key=lambda n: n.sent or datetime.min)
        timed_ = [n for n in group if n.start]
        latest = timed_[-1] if timed_ else group[-1]
        windows = list(dict.fromkeys((n.start, n.end) for n in timed_))
        rows.append({
            "sr": sr, "priority": next((n.priority for n in group if n.priority), None),
            "kinds": sorted({n.kind for n in group}),
            "start": latest.start.isoformat() if latest.start else None,
            "end": latest.end.isoformat() if latest.end else None,
            "duration_hours": latest.duration_hours,
            "revised": len(windows) > 1,
            "superseded_windows": [[a.isoformat(), b.isoformat()] for a, b in windows[:-1] if a and b],
            "buildings": sorted({b for n in group for b in n.buildings}),
            "buildings_partial": any(n.buildings_partial for n in group),
            "buildings_raw": latest.buildings_raw,
            "systems": sorted({s for n in group for s in n.systems}),
            "reason": latest.reason or next((n.reason for n in group if n.reason), ""),
            "scope": latest.scope or next((n.scope for n in group if n.scope), ""),
            "exclusions": sorted({e for n in group for e in n.exclusions}),
            "first_notice": group[0].sent.isoformat() if group[0].sent else None,
            "messages": len(group), "threads": sorted({n.sender for n in group}),
        })
    return sorted(rows, key=lambda r: r["start"] or "")


@lru_cache(maxsize=1)
def load_corpus(source_dir: str = str(SOURCE_DIR)) -> Corpus:
    resolver = BuildingResolver()
    messages: list[Message] = []
    with timed(log, "correspondence loaded"):
        for path in sorted(Path(source_dir).glob("*.rtf")):
            msgs = split_messages(rtf_file_text(path), thread=path.name)
            messages.extend(msgs)
        # identical (sender, subject, date) blocks appear when threads overlap
        seen, unique = set(), []
        for m in messages:
            key = (m.sender, m.subject, m.date)
            if key not in seen:
                seen.add(key)
                unique.append(m)
        notices = []
        for m in unique:
            if "SR-" in m.subject or "Outage Request ID" in m.full_text[:3000]:
                n = parse_notice(m.full_text[:6000], resolver, m.subject)
                n.sent, n.sender = m.date, m.sender
                notices.append(n)
        history = _history(notices)
    log.info(f"corpus messages={len(unique)} duplicates_removed={len(messages) - len(unique)} "
             f"tririga_notices={len(notices)} distinct_outage_requests={len(history)}")
    return Corpus(unique, notices, history)
