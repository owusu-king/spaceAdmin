"""Scenario library: realistic inputs built from the supplied materials.

Outage scenarios use the actual RTF threads or single TRIRIGA messages; space scenarios
follow the request types James described. Each scenario carries the *expectations* used
by app/evaluation.py, written from James's documented examples (not from agent output).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import SOURCE_DIR
from app.data.notices import Corpus, Message, rtf_file_text

OUTAGE_KEYWORDS = re.compile(r"(SR-\d|outage|elevator|shut ?down|steam|water|exhaust|power|\bAC\b|HVAC|chilled|air handler|"
                             r"maintenance|out of (order|service)|AHU)", re.I)


def message_text(m: Message) -> str:
    """Re-create a single message with its header so the agent can date it."""
    date = m.date.strftime("%B %d, %Y at %I:%M:%S %p") if m.date else ""
    return f"From: {m.sender}\nSubject: {m.subject}\nDate: {date}\nTo: {m.to}\n{m.full_text}"


def outage_sources(corpus: Corpus) -> list[dict]:
    """Outage-related top-level messages offered as case inputs in the UI."""
    out = []
    for i, m in enumerate(corpus.messages):
        if OUTAGE_KEYWORDS.search(m.subject or ""):
            out.append({"id": i, "subject": m.subject, "sender": m.sender.split("<")[0].strip(),
                        "date": m.date.isoformat() if m.date else None, "thread": m.thread,
                        "sr": m.sr_ids[0] if m.sr_ids else None})
    return sorted(out, key=lambda x: x["date"] or "", reverse=True)


@dataclass
class Scenario:
    id: str
    title: str
    kind: str                      # space | outage
    description: str
    payload: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)


def _thread(name: str) -> str:
    return rtf_file_text(SOURCE_DIR / name)


def scenarios() -> list[Scenario]:
    return [
        Scenario("OUT-OES-EXHAUST", "Planned exhaust reduction · OES", "outage",
                 "Two-hour exhaust-flow reduction; the notice states fume hoods and snorkels are unaffected.",
                 {"text": _thread("OES System Notice - Notification of Planned Outage_ SR-1119085 - Scheduled Outage.rtf"),
                  "source": "OES SR-1119085"},
                 {"sr": "SR-1119085", "building": "O'Connor Engineering and Science", "start": "2026-11-19T07:00:00",
                  "exclusions_include": ["fume hoods", "snorkels"], "no_full_shutdown_claim": True}),
        Scenario("OUT-ABL-STEAM", "Emergency steam repair · ABL", "outage",
                 "Steam leak: early emails list heating and hot water as possible impacts; the confirmed scope is autoclaves only.",
                 {"text": _thread("ABL System Notice - ALERT! Unscheduled Outage_ SR-1118242 - Emergency Outage.rtf"),
                  "source": "ABL thread"},
                 {"sr": "SR-1118242", "building": "Biology", "start": "2026-10-27T08:00:00",
                  "confirmed_include": ["autoclave"], "preliminary_not_confirmed": ["heating", "hot water"],
                  "other_incidents_set_aside": True}),
        Scenario("OUT-BRC-POWER", "Utility power transfer · BRC and others", "outage",
                 "TRIRIGA approval request listing several buildings with a truncated building list.",
                 {"source": "BRC SR-1119169", "sr_message": "SR-1119169"},
                 {"sr": "SR-1119169", "building": "BRC", "partial_buildings": True, "start": "2026-12-01T06:00:00"}),
        Scenario("OUT-GRB-ELEVATOR", "Elevator out of order · GRB", "outage",
                 "Operational email chain without a TRIRIGA notice; the latest message confirms restoration.",
                 {"text": _thread("GRB System Notice - Re_ GRB East elevator is out of order..rtf"), "source": "GRB thread"},
                 {"building": "George R. Brown", "restoration_confirmed": True}),
        Scenario("OUT-KECK-ELEVATOR", "Elevator pit maintenance · Keck", "outage",
                 "Recurring elevator work being planned; dates not yet set. Other buildings appear only in quoted history.",
                 {"text": _thread("KECK System Notice - KECK_ Re_ Elevator maintenance for Keck.rtf"), "source": "KECK thread"},
                 {"building": "Keck", "only_building": True, "asks_for_timing": True}),
        Scenario("OUT-BRK-ELEVATOR", "Elevator work during Thanksgiving recess · BRK", "outage",
                 "Mandatory elevator maintenance proposed for Wednesday, November 25 in prose, with no time of day. "
                 "Another email says that date starts the Thanksgiving recess.",
                 {"text": _thread("BRK System Notice - Re_ Elevator maintenance for Brockman.rtf"), "source": "BRK thread"},
                 {"building": "Brockman Hall", "date_only_start": "2026-11-25", "conflict_title": "recess"}),
        Scenario("SPA-NEAR", "Office near a faculty member · new hire", "space",
                 "A new hire in Professor Bartel's group needs an office close to hers.",
                 {"request_type": "new_office", "near": "Bonnie Bartel", "function": "Office"},
                 {"near_building": "George R. Brown", "near_floor": "03"}),
        Scenario("SPA-NEW-FACULTY", "New faculty office · Chemistry", "space",
                 "A new Chemistry faculty member needs an office of at least 120 sf.",
                 {"request_type": "new_office", "department": "Chemistry", "function": "Office", "min_area": 120},
                 {"min_area": 120, "no_unconditional_vacancy": True}),
        Scenario("SPA-LAB-BIOS", "Research lab for a new program · BioSciences", "space",
                 "A researcher needs laboratory space of at least 400 sf; lab infrastructure is not in the records.",
                 {"request_type": "department_function", "department": "BioSciences", "function": "Research Lab",
                  "min_area": 400, "lab_requirements": ["fume hood", "ventilation"]},
                 {"min_area": 400, "lab_conditional": True, "ehs_approval": True}),
        Scenario("SPA-VACANCY-NS", "Vacancy review · Natural Sciences", "space",
                 "School leadership reviews recorded vacancies and possible reassignment options.",
                 {"request_type": "vacancy_review"}, {"vacancy_cross_checked": True}),
        Scenario("SPA-CHANGE", "Assignment change · existing faculty", "space",
                 "A faculty member needs to change an existing room assignment.",
                 {"request_type": "change_assignment", "netid": "ljd1", "function": "Office"},
                 {"excludes_current_room": True}),
    ]


def resolve_payload(sc: Scenario, corpus: Corpus) -> dict:
    """Materialize payloads that reference a single corpus message."""
    p = dict(sc.payload)
    if "sr_message" in p:
        sr = p.pop("sr_message")
        msg = next(m for m in corpus.messages if sr in m.subject)
        p["text"] = message_text(msg)
    return p
