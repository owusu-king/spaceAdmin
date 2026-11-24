"""Outage Agent (Outage Coordination).

Objective: take an outage notice or operational email, reconstruct what is actually
happening, assess impact, identify Natural Sciences departments and their contacts,
review schedule conflicts, and return a coordination plan with draft communications for
James's approval. It never sends messages or edits TRIRIGA.

Decision pipeline
  parse -> reconstruct timeline -> resolve scope -> impact register (+ model-inferred lab use)
  -> departments -> stakeholders -> conflicts & coverage -> recurrence forecast
  -> (optional) request temporary-space support from the Space Administrator Agent
  -> coordination plan, follow-up questions, drafts
"""
from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Callable

import pandas as pd

from app.agents.base import Agent, AgentMessage, Evidence, ToolResult, tool
from app.agents.schemas import OutageRequest
from app.config import AGENTS, NATURAL_SCIENCES_DEPARTMENTS
from app.data.buildings import BuildingResolver
from app.data.loader import Dataset
from app.data.notices import PRELIMINARY, RESTORED, Corpus, Message, OutageNotice, parse_notice, sentences, split_messages
from app.ml.inference import LAB_GROUPS, SpacePredictor
from app.services import scheduling
from app.services.events import EventStore, extract_dates, extract_duration_hours
from app.services.stakeholders import resolve_stakeholders

LAB_SYSTEMS = {"Steam", "Fume Hood Exhaust", "Building Exhaust", "Control Air", "Compressed Air", "Nitrogen",
               "Lab Air", "Clean Room", "Chilled Water", "Vacuum"}
BUILDING_WIDE = {"Electrical", "Emergency Power", "Air Conditioning", "Heating Hot Water", "Fire Alarm", "Fire",
                 "IT Related", "Outside Air", "Pumps"}
WATER = {"Domestic Hot and Cold Water", "Domestic Hot Water only", "Domestic Hot Water", "Domestic Cold Water"}
TEACHING = {"Classroom", "Teaching Lab"}
RELOCATION_SYSTEMS = {"Air Conditioning", "Heating Hot Water", "Electrical", "Outside Air"}   # occupancy-limiting
GENERIC_WORDS = {"affected", "service", "services", "outage", "shutdown", "building", "will", "would", "there", "their"}


def _subject_words(subject: str) -> set[str]:
    stop = {"re", "fw", "fwd", "the", "for", "and", "of", "in", "is", "a", "to", "outage", "notification", "notice",
            "approval", "required", "request", "an", "scheduled", "urgent", "emergency", "alert", "unscheduled", "planned"}
    return {w for w in re.findall(r"[a-z]{3,}", subject.lower()) if w not in stop}


class OutageAgent(Agent):
    name = "outage"
    objectives = [
        "Read notices and emails; confirm scope and timing.",
        "Sort impacts: confirmed, potential, unaffected.",
        "Find affected departments, contacts and people.",
        "Check conflicts; suggest other times.",
        "Draft messages for approval; never send.",
    ]
    permissions = {"parse_outage_notice", "isolate_incident", "reconstruct_event_timeline", "find_related_outages",
                   "resolve_outage_scope", "assess_outage_impact", "find_affected_departments", "get_stakeholders",
                   "check_schedule_conflicts", "forecast_recurring_maintenance", "request_space_support", "find_affected_people",
                   "draft_communication"}
    data_scopes = ["Supplied correspondence and TRIRIGA notices", "Space records (building-level)",
                   "People records (department, role, location)", "Space-function model predictions",
                   "Outage history derived from notices"]

    def __init__(self, ds: Dataset, predictor: SpacePredictor, resolver: BuildingResolver, corpus: Corpus,
                 as_of: datetime | None = None, events: EventStore | None = None) -> None:
        self.events = events
        self.ds, self.predictor, self.resolver, self.corpus = ds, predictor, resolver, corpus
        self.as_of = as_of or datetime.now()
        self.support_channel: Callable[[AgentMessage], AgentMessage] | None = None   # set by the coordinator
        super().__init__()

    # =========================================================================== tools
    @tool("parse_outage_notice")
    def parse_outage_notice(self, text: str) -> ToolResult:
        """Split the input into messages and extract structured fields from each."""
        msgs = split_messages(text, include_quoted=True)
        parsed = []
        for m in msgs:
            tririga = "SR-" in m.subject or "Outage Request ID" in m.full_text[:4000]
            n = parse_notice(m.full_text[:8000] if tririga else m.own_text, self.resolver, m.subject)
            n.sent, n.sender = m.date, m.sender
            parsed.append((m, n))
        tri = [n for _, n in parsed if n.kind != "informal"]
        status = "success" if (tri or any(n.buildings or n.systems for _, n in parsed)) else "partial"
        warn = [] if tri else ["No TRIRIGA notice found; working from operational email only"]
        return ToolResult("parse_outage_notice", status, parsed, warnings=warn)

    @tool("isolate_incident")
    def isolate_incident(self, parsed: list[tuple[Message, OutageNotice]]) -> ToolResult:
        """Pick the current notice and keep only messages about the same incident."""
        top = [(m, n) for m, n in parsed if not m.quoted]
        relevant = [(m, n) for m, n in top if n.sr or n.systems] or top
        lm, ln = max(relevant, key=lambda mn: (mn[0].date or datetime.min))
        # Prefer the formal TRIRIGA notice when it describes the same incident as the latest email.
        tri = [(m, n) for m, n in top if n.kind != "informal" and m.date and lm.date
               and abs((lm.date - m.date).days) <= 14
               and (ln.kind != "informal" or set(n.systems) & set(ln.systems) or set(n.buildings) & set(ln.buildings))]
        pm, primary = max(tri, key=lambda mn: mn[0].date) if tri else (lm, ln)
        p_date = pm.date or datetime.max
        words = _subject_words(primary.subject)
        keep, dropped = [], 0
        for m, n in parsed:
            same_sr = primary.sr and primary.sr in m.sr_ids
            close = m.date is None or p_date == datetime.max or timedelta(days=-7) <= (p_date - m.date) <= timedelta(days=14)
            overlap = bool(set(n.systems) & set(primary.systems)) or bool(words & _subject_words(m.subject))
            if m is pm or same_sr or (close and overlap and not (n.sr and primary.sr and n.sr != primary.sr)):
                keep.append((m, n))
            else:
                dropped += 1
        warn = [f"{dropped} message(s) in this input belong to other incidents and were set aside"] if dropped else []
        return ToolResult("isolate_incident", "success", {"primary": primary, "messages": keep, "set_aside": dropped},
                          [Evidence(f"Current notice: {primary.subject or 'operational email'} "
                                    f"({pm.date:%b %d %Y} from {pm.sender.split('<')[0].strip() or 'unknown'})" if pm.date else
                                    "Current notice identified", "recorded", "correspondence", primary.sr or "")], warnings=warn)

    @tool("reconstruct_event_timeline")
    def reconstruct_event_timeline(self, parsed: list[tuple[Message, OutageNotice]]) -> ToolResult:
        """Order messages; classify planned / revised / confirmed / preliminary / restoration nodes."""
        items = sorted(parsed, key=lambda mn: mn[0].date or datetime.min)
        nodes, windows, ev, seen = [], [], [], set()
        for m, n in items:
            own = m.own_text or m.full_text
            restore = [x for x in sentences(own) if RESTORED.search(x)]
            if restore and n.kind == "informal":
                future = all(re.search(r"\b(will|should|expected|by \d)", x, re.I) for x in restore)
                ntype = "restoration_estimate" if future else "restoration_confirmed"
            elif n.kind == "planned_notification" or n.kind == "unscheduled_alert":
                ntype = "confirmed"
            elif n.kind == "approval_request":
                ntype = "planned"
            elif PRELIMINARY.search(own):
                ntype = "preliminary"
            else:
                ntype = "update"
            sig = (m.date.replace(second=0) if m.date else None, n.kind, n.sr, n.start)
            if sig in seen:          # forwarded copies of the same notice are one event
                continue
            seen.add(sig)
            window = (n.start, n.end) if n.start else None
            if window and windows and window != windows[-1]:
                ntype = "revised" if ntype in ("planned", "update") else ntype
            if window:
                windows.append(window)
            nodes.append({"date": m.date.isoformat() if m.date else None, "sender": m.sender, "subject": m.subject,
                          "type": ntype, "sr": n.sr, "start": n.start.isoformat() if n.start else None,
                          "end": n.end.isoformat() if n.end else None, "superseded": False,
                          "excerpt": " ".join(sentences(own)[:2])[:280]})
        current = windows[-1] if windows else None
        for node in nodes:
            if node["start"] and current and (node["start"], node["end"]) != (current[0].isoformat(), current[1].isoformat()):
                node["superseded"] = True
        if current:
            ev.append(Evidence(f"Current timing {current[0]:%b %d %Y %H:%M} to {current[1]:%H:%M}",
                               "confirmed" if any(n["type"] == "confirmed" for n in nodes) else "recorded",
                               "correspondence", nodes[-1]["date"] or ""))
        if any(n["superseded"] for n in nodes):
            ev.append(Evidence("Earlier timing superseded by a later notice", "historical", "correspondence"))
        return ToolResult("reconstruct_event_timeline", "success",
                          {"nodes": nodes, "current_window": current}, ev,
                          missing_information=[] if current else ["Outage start and end times not stated"])

    @tool("find_related_outages")
    def find_related_outages(self, sr: str | None, buildings: list[str], systems: list[str]) -> ToolResult:
        """Historical outages for the same request or the same building and systems."""
        same = [h for h in self.corpus.history if sr and h["sr"] == sr]
        similar = [h for h in self.corpus.history if h["sr"] != sr and set(h["buildings"]) & set(buildings)
                   and set(h["systems"]) & set(systems)]
        ev = [Evidence(f"{h['sr']} ({h['start'][:10]}): {', '.join(h['systems'])} in {', '.join(h['buildings'])}",
                       "historical", "correspondence", h["sr"]) for h in similar[-5:]]
        return ToolResult("find_related_outages", "success", {"same_request": same, "similar": similar}, ev,
                          warnings=["Similar historical outages are context, not proof of this outage's impact"] if similar else [])

    @tool("resolve_outage_scope")
    def resolve_outage_scope(self, notice: OutageNotice, parsed: list[tuple[Message, OutageNotice]]) -> ToolResult:
        """Resolve affected buildings and systems from the notice contents (not filenames)."""
        buildings = list(notice.buildings)
        ambiguous, ev, missing = [], [], []
        mentioned = sorted({b for _, n in parsed for b in n.buildings} - set(buildings))
        if not buildings:
            buildings, mentioned = mentioned, []
        text = " ".join(m.own_text for m, _ in parsed)
        for match in self.resolver.find_in_text(notice.subject + " " + text):
            if match.status == "ambiguous":
                ambiguous.append(match.candidates)
        systems = list(notice.systems) or sorted({s for _, n in parsed for s in n.systems})
        for b in buildings:
            ev.append(Evidence(f"Building in scope: {self.resolver.info(b)['name']}", "confirmed" if notice.kind != "informal" else "recorded",
                               "correspondence", notice.sr or notice.subject))
        if notice.buildings_partial:
            missing.append("TRIRIGA building list is truncated ('...More'). Open the linked record for the full list.")
        if not buildings:
            missing.append("Affected building could not be identified from the notice contents")
        if not systems:
            missing.append("Affected systems not stated")
        status = "partial" if missing else ("ambiguous" if ambiguous and not buildings else "success")
        return ToolResult("resolve_outage_scope", status, {"buildings": buildings, "systems": systems,
                          "ambiguous": ambiguous, "partial": notice.buildings_partial,
                          "mentioned_elsewhere": mentioned}, ev, missing_information=missing)

    @tool("assess_outage_impact")
    def assess_outage_impact(self, notice: OutageNotice, buildings: list[str], systems: list[str],
                             timeline: list[dict], parsed: list[tuple[Message, OutageNotice]]) -> ToolResult:
        """Impact register: confirmed, potential, explicitly unaffected and unknown."""
        reg = {"confirmed": [], "potential": [], "explicitly_unaffected": [], "unknown": [], "preliminary_superseded": []}
        sp = self.ds.spaces[self.ds.spaces["building"].isin(buildings)]
        for s in notice.impact_statements:
            reg["confirmed"].append({"entity": "service", "statement": s, "classification": "confirmed",
                                     "source": notice.sr or "notice"})
        for s in systems:
            reg["confirmed"].append({"entity": "system", "statement": f"{s} impacted", "classification": "confirmed",
                                     "source": notice.sr or "notice"})
        for e in sorted({x for _, n in parsed for x in n.exclusions}):
            reg["explicitly_unaffected"].append({"entity": "service", "statement": e, "classification": "confirmed",
                                                 "source": notice.sr or "notice"})
        confirmed_text = " ".join(notice.impact_statements + [notice.scope, notice.equipment]).lower()
        for m, n in parsed:
            node = next((x for x in timeline if x["date"] == (m.date.isoformat() if m.date else None)), None)
            if node and node["type"] == "preliminary":
                for s in sentences(m.own_text):
                    if PRELIMINARY.search(s) or not any(k in s.lower() for k in ("affected", "out of service", "shut", "unavailable")):
                        continue
                    key_words = {w for w in re.findall(r"[a-z]{5,}", s.lower())} - GENERIC_WORDS
                    later = bool(key_words) and any(w in confirmed_text for w in key_words)
                    reg["preliminary_superseded"].append({
                        "entity": "service", "statement": s, "classification": "historical",
                        "later_confirmed": later,
                        "note": "Later confirmed in current scope" if later else "Not in the confirmed scope",
                        "source": f"{m.sender.split('<')[0].strip()} {node['date'][:10] if node['date'] else ''}"})
        ev = []
        labs = sp[sp["class_group"].isin(LAB_GROUPS)]
        lab_like = pd.DataFrame()
        if set(systems) & (LAB_SYSTEMS | BUILDING_WIDE):
            if len(labs):
                reg["potential"].append({"entity": "room", "statement": f"{len(labs)} recorded laboratory spaces in scope "
                                         f"may depend on {', '.join(sorted(set(systems) & (LAB_SYSTEMS | BUILDING_WIDE)))}",
                                         "classification": "recorded", "source": "space_records", "count": int(len(labs))})
            pred = self.predictor.table.reindex(sp["space_id"])
            cand = pred[(~pred["recorded_group"].isin(LAB_GROUPS)) & (pred["p_lab"] >= AGENTS.lab_like_threshold)]
            lab_like = sp[sp["space_id"].isin(cand.index)]
            if len(lab_like):
                rooms = ", ".join(lab_like["room_key"].head(6))
                reg["potential"].append({"entity": "room", "statement": f"{len(lab_like)} non-lab records with a lab-like "
                                         f"profile (model): {rooms}{'…' if len(lab_like) > 6 else ''}",
                                         "classification": "inferred", "source": "model:SpaceFunctionNet",
                                         "count": int(len(lab_like)), "rooms": lab_like["room_key"].tolist()})
                ev.append(Evidence(f"Model flags {len(lab_like)} rooms recorded as non-lab with lab-like use", "inferred", "model"))
        if set(systems) & WATER:
            rest = sp[sp["space_class"].fillna("").str.startswith(("X03", "919", "X08"))]
            reg["potential"].append({"entity": "service", "statement": f"Restrooms/showers ({len(rest)} recorded) and lab sinks",
                                     "classification": "recorded", "source": "space_records"})
        if "Elevator" in systems:
            upper = sp[~sp["floor"].isin(["01", "B", "B1", "B2", "M"])]
            reg["potential"].append({"entity": "service", "statement": f"Accessibility: {upper['floor'].nunique()} upper floor(s) "
                                     "lose elevator access; deliveries and mobility needs", "classification": "inferred",
                                     "source": "space_records"})
        if "Other" in systems:
            reg["unknown"].append({"entity": "system", "statement": "System listed as 'Other': confirm the affected service",
                                   "classification": "unknown", "source": notice.sr or "notice"})
        if notice.buildings_partial:
            reg["unknown"].append({"entity": "building", "statement": "Full building list not shown in the notice",
                                   "classification": "unknown", "source": notice.sr or "notice"})
        if not notice.end:
            reg["unknown"].append({"entity": "service", "statement": "Restoration time not stated",
                                   "classification": "unknown", "source": "notice"})
        return ToolResult("assess_outage_impact", "success",
                          {"register": reg, "lab_spaces": int(len(labs)), "lab_like_rooms": int(len(lab_like)),
                           "teaching_spaces": int(sp["class_group"].isin(TEACHING).sum()), "spaces_in_scope": int(len(sp))}, ev)

    @tool("find_affected_departments")
    def find_affected_departments(self, buildings: list[str], systems: list[str]) -> ToolResult:
        """Departments with allocated space in scope; Natural Sciences group separated."""
        sp = self.ds.spaces[self.ds.spaces["building"].isin(buildings)].copy()
        if set(systems) and set(systems) <= LAB_SYSTEMS:
            sp = sp[sp["class_group"].isin(LAB_GROUPS) | self.predictor.table.reindex(sp["space_id"])["p_lab"].ge(0.5).values]
        alloc = sp["allocation"].dropna().str.split(", ").explode()
        counts = alloc[alloc != "Unassigned"].value_counts()
        rows = [{"department": d, "spaces": int(c), "natural_sciences": d in NATURAL_SCIENCES_DEPARTMENTS}
                for d, c in counts.items()]
        ns = [r for r in rows if r["natural_sciences"]]
        ev = [Evidence(f"{r['department']} holds {r['spaces']} allocated space(s) in scope", "recorded", "space_records")
              for r in ns]
        missing = [] if rows else ["No departmental allocations on record for the affected scope"]
        return ToolResult("find_affected_departments", "success" if rows else "partial",
                          {"all": rows, "natural_sciences": ns}, ev, missing_information=missing)

    @tool("get_stakeholders")
    def get_stakeholders(self, buildings: list[str], departments: list[str]) -> ToolResult:
        """Formal approvers, coordination contacts and notification recipients per department."""
        res = resolve_stakeholders(self.ds.people, self.ds.spaces, buildings, departments)
        status = "partial" if res["gaps"] else "success"
        return ToolResult("get_stakeholders", status, res, missing_information=res["gaps"] +
                          ["Formal approver roster is not in the supplied data (TRIRIGA)"])

    def review_window(self, buildings: list[str], start: datetime | None, end: datetime | None, sr: str | None = None,
                      date_only: bool = False, duration_hours: float | None = None) -> dict:
        """Conflict review for any proposed window (used by the agent and when James proposes another time)."""
        sp = self.ds.spaces[self.ds.spaces["building"].isin(buildings)]
        teaching = int(sp["class_group"].isin(TEACHING).sum())
        labs = int(sp["class_group"].isin(LAB_GROUPS).sum())
        events = self.events.overlapping(buildings, start, end) if (self.events and start and end) else []
        res = scheduling.check_conflicts(start, end, buildings, self.corpus.history, teaching, labs, exclude_sr=sr,
                                         events=events, calendar=self.events.count() if self.events else None,
                                         date_only=date_only)
        res["alternatives"] = []
        if start and (res["conflicts"] or res["window"]["business_hours"]):
            span = timedelta(hours=duration_hours) if date_only and duration_hours else (None if date_only else end - start)
            if span:
                avoid = (lambda s, e: bool(self.events.overlapping(buildings, s, e))) if self.events else None
                res["alternatives"] = scheduling.propose_alternatives(start, start + span, buildings, self.corpus.history,
                                                                      exclude_sr=sr, avoid=avoid)
        return res

    def _proposed_dates(self, parsed: list[tuple[Message, OutageNotice]], buildings: list[str]):
        """Dates in prose ('Wednesday, November 25th') that refer to the scope buildings, newest message first."""
        for m, n in sorted(parsed, key=lambda mn: mn[0].date or datetime.min, reverse=True):
            if not m.date:
                continue
            sents = sentences(m.own_text)
            for i, s in enumerate(sents):
                found = [d for d in extract_dates(s, m.date) if d[0] >= m.date.date()]
                if not found:
                    continue
                context = " ".join(sents[max(0, i - 1): i + 1])
                ctx_b = {b.export_name for b in self.resolver.find_in_text(context) if b.status == "resolved"}
                msg_b = set(n.buildings)
                if (ctx_b and ctx_b & set(buildings)) or (not ctx_b and msg_b and msg_b <= set(buildings)):
                    duration = extract_duration_hours(m.own_text) or next(
                        (h for h in (extract_duration_hours(x.own_text) for x, _ in parsed) if h), None)
                    return found[0], duration, s
        return None

    @tool("check_schedule_conflicts")
    def check_schedule_conflicts(self, window, buildings: list[str], sr: str | None, date_only: bool = False,
                                 duration_hours: float | None = None) -> ToolResult:
        """Known conflicts (events calendar, blocked times, other outages), risks, coverage and alternative windows."""
        start, end = window if window else (None, None)
        res = self.review_window(buildings, start, end, sr, date_only, duration_hours)
        ev = [Evidence(f"Conflict: {c['title']} ({c['building'] if c.get('building') else ', '.join(buildings)}, "
                       f"{c['start'][:10]})", "confirmed" if c["classification"] == "confirmed" else "recorded",
                       c.get("source", "calendar")) for c in res["conflicts"]]
        gaps = [c["source"] for c in res["coverage"] if c["status"] != "covered"]
        return ToolResult("check_schedule_conflicts", "partial", res, ev,
                          missing_information=[f"Schedule coverage gap: {g}" for g in gaps])

    @tool("forecast_recurring_maintenance")
    def forecast_recurring_maintenance(self, buildings: list[str]) -> ToolResult:
        """Statistical estimate of upcoming recurring preventive-maintenance windows."""
        fc = scheduling.recurring_maintenance_forecast(self.corpus.history, self.as_of, buildings)
        ev = [Evidence(f"Estimated next {', '.join(f['systems'])} maintenance in {f['building']} around "
                       f"{f['estimated_next']} (±{f['window_days']:.0f} days, {f['confidence']} confidence)",
                       "inferred", "rule:recurrence", f["building"]) for f in fc]
        return ToolResult("forecast_recurring_maintenance", "success", fc, ev)

    @tool("request_space_support")
    def request_space_support(self, payload: dict) -> ToolResult:
        """Ask the Space Administrator Agent (via the coordinator) for temporary space options."""
        if not self.support_channel:
            return ToolResult("request_space_support", "unauthorized",
                              warnings=["No coordinator channel: cross-agent requests are disabled"])
        msg = AgentMessage(self.name, "space_admin", "temporary_space_request", payload)
        reply = self.support_channel(msg)
        return ToolResult("request_space_support", "success", {"request": vars(msg), "reply": vars(reply)})

    @tool("find_affected_people")
    def find_affected_people(self, buildings: list[str], systems: list[str]) -> ToolResult:
        """People recorded in the affected spaces, with the reason each one is affected."""
        sp = self.ds.spaces[self.ds.spaces["building"].isin(buildings)]
        pred = self.predictor.table.reindex(sp["space_id"])
        lab_only = bool(systems) and set(systems) <= LAB_SYSTEMS
        if lab_only:   # lab systems: only people in recorded labs or rooms the model infers are lab-like
            lab_mask = sp["class_group"].isin(LAB_GROUPS).values | (pred["p_lab"].fillna(0).values >= AGENTS.lab_like_threshold)
            rooms = sp[lab_mask]
            rule = f"People in laboratory and lab-like rooms ({', '.join(systems)})"
        else:
            rooms = sp
            rule = f"Everyone in the building ({', '.join(systems) or 'building-wide'})"
        recorded_lab = set(sp.loc[sp["class_group"].isin(LAB_GROUPS), "room_key"])
        people = self.ds.people[self.ds.people["room_key"].isin(set(rooms["room_key"]))]
        rows = []
        for _, p in people.iterrows():
            why = (f"Works in {p.room_key} " + ("(recorded laboratory)" if p.room_key in recorded_lab else
                   "(model infers lab-like use)") if lab_only else f"Recorded in {p.room_key}")
            rows.append({"name": p["name"], "netid": p.netid, "email": p.email, "department": p.department,
                         "role": p.role, "room": p.room_key, "reason": why,
                         "natural_sciences": p.department in NATURAL_SCIENCES_DEPARTMENTS})
        rows.sort(key=lambda r: (not r["natural_sciences"], r["department"] or "", r["room"], r["name"] or ""))
        by_dept = pd.Series([r["department"] for r in rows], dtype="object").value_counts()
        ns = [r for r in rows if r["natural_sciences"]]
        return ToolResult("find_affected_people", "partial" if rows else "not_found",
                          {"scope_rule": rule, "total": len(rows), "natural_sciences": len(ns),
                           "by_department": [{"department": d, "people": int(n), "natural_sciences": d in NATURAL_SCIENCES_DEPARTMENTS}
                                             for d, n in by_dept.items()], "people": rows},
                          [Evidence(f"{len(rows)} people recorded in affected spaces ({len(ns)} in Natural Sciences)",
                                    "recorded", "people_records")],
                          missing_information=["People without a recorded location (many students and visitors) cannot be "
                                               "identified from the records; departmental lists are still needed"])

    @tool("draft_communication")
    def draft_communication(self, notice: OutageNotice, window, register: dict, stakeholders: dict,
                            questions: list[str], people: dict | None = None, buildings: list[str] | None = None,
                            date_only: bool = False, systems: list[str] | None = None) -> ToolResult:
        """Draft the coordination email and the notice to affected people. Drafts are never sent."""
        to = [c for d in stakeholders["departments"] if d["natural_sciences"] for c in d["coordination_contacts"]]
        if not window:
            when = "timing to be confirmed"
        elif date_only:
            when = f"{window[0]:%A %b %d, %Y} (hours to be confirmed)"
        else:
            when = f"{window[0]:%A %b %d, %Y %H:%M} to {window[1]:%H:%M}"
        bldg = ", ".join(self.resolver.info(b)["name"] for b in (buildings or notice.buildings or [])) or "the affected building"
        notice.systems = notice.systems or list(systems or [])
        lines = [f"Subject: Coordination: {notice.sr or 'outage'} at {bldg}", "",
                 "Hello,", "", f"Facilities has {'scheduled' if notice.kind != 'informal' else 'reported'} work affecting "
                 f"{', '.join(notice.systems) or 'building systems'} at {bldg}, {when}."]
        conf = [i["statement"] for i in register["confirmed"] if i["entity"] == "service"][:3]
        if conf:
            lines += ["", "What to expect:"] + [f"- {c}" for c in conf]
        excl = [i["statement"] for i in register["explicitly_unaffected"]][:3]
        if excl:
            lines += ["", "Not affected:"] + [f"- {e}" for e in excl]
        lines += ["", "Does this window conflict with any experiments, classes, deliveries or accessibility needs?"]
        lines += [f"- {q}" for q in questions if not q.startswith("Who coordinates")][:2]
        lines += ["", "Thank you,", "James"]
        coord_draft = {"kind": "coordination", "title": "Coordination with departments", "status": "DRAFT — NOT SENT",
                       "recipients": {"coordination_contacts": [f"{c['name']} <{c['email']}> ({c['verification']})" for c in to],
                                      "formal_approver": "Not on record — confirm in TRIRIGA"},
                       "body": "\n".join(lines), "locked_action": "send_email"}
        drafts = [coord_draft]
        ns_people = [p for p in (people or {}).get("people", []) if p["natural_sciences"]]
        if ns_people:
            n = ["Subject: " + (f"Building notice: {', '.join(notice.systems) or 'service'} outage at {bldg}, {when}"),
                 "", "Hello,", "", f"Please be aware of upcoming work at {bldg}.", "",
                 f"When: {when}", f"Systems: {', '.join(notice.systems) or 'to be confirmed'}"]
            if notice.reason:
                n.append(f"Why: {notice.reason}")
            if conf:
                n += ["", "What to expect:"] + [f"- {c}" for c in conf]
            if excl:
                n += ["", "Not affected:"] + [f"- {e}" for e in excl]
            n += ["", "If this affects your work, please reply.", "", "Thank you,", "James"]
            drafts.append({"kind": "notification", "title": "Notice to affected people", "status": "DRAFT — NOT SENT",
                           "recipients": {"affected_people": len(ns_people),
                                          "departments": sorted({p["department"] for p in ns_people}),
                                          "delivery": "Send as BCC from your own mailbox; export the list as CSV"},
                           "body": "\n".join(n), "locked_action": "send_email"})
        return ToolResult("draft_communication", "success", drafts)

    # =========================================================================== workflow
    def run(self, req: OutageRequest, cooperate: bool = False) -> dict:
        """cooperate=True asks the Space agent for temporary space (on demand only; James rarely needs it)."""
        self.reset()
        self.log.info(f"workflow start source={req.source} chars={len(req.text)}")
        parsed_all = self.call("parse_outage_notice", text=req.text).data
        inc = self.call("isolate_incident", parsed=parsed_all)
        primary, parsed = inc.data["primary"], inc.data["messages"]
        tl = self.call("reconstruct_event_timeline", parsed=parsed)
        window = tl.data["current_window"]
        scope = self.call("resolve_outage_scope", notice=primary, parsed=parsed)
        buildings, systems = scope.data["buildings"], scope.data["systems"]
        impact = self.call("assess_outage_impact", notice=primary, buildings=buildings, systems=systems,
                           timeline=tl.data["nodes"], parsed=parsed)
        related = self.call("find_related_outages", sr=primary.sr, buildings=buildings, systems=systems)
        depts = self.call("find_affected_departments", buildings=buildings, systems=systems)
        ns_names = [d["department"] for d in depts.data["natural_sciences"]]
        stake = self.call("get_stakeholders", buildings=buildings, departments=ns_names)
        date_only, duration, date_basis = False, None, None
        if not window and buildings:
            found = self._proposed_dates(parsed, buildings)
            if found:
                (d1, d2), duration, date_basis = found
                window = (datetime.combine(d1, datetime.min.time()), datetime.combine(d2 + timedelta(days=1), datetime.min.time()))
                date_only = True
        sched = self.call("check_schedule_conflicts", window=window, buildings=buildings, sr=primary.sr,
                          date_only=date_only, duration_hours=duration)
        fc = self.call("forecast_recurring_maintenance", buildings=buildings)

        questions = self._questions(primary, scope, stake, sched, depts, impact)
        support = None
        hours = (duration or 0) if date_only else ((window[1] - window[0]).total_seconds() / 3600 if window else 0)
        lab_sys = bool(set(systems) & LAB_SYSTEMS)
        needs_space = lab_sys or bool(set(systems) & RELOCATION_SYSTEMS)
        if cooperate and ns_names and needs_space and hours >= AGENTS.relocation_min_hours and buildings:
            payload = {"request_type": "temporary_relocation", "department": ns_names[0],
                       "function": "Research Lab" if lab_sys else "Workstation",
                       "exclude_buildings": buildings, "reason": f"{primary.sr or 'Outage'}: {', '.join(systems)} "
                       f"for {hours:.0f} h", "duration_hours": hours}
            support = self.call("request_space_support", payload=payload).data
        people = self.call("find_affected_people", buildings=buildings, systems=systems)
        draft = self.call("draft_communication", notice=primary, window=window, register=impact.data["register"],
                          stakeholders=stake.data, questions=questions, people=people.data, buildings=buildings,
                          date_only=date_only, systems=systems)
        plan = self._plan(primary, sched.data, stake.data, impact.data, support, hours)
        findings = [vars(e) for r in (tl, scope, impact, related, depts, sched, fc) for e in r.evidence]
        self.log.info(f"workflow done sr={primary.sr} buildings={len(buildings)} ns_departments={len(ns_names)} "
                      f"conflicts={len(sched.data['conflicts'])} questions={len(questions)} support={'yes' if support else 'no'}")
        return {
            "case_type": "outage_coordination", "agent": self.name,
            "outage": {"request_reference": primary.sr, "kind": primary.kind, "priority": primary.priority,
                       "buildings": [self.resolver.info(b) for b in buildings], "affected_systems": systems,
                       "proposed_start": window[0].isoformat() if window else None,
                       "proposed_end": window[1].isoformat() if window else None,
                       "duration_hours": round(hours, 2) if hours else None,
                       "date_only": date_only, "date_basis": date_basis,
                       "reason": primary.reason, "scope_of_work": primary.scope, "equipment": primary.equipment,
                       "shops": primary.shops, "status": "under_review"},
            "summary": self._summary(primary, buildings, systems, impact.data, ns_names, sched.data, window),
            "timeline": tl.data["nodes"], "impacts": impact.data["register"],
            "incident": {"messages_used": len(parsed), "set_aside": inc.data["set_aside"],
                         "mentioned_elsewhere": scope.data["mentioned_elsewhere"]},
            "impact_counts": {k: impact.data[k] for k in ("lab_spaces", "lab_like_rooms", "teaching_spaces", "spaces_in_scope")},
            "affected_departments": depts.data, "stakeholders": stake.data, "scheduling": sched.data,
            "related_outages": {"same_request": related.data["same_request"], "similar": related.data["similar"][-6:]},
            "maintenance_forecast": fc.data, "space_support": support,
            "missing_information": list(dict.fromkeys(m for r in (tl, scope, depts, stake) for m in r.missing_information)),
            "follow_up_questions": questions, "recommended_actions": plan, "communication_drafts": draft.data,
            "affected_people": people.data,
            "findings": findings, "required_approvals": [
                "James (review)", "TRIRIGA approver (outage)"],
            "review_status": "pending", "trace": self.trace,
            "decision_note": "Recommendation pending review. The agent does not approve, send or reschedule.",
        }

    # ---------------------------------------------------------------------------- helpers
    def _questions(self, n, scope, stake, sched, depts, impact) -> list[str]:
        q = []
        if n.buildings_partial:
            q.append(f"{n.sr} building list is truncated. Which other buildings are in scope?")
        if not scope.data["buildings"]:
            q.append("Which building does this outage affect?")
        if not n.start and not (sched.data["window"] and sched.data["window"].get("date_only")):
            q.append("When does the work start and end?")
        for g in stake.data["gaps"]:
            q.append(f"Who coordinates for {g.replace('No coordination contact on record for ', '')}?")
        if impact.data["lab_like_rooms"]:
            q.append("Do the rooms the model marks as lab-like hold equipment on these systems?")
        if "Other" in scope.data["systems"]:
            q.append("Which service does 'Other' refer to?")
        if sched.data["window"] and sched.data["window"]["weekend"]:
            q.append("Any weekend experiments, deliveries or lab sessions in this window?")
        if not depts.data["natural_sciences"] and scope.data["buildings"]:
            q.append("No Natural Sciences space in scope. Notify anyone anyway?")
        return q

    def _plan(self, n, sched, stake, impact, support, hours) -> list[dict]:
        """Short, actionable next steps (details live in the report sections)."""
        acts = [{"action": "Review and approve the drafts", "owner": "James", "status": "pending"}]
        if sched["conflicts"]:
            first = sched["conflicts"][0]["title"].split(" (")[0]
            more = f" +{len(sched['conflicts']) - 1}" if len(sched["conflicts"]) > 1 else ""
            acts.append({"action": f"Resolve conflict: {first}{more}", "owner": "James / Facilities", "status": "pending"})
        if sched.get("alternatives"):
            acts.append({"action": f"Propose another time ({len(sched['alternatives'])} suggested)", "owner": "James",
                         "status": "proposal"})
        missing = [d["department"] for d in stake["departments"] if d["natural_sciences"] and not d["coordination_contacts"]]
        if missing:
            acts.append({"action": f"Find contacts for {len(missing)} department(s)", "owner": "James", "status": "pending"})
        if n.kind == "approval_request":
            acts.append({"action": f"Gather input before approving {n.sr}", "owner": "James", "status": "pending"})
        if impact["teaching_spaces"] and sched.get("window") and sched["window"]["business_hours"]:
            acts.append({"action": "Check with the Registrar", "owner": "James", "status": "pending"})
        if support and support["reply"]["payload"].get("candidates"):
            acts.append({"action": f"Share {len(support['reply']['payload']['candidates'])} temporary space options",
                         "owner": "James", "status": "proposal"})
        acts.append({"action": "Track restoration", "owner": "Outage Agent", "status": "open"})
        return acts

    def _summary(self, n, buildings, systems, impact, ns, sched, window) -> str:
        where = ", ".join(self.resolver.info(b)["code"] or b for b in buildings) or "unidentified building"
        when = ("timing not stated" if not window else f"{window[0]:%b %d} (time not stated)"
                if window[1] - window[0] >= timedelta(hours=23) else f"{window[0]:%b %d %H:%M}–{window[1]:%H:%M}")
        return (f"{n.sr or 'Outage'} · {', '.join(systems) or 'systems not stated'} · {where} · {when}. "
                f"{len(impact['register']['confirmed'])} confirmed and {len(impact['register']['potential'])} potential impact "
                f"items, {len(impact['register']['explicitly_unaffected'])} explicit exclusion(s). "
                f"{len(ns)} Natural Sciences department(s); {len(sched['conflicts'])} recorded conflict(s).")
