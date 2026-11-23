"""Scheduling Engine: conflict review, schedule coverage, rescheduling proposals and
recurring preventive-maintenance estimates.

Rules from James:
* A weekend or an empty class schedule does not establish that a building is available.
* Missing scheduling information is reported as a coverage gap, never as "no conflicts".
* Estimated future maintenance windows are PREDICTIONS and are labelled as such;
  confirmed dates come from maintenance notices.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median

BUSINESS_START, BUSINESS_END = 8, 18


def _dt(v) -> datetime | None:
    return datetime.fromisoformat(v) if isinstance(v, str) else v


def overlapping_outages(history: list[dict], buildings: list[str], start: datetime, end: datetime,
                        exclude_sr: str | None = None) -> list[dict]:
    out = []
    for h in history:
        hs, he = _dt(h["start"]), _dt(h["end"])
        if not hs or not he or h["sr"] == exclude_sr:
            continue
        if set(h["buildings"]) & set(buildings) and hs < end and he > start:
            out.append(h)
    return out


def check_conflicts(start: datetime | None, end: datetime | None, buildings: list[str], history: list[dict],
                    teaching_spaces: int, lab_spaces: int, exclude_sr: str | None = None,
                    events: list[dict] | None = None, calendar: dict | None = None, date_only: bool = False) -> dict:
    """Assess known conflicts for a window. Returns conflicts, risks and schedule coverage.

    events   = calendar entries overlapping the window (already filtered to these buildings or campus-wide)
    calendar = counts of calendar entries by source, to report what was checked"""
    coverage = [
        {"source": "Outage history (supplied TRIRIGA notices)", "status": "covered",
         "period": _history_period(history)},
        {"source": "Class schedule (Registrar)", "status": "not_available",
         "period": "No schedule source covers this period. Internal class scheduling is a future integration."},
        _calendar_coverage(calendar),
        {"source": "Departmental input (experiments, deliveries, setup)", "status": "pending",
         "period": "Requires response from coordination contacts."},
    ]
    if not start or not end:
        return {"conflicts": [], "risks": ["Outage timing not stated; conflicts cannot be assessed"],
                "coverage": coverage, "window": None}
    conflicts = [{"type": "overlapping_outage", "title": f"Other outage {h['sr']}", "sr": h["sr"], "start": h["start"],
                  "end": h["end"], "systems": h["systems"], "classification": "recorded", "source": "TRIRIGA notices"}
                 for h in overlapping_outages(history, buildings, start, end, exclude_sr)]
    for e in events or []:
        conflicts.append({"type": e["kind"], "title": e["title"], "start": e["start"], "end": e["end"],
                          "building": e["building"] or "Campus-wide", "source": e["source"], "note": e["note"],
                          "classification": "confirmed" if e["source"] == "blocked" else "recorded"})
    risks = []
    if date_only:
        risks.append("Time of day not stated; whole day checked.")
    weekend = start.weekday() >= 5
    business = not weekend and (date_only or (start.hour < BUSINESS_END and end.hour > BUSINESS_START))
    if business and teaching_spaces:
        risks.append(f"Weekday hours, {teaching_spaces} teaching spaces: check with the Registrar.")
    if business and not teaching_spaces:
        risks.append("Weekday business hours: normal operations affected.")
    if weekend:
        risks.append("Weekend: still confirm experiments, deliveries and lab work.")
    if lab_spaces:
        risks.append(f"{lab_spaces} labs: confirm no experiments depend on these systems.")
    return {"conflicts": conflicts, "risks": risks, "coverage": coverage,
            "window": {"start": start.isoformat(), "end": end.isoformat(), "weekend": weekend,
                       "business_hours": business, "date_only": date_only},
            "verdict": "conflict" if conflicts else ("review" if risks else "clear_of_known_conflicts")}


def _calendar_coverage(calendar: dict | None) -> dict:
    if not calendar:
        return {"source": "Events calendar", "status": "not_available",
                "period": "No events loaded. Import the Rice events calendar (.ics) or add events."}
    parts = ", ".join(f"{n} {k}" for k, n in sorted(calendar.items()))
    return {"source": "Events calendar", "status": "covered",
            "period": f"Checked {parts}. Events not in the calendar cannot be detected."}


def propose_alternatives(start: datetime, end: datetime, buildings: list[str], history: list[dict],
                         n: int = 3, exclude_sr: str | None = None, avoid=None) -> list[dict]:
    """Alternative windows of equal duration: early weekday morning or Saturday morning,
    skipping windows that overlap other recorded outages. Proposals need departmental input."""
    duration = end - start
    hours = duration.total_seconds() / 3600
    proposals, day = [], start.date()
    for offset in range(1, 21):
        d = day + timedelta(days=offset)
        if d.weekday() == 6:
            continue
        if d.weekday() < 5:
            if hours > 4:            # cannot finish before business hours; offer Saturdays only
                continue
            s = datetime(d.year, d.month, d.day) + timedelta(hours=BUSINESS_START) - duration
            label = "Early weekday, finishing before business hours"
        else:
            s = datetime(d.year, d.month, d.day, 6)
            label = "Saturday morning (weekend still requires departmental review)"
        e = s + duration
        if overlapping_outages(history, buildings, s, e, exclude_sr) or (avoid and avoid(s, e)):
            continue
        proposals.append({"start": s.isoformat(), "end": e.isoformat(), "rationale": label,
                          "status": "proposal_requires_department_input"})
        if len(proposals) >= n:
            break
    return proposals


def _recurring_groups(history: list[dict], buildings: list[str] | None = None, level: str = "systems") -> dict[tuple, list[datetime]]:
    """Scheduled outages grouped by (building, identical system set), or by building alone when
    level="building"; dates de-duplicated and sorted."""
    groups: dict[tuple, list[datetime]] = {}
    for h in history:
        if h["priority"] != "Scheduled" or not h["start"]:
            continue
        for b in h["buildings"]:
            if buildings and b not in buildings:
                continue
            key = (b, tuple(sorted(h["systems"]))) if level == "systems" else (b, ("Any scheduled work",))
            groups.setdefault(key, []).append(_dt(h["start"]))
    return {k: sorted(set(d.replace(minute=0, second=0) for d in v)) for k, v in groups.items()}


def _interval_rule(dates: list[datetime]) -> tuple[float, float] | None:
    """Median interval and half-width of the window, from earlier dates only. None if too few."""
    gaps = [(b2 - a).days for a, b2 in zip(dates, dates[1:]) if (b2 - a).days >= 5]
    if not gaps:
        return None
    step = median(gaps)
    spread = (max(gaps) - min(gaps)) / 2 if len(gaps) > 1 else step * 0.25
    return step, max(spread, 7)


def forecast_backtest(history: list[dict], level: str = "systems") -> dict:
    """Rolling-origin backtest of the recurring-maintenance rule.

    For every recurring group, each occurrence after the second is predicted using only the
    occurrences before it (no look-ahead). Reports the absolute error in days, how often the actual
    date fell inside the predicted window, and a naive baseline that uses one pooled interval for
    every group (median of all gaps seen before the predicted date).

    level="systems" tests the deployed rule (same building and systems); level="building" tests the
    same rule on "next scheduled outage in this building", which has more repeats to learn from.
    """
    groups = _recurring_groups(history, level=level)
    trials = []
    for (b, systems), dates in groups.items():
        for k in range(2, len(dates)):
            rule = _interval_rule(dates[:k])
            if not rule:
                continue
            step, half = rule
            predicted = dates[k - 1] + timedelta(days=step)
            pooled = [(y - x).days for ds in groups.values() for x, y in zip(ds, ds[1:])
                      if y <= dates[k - 1] and (y - x).days >= 5]
            base = dates[k - 1] + timedelta(days=median(pooled)) if pooled else None
            err = abs((dates[k] - predicted).days)
            trials.append({"building": b, "systems": list(systems), "actual": dates[k].date().isoformat(),
                           "predicted": predicted.date().isoformat(), "window_days": round(half),
                           "error_days": err, "within_window": err <= half,
                           "baseline_error_days": abs((dates[k] - base).days) if base else None})
    if not trials:
        return {"n": 0, "level": level, "trials": [], "note": "Needs a series with 3 or more scheduled outages."}
    errs = [t["error_days"] for t in trials]
    base = [t["baseline_error_days"] for t in trials if t["baseline_error_days"] is not None]
    return {"n": len(trials), "level": level, "groups": len({(t["building"], tuple(t["systems"])) for t in trials}),
            "median_error_days": round(median(errs), 1), "mean_error_days": round(sum(errs) / len(errs), 1),
            "within_window": round(sum(t["within_window"] for t in trials) / len(trials), 3),
            "baseline_median_error_days": round(median(base), 1) if base else None,
            "method": "Rolling origin: each occurrence predicted from earlier occurrences only",
            "baseline": "One pooled interval for all buildings and systems",
            "trials": sorted(trials, key=lambda t: t["actual"])}


def recurring_maintenance_forecast(history: list[dict], as_of: datetime, buildings: list[str] | None = None) -> list[dict]:
    """Estimate next windows for recurring *scheduled* maintenance by (building, system set).

    Method: for each building and identical system signature with >= 2 scheduled outages, take the
    median interval between start dates and project forward from the last occurrence. This is a
    simple statistical estimate, not a trained model; with 2-3 occurrences confidence is low.
    """
    out = []
    for (b, systems), dates in _recurring_groups(history, buildings).items():
        rule = _interval_rule(dates)
        if len(dates) < 2 or not rule:
            continue
        step, spread = rule
        nxt = dates[-1] + timedelta(days=step)
        while nxt < as_of:
            nxt += timedelta(days=step)
        out.append({"building": b, "systems": list(systems), "occurrences": [d.date().isoformat() for d in dates],
                    "median_interval_days": round(step, 1), "estimated_next": nxt.date().isoformat(),
                    "window_days": round(max(spread, 7), 0),
                    "confidence": "low" if len(dates) < 4 else "moderate",
                    "classification": "inferred", "label": "Prediction: estimated window, not a confirmed date"})
    return sorted(out, key=lambda x: x["estimated_next"])


def _history_period(history: list[dict]) -> str:
    starts = sorted(h["start"] for h in history if h["start"])
    return f"{starts[0][:10]} to {starts[-1][:10]} (incomplete maintenance calendar)" if starts else "none"
