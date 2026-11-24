"""Proximity ranking, prose dates, events calendar and outage conflict detection."""
from datetime import date, datetime

from app.agents.schemas import OutageRequest, SpaceRequest
from app.scenarios import resolve_payload, scenarios
from app.services.events import EventStore, extract_dates, extract_duration_hours, parse_ics

ICS = """BEGIN:VCALENDAR
BEGIN:VEVENT
SUMMARY:Chemistry Department Symposium
DTSTART:20261203T090000
DTEND:20261203T170000
LOCATION:BioScience Research Collaborative\\, Room 280
END:VEVENT
BEGIN:VEVENT
SUMMARY:Last Day of Classes
DTSTART;VALUE=DATE:20261204
END:VEVENT
END:VCALENDAR"""


def test_prose_dates_and_durations():
    ref = datetime(2026, 10, 22)
    assert extract_dates("I was able to secure Wednesday, November 25th.", ref) == [(date(2026, 11, 25), date(2026, 11, 25))]
    assert extract_dates("Fire Alarm Testing - February 18-19, 2026", ref)[0] == (date(2026, 2, 18), date(2026, 2, 19))
    assert extract_duration_hours("a full 8-hour shutdown") == extract_duration_hours("an eight-hour shutdown") == 8.0


def test_ics_import_resolves_buildings(system):
    ev = parse_ics(ICS)
    assert len(ev) == 2 and ev[1]["all_day"]
    store = EventStore()
    assert store.import_ics(ICS, system.space.resolver) == {"imported": 2, "skipped": 0}
    assert store.import_ics(ICS, system.space.resolver) == {"imported": 0, "skipped": 2}      # importing again is safe
    hits = store.overlapping(["BRC"], datetime(2026, 12, 3, 10), datetime(2026, 12, 3, 12))
    assert [h["title"] for h in hits] == ["Chemistry Department Symposium"]
    assert store.overlapping(["Keck"], datetime(2026, 12, 4, 8), datetime(2026, 12, 4, 9))[0]["building"] is None   # campus-wide


def _outage(system, sid):
    sc = next(s for s in scenarios() if s.id == sid)
    return system.run_outage(OutageRequest(**resolve_payload(sc, system.corpus)))


def test_brk_elevator_conflicts_with_recess(system):
    r = _outage(system, "OUT-BRK-ELEVATOR")
    assert r["outage"]["date_only"] and r["outage"]["proposed_start"].startswith("2026-11-25")
    s = r["scheduling"]
    assert s["verdict"] == "conflict" and any("recess" in c["title"].lower() for c in s["conflicts"])
    assert s["alternatives"] and all(datetime.fromisoformat(a["start"]).weekday() == 5 for a in s["alternatives"])  # 8 h -> Saturdays


def test_blocked_time_is_flagged_and_avoided(system):
    store = system.outage.events
    ev = store.add("Symposium (blocked)", datetime(2026, 11, 20, 0), datetime(2026, 11, 21, 0), "O'Connor Engineering and Science",
                   "blocked", "blocked", "James: no outage")
    try:
        r = system.outage.review_window(["O'Connor Engineering and Science"], datetime(2026, 11, 20, 6), datetime(2026, 11, 20, 8))
        assert r["verdict"] == "conflict" and r["conflicts"][0]["source"] == "blocked"
        assert all(not a["start"].startswith("2026-11-20") for a in r["alternatives"])
    finally:
        store.delete(ev["id"])


def test_no_timing_means_no_claim(system):
    r = _outage(system, "OUT-KECK-ELEVATOR")
    assert r["scheduling"]["window"] is None and r["outage"]["proposed_start"] is None
    assert any("When does the work" in q for q in r["follow_up_questions"])


def test_proximity_to_reference_office(system):
    r = system.run_space(SpaceRequest(request_type="new_office", near="Bonnie Bartel", function="Office"))
    top = r["candidates"][0]
    assert top["building"] == "George R. Brown" and top["floor"] == "03"
    scores = [c["criteria"]["proximity"] for c in r["candidates"]]
    assert scores[0] == max(scores)
    assert any(q["field"] == "Close to" for q in r["requirements"])


def test_ambiguous_reference_person_defers_ranking(system):
    r = system.run_space(SpaceRequest(request_type="new_office", near="Ajo-Franklin", function="Office"))
    assert r["candidates"] == [] and "deferred" in r["summary"]
    assert any("Which person" in q for q in r["verification_questions"])


def test_forecast_backtest_uses_only_earlier_outages():
    from app.services.scheduling import forecast_backtest
    h = lambda d: {"priority": "Scheduled", "start": f"{d}T08:00:00", "buildings": ["Biology"], "systems": ["Electrical"]}
    # perfectly regular 30-day series: every held-out date is predicted exactly from the dates before it
    r = forecast_backtest([h("2026-01-01"), h("2026-01-31"), h("2026-03-02"), h("2026-04-01")])
    assert r["n"] == 2 and r["median_error_days"] == 0 and r["within_window"] == 1.0
    # the first prediction only knows the first two dates, so a later change in rhythm is not seen in advance
    r = forecast_backtest([h("2026-01-01"), h("2026-01-31"), h("2026-05-01")])
    assert r["n"] == 1 and r["trials"][0]["predicted"] == "2026-03-02" and r["trials"][0]["error_days"] == 60
    # too few repeats is reported, not invented
    assert forecast_backtest([h("2026-01-01"), h("2026-01-31")])["n"] == 0
