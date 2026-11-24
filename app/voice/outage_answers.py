"""Spoken answers about outages, holidays and dates, built from the records, never from the language model's memory.

    "How many outages do we have in the next two weeks?"
    "When is Thanksgiving, and does any outage conflict with it?"

The time window comes from the question (a holiday, "this week", "next two weeks", ...; two weeks by default).
Outages are the TRIRIGA requests on record; holidays are computed (US federal calendar) and widened by campus
calendar entries such as "Thanksgiving recess". Each check is reported as a step, then a short spoken answer.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "couple of": 2, "few": 3}
HOLIDAY_NAMES = [  # (words in the question, holiday name, words that mark campus calendar entries for it)
    (r"thanksgiving", "Thanksgiving", r"thanksgiving"),
    (r"christmas|xmas", "Christmas", r"christmas|winter (?:break|recess)|holiday break"),
    (r"new year", "New Year's Day", r"new year"),
    (r"labou?r day", "Labor Day", r"labor day"),
    (r"memorial day", "Memorial Day", r"memorial day"),
    (r"independence day|fourth of july|4th of july|july (?:4|fourth)", "Independence Day", r"independence day|july 4"),
    (r"juneteenth", "Juneteenth", r"juneteenth"),
    (r"\bmlk\b|martin luther king", "Martin Luther King Jr. Day", r"\bmlk\b|martin luther king"),
    (r"veterans day", "Veterans Day", r"veterans day"),
]
HOLIDAY_RX = re.compile("|".join(p for p, _, _ in HOLIDAY_NAMES) + r"|\bholidays?\b|\brecess\b", re.I)
OUTAGE_RX = re.compile(r"\b(outages?|shut ?downs?|shut-downs?|power (?:off|cut|outage)|maintenance|interruptions?|"
                       r"service disruptions?|planned work|scheduled work)\b", re.I)
DATE_Q_RX = re.compile(r"\b(when|what day|which day|what date|date of|is .* coming|conflict|clash|overlap)\b", re.I)


def is_outage_question(text: str) -> bool:
    """Outage questions, and questions about when a holiday is (and what it conflicts with)."""
    return bool(OUTAGE_RX.search(text) or (HOLIDAY_RX.search(text) and DATE_Q_RX.search(text)))


# ------------------------------------------------------------------------------ calendar arithmetic
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th weekday (Mon=0) of a month; n = -1 for the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def holidays(year: int) -> dict[str, date]:
    return {"New Year's Day": date(year, 1, 1), "Martin Luther King Jr. Day": _nth_weekday(year, 1, 0, 3),
            "Memorial Day": _nth_weekday(year, 5, 0, -1), "Juneteenth": date(year, 6, 19),
            "Independence Day": date(year, 7, 4), "Labor Day": _nth_weekday(year, 9, 0, 1),
            "Veterans Day": date(year, 11, 11), "Thanksgiving": _nth_weekday(year, 11, 3, 4),
            "Christmas": date(year, 12, 25)}


def next_holiday(name: str, today: date) -> date:
    d = holidays(today.year)[name]
    return d if d >= today else holidays(today.year + 1)[name]


def spoken_date(d: date | datetime, today: date) -> str:
    d = d.date() if isinstance(d, datetime) else d
    return f"{d:%A}, {d:%B} {d.day}" + (f", {d.year}" if d.year != today.year else "")


def _clock(t: datetime) -> str:
    if t.hour == 12 and t.minute == 0:
        return "noon"
    h = t.hour % 12 or 12
    return f"{h}{f':{t.minute:02d}' if t.minute else ''} {'a.m.' if t.hour < 12 else 'p.m.'}"


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    return "" if not items else items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def describe(o: dict, today: date) -> str:
    """'an air conditioning, electrical and elevator outage at BRC and Tudor Fieldhouse on Tuesday, December 1,
    from 6 to 8 a.m.'"""
    systems = [s.lower() for s in o.get("systems") or []]
    what = _join(systems[:3] + (["other"] if len(systems) > 3 else []))
    where = _join((o.get("buildings") or [])[:3]) or "a building not yet identified"
    s, e = datetime.fromisoformat(o["start"]), datetime.fromisoformat(o["end"]) if o.get("end") else None
    when = spoken_date(s, today)
    if e and e.date() == s.date():
        a, b = _clock(s), _clock(e)
        if a[-4:] == b[-4:] and a[-4:] in ("a.m.", "p.m."):        # "from 6 to 8 a.m."
            a = a[:-5]
        when += f", from {a} to {b}"
    elif e:
        when += f" at {_clock(s)} until {spoken_date(e, today)}"
    article = "an" if what[:1] in "aeiou" else "a"
    return f"{article} {what + ' ' if what else ''}outage at {where} on {when}"


# ------------------------------------------------------------------------------ the question's time window
def window(text: str, today: date, calendar=None) -> dict:
    """{start, end (exclusive), label, holiday, calendar_entries} for the question."""
    low = text.lower()
    for pattern, name, cal_words in HOLIDAY_NAMES:
        if re.search(pattern, low):
            day = next_holiday(name, today)
            start, end, entries = day, day + timedelta(days=1), []
            if calendar is not None:                       # campus entries such as "Thanksgiving recess"
                around = calendar.overlapping([], datetime.combine(day - timedelta(days=7), datetime.min.time()),
                                              datetime.combine(day + timedelta(days=7), datetime.min.time()))
                for ev in around:
                    if re.search(cal_words, ev["title"], re.I):
                        entries.append(ev)
                        start = min(start, datetime.fromisoformat(ev["start"]).date())
                        end = max(end, datetime.fromisoformat(ev["end"]).date())
            return {"start": start, "end": end, "label": name, "holiday": (name, day), "calendar_entries": entries}
    start = today
    if re.search(r"\btomorrow\b", low):
        start = today + timedelta(days=1)
        return {"start": start, "end": start + timedelta(days=1), "label": "tomorrow", "holiday": None, "calendar_entries": []}
    if re.search(r"\btoday\b", low) and not re.search(r"\b(?:next|coming|week|weeks|month|months|days)\b", low):
        return {"start": today, "end": today + timedelta(days=1), "label": "today", "holiday": None, "calendar_entries": []}
    if re.search(r"\bnext week\b", low):
        s = today + timedelta(days=7 - today.weekday())
        return {"start": s, "end": s + timedelta(days=7), "label": "next week", "holiday": None, "calendar_entries": []}
    if re.search(r"\bthis week\b", low):
        return {"start": today, "end": today + timedelta(days=7 - today.weekday()), "label": "this week",
                "holiday": None, "calendar_entries": []}
    if re.search(r"\bthis month\b", low):
        nxt = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
        return {"start": today, "end": nxt, "label": "this month", "holiday": None, "calendar_entries": []}
    m = re.search(r"\b(\d+|a couple of|couple of|a few|few|an|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                  r"(day|week|month)s?\b", low)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUMBERS.get(m.group(1).replace("a ", ""), 1)
        days = n * {"day": 1, "week": 7, "month": 30}[m.group(2)]
        return {"start": today, "end": today + timedelta(days=days + 1), "label": f"the next {m.group(0)}",
                "holiday": None, "calendar_entries": []}
    if re.search(r"\bnext month\b", low):
        s = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
        e = date(s.year + (s.month == 12), s.month % 12 + 1, 1)
        return {"start": s, "end": e, "label": "next month", "holiday": None, "calendar_entries": []}
    return {"start": today, "end": today + timedelta(days=15), "label": "the next two weeks", "holiday": None,
            "calendar_entries": []}


def _overlaps(o: dict, start: date, end: date) -> bool:
    if not o.get("start"):
        return False
    s = datetime.fromisoformat(o["start"]).date()
    e = datetime.fromisoformat(o["end"]).date() if o.get("end") else s
    return s < end and e >= start


# ------------------------------------------------------------------------------ the answer
def answer(question: str, history: list[dict], calendar, today: date) -> tuple[list[str], str]:
    """(steps shown while working, spoken reply)."""
    w = window(question, today, calendar)
    steps, said = [], []
    dated = [o for o in history if o.get("start")]
    if w["holiday"]:
        name, day = w["holiday"]
        steps.append(f"{name} is {spoken_date(day, today)}")
        said.append(f"{name} is {spoken_date(day, today)}.")
        for ev in w["calendar_entries"]:
            title = re.sub(r"\s*\(per email.*$", "", ev["title"]).strip()
            steps.append(f"Campus calendar: {title}, {spoken_date(datetime.fromisoformat(ev['start']), today)}")
        if w["calendar_entries"]:
            first = w["calendar_entries"][0]
            title = re.sub(r"\s*\(per email.*$", "", first["title"]).strip()
            said.append(f"The campus calendar also shows {title} on {spoken_date(datetime.fromisoformat(first['start']), today)}.")
    hits = sorted((o for o in dated if _overlaps(o, w["start"], w["end"])), key=lambda o: o["start"])
    last_day = w["end"] - timedelta(days=1)
    span = (f"between {spoken_date(w['start'], today)} and {spoken_date(last_day, today)}"
            if last_day > w["start"] else f"on {spoken_date(w['start'], today)}")
    steps.append(f"Checked {len(history)} outage requests on record: {len(hits)} {span}")

    if w["holiday"]:
        if hits:
            said.append(f"Yes, there {'is' if len(hits) == 1 else 'are'} {len(hits)} outage{'s' if len(hits) > 1 else ''} "
                        f"in that time: {describe(hits[0], today)}.")
        else:
            later = [o for o in dated if datetime.fromisoformat(o["start"]).date() >= w["end"]]
            near = min(later, key=lambda o: o["start"]) if later else None
            said.append("No outage on record falls in that time" +
                        (f"; the next one is {describe(near, today)}." if near else "."))
        return steps, _tidy(" ".join(said))

    if not hits:
        said.append(f"There are no outages on record {span}.")
    elif len(hits) == 1:
        said.append(f"There is one outage on record {span}: {describe(hits[0], today)}.")
    else:
        said.append(f"There are {len(hits)} outages on record {span}. The first is {describe(hits[0], today)}" +
                    (f", and the last is {describe(hits[-1], today)}." if len(hits) > 2 else
                     f", and the second is {describe(hits[1], today)}."))
    if hits and calendar is not None:                     # does any of them clash with a campus event?
        clashes = []
        for o in hits:
            s = datetime.fromisoformat(o["start"])
            e = datetime.fromisoformat(o["end"]) if o.get("end") else s + timedelta(hours=1)
            clashes += [ev for ev in calendar.overlapping(o.get("buildings") or [], s, e)]
        steps.append(f"Checked the campus calendar: {len(clashes)} conflicting event{'s' if len(clashes) != 1 else ''}")
        if clashes:
            title = re.sub(r"\s*\(per email.*$", "", clashes[0]["title"]).strip()
            said.append(f"It overlaps {title} on the calendar." if len(hits) == 1 else
                        f"At least one overlaps {title} on the calendar.")
        else:
            said.append("None of them conflicts with an event on the campus calendar." if len(hits) > 1 else
                        "It doesn't conflict with any event on the campus calendar.")
    if len(hits) > 2:
        said.append("The full list is on the Outages page.")
    return steps, _tidy(" ".join(said))


def _tidy(text: str) -> str:
    """'8 a.m..' -> '8 a.m.'; '8 a.m., and' stays."""
    return re.sub(r"\.{2,}", ".", text)
