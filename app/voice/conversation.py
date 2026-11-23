"""Spoken conversation with the Space Administrator Agent, as a LangGraph graph.

    user turn ─► understand ─┬─► chat ─────────────────────────────────► END   small talk, stop
                             ├─► save ─────────────────────────────────► END   "save it as a case"
                             ├─► refuse ───────────────────────────────► END   approve / assign: stays with James
                             └─► resolve ─┬─► ask ─────────────────────► END   missing or unclear detail
                                          └─► search ─► respond ───────► END   agent ran; spoken summary

* understand  the local LLM turns what James said into an intent and request fields (JSON schema output).
              Rules back it up: availability questions, buildings, "save it as a case", approvals.
              It only extracts; it never decides anything about space.
* resolve     deterministic checks against the records: person, department, buildings.
              Speech-to-text misspells names ("Bartle"), so near-matches are corrected and said aloud.
* search      says what it is about to do, then runs the existing Space Administrator Agent unchanged
              (same rules, evidence, limits). Each agent step is streamed live so James sees the process.
* respond     the LLM summarizes the agent's report in 2-3 spoken sentences, using only the report,
              including briefly how the result was reached.
* save        "save it as a case": saves the latest agent result (the rooms it found) as a space case for review.
* refuse      approving or assigning is never done by the assistant; it says where to do it.

State is kept per conversation (thread) by the checkpointer, so follow-ups such as "make it bigger"
or "what about BRC instead" refine the same request. Replies stream sentence by sentence through
LangGraph's custom stream so speech can start early; an interruption cancels the stream.
"""
from __future__ import annotations

import difflib
import json
import operator
import re
import threading
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig

from app.agents.schemas import FUNCTIONS, SpaceRequest
from app.logging_setup import get_logger

log = get_logger("voice.conversation")
REQUEST_TYPES = ["new_office", "department_function", "change_assignment", "vacancy_review"]
INTENTS = ["space_request", "new_request", "save_case", "approve_or_assign", "small_talk", "stop"]
SLOT_KEYS = ["request_type", "function", "department", "near", "min_area", "seats", "preferred_buildings",
             "exclude_buildings", "netid"]

UNDERSTAND_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "request_type": {"type": "string", "enum": REQUEST_TYPES},
        "function": {"type": "string", "enum": FUNCTIONS},
        "department": {"type": "string"},
        "near": {"type": "string"},
        "min_area": {"type": "number"},
        "seats": {"type": "integer"},
        "preferred_buildings": {"type": "array", "items": {"type": "string"}},
        "exclude_buildings": {"type": "array", "items": {"type": "string"}},
        "netid": {"type": "string"},
    },
    "required": ["intent"],
}

UNDERSTAND_PROMPT = """You extract a space request from what a university facilities administrator (James) says.
Return JSON only. Include a field ONLY if James stated it in his latest message; never guess or fill defaults.

intent:
- space_request: he asks for space, asks what is available, adds or changes a detail, or answers your question
- new_request: he starts a different request ("new one", "forget that", "another request")
- save_case: he wants the result (the rooms found) saved as a case
- approve_or_assign: he asks you to approve, assign, book or reserve
- small_talk: greetings, thanks, questions about what you can do
- stop: he wants to end ("stop", "that's all", "cancel")

fields:
- request_type: new_office (an office or room for a person or new hire), department_function (space for a function,
  e.g. a lab), change_assignment (move an existing person), vacancy_review (what space is available or vacant)
- function: one of {functions}. "office"/"room for someone" = Office; "lab" = Research Lab
- department: the department name as said, e.g. "Chemistry", "BioSciences", "Physics and Astronomy"
- near: name of the person whose office the space should be close to, without titles ("Professor", "Dr")
- min_area: minimum area in square feet (number)
- seats: number of people/seats
- preferred_buildings / exclude_buildings: building names or codes as said
- netid: a NetID if he spells one

Examples (latest message -> JSON):
"I need a room for Professor Smith's new hire" -> {{"intent": "space_request", "request_type": "new_office", "function": "Office", "near": "Smith"}}
"An office near Dr. Jane Doe, at least 120 square feet" -> {{"intent": "space_request", "request_type": "new_office", "function": "Office", "near": "Jane Doe", "min_area": 120}}
"Chemistry needs a research lab of 400 square feet" -> {{"intent": "space_request", "request_type": "department_function", "function": "Research Lab", "department": "Chemistry", "min_area": 400}}
"Do you have any available space in Keck Hall?" -> {{"intent": "space_request", "request_type": "vacancy_review", "preferred_buildings": ["Keck Hall"]}}
"Make it in Keck instead" -> {{"intent": "space_request", "preferred_buildings": ["Keck"]}}
"Okay, save it as a case" -> {{"intent": "save_case"}}
"Go ahead and approve it" -> {{"intent": "approve_or_assign"}}
"Hi, what can you do?" -> {{"intent": "small_talk"}}

The request so far (for context, do not repeat unchanged fields): {slots}
Your last question to him: {last_question}"""

RESPOND_PROMPT = """You are the voice of a campus space assistant talking to James, a facilities administrator.
Speak naturally in 2 or 3 short sentences. No lists, markdown, symbols or abbreviations; say "square feet".
Use ONLY facts in REPORT. In a few words, say how you got there, using PROCESS.
If REPORT has candidates: name the best as "room <room> on floor <floor> of <building>" and give one reason (closeness,
size or recorded vacancy); mention the most important condition briefly.
If REPORT has vacancies: say how many recorded vacancies there are, name one or two example rooms, and say that a recorded
vacancy still has to be confirmed with the department.
These are candidates for review, never assignments. End by offering to save this as a case or to refine the search.
{note}Never read out labels or field names.
PROCESS: {process}
REPORT: {report}"""

CHAT_PROMPT = """You are a campus space assistant talking to James, a facilities administrator, by voice.
You help find offices, labs and other space, and check what is recorded as available, using the university's space and
people records; a separate agent ranks candidates, and James saves results as cases and approves them himself.
Reply in one or two short spoken sentences, no lists or symbols."""

# --------------------------------------------------------------------------- rules that back up the LLM
SAVE_RX = re.compile(r"\b(save|create|open|make|log|file)\b[^.?!]{0,40}\bcase\b", re.I)
APPROVE_RX = re.compile(r"\b(approve|book|reserve)\b|\bassign (it|this|that|the room|the space)\b", re.I)
AVAIL_RX = re.compile(r"\b(available|vacant|vacancy|vacancies|free|empty|unused)\b", re.I)
SPACE_RX = re.compile(r"\b(space|spaces|room|rooms|office|offices|lab|labs|desk|desks|seat|seats)\b", re.I)
PLACE_RX = re.compile(r"\b(?:in|at|on|inside|within)\s+((?:the\s+)?[A-Z][\w.'’-]*(?:\s+[A-Z0-9][\w.'’-]*){0,5})")


def _skeleton(word: str) -> str:
    """Consonant outline of a word with repeats collapsed: bartle, bartel -> brtl; bartlett -> brtlt."""
    out = ""
    for ch in re.sub(r"[aeiouy]", "", word):
        if not out or out[-1] != ch:
            out += ch
    return out


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "twelve": 12, "twenty": 20, "fifty": 50, "hundred": 100}


def _numbers_in(text: str) -> set[float]:
    nums = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    low = text.lower()
    nums |= {float(v) for w, v in _NUM_WORDS.items() if re.search(rf"\b{w}\b", low)}
    if re.search(r"\bhundred\b", low):           # "one hundred fifty": STT usually writes digits anyway
        nums |= {n * 100 for n in nums if n < 10}
    return nums


def _said(value: str, text: str) -> bool:
    """True if every word of value (3+ letters) appears, or nearly appears, in what was said."""
    words = [w for w in re.findall(r"[a-z0-9]{3,}", value.lower())]
    spoken = re.findall(r"[a-z0-9]{2,}", text.lower())
    return bool(words) and all(any(w == s or difflib.SequenceMatcher(None, w, s).ratio() >= 0.8 for s in spoken)
                               for w in words) or value.strip().upper() in text.upper().split()


def _grounded(key: str, value, text: str):
    """Keep an extracted value only if James actually said it (small models fill schema fields eagerly).
    Categories (request type, function) are interpretations and are kept."""
    if key in ("min_area", "seats"):
        return value if float(value) in _numbers_in(text) else None
    if key in ("department", "near", "netid"):
        return value if _said(str(value), text) else None
    if key in ("preferred_buildings", "exclude_buildings"):
        return [b for b in value if _said(str(b), text)]
    return value


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


class ConvState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    slots: dict
    intent: str
    question: str | None
    notes: list[str]
    report: dict | None          # compact view the LLM may speak about
    full_report: dict | None     # the agent's complete report (saved as a case on request)
    saved: dict | None           # {"case_id", "run_id"} of the last save
    reply: str
    interrupted: bool


def _cfg(config: RunnableConfig, key: str):
    return (config.get("configurable") or {}).get(key)


# A sentence ends at . ! or ? followed by whitespace, but not after a lone capital ("George R. Brown").
SENTENCE_END = re.compile(r"(?<!\s[A-Z])[.!?]\s+")


def _speak(text: str) -> None:
    """Send finished sentences to the speech stream."""
    w, rest = get_stream_writer(), text.strip() + " "
    while (m := SENTENCE_END.search(rest)):
        w({"sentence": rest[: m.end()].strip()})
        rest = rest[m.end():]
    if rest.strip():
        w({"sentence": rest.strip()})


def _stream_reply(llm, messages: list[dict], cancel: threading.Event | None) -> tuple[str, bool]:
    """Stream the LLM reply, emitting each complete sentence as soon as it is ready."""
    w, buf, said = get_stream_writer(), "", []
    for piece in llm.chat_stream(messages, cancel):
        buf += piece
        while (m := SENTENCE_END.search(buf)):
            sentence, buf = buf[: m.end()].strip(), buf[m.end():]
            said.append(sentence)
            w({"sentence": sentence})
    if cancel is not None and cancel.is_set():
        return " ".join(said), True
    if buf.strip():
        said.append(buf.strip())
        w({"sentence": buf.strip()})
    return " ".join(said), False


def step_label(tool: str, res) -> str | None:
    """One readable line per agent tool call, shown live while the agent works."""
    d = res.data
    if tool == "resolve_building":
        return f"Matched building {d}" if res.status == "success" else None
    if tool == "get_department":
        return f"Confirmed department {d['name']}" if res.status == "success" else None
    if tool in ("find_person", "get_person"):
        if res.status not in ("success", "partial"):
            return None
        rooms = d.get("rooms") or []
        return f"Found {d['name']}'s office ({rooms[0].replace(' · ', ' ')})" if rooms else f"Found {d['name']}"
    if tool == "search_spaces":
        return f"Searched the space records: {len(d or [])} compatible rooms"
    if tool == "model_space_assessment":
        flagged = len(res.evidence)
        return f"Checked actual use with the model ({len(d or {})} rooms" + (f", {flagged} flagged)" if flagged else ")")
    if tool == "evaluate_space_constraints":
        return "Applied the size and capacity rules"
    if tool == "measure_proximity":
        return "Measured closeness to the reference office"
    if tool == "rank_candidate_spaces":
        return f"Ranked {len(d or [])} rooms by suitability and evidence"
    if tool == "check_planning_reservations":
        return "Checked planning reservations" + (f": {len(d)} flagged" if d else "")
    if tool == "get_floor_plan":
        return "Found the floor plans" if res.status == "success" else None
    return None


class SpaceConversation:
    """Builds and runs the graph. `coord` is the app's Coordinator (its Space agent and records)."""

    def __init__(self, coord) -> None:
        self.coord = coord
        self.space = coord.space
        ppl = coord.ds.people.dropna(subset=["name"]).drop_duplicates("netid")
        self._persons = [{"name": r["name"], "netid": r["netid"], "department": r["department"] or "unknown department",
                          "faculty": "Faculty" in str(r["role"]) or str(r["role"]).startswith("Dean"),
                          "tokens": re.split(r"[\s,.'’-]+", r["name"].lower())} for _, r in ppl.iterrows()]
        self._by_netid = {p["netid"].lower(): p for p in self._persons}
        self.graph = self._build().compile(checkpointer=InMemorySaver())

    # ------------------------------------------------------------------------------ nodes
    def understand(self, state: ConvState, config: RunnableConfig) -> dict:
        llm = _cfg(config, "llm")
        user = state["messages"][-1]["content"]
        prompt = UNDERSTAND_PROMPT.format(functions=", ".join(FUNCTIONS), slots=json.dumps(state.get("slots") or {}),
                                          last_question=state.get("question") or "none")
        out = llm.chat_json([{"role": "system", "content": prompt}, {"role": "user", "content": user}], UNDERSTAND_SCHEMA)
        intent = out.get("intent", "space_request")
        if intent not in INTENTS:
            intent = "space_request"
        # rules first for actions: saving and approving are recognized reliably, whatever the model says
        if SAVE_RX.search(user) or intent == "save_case":
            log.info("understood intent=save_case")
            return {"intent": "save_case", "notes": [], "question": None}
        if APPROVE_RX.search(user):
            return {"intent": "approve_or_assign", "notes": [], "question": None}

        slots = {} if intent == "new_request" else dict(state.get("slots") or {})
        kept, dropped = [], []
        for k in SLOT_KEYS:
            v = out.get(k)
            if v in (None, "", []):
                continue
            v = _grounded(k, v, user)
            if v in (None, []):
                dropped.append(k)
                continue
            slots[k] = v
            kept.append(k)
        if "preferred_buildings" not in kept:                 # rule backup: "in/at/on <Building>"
            places = self._places(user)
            if places:
                slots["preferred_buildings"] = places
                kept.append("preferred_buildings (rule)")
        if "near" not in kept:                                # rule backup: "Professor X", "near X"
            named = self._named_person(user)
            if named:
                slots["near"] = named
                kept.append("near (rule)")
        about_space = bool(SPACE_RX.search(user) or AVAIL_RX.search(user) or any(k for k in kept if "(rule)" in k))
        if intent in ("small_talk", "stop") and about_space:   # e.g. "do you have any available space in X?"
            intent = "space_request"
        if AVAIL_RX.search(user) and not {"near", "department", "near (rule)", "function"} & set(kept):
            slots["request_type"] = "vacancy_review"           # "what is available" = review recorded vacancies
        log.info(f"understood intent={intent} kept={kept} dropped_unsaid={dropped}")
        if intent in ("space_request", "new_request"):
            get_stream_writer()({"step": "Understood: " + self._describe(slots), "tool": "understand"})
        return {"intent": intent, "slots": slots, "notes": [], "question": None}

    def warm(self, llm) -> None:
        """Run one schema-bound call so the model's JSON grammar is ready before James speaks."""
        llm.chat_json([{"role": "system", "content": UNDERSTAND_PROMPT.format(functions=", ".join(FUNCTIONS), slots="{}",
                                                                             last_question="none")},
                       {"role": "user", "content": "Hello"}], UNDERSTAND_SCHEMA)

    def resolve(self, state: ConvState) -> dict:
        slots, notes = dict(state["slots"]), []
        if slots.get("near") and self._is_netid(slots["near"]):      # a NetID (e.g. "bartel" is also a surname)
            p = self._by_netid[slots["near"].strip().lower()]
            slots["near"], slots["near_name"] = p["netid"], p["name"]
        elif slots.get("near"):
            said = slots["near"]
            title = bool(re.search(r"\b(prof|professor|dr|doctor)\b", state["messages"][-1]["content"], re.I)) or bool(slots.get("near_title"))
            slots["near_title"] = title
            hits = self._person_candidates(said, prefer_faculty=title)
            if len(hits) == 1:
                p = hits[0]
                slots["near"] = p["netid"]            # NetID: unambiguous for the agent
                slots["near_name"] = p["name"]
                if p["exact"] is False:
                    notes.append(f"I took {said} to mean {p['name']}.")
            elif hits:
                opts = "; or ".join(f"{p['name']} in {p['department']}" for p in hits[:3])
                return {"question": f"Did you mean {opts}?", "slots": slots}
            else:
                return {"question": f"I couldn't find {said} in the people records. Could you spell the last name?",
                        "slots": {k: v for k, v in slots.items() if k != "near"}}
        if slots.get("department"):
            r = self.space.get_department(slots["department"])
            if r.status == "success":
                slots["department"] = r.data["name"]
            else:
                close = difflib.get_close_matches(slots["department"], self.space.departments, n=3, cutoff=0.6)
                if len(close) == 1 or (r.status == "ambiguous" and r.data and len(r.data) == 1):
                    notes.append(f"I used the department {close[0]}.")
                    slots["department"] = close[0]
                else:
                    opts = close or (r.data[:3] if r.status == "ambiguous" else [])
                    return {"question": (f"Which department do you mean: {', '.join(opts)}?" if opts else
                                         f"I couldn't find the department {slots['department']}. What is its official name?"),
                            "slots": slots}
        for key in ("preferred_buildings", "exclude_buildings"):
            known = []
            for b in slots.get(key) or []:
                m = self.space.resolver.resolve(b)
                if m.status == "resolved":
                    known.append(m.export_name)
                elif m.status == "ambiguous":
                    return {"question": f"Which building do you mean by {b}: {' or '.join(m.candidates)}?", "slots": slots}
                elif slots.get("near_name", slots.get("near", "")).lower().find(b.lower()) < 0:
                    return {"question": f"I don't know a building called {b}. Which building?",
                            "slots": {**slots, key: [x for x in slots[key] if x != b]}}
            slots[key] = list(dict.fromkeys(known))
        rt = slots.get("request_type", "new_office")
        if rt != "vacancy_review" and not (slots.get("department") or slots.get("near") or slots.get("preferred_buildings")):
            return {"question": "Which department is this for, or whose office should it be close to?", "slots": slots}
        if rt != "vacancy_review":                     # a vacancy review covers all functions
            slots.setdefault("function", "Office")
        return {"slots": slots, "notes": notes, "question": None}

    def ask(self, state: ConvState) -> dict:
        text = " ".join(state.get("notes") or []) + " " + state["question"]
        _speak(text)
        return {"reply": text.strip(), "messages": [{"role": "assistant", "content": text.strip()}], "interrupted": False}

    def chat(self, state: ConvState, config: RunnableConfig) -> dict:
        if state["intent"] == "stop":
            text = "Okay. I'm here when you need me."
            _speak(text)
            return {"reply": text, "messages": [{"role": "assistant", "content": text}], "interrupted": False}
        history = [m for m in state["messages"][-6:]]
        text, cut = _stream_reply(_cfg(config, "llm"), [{"role": "system", "content": CHAT_PROMPT}, *history], _cfg(config, "cancel"))
        return {"reply": text, "messages": [{"role": "assistant", "content": text}], "interrupted": cut}

    def search(self, state: ConvState) -> dict:
        slots = state["slots"]
        req = SpaceRequest(**{k: v for k, v in slots.items() if k in SpaceRequest.model_fields})
        w = get_stream_writer()
        w({"status": "searching", "request": req.model_dump(exclude_defaults=True)})
        _speak(self._lead_in(slots))                                  # say what is about to happen
        steps: list[str] = []

        def observe(tool, res):
            label = step_label(tool, res)
            if label and label not in steps:
                steps.append(label)
                w({"step": label, "tool": tool, "ok": res.status in ("success", "partial")})
        full = self.coord.run_space(req, observer=observe)
        report = {"summary": full["summary"], "open_questions": full["verification_questions"][:2],
                  "requirements": {r["field"]: r["value"] for r in full["requirements"]}}
        if full.get("vacancies") is not None:
            vac = full["vacancies"]
            clean = [v for v in vac if not v["flags"]] or vac
            report["vacancies"] = {"total": len(vac), "with_flags": sum(1 for v in vac if v["flags"]),
                                   "examples": [{"room": v["room"], "floor": v["floor"].lstrip("0") or "0",
                                                 "building": v["building"], "area_sf": round(v["area"]) if v["area"] else None,
                                                 "seat_status": v["seat_status"], "allocation": v["allocation"]}
                                                for v in clean[:3]]}
        else:
            report["candidates"] = [{"room": c["room"], "floor": c["floor"].lstrip("0") or "0", "building": c["building"],
                                     "area_sf": round(c["area"]) if c["area"] else None,
                                     "status": c["status"].replace("_", " "), "closeness": (c.get("proximity") or {}).get("label"),
                                     "allocation": c["allocation"], "conditions": c.get("conditions", [])[:2],
                                     "recorded_vacant": c["seat_status"] == "Vacant"} for c in full["candidates"][:3]]
        report["process"] = steps
        return {"report": report, "full_report": full}

    def respond(self, state: ConvState, config: RunnableConfig) -> dict:
        notes = " ".join(state.get("notes") or [])
        r = dict(state["report"])
        process = "; ".join(r.pop("process", []))
        prompt = RESPOND_PROMPT.format(report=json.dumps(r), process=process,
                                       note=f'Begin with this sentence, as is: "{notes}"' + chr(10) if notes else "")
        user = state["messages"][-1]["content"]
        text, cut = _stream_reply(_cfg(config, "llm"), [{"role": "system", "content": prompt},
                                                        {"role": "user", "content": user}], _cfg(config, "cancel"))
        return {"reply": text, "messages": [{"role": "assistant", "content": text}], "interrupted": cut}

    def save(self, state: ConvState, config: RunnableConfig) -> dict:
        full, saved = state.get("full_report"), state.get("saved") or {}
        if not full:
            text = "There's nothing to save yet. Tell me what space you need, and I'll search first."
        elif saved.get("run_id") == full.get("run_id"):
            text = f"This search is already saved as case {saved['case_id']}."
        else:
            save_case = _cfg(config, "save_case")
            if not save_case:
                text = "I can't save cases from here. Use the Create case button."
            else:
                cid = save_case(state)
                saved = {"case_id": cid, "run_id": full.get("run_id")}
                get_stream_writer()({"saved": cid})
                top = (full.get("candidates") or [None])[0]
                what = (f" with room {top['room']} in {top['building']} as the top candidate" if top
                        else f" with {len(full.get('vacancies') or [])} recorded vacancies" if full.get("vacancies") is not None else "")
                text = f"Saved as a case{what}. It's on the Cases page, ready for your review."
        _speak(text)
        return {"reply": text, "messages": [{"role": "assistant", "content": text}], "saved": saved, "interrupted": False}

    def refuse(self, state: ConvState) -> dict:
        text = ("I can't approve or assign space; that decision stays with you. "
                "Save this as a case, then approve it on the case page.")
        _speak(text)
        return {"reply": text, "messages": [{"role": "assistant", "content": text}], "interrupted": False}

    # ------------------------------------------------------------------------------ graph
    def _build(self) -> StateGraph:
        g = StateGraph(ConvState)
        for name in ("understand", "resolve", "ask", "chat", "search", "respond", "save", "refuse"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "understand")
        route = {"small_talk": "chat", "stop": "chat", "save_case": "save", "approve_or_assign": "refuse"}
        g.add_conditional_edges("understand", lambda s: route.get(s["intent"], "resolve"),
                                ["chat", "save", "refuse", "resolve"])
        g.add_conditional_edges("resolve", lambda s: "ask" if s.get("question") else "search", ["ask", "search"])
        g.add_edge("search", "respond")
        for end in ("ask", "chat", "respond", "save", "refuse"):
            g.add_edge(end, END)
        return g

    # ------------------------------------------------------------------------------ helpers
    @staticmethod
    def _describe(slots: dict) -> str:
        """What was understood, in a few words: 'available space · Anderson Hall'."""
        kind = {"vacancy_review": "available space", "department_function": (slots.get("function") or "space").lower(),
                "change_assignment": "reassignment"}.get(slots.get("request_type"), (slots.get("function") or "office").lower())
        parts = [kind]
        if slots.get("near"):
            parts.append(f"near {slots.get('near_name') or slots['near']}")
        if slots.get("department"):
            parts.append(str(slots["department"]))
        if slots.get("min_area"):
            parts.append(f"at least {slots['min_area']:g} sf")
        if slots.get("preferred_buildings"):
            parts.append(", ".join(slots["preferred_buildings"]))
        return " · ".join(parts)

    def _lead_in(self, slots: dict) -> str:
        """A short spoken line before the agent runs, so James knows what is happening."""
        where = _join(slots.get("preferred_buildings") or []) if slots.get("preferred_buildings") else ""
        fn = (slots.get("function") or "office").lower()
        if slots.get("request_type") == "vacancy_review":
            return f"Checking recorded vacancies in {where or 'the Natural Sciences buildings'}."
        if slots.get("near_name") or slots.get("near"):
            return f"Looking for {fn} space near {slots.get('near_name') or slots['near']}{' in ' + where if where else ''}."
        if slots.get("department"):
            return f"Searching {fn} space for {slots['department']}{' in ' + where if where else ''}."
        return f"Searching {fn} space{' in ' + where if where else ''}."

    def _places(self, text: str) -> list[str]:
        """Buildings named after in/at/on, matched against the records (longest name that resolves)."""
        out = []
        for m in PLACE_RX.finditer(text):
            words = re.sub(r"^the\s+", "", m.group(1)).rstrip(".,?!").split()
            for n in range(len(words), 0, -1):
                hit = self.space.resolver.resolve(" ".join(words[:n]))
                if hit.status in ("resolved", "ambiguous"):
                    out.append(" ".join(words[:n]))
                    break
        return out

    # Titles match in any case; the name itself must be capitalized (speech-to-text capitalizes names).
    _TITLED = re.compile(r"\b(?i:professor|prof\.?|dr\.?|doctor)\s+([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)")
    _NEAR = re.compile(r"\b(?i:close to|near|next to|beside)\s+([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)")

    def _named_person(self, text: str) -> str | None:
        """A person named with a title or 'near/close to' (not a building), e.g. "Professor Bartle's" -> "Bartle"."""
        for rx in (self._TITLED, self._NEAR):
            m = rx.search(text)
            if m:
                name = re.sub(r"['’]s$", "", m.group(1)).strip()
                if self.space.resolver.resolve(name).status != "resolved":
                    return name
        return None

    def _is_netid(self, text: str) -> bool:
        return bool(self._by_netid.get(text.strip().lower()))

    def _person_candidates(self, said: str, prefer_faculty: bool) -> list[dict]:
        """People matching a spoken name. Exact name tokens win; otherwise names that sound alike
        (speech-to-text writes "Bartle" for "Bartel"). With "Professor"/"Dr", faculty are preferred."""
        tokens = [t for t in re.split(r"[\s,.'’]+", said.lower()) if t and t not in ("professor", "prof", "dr", "doctor", "s")]
        if not tokens:
            return []
        def score(parts: list[str], tok: str) -> float:
            # spelling similarity, plus a bonus when the consonants match (Bartle / Bartel sound alike)
            return max((difflib.SequenceMatcher(None, tok, p).ratio() + (0.2 if _skeleton(tok) == _skeleton(p) else 0)
                        for p in parts), default=0)
        exact, close = [], []
        for p in self._persons:
            parts = p["tokens"]
            if all(t in parts for t in tokens):
                exact.append({**p, "exact": True})
            elif all(score(parts, t) >= 0.8 for t in tokens):
                close.append({**p, "exact": False, "score": sum(score(parts, t) for t in tokens)})
        for group in (exact, sorted(close, key=lambda x: -x["score"])):
            if prefer_faculty:
                fac = [p for p in group if p["faculty"]]
                group = fac or group
            if group and not group[0]["exact"] and len(group) > 1 and group[0]["score"] - group[1]["score"] >= 0.05:
                return group[:1]                       # one candidate clearly sounds closest
            if group:
                return group[:5]
        return []

    # ------------------------------------------------------------------------------ run
    def state(self, thread_id: str) -> dict:
        return self.graph.get_state({"configurable": {"thread_id": thread_id}}).values

    def mark_saved(self, thread_id: str, case_id: str, state: dict) -> None:
        """Record a save made outside the conversation (the Create case button)."""
        run_id = (state.get("full_report") or {}).get("run_id")
        self.graph.update_state({"configurable": {"thread_id": thread_id}}, {"saved": {"case_id": case_id, "run_id": run_id}})

    def turn(self, thread_id: str, text: str, llm, cancel: threading.Event | None = None, on_event=None,
             save_case=None) -> dict:
        """Run one user turn. on_event receives {"sentence"}, {"status"}, {"step"} and {"saved"} events as they
        happen. save_case(state) -> case id lets the assistant save by voice. Returns the state after the turn."""
        config = {"configurable": {"thread_id": thread_id, "llm": llm, "cancel": cancel, "save_case": save_case}}
        for mode, chunk in self.graph.stream({"messages": [{"role": "user", "content": text}]}, config,
                                             stream_mode=["custom", "updates"]):
            if mode == "custom" and on_event:
                on_event(chunk)
            if cancel is not None and cancel.is_set():
                break
        return self.graph.get_state(config).values
