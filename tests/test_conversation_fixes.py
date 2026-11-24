"""Fixes from a real conversation (King's session, Nov 2026): the speaker's own name, corrections, acknowledgements,
half sentences, and outage and holiday questions answered from the records instead of the model's memory."""
import threading
import time
from datetime import date

import pytest

from tests.test_voice import ScriptedLLM

TODAY = date(2026, 11, 23)


@pytest.fixture(scope="module")
def conv(system):
    from app.voice.conversation import SpaceConversation
    return SpaceConversation(system, today=TODAY)


def test_own_name_is_never_the_person_to_sit_near(conv):
    llm = ScriptedLLM({"This is King": {"intent": "small_talk", "near": "King"}})
    st = conv.turn("k1", "Well, I'm not Jones, it's a different person. This is King.", llm)
    assert st["user_name"] == "King" and not (st.get("slots") or {}).get("near")      # small talk never edits slots
    assert "King" in llm.prompts[-1] and "James" not in llm.prompts[-1]              # called by their own name


def test_unknown_name_is_not_assumed(conv):
    llm = ScriptedLLM({})
    conv.turn("k2", "Hey buddy.", llm)
    assert "James" not in llm.prompts[-1] and "don't use one" in llm.prompts[-1]


def test_correction_takes_the_detail_out(conv):
    llm = ScriptedLLM({"near King": {"intent": "space_request", "request_type": "new_office", "near": "King",
                                     "department": "Chemistry"},
                       "didn't say": {"intent": "space_request", "near": "King"}})
    st = conv.turn("k3", "I need an office near King in Chemistry", llm)
    assert st["slots"]["near"] == "King" and st["question"].startswith("Did you mean")
    st = conv.turn("k3", "I didn't say king, king was my name.", llm)
    assert "near" not in st["slots"] and st["slots"]["department"] == "Chemistry"
    assert st["reply"].startswith("Sorry about that, I've taken King out of the request.")


def test_professor_with_the_same_name_is_kept(conv):
    llm = ScriptedLLM({"This is King": {"intent": "small_talk"},
                       "Professor King": {"intent": "space_request", "request_type": "new_office", "near": "King"}})
    conv.turn("k4", "This is King.", llm)
    st = conv.turn("k4", "Find an office near Professor King", llm)
    assert st["slots"]["near"] == "King" and "Cynthia King" in st["question"]       # faculty, not the speaker


def test_same_name_and_department_is_offered_once(conv):
    llm = ScriptedLLM({"Abigail": {"intent": "space_request", "request_type": "new_office", "near": "Abigail King"}})
    st = conv.turn("k5", "An office near Abigail King", llm)
    assert st["slots"].get("near_name") == "Abigail King" or st["question"].count("Abigail King") == 1


def test_acknowledgements_get_no_reply_and_stop_must_be_said(conv):
    llm = ScriptedLLM({"Great": {"intent": "stop"}, "Not probably": {"intent": "stop"}, "finished": {"intent": "stop"}})
    assert conv.turn("k6", "Great.", llm)["reply"] == ""
    assert conv.turn("k6", "Right.", llm)["reply"] == ""
    st = conv.turn("k6", "Not probably the time soon.", llm)
    assert st["intent"] == "small_talk"                                             # the model's "stop" is not enough
    assert conv.turn("k6", "Okay, so we're finished.", llm)["intent"] == "stop"


def test_outage_and_holiday_questions_come_from_the_records(conv):
    llm = ScriptedLLM({})
    steps = []
    st = conv.turn("k7", "When is Thanksgiving, and is there any outage that might conflict with it?", llm,
                   on_event=lambda e: "step" in e and steps.append(e["step"]))
    assert st["intent"] == "outage_question" and not llm.prompts                    # no model involved
    assert "Thanksgiving is Thursday, November 26" in st["reply"]                    # not the 28th
    assert any(s.startswith("Checked") and "outage requests" in s for s in steps)
    st = conv.turn("k7", "Can we find how many outages we have between today and next two weeks?", llm)
    assert "BRC" in st["reply"] and "December 1" in st["reply"] and "one moment" not in st["reply"].lower()


def test_status_question_says_what_is_happening(conv):
    llm = ScriptedLLM({})
    st = conv.turn("k8", "I'm saying you need to let me know what is going on, you took a long time.", llm)
    assert st["intent"] == "status" and st["reply"].startswith("Nothing is running right now.")


def test_chat_never_promises_to_look_things_up(conv):
    llm = ScriptedLLM({})
    conv.turn("k9", "What's up?", llm)
    assert "one moment" in llm.prompts[-1] and "Never invent dates" in llm.prompts[-1]
    assert "Monday, November 23, 2026" in llm.prompts[-1]                           # knows today's date


# ------------------------------------------------------------------------------ outage answers (pure)
def test_holiday_dates_are_computed():
    from app.voice.outage_answers import holidays
    h = holidays(2026)
    assert h["Thanksgiving"] == date(2026, 11, 26) and h["Labor Day"] == date(2026, 9, 7)
    assert h["Memorial Day"] == date(2026, 5, 25) and h["Martin Luther King Jr. Day"] == date(2026, 1, 19)


def test_time_windows_from_the_question():
    from app.voice.outage_answers import window
    assert window("outages in the next two weeks", TODAY)["end"] == date(2026, 12, 8)
    assert window("anything this week?", TODAY)["end"] == date(2026, 11, 30)
    assert window("next month", TODAY)["start"] == date(2026, 12, 1)
    assert window("what about tomorrow", TODAY)["start"] == date(2026, 11, 24)


def test_outage_answer_wording():
    from app.voice.outage_answers import answer
    hist = [{"sr": "SR-1", "start": "2026-12-01T06:00:00", "end": "2026-12-01T08:00:00",
             "buildings": ["BRC", "Tudor Fieldhouse"], "systems": ["Air Conditioning", "Electrical"]}]
    steps, text = answer("how many outages in the next two weeks", hist, None, TODAY)
    assert text == ("There is one outage on record between Monday, November 23 and Monday, December 7: an air "
                    "conditioning and electrical outage at BRC and Tudor Fieldhouse on Tuesday, December 1, from 6 to 8 a.m.")


# ------------------------------------------------------------------------------ half sentences (session)
def test_half_sentences_are_detected():
    from app.voice.session import looks_unfinished
    assert looks_unfinished("So I'm looking for") and looks_unfinished("That person is") and looks_unfinished("So, um")
    assert not looks_unfinished("The workstation in the chemistry department.") and not looks_unfinished("Keck Hall")


def test_a_paused_sentence_is_joined_before_answering(monkeypatch):
    import app.voice.session as vs
    monkeypatch.setattr(vs, "HOLD_SECONDS", 0.3)
    heard = iter(["So I'm looking for", "The workstation in the chemistry department."])
    seen = []

    class Conv:
        def turn(self, tid, text, llm, cancel, on_event, save_case=None, spoken=False):
            seen.append(text)
            return {"reply": "", "slots": {}, "intent": "ack"}
    s = vs.VoiceSession(Conv(), None, lambda a: next(heard), lambda t: b"", None, lambda st: "")
    s._emit = lambda e: None
    s._start_turn(audio=b"x")
    time.sleep(0.1)
    s._start_turn(audio=b"x")
    time.sleep(0.2)
    assert seen == ["So I'm looking for The workstation in the chemistry department."]


def test_a_lone_half_sentence_is_answered_after_the_wait(monkeypatch):
    import app.voice.session as vs
    monkeypatch.setattr(vs, "HOLD_SECONDS", 0.2)
    seen = []

    class Conv:
        def turn(self, tid, text, llm, cancel, on_event, save_case=None, spoken=False):
            seen.append(text)
            return {"reply": "", "slots": {}, "intent": "ack"}
    s = vs.VoiceSession(Conv(), None, lambda a: "That person is", lambda t: b"", None, lambda st: "")
    s._emit = lambda e: None
    s._start_turn(audio=b"x")
    time.sleep(0.6)
    assert seen == ["That person is"]


# ------------------------------------------------------------------------------ buildings and references
def test_building_questions_come_from_the_records(conv):
    llm = ScriptedLLM({})
    st = conv.turn("b1", "What is BRC?", llm)
    assert st["intent"] == "building_info" and not llm.prompts                       # no model, nothing invented
    assert st["reply"].startswith("BRC is the BioScience Research Collaborative.") and "rooms" in st["reply"]
    assert conv.turn("b1", "Tell me about Keck Hall", llm)["reply"].startswith("Keck Hall has")


def test_that_building_means_the_last_one_mentioned(conv):
    llm = ScriptedLLM({"free space": {"intent": "space_request", "request_type": "vacancy_review",
                                      "preferred_buildings": ["that building"]}})
    conv.turn("b2", "What is BRC?", llm)
    st = conv.turn("b2", "Well, can you find any free space in that building?", llm)
    assert st["slots"]["preferred_buildings"] == ["BRC"] and st.get("full_report")    # searched BRC, no question


def test_in_there_follows_the_last_search(conv):
    llm = ScriptedLLM({"lab space in Keck": {"intent": "space_request", "request_type": "vacancy_review",
                                             "preferred_buildings": ["Keck"]},
                       "office in there": {"intent": "space_request", "function": "Office", "department": "Chemistry"}})
    conv.turn("b3", "Is there any lab space in Keck?", llm)
    st = conv.turn("b3", "What about an office in there for Chemistry?", llm)
    assert st["slots"]["preferred_buildings"] == ["Keck"] and st["slots"]["department"] == "Chemistry"


def test_unrelated_questions_are_not_outage_questions(conv):
    llm = ScriptedLLM({"weather": {"intent": "outage_question"}})
    assert conv.turn("b4", "What is the weather?", llm)["intent"] == "small_talk"
