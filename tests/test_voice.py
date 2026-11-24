"""Voice assistant: the LangGraph conversation (with a scripted LLM), settings, the WebSocket and barge-in.

The LLM is faked so the tests are deterministic and need no Ollama. Speech-detection tests use the
real local models when they are installed (models/voice) and are skipped otherwise."""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.voice import engines

HAVE_SPEECH = engines.vad_ready() and bool(engines.tts_voices())


class ScriptedLLM:
    """Returns the JSON given for the first matching phrase; replies with fixed sentences."""

    def __init__(self, script: dict, reply: str = "The best room is room 300 on floor 3. Shall I save it as a case?"):
        self.script, self.reply, self.prompts = script, reply, []

    def chat_json(self, messages, schema):
        user = messages[-1]["content"]
        return next((v for k, v in self.script.items() if k in user), {"intent": "small_talk"})

    def chat_stream(self, messages, cancel=None):
        self.prompts.append(messages[0]["content"])
        for word in self.reply.split(" "):
            if cancel is not None and cancel.is_set():
                return
            yield word + " "


@pytest.fixture(scope="module")
def conv(system):
    from app.voice.conversation import SpaceConversation
    return SpaceConversation(system)


def test_unsaid_details_are_dropped(conv):
    llm = ScriptedLLM({"office": {"intent": "space_request", "request_type": "new_office", "function": "Office",
                                  "department": "Chemistry", "min_area": 200, "seats": 1, "preferred_buildings": ["Hegeman"]}})
    st = conv.turn("g1", "I need an office for Chemistry.", llm)
    assert st["slots"]["department"] == "Chemistry"
    assert "min_area" not in st["slots"] and "seats" not in st["slots"]          # never said
    assert not st["slots"].get("preferred_buildings")


def test_misheard_name_is_matched_and_said_aloud(conv):
    llm = ScriptedLLM({"Bartle": {"intent": "space_request", "request_type": "new_office", "near": "Bartle"}})
    said = []
    st = conv.turn("g2", "I need a room for Professor Bartle's new hire.", llm, on_event=lambda e: said.append(e))
    assert st["slots"]["near"] == "bartel" and st["slots"]["near_name"] == "Bonnie Bartel"
    assert st["full_report"]["requirements"][-1]["value"] != ""                   # the agent ran
    assert "Bonnie Bartel" in llm.prompts[-1]                                     # correction is spoken first
    assert any("sentence" in e for e in said)


def test_missing_department_or_person_asks_first(conv):
    llm = ScriptedLLM({"office": {"intent": "space_request", "request_type": "new_office", "function": "Office"}})
    st = conv.turn("g3", "I need an office.", llm)
    assert st["question"] and "department" in st["reply"].lower() and not st.get("full_report")


def test_follow_up_refines_the_same_request(conv):
    llm = ScriptedLLM({"Chemistry office": {"intent": "space_request", "department": "Chemistry", "function": "Office"},
                       "150": {"intent": "space_request", "min_area": 150}})
    conv.turn("g4", "A Chemistry office please.", llm)
    st = conv.turn("g4", "At least 150 square feet.", llm)
    assert st["slots"]["department"] == "Chemistry" and st["slots"]["min_area"] == 150
    assert st["full_report"]["request"]["min_area"] == 150


def test_unknown_building_is_questioned_not_searched(conv):
    llm = ScriptedLLM({"Atlantis": {"intent": "space_request", "department": "Chemistry", "preferred_buildings": ["Atlantis"]}})
    st = conv.turn("g5", "Chemistry office in Atlantis.", llm)
    assert "Atlantis" in st["reply"] and not st.get("full_report")


def test_availability_question_runs_the_agent_and_shows_its_steps(conv):
    llm = ScriptedLLM({})                       # the model mistakes it for small talk; the rules recover it
    events = []
    st = conv.turn("g7", "Do you have any available space on Anderson Hall?", llm, on_event=events.append)
    assert st["slots"]["request_type"] == "vacancy_review" and st["slots"]["preferred_buildings"] == ["Anderson Hall"]
    assert st["full_report"]["vacancies"] is not None and st["report"]["vacancies"]["total"] == len(st["full_report"]["vacancies"])
    said = [e["sentence"] for e in events if "sentence" in e]
    steps = [e["step"] for e in events if "step" in e and e.get("tool") != "understand"]
    assert any(e.get("step", "").startswith("Understood: available space") for e in events)   # what was understood
    assert said[0].startswith("Checking recorded vacancies in Anderson Hall")        # spoken lead-in
    assert any("model" in s for s in steps)                                           # live agent steps
    assert "PROCESS:" in llm.prompts[-1] and steps[0] in llm.prompts[-1]              # the answer explains how


def test_save_by_voice_and_never_approve(conv):
    llm = ScriptedLLM({"Chemistry": {"intent": "space_request", "department": "Chemistry"}})
    saved_states = []
    def save_case(state):
        saved_states.append(state)
        return "SA-9999"
    assert "nothing to save" in conv.turn("g8", "Okay, save it as a case.", llm, save_case=save_case)["reply"]
    conv.turn("g8", "A Chemistry office please.", llm, save_case=save_case)
    events = []
    st = conv.turn("g8", "Okay, save it as a case.", llm, on_event=events.append, save_case=save_case)
    assert len(saved_states) == 1 and {"saved": "SA-9999"} in events and "top candidate" in st["reply"]
    assert "already saved" in conv.turn("g8", "Save it as a case please.", llm, save_case=save_case)["reply"]
    assert len(saved_states) == 1                                                      # no duplicate case
    st = conv.turn("g8", "Great, approve it.", llm, save_case=save_case)
    assert st["intent"] == "approve_or_assign" and "stays with you" in st["reply"]


def test_only_non_words_count_as_noise():
    from app.voice.session import is_noise
    assert all(is_noise(t) for t in ["", "Uh.", "hmm", "Mm-hmm.", "...", "Um, uh"[:3]])
    assert not any(is_noise(t) for t in ["Yes.", "No.", "Yeah.", "Keck", "OK", "Find me an office in Keck"])


def test_yes_after_the_save_offer_saves_and_is_never_an_approval(conv):
    llm = ScriptedLLM({"Chemistry": {"intent": "space_request", "department": "Chemistry"},
                       "Yes": {"intent": "approve_or_assign"}},                        # the model misreads "yes"
                      reply="Room 300 is the best fit. Would you like me to save this as a case or refine the search?")
    saved = []
    conv.turn("g9", "Chemistry office please.", llm, spoken=True, save_case=lambda st: saved.append(1) or "SA-0001")
    st = conv.turn("g9", "Yes.", llm, spoken=True, save_case=lambda st: saved.append(1) or "SA-0001")
    assert st["intent"] == "save_case" and saved and "Saved as a case" in st["reply"]


class _ScriptedVAD:
    """Speech probability per frame from a script: 1 = speech, 0 = silence."""
    frame = 512
    def __init__(self, script): self.script = list(script)
    def __call__(self, frame): return float(self.script.pop(0)) if self.script else 0.0


def _session(vad):
    from app.voice.session import VoiceSession
    s = VoiceSession(None, None, lambda a: "", lambda t: b"", vad, lambda st: "")
    s.events, s.turns = [], []
    s._emit = s.events.append
    s._start_turn = lambda audio=None, text=None: s.turns.append(len(audio))
    return s


def _feed(s, n):
    import numpy as np
    s._feed(np.zeros(512 * n, "<i2").tobytes())


def test_a_noise_blip_does_not_interrupt_the_assistant():
    s = _session(_ScriptedVAD([1] * 5 + [0] * 30))                                   # ~160 ms of "speech": a cough
    s.client_playing = True                                                            # the assistant is talking
    _feed(s, 35)
    assert not any(isinstance(e, dict) and e.get("type") == "interrupt" for e in s.events) and not s.turns


def test_real_speech_interrupts_the_assistant_and_is_kept_whole():
    s = _session(_ScriptedVAD([1] * 20 + [0] * 30))                                   # ~650 ms of speech
    s.client_playing = True
    _feed(s, 50)
    assert any(isinstance(e, dict) and e.get("type") == "interrupt" for e in s.events) and len(s.turns) == 1
    assert s.turns[0] >= 20 * 512                                                      # captured from the start


def test_speech_starts_as_quickly_as_before_when_the_assistant_is_quiet():
    s = _session(_ScriptedVAD([1] * 10 + [0] * 30))                                   # ~320 ms of speech
    _feed(s, 40)
    assert len(s.turns) == 1 and not any(isinstance(e, dict) and e.get("type") == "interrupt" for e in s.events)


def test_muted_microphone_means_text_only_replies():
    import asyncio, json
    from app.voice.session import VoiceSession
    class Conv:
        def turn(self, tid, text, llm, cancel, on_event, save_case=None, spoken=False):
            on_event({"sentence": "Here is the answer."})
            return {"reply": "Here is the answer.", "slots": {}, "intent": "small_talk"}
    spoken = []
    s = VoiceSession(Conv(), None, lambda a: "", lambda t: spoken.append(t) or b"\x00\x00", None, lambda st: "")
    events = []
    s._emit = events.append
    s.loop = None
    import threading as th
    th.Thread(target=s._tts_worker, daemon=True).start()
    async def go():
        await s._control(json.dumps({"type": "mic_mute"}))
        await s._control(json.dumps({"type": "text", "text": "Any office in Keck?"}))
    asyncio.run(go())
    time.sleep(0.5)
    assert any(e.get("type") == "say" for e in events if isinstance(e, dict))         # the text still arrives
    assert not spoken and not any(isinstance(e, bytes) for e in events)               # but nothing is spoken
    s._tts_q.put(None)


def test_warm_up_never_changes_the_model_load_options():
    from app.voice.llm import OllamaChat
    llm = OllamaChat("qwen3:8b")
    warm = llm._body([], options={"num_predict": 1})["options"]
    normal = llm._body([])["options"]
    assert warm["num_ctx"] == normal["num_ctx"] and warm["num_predict"] == 1   # same context: no reload


def test_interrupt_cancels_the_reply(conv):
    llm = ScriptedLLM({"Chemistry": {"intent": "space_request", "department": "Chemistry"}},
                      reply=" ".join(["word."] * 200))
    cancel, said = threading.Event(), []
    def on_event(e):
        if "sentence" in e:
            said.append(e["sentence"])
            if len(said) == 3:
                cancel.set()
    conv.turn("g6", "Chemistry office.", llm, cancel=cancel, on_event=on_event)
    assert 3 <= len(said) < 200


# ------------------------------------------------------------------------------ settings and API
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import api
    from app.services.cases import CaseStore
    with TestClient(api.app) as c:
        api.STATE["store"] = CaseStore(tmp_path_factory.mktemp("db") / "voice.db")
        api.attach_switches(api.STATE["store"])
        yield c


def _fake_models(monkeypatch, tools=True):
    import app.voice.service as svc
    monkeypatch.setattr(svc, "installed_models", lambda: ([{"name": "qwen3:8b", "size_gb": 5.2, "tools": tools,
                                                             "family": "qwen3", "parameters": "8B"},
                                                            {"name": "dolphin3:8b", "size_gb": 4.9, "tools": False,
                                                             "family": "llama", "parameters": "8B"}], None))


def test_voice_switch_needs_no_unlock_key_but_needs_models(client, monkeypatch):
    monkeypatch.delenv("FAI_ALLOW_UNLOCK", raising=False)
    from app import api
    _fake_models(monkeypatch, tools=False)
    api.STATE["voice"]._ready_cache = None
    r = client.put("/api/v1/settings/voice_assistant", json={"on": True})
    assert r.status_code == 409 and "tool calling" in r.json()["detail"]
    if not HAVE_SPEECH:
        pytest.skip("speech models not installed")
    _fake_models(monkeypatch, tools=True)
    api.STATE["voice"]._ready_cache = None
    assert client.put("/api/v1/settings/voice_assistant", json={"on": True}).status_code == 200
    client.put("/api/v1/settings/voice_assistant", json={"on": False})


def test_only_installed_tool_models_can_be_chosen(client, monkeypatch):
    _fake_models(monkeypatch)
    o = client.get("/api/v1/voice/options").json()
    assert o["selected"]["voice_llm"] == "qwen3:8b"
    assert client.put("/api/v1/voice/options", json={"voice_llm": "dolphin3:8b"}).status_code == 422   # no tools
    assert client.put("/api/v1/voice/options", json={"voice_llm": "llama9:400b"}).status_code == 422   # not installed


def test_socket_refuses_when_switched_off(client):
    with client.websocket_connect("/api/v1/voice/ws") as ws:
        assert "switched off" in ws.receive_json()["message"]


@pytest.mark.skipif(not HAVE_SPEECH, reason="speech models not installed")
def test_socket_text_turn_speaks_and_saves_a_case(client, monkeypatch):
    from app import api
    from app.voice.session import VoiceSession
    _fake_models(monkeypatch)
    api.STATE["voice"]._ready_cache = None
    client.put("/api/v1/settings/voice_assistant", json={"on": True})
    llm = ScriptedLLM({"Chemistry": {"intent": "space_request", "request_type": "new_office", "department": "Chemistry"}})
    monkeypatch.setattr(api.STATE["voice"], "build_session", lambda save: VoiceSession(
        api.STATE["voice"].conversation(), llm, lambda a: "", lambda t: b"\x00\x00" * 240, engines.SpeechDetector(), save))
    with client.websocket_connect("/api/v1/voice/ws") as ws:
        ws.send_json({"type": "text", "text": "An office for Chemistry."})
        seen, audio = [], 0
        while True:
            m = ws.receive()
            if m.get("bytes"):
                audio += 1
                continue
            j = __import__("json").loads(m["text"])
            seen.append(j["type"])
            if j["type"] == "reply":
                assert j["report"]["candidates"] and j["can_save"]
                break
        assert "transcript" in seen and "say" in seen
        ws.send_json({"type": "save"})
        while (j := ws.receive_json())["type"] != "saved":
            pass
    case = client.get(f"/api/v1/cases/{j['case_id']}").json()
    assert case["kind"] == "space" and case["input"]["source"] == "voice" and case["status"] == "ReadyForReview"
    client.put("/api/v1/settings/voice_assistant", json={"on": False})


# ------------------------------------------------------------------------------ listening and barge-in (real models)
@pytest.mark.skipif(not HAVE_SPEECH, reason="speech models not installed")
def test_speech_is_segmented_and_interrupts_the_assistant():
    from app.voice.session import VoiceSession
    speaker = engines.Speaker("af_heart")
    pcm24 = np.frombuffer(speaker("Find me an office close to Professor Bartel."), "<i2").astype(np.float32)
    import scipy.signal as ss
    speech16 = ss.resample_poly(pcm24, 2, 3).astype("<i2")
    silence = np.zeros(16000, "<i2")

    heard, events = [], []
    class Conv:
        def turn(self, *a, **k):
            return {"reply": "ok", "slots": {}}
    s = VoiceSession(Conv(), None, lambda audio: heard.append(len(audio)) or "find me an office", lambda t: b"",
                     engines.SpeechDetector(), lambda st: "")
    s._emit = events.append
    s._feed(np.concatenate([silence, speech16, silence]).tobytes())
    time.sleep(0.5)
    assert len(heard) == 1 and heard[0] > len(speech16) * 0.8                  # one utterance, not cut short
    s.client_playing = True                                                      # assistant is talking...
    s._feed(np.concatenate([speech16[:16000], silence]).tobytes())               # ...and James talks over it
    assert any(e.get("type") == "interrupt" for e in events if isinstance(e, dict))

    # tap to speak: mic_on starts fresh; tapping again (mic_off) sends speech that has not ended in silence yet
    import asyncio, json
    time.sleep(0.3)
    heard.clear()
    asyncio.run(s._control(json.dumps({"type": "mic_on"})))
    s._feed(np.concatenate([silence[:4000], speech16]).tobytes())               # still talking, no closing silence
    assert not heard
    asyncio.run(s._control(json.dumps({"type": "mic_off"})))
    time.sleep(0.5)
    assert len(heard) == 1
