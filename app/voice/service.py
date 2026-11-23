"""Voice assistant wiring: which installed models are chosen, whether everything is ready, and
building a session. The choices are stored with the Settings switches (app/services/settings.py)."""
from __future__ import annotations

import threading
import time

from app.logging_setup import get_logger
from app.voice import engines
from app.voice.llm import OllamaChat, installed_models

log = get_logger("voice.service")
PREFERRED_LLM, PREFERRED_VOICE = "qwen3:8b", "af_heart"
CHOICES = ("voice_llm", "voice_stt", "voice_voice")


class VoiceService:
    def __init__(self, coord, settings) -> None:
        self.coord, self.settings = coord, settings
        self._conv = None
        self._lock = threading.Lock()
        self._ready_cache: tuple[float, tuple[bool, str]] | None = None

    # ------------------------------------------------------------------------------ options
    def options(self) -> dict:
        llms, llm_error = installed_models()
        stt, voices = engines.stt_models(), engines.tts_voices()
        sel = self.selection(llms, stt, voices)
        problems = []
        if llm_error:
            problems.append(llm_error)
        elif not any(m["tools"] for m in llms):
            problems.append("No installed Ollama model supports tool calling. Install one, e.g. `ollama pull qwen3:8b`.")
        if not stt:
            problems.append("No speech-to-text model in models/voice/stt (see README, voice setup).")
        if not voices:
            problems.append("Kokoro voice files missing in models/voice/tts (see README, voice setup).")
        if not engines.vad_ready():
            problems.append("Speech detector missing: models/voice/vad/silero/silero_vad.onnx.")
        return {"llms": llms, "stt": stt, "voices": voices, "selected": sel, "problems": problems,
                "ready": not problems and all(sel.values())}

    def selection(self, llms=None, stt=None, voices=None) -> dict:
        """Stored choice if still installed, else the recommended default, else the first suitable one."""
        llms = installed_models()[0] if llms is None else llms
        stt = engines.stt_models() if stt is None else stt
        voices = engines.tts_voices() if voices is None else voices
        tool_llms = [m["name"] for m in llms if m["tools"]]
        def pick(key, available, preferred):
            stored = self.settings.get_choice(key)
            if stored in available:
                return stored
            if preferred in available:
                return preferred
            return available[0] if available else None
        return {"voice_llm": pick("voice_llm", tool_llms, PREFERRED_LLM),
                "voice_stt": pick("voice_stt", [m["id"] for m in stt], "parakeet-tdt-0.6b-v2"),
                "voice_voice": pick("voice_voice", [v["id"] for v in voices], PREFERRED_VOICE)}

    def readiness(self) -> tuple[bool, str]:
        """Cached for a few seconds: Settings asks often and each check queries Ollama."""
        now = time.monotonic()
        if self._ready_cache and now - self._ready_cache[0] < 10:
            return self._ready_cache[1]
        o = self.options()
        res = (True, "") if o["ready"] else (False, o["problems"][0] if o["problems"] else "Choose the models first")
        self._ready_cache = (now, res)
        return res

    def choose(self, key: str, value: str, actor: str) -> None:
        o = self.options()
        allowed = {"voice_llm": [m["name"] for m in o["llms"] if m["tools"]],
                   "voice_stt": [m["id"] for m in o["stt"]], "voice_voice": [v["id"] for v in o["voices"]]}[key]
        if value not in allowed:
            raise ValueError(f"'{value}' is not an installed option for {key} (installed: {', '.join(allowed) or 'none'})")
        self.settings.set_choice(key, value, actor)
        self._ready_cache = None

    # ------------------------------------------------------------------------------ sessions
    def conversation(self):
        with self._lock:
            if self._conv is None:
                from app.voice.conversation import SpaceConversation
                self._conv = SpaceConversation(self.coord)
            return self._conv

    def prewarm(self) -> None:
        """Load the chosen models in the background so the first conversation starts quickly."""
        def run():
            try:
                sel = self.selection()
                engines.Transcriber(sel["voice_stt"]); engines.Speaker(sel["voice_voice"])
                llm = OllamaChat(sel["voice_llm"])
                llm.warm()
                self.conversation().warm(llm)            # first schema-bound call is slow; do it now
                log.info("voice models preloaded")
            except Exception as exc:
                log.warning(f"voice preload skipped error={exc!r}")
        threading.Thread(target=run, daemon=True, name="voice-prewarm").start()

    def build_session(self, save_case):
        """Load the chosen engines (first time takes a few seconds) and return a VoiceSession."""
        from app.voice.session import VoiceSession
        sel = self.selection()
        llm = OllamaChat(sel["voice_llm"])
        llm.warm()
        try:
            self.conversation().warm(llm)
        except Exception as exc:
            log.warning(f"conversation warm-up failed error={exc!r}")
        return VoiceSession(self.conversation(), llm, engines.Transcriber(sel["voice_stt"]),
                            engines.Speaker(sel["voice_voice"]), engines.SpeechDetector(), save_case)
