"""One live voice conversation over a WebSocket.

Browser -> server
  binary         16-bit PCM mono at 16 kHz from the microphone (echo cancellation on in the browser)
  {"type": "text", "text": ...}   typed instead of spoken
  {"type": "played"}              the browser finished playing everything it was sent
  {"type": "stop"}                stop talking now
  {"type": "save"}                save the current result as a case

Server -> browser
  {"type": "status", "state": loading|listening|hearing|transcribing|thinking|searching|speaking}
  {"type": "transcript", "text": ...}           what was heard
  {"type": "say", "text": ..., "turn": n}        a sentence of the reply (shown at once when replies are text only)
  {"type": "speech", "text": ..., "turn": n, "audio": bool}   the words of the audio that follows, so the browser
                                                shows them as they are spoken (audio false: synthesis failed)
  binary  4-byte little-endian turn number + 16-bit PCM mono at 24 kHz for that piece
  {"type": "reply", ...}                        the full reply, the request so far and the agent's top candidates
  {"type": "interrupt", "turn": n}              stop playing everything from turns before n
  {"type": "saved", "case_id": ...} | {"type": "error", "message": ...}

Interruption (barge-in): while the assistant is thinking or speaking, speech detected on the microphone
cancels the reply (the LLM stream stops, queued speech is dropped) and the new utterance becomes the next turn.
No audio is stored; only the text of each turn is logged.
"""
from __future__ import annotations

import asyncio
import queue
import re
import struct
import threading
import uuid
from collections import deque

import numpy as np

from app.logging_setup import get_logger, kv, run_context

log = get_logger("voice.session")
# Frames are 32 ms. Silero VAD scores speech, not loudness, so most background noise never starts an utterance.
START_PROB, END_PROB = 0.5, 0.35
START_FRAMES, END_FRAMES = 3, 22          # ~100 ms of speech to start, ~700 ms of silence to end
PREROLL_FRAMES, MAX_FRAMES = 10, 940      # keep ~320 ms before speech starts; cap one utterance at ~30 s
MIN_SPEECH_FRAMES = 8                     # ignore blips shorter than ~250 ms
BARGE_FRAMES = 8                          # interrupt the assistant only after ~250 ms of actual speech (not a cough
                                          # or a door); the words are captured from the start either way
HOLD_SECONDS = 2.2                        # a sentence cut off mid-way ("So I'm looking for") waits this long for the rest
MAX_HOLDS = 2
DANGLING = {"a", "an", "the", "to", "of", "for", "in", "on", "at", "with", "from", "by", "about", "near", "into",
            "and", "or", "but", "so", "because", "if", "that", "which", "who", "where", "when",
            "is", "are", "was", "were", "be", "am", "will", "would", "can", "could", "should", "need", "needs", "want",
            "i", "we", "you", "he", "she", "they", "it", "my", "our", "your", "his", "her", "their", "this", "these",
            "those", "some", "any", "um", "uh", "er", "like", "looking"}
SPLIT_WORDS = 16                          # long sentences are spoken in two pieces, so speech starts sooner
NON_WORDS = {"uh", "um", "umm", "hmm", "hm", "mm", "mhm", "mm hmm", "uh huh", "ah", "eh", "er", "erm", "huh"}


def speech_pieces(sentence: str) -> list[str]:
    """Split a long sentence at a comma, semicolon or dash near its middle (at least 5 words each side).
    The first piece is synthesized and playing while the second is made, so long sentences start sooner."""
    words = sentence.split()
    if len(words) <= SPLIT_WORDS:
        return [sentence]
    cuts = [i for i, w in enumerate(words[:-1], 1) if w[-1] in ",;:" or words[i] in ("—", "-", "–")]
    cuts = [i for i in cuts if 5 <= i <= len(words) - 5]
    if not cuts:
        return [sentence]
    i = min(cuts, key=lambda c: abs(c - len(words) * 0.45))
    return [" ".join(words[:i]), " ".join(words[i:])]


def looks_unfinished(text: str) -> bool:
    """Speech-to-text ends a sentence with . ? or ! when it is complete. Without one, a last word that cannot end a
    sentence ("for", "is", "the", "um") or a trailing comma means the speaker paused mid-sentence."""
    t = text.strip()
    if not t or t[-1] in ".?!":
        return False
    words = re.findall(r"[a-z']+", t.lower())
    return t.endswith(",") or bool(words) and words[-1] in DANGLING


def is_noise(text: str) -> bool:
    """A transcript that holds no words: empty, punctuation only, or a hesitation sound. Real words, however
    short ("yes", "no", "Keck"), are never dropped."""
    t = " ".join(text.lower().replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").replace("-", " ").split())
    return not any(ch.isalpha() for ch in t) or t in NON_WORDS


class VoiceSession:
    def __init__(self, conversation, llm, transcriber, speaker, detector, save_case) -> None:
        self.conv, self.llm, self.stt, self.tts, self.vad = conversation, llm, transcriber, speaker, detector
        self.save_case = save_case
        self.thread_id = uuid.uuid4().hex[:8]
        self.turn = 0                         # current turn number; audio from older turns is discarded
        self.cancel = threading.Event()
        self.busy = False                     # a turn is being worked on
        self.client_playing = False           # the browser is still playing our audio
        self._pending = np.zeros(0, np.int16)
        self._preroll: deque = deque(maxlen=PREROLL_FRAMES)
        self._speech: list[np.ndarray] | None = None
        self._above = self._below = self._voiced = 0
        self._barge_pending = False           # speech began while the assistant was busy: interrupt once it is real
        self.speak = True                     # False while the mic is muted: replies come as text only
        self._held, self._holds, self._hold_token = "", 0, 0   # first half of a sentence, waiting for the rest
        self._tts_q: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------------------ transport
    async def run(self, ws) -> None:
        self.loop = asyncio.get_running_loop()
        self.out: asyncio.Queue = asyncio.Queue()
        sender = asyncio.create_task(self._send_loop(ws))
        threading.Thread(target=self._tts_worker, daemon=True, name=f"tts-{self.thread_id}").start()
        log.info("voice session open " + kv(id=self.thread_id))
        self._emit({"type": "status", "state": "listening"})
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    self._feed(msg["bytes"])
                elif msg.get("text"):
                    await self._control(msg["text"])
        finally:
            self.cancel.set()
            self._tts_q.put(None)
            sender.cancel()
            log.info("voice session closed " + kv(id=self.thread_id, turns=self.turn))

    async def _send_loop(self, ws) -> None:
        while True:
            item = await self.out.get()
            try:
                await (ws.send_bytes(item) if isinstance(item, bytes) else ws.send_json(item))
            except Exception:
                return

    def _emit(self, item) -> None:
        """Thread-safe send."""
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)

    async def _control(self, raw: str) -> None:
        import json
        m = json.loads(raw)
        kind = m.get("type")
        if kind == "text" and m.get("text", "").strip():
            self._drop_held()
            self._interrupt()
            self._emit({"type": "transcript", "text": m["text"].strip(), "typed": True})
            self._start_turn(text=m["text"].strip())
        elif kind == "played":
            self.client_playing = False
            if not self.busy:
                self._emit({"type": "status", "state": "listening"})
        elif kind == "stop":
            self._interrupt()
            self._emit({"type": "status", "state": "listening"})
        elif kind == "mic_mute":                     # typing only: answer in text, stop any speech now
            self.speak = False
            self._drop_held()
            self._interrupt_audio()
            self._speech, self._above, self._barge_pending = None, 0, False
        elif kind == "mic_on":                       # microphone (back) on: voice replies, fresh utterance
            self.speak = True
            self._drop_held()
            self._interrupt()
            self.vad.reset()
            self._pending, self._speech, self._above, self._below = np.zeros(0, np.int16), None, 0, 0
            self._preroll.clear()
            self._emit({"type": "status", "state": "listening"})
        elif kind == "mic_off":                      # tapped again: send what was said so far
            frames, self._speech, self._above = self._speech, None, 0
            if frames and len(frames) >= MIN_SPEECH_FRAMES:
                self._start_turn(audio=np.concatenate(frames))
            elif not self.busy:
                self._emit({"type": "status", "state": "idle"})
        elif kind == "save":
            try:
                state = self.conv.state(self.thread_id)
                cid = await self.loop.run_in_executor(None, self.save_case, state)
                self.conv.mark_saved(self.thread_id, cid, state)      # "save it" later will not save twice
                self._emit({"type": "saved", "case_id": cid, "via": "button"})
            except Exception as exc:
                self._emit({"type": "error", "message": str(exc)})

    # ------------------------------------------------------------------------------ listening
    def _feed(self, data: bytes) -> None:
        pcm = np.concatenate([self._pending, np.frombuffer(data, "<i2")])
        n = len(pcm) // self.vad.frame * self.vad.frame
        self._pending = pcm[n:]
        for frame in pcm[:n].reshape(-1, self.vad.frame):
            self._frame(frame)

    def _frame(self, frame: np.ndarray) -> None:
        f = frame.astype(np.float32) / 32768.0
        p = self.vad(f)
        if self._speech is None:
            self._preroll.append(f)
            self._above = self._above + 1 if p >= START_PROB else 0
            if self._above >= START_FRAMES:                  # speech started
                self._barge_pending = self.busy or self.client_playing or not self._tts_q.empty()
                self._speech, self._below, self._voiced = list(self._preroll), 0, self._above
                if not self._barge_pending:
                    self._emit({"type": "status", "state": "hearing"})
            return
        self._speech.append(f)
        self._below = self._below + 1 if p < END_PROB else 0
        if p >= END_PROB:
            self._voiced += 1
        if self._barge_pending and self._voiced >= BARGE_FRAMES:   # real speech over the assistant: barge-in
            self._barge_pending = False
            self._interrupt()
            self._emit({"type": "status", "state": "hearing"})
        if self._below >= END_FRAMES or len(self._speech) >= MAX_FRAMES:
            frames, self._speech, self._above = self._speech, None, 0
            self._preroll.clear()
            if len(frames) - self._below >= MIN_SPEECH_FRAMES:
                if self._barge_pending:                       # short but real: stop the assistant first
                    self._interrupt()
                self._barge_pending = False
                self._start_turn(audio=np.concatenate(frames))
            else:
                pending, self._barge_pending = self._barge_pending, False
                if not pending:                               # a blip: the assistant carries on untouched
                    self._emit({"type": "status", "state": "listening"})

    # ------------------------------------------------------------------------------ half sentences
    def _drop_held(self) -> None:
        self._held, self._holds = "", 0
        self._hold_token += 1

    def _hold(self, text: str) -> None:
        """Keep the first half of a sentence and wait for the rest; if none comes, answer what was said."""
        self._held, self._holds = text, self._holds + 1
        self._hold_token += 1
        token = self._hold_token
        log.info("waiting for the rest of the sentence " + kv(id=self.thread_id, heard=repr(text[-60:])))
        threading.Timer(HOLD_SECONDS, self._release_held, args=(token,)).start()

    def _release_held(self, token: int) -> None:
        if token != self._hold_token or not self._held:
            return                                          # the rest arrived, or the turn moved on
        if self._speech is not None:                        # still talking: look again shortly
            threading.Timer(0.8, self._release_held, args=(token,)).start()
            return
        text, self._held, self._holds = self._held, "", 0
        self._hold_token += 1
        self._start_turn(text=text)

    # ------------------------------------------------------------------------------ turns
    def _interrupt_audio(self) -> None:
        """Stop speech only (the reply text still arrives)."""
        while not self._tts_q.empty():
            try:
                self._tts_q.get_nowait()
            except queue.Empty:
                break
        self.client_playing = False

    def _interrupt(self) -> None:
        if self.busy or self.client_playing or not self._tts_q.empty():
            log.info("interrupted " + kv(id=self.thread_id, turn=self.turn))
        self.cancel.set()
        self.turn += 1
        self.busy = self.client_playing = False
        self._emit({"type": "interrupt", "turn": self.turn})

    def _start_turn(self, audio: np.ndarray | None = None, text: str | None = None) -> None:
        self.cancel = threading.Event()
        self.turn += 1
        self.busy = True
        turn, cancel = self.turn, self.cancel
        threading.Thread(target=self._run_turn, args=(turn, cancel, audio, text), daemon=True).start()

    def _run_turn(self, turn: int, cancel: threading.Event, audio, text) -> None:
        with run_context():
            try:
                if text is None:
                    self._emit({"type": "status", "state": "transcribing"})
                    text = self.stt(audio)
                    if cancel.is_set():
                        return
                    if not text or is_noise(text):                # a sound, not a request: say nothing
                        log.info("ignored noise " + kv(id=self.thread_id, heard=repr(text)))
                        if self._held:                            # still waiting for the rest of a sentence
                            self._release_held(self._hold_token)
                        else:
                            self._emit({"type": "status", "state": "listening"})
                        return
                    self._emit({"type": "transcript", "text": text})
                    if self._held:                                # the rest of a sentence cut off by a pause
                        text = f"{self._held} {text}"
                        self._held = ""
                        self._hold_token += 1
                    if looks_unfinished(text) and self._holds < MAX_HOLDS:
                        self._hold(text)
                        self._emit({"type": "status", "state": "listening"})
                        return
                    self._holds = 0
                log.info("turn " + kv(id=self.thread_id, n=turn, heard=repr(text[:120])))
                self._emit({"type": "status", "state": "thinking"})

                said = []

                def on_event(e: dict) -> None:
                    if cancel.is_set():
                        return
                    if "sentence" in e:
                        said.append(e["sentence"])
                        self._emit({"type": "say", "text": e["sentence"], "turn": turn})
                        if self.speak:                       # muted: text only
                            self._tts_q.put((turn, e["sentence"]))
                    elif "step" in e:                       # the agent's progress, shown live
                        self._emit({"type": "step", "text": e["step"], "tool": e.get("tool"), "turn": turn})
                    elif "saved" in e:                      # saved by voice ("save it as a case")
                        self._emit({"type": "saved", "case_id": e["saved"], "via": "voice"})
                    elif e.get("status"):
                        self._emit({"type": "status", "state": e["status"]})

                state = self.conv.turn(self.thread_id, text, self.llm, cancel, on_event, save_case=self.save_case,
                                       spoken=audio is not None)
                if cancel.is_set():
                    return
                report = state.get("report") if state.get("intent") not in ("small_talk", "stop") else None
                self._emit({"type": "reply", "turn": turn, "text": state.get("reply", ""),
                            "slots": {k: v for k, v in (state.get("slots") or {}).items() if k != "near_title"},
                            "question": state.get("question"), "report": report,
                            "can_save": bool(state.get("full_report"))})
                if not self.speak or not said:              # nothing will play (text only, or nothing to say): done
                    self._emit({"type": "status", "state": "listening"})
            except Exception as exc:
                log.error(f"voice turn failed error={exc!r}")
                self._emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                self._emit({"type": "status", "state": "listening"})
            finally:
                if turn == self.turn:
                    self.busy = False

    def _tts_worker(self) -> None:
        while True:
            item = self._tts_q.get()
            if item is None:
                return
            turn, sentence = item
            for piece in speech_pieces(sentence):
                if turn != self.turn or not self.speak:
                    break                                   # interrupted or muted: drop speech
                try:
                    pcm = self.tts(piece)
                except Exception as exc:
                    log.error(f"speech synthesis failed error={exc!r}")
                    self._emit({"type": "speech", "text": piece, "turn": turn, "audio": False})
                    continue
                if turn == self.turn:
                    self.client_playing = True
                    self._emit({"type": "status", "state": "speaking"})
                    self._emit({"type": "speech", "text": piece, "turn": turn, "audio": True})
                    self._emit(struct.pack("<I", turn) + pcm)
