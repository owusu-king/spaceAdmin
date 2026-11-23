"""Local speech engines: what is installed, and loading it on first use.

Models live in models/voice/ (git-ignored):
  stt/<name>/   an onnx-asr model folder (config.json + ONNX files), e.g. parakeet-tdt-0.6b-v2
  tts/          Kokoro: kokoro-v1.0.onnx + voices-v1.0.bin
  vad/silero/   silero_vad.onnx

Only models found here are offered in Settings, so the user can only pick what is installed.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache

import numpy as np

from app.config import MODEL_DIR
from app.logging_setup import get_logger, timed

log = get_logger("voice.engines")
VOICE_DIR = MODEL_DIR / "voice"
STT_DIR, TTS_DIR, VAD_DIR = VOICE_DIR / "stt", VOICE_DIR / "tts", VOICE_DIR / "vad" / "silero"
KOKORO_MODEL, KOKORO_VOICES = TTS_DIR / "kokoro-v1.0.onnx", TTS_DIR / "voices-v1.0.bin"
STT_RATE = 16000                      # Parakeet and Silero expect 16 kHz mono
_ACCENT = {"a": "American", "b": "British"}
_GENDER = {"f": "female", "m": "male"}
_lock = threading.Lock()


# ------------------------------------------------------------------------------ catalog
def stt_models() -> list[dict]:
    """Installed speech-to-text model folders."""
    out = []
    for d in sorted(STT_DIR.glob("*")) if STT_DIR.exists() else []:
        cfg = d / "config.json"
        if d.is_dir() and cfg.exists() and any(d.glob("*.onnx")):
            kind = json.loads(cfg.read_text(encoding="utf-8")).get("model_type", "")
            out.append({"id": d.name, "label": d.name.replace("-", " "), "model_type": kind,
                        "int8": any(d.glob("*.int8.onnx"))})
    return out


def tts_voices() -> list[dict]:
    """English Kokoro voices, if Kokoro is installed."""
    if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists()):
        return []
    out = []
    for v in sorted(_kokoro().get_voices()):
        prefix, _, name = v.partition("_")
        if len(prefix) == 2 and prefix[0] in _ACCENT and prefix[1] in _GENDER:
            out.append({"id": v, "label": f"{name.title()} ({_ACCENT[prefix[0]]}, {_GENDER[prefix[1]]})",
                        "lang": "en-us" if prefix[0] == "a" else "en-gb"})
    return out


def vad_ready() -> bool:
    return (VAD_DIR / "silero_vad.onnx").exists()


# ------------------------------------------------------------------------------ loaders
@lru_cache(maxsize=1)
def _kokoro():
    from kokoro_onnx import Kokoro
    with timed(log, "tts model loaded", model="kokoro-v1.0"):
        return Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))


@lru_cache(maxsize=2)
def _stt(model_id: str):
    import onnx_asr
    m = next((x for x in stt_models() if x["id"] == model_id), None)
    if not m:
        raise FileNotFoundError(f"Speech model '{model_id}' is not installed in {STT_DIR}")
    with timed(log, "stt model loaded", model=model_id):
        return onnx_asr.load_model(m["model_type"], STT_DIR / model_id, quantization="int8" if m["int8"] else None)


class Transcriber:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        with _lock:
            self._model = _stt(model_id)

    def __call__(self, audio: np.ndarray) -> str:
        """audio: float32 mono at 16 kHz."""
        if len(audio) < STT_RATE // 4:
            return ""
        return str(self._model.recognize(audio.astype(np.float32), sample_rate=STT_RATE)).strip()


class Speaker:
    rate = 24000                      # Kokoro output rate

    def __init__(self, voice: str) -> None:
        self.voice = voice
        self.lang = "en-gb" if voice.startswith("b") else "en-us"
        with _lock:
            self._k = _kokoro()

    def __call__(self, text: str) -> bytes:
        """Synthesize one sentence; returns 16-bit PCM mono at 24 kHz."""
        audio, _ = self._k.create(text, voice=self.voice, speed=1.05, lang=self.lang)
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


class SpeechDetector:
    """Streaming Silero VAD (v5): feed 512-sample frames at 16 kHz, get speech probability per frame."""
    frame = 512
    _context = 64

    def __init__(self) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self._s = ort.InferenceSession(str(VAD_DIR / "silero_vad.onnx"), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._ctx = np.zeros((1, self._context), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._ctx, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._s.run(None, {"input": x, "state": self._state, "sr": np.array(STT_RATE, np.int64)})
        self._ctx = x[:, -self._context:]
        return float(out[0][0])
