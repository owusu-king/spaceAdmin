"""Client for the local Ollama server (FAI_OLLAMA_URL, default http://127.0.0.1:11434).

Only models already installed in Ollama are listed. A model qualifies for the voice assistant
when Ollama reports the 'tools' capability: those are instruction-tuned models that follow
schemas reliably. Thinking is switched off so replies come fast enough for speech.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Iterator

import httpx

from app.config import ENV_FILE  # noqa: F401  importing config loads .env
from app.logging_setup import get_logger

log = get_logger("voice.llm")


def ollama_url() -> str:
    return (os.environ.get("FAI_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")


def installed_models(timeout: float = 3.0) -> tuple[list[dict], str | None]:
    """([{name, size_gb, tools, family}], error). Never raises."""
    try:
        with httpx.Client(base_url=ollama_url(), timeout=timeout) as c:
            tags = c.get("/api/tags").json().get("models", [])
            out = []
            for m in tags:
                show = c.post("/api/show", json={"model": m["name"]}).json()
                caps = show.get("capabilities") or []
                out.append({"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1), "tools": "tools" in caps,
                            "family": (m.get("details") or {}).get("family"),
                            "parameters": (m.get("details") or {}).get("parameter_size")})
        return sorted(out, key=lambda x: (not x["tools"], x["name"])), None
    except Exception as exc:  # Ollama not running or unreachable
        return [], f"Ollama is not reachable at {ollama_url()} ({type(exc).__name__})"


class OllamaChat:
    """The two calls the conversation needs: a JSON answer that follows a schema, and a streamed reply."""

    def __init__(self, model: str, timeout: float = 120.0) -> None:
        self.model, self.timeout = model, timeout

    # Every request must use the same load options (context size): a different num_ctx makes Ollama reload the model.
    OPTIONS = {"temperature": 0.2, "num_ctx": 8192}
    KEEP_ALIVE = "2h"          # stay loaded between conversations; Ollama frees the memory after two idle hours

    def _body(self, messages: list[dict], options: dict | None = None, **extra) -> dict:
        return {"model": self.model, "messages": messages, "think": False, "keep_alive": self.KEEP_ALIVE,
                "options": {**self.OPTIONS, **(options or {})}, **extra}

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        with httpx.Client(base_url=ollama_url(), timeout=self.timeout) as c:
            r = c.post("/api/chat", json=self._body(messages, stream=False, format=schema))
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"] or "{}")

    def chat_stream(self, messages: list[dict], cancel: threading.Event | None = None) -> Iterator[str]:
        with httpx.Client(base_url=ollama_url(), timeout=self.timeout) as c:
            with c.stream("POST", "/api/chat", json=self._body(messages, stream=True)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if cancel is not None and cancel.is_set():
                        log.info("reply cancelled (interrupted)")
                        return
                    if line:
                        chunk = json.loads(line)
                        piece = (chunk.get("message") or {}).get("content", "")
                        if piece:
                            yield piece
                        if chunk.get("done"):
                            return

    def warm(self) -> None:
        """Load the model into memory so the first spoken turn is not slow."""
        try:
            with httpx.Client(base_url=ollama_url(), timeout=self.timeout) as c:
                c.post("/api/chat", json=self._body([{"role": "user", "content": "hi"}], options={"num_predict": 1},
                                                     stream=False))
        except Exception as exc:
            log.warning(f"model warm-up failed model={self.model} error={exc!r}")
