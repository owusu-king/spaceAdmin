"""Structured console logging with execution identifiers.

Format:  2026-11-22 20:41:07.123 | INFO  | run=3f9a2c | agents.space | message  key=value ...

`run_context(...)` scopes a short run id to a workflow so every log line it emits
can be correlated, and `timed(...)` records execution duration.
"""
from __future__ import annotations

import contextvars
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")
_configured = False


class _RunFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.addFilter(_RunFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-5s | run=%(run_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger("fai")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"fai.{name}")


def kv(**fields) -> str:
    """Render key=value pairs for a log line."""
    return " ".join(f"{k}={v}" for k, v in fields.items())


def current_run_id() -> str:
    return _run_id.get()


@contextmanager
def run_context(run_id: str | None = None) -> Iterator[str]:
    rid = run_id or uuid.uuid4().hex[:6]
    token = _run_id.set(rid)
    try:
        yield rid
    finally:
        _run_id.reset(token)


@contextmanager
def timed(logger: logging.Logger, label: str, **fields) -> Iterator[dict]:
    info: dict = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        logger.info(f"{label} {kv(**fields, duration_ms=info['duration_ms'])}")
