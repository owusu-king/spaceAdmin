"""Shared agent framework: evidence, tool results, permissions and bounded execution.

An agent is a set of objectives plus a registry of tools it is *permitted* to call.
Tools return the architecture's tool-response contract (§13.4). Every call is
permission-checked, counted against an iteration limit, timed, logged and kept in the
agent's trace so the UI and notebooks can show exactly what the agent did.

Actions that would change institutional state (sending email, writing TRIRIGA records,
assigning space) are not tools at all: they are LOCKED_ACTIONS and always refused.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

from app.config import AGENTS
from app.logging_setup import current_run_id, get_logger, kv

EVIDENCE_CLASSES = ("confirmed", "recorded", "inferred", "historical", "conflicting", "unknown")
LOCKED_ACTIONS = {
    "send_email": "Agents cannot send email.",
    "write_tririga": "Agents cannot write to TRIRIGA.",
    "assign_space": "Agents cannot assign space.",
    "approve_outage": "Agents cannot approve outages.",
}


class PermissionDenied(PermissionError):
    pass


class ExecutionLimitReached(RuntimeError):
    pass


@dataclass
class Evidence:
    claim: str
    classification: str                  # one of EVIDENCE_CLASSES
    source: str                          # space_records | people_records | correspondence | model | config | rule
    reference: str = ""                  # row key, SR id, message date, model name ...
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        if self.classification not in EVIDENCE_CLASSES:
            raise ValueError(f"invalid evidence classification {self.classification}")


@dataclass
class ToolResult:
    tool_name: str
    status: str                          # success | partial | not_found | ambiguous | unauthorized | error
    data: Any = None
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("data")        # traces keep metadata only; data goes into the agent's report
        return d


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    intent: str
    payload: dict
    run_id: str = field(default_factory=current_run_id)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class Agent:
    """Base class. Subclasses declare `name`, `objectives`, `permissions` and register tools."""

    name: str = "agent"
    objectives: list[str] = []
    permissions: set[str] = set()          # tool names this agent may call
    data_scopes: list[str] = []            # human-readable list of data it may read

    def __init__(self, max_tool_calls: int = AGENTS.max_tool_calls) -> None:
        self.log = get_logger(f"agents.{self.name}")
        self._tools: dict[str, Callable[..., ToolResult]] = {}
        self.trace: list[dict] = []
        self.observer: Callable[[str, ToolResult], None] | None = None   # called after every tool call
        self.max_tool_calls = max_tool_calls
        self._register_tools()
        unknown = self.permissions - set(self._tools)
        if unknown:
            raise ValueError(f"{self.name}: permissions reference unknown tools {unknown}")
        self.log.info("agent initialized " + kv(tools=len(self._tools), permitted=len(self.permissions)))

    # -- registry --------------------------------------------------------------------------
    def _register_tools(self) -> None:
        for attr in dir(self):
            fn = getattr(self, attr)
            if callable(fn) and getattr(fn, "_is_tool", False):
                self._tools[fn._tool_name] = fn

    @property
    def tools(self) -> dict[str, str]:
        return {n: (f.__doc__ or "").strip().splitlines()[0] for n, f in sorted(self._tools.items())}

    def reset(self) -> None:
        self.trace = []

    # -- execution -------------------------------------------------------------------------
    def call(self, tool: str, **kwargs) -> ToolResult:
        if tool in LOCKED_ACTIONS:
            self.log.warning(f"locked action refused tool={tool}")
            result = ToolResult(tool, "unauthorized", warnings=[LOCKED_ACTIONS[tool]])
            self.trace.append({**result.to_dict(), "step": len(self.trace) + 1})
            return result
        if tool not in self.permissions:
            self.log.warning(f"permission denied tool={tool}")
            raise PermissionDenied(f"{self.name} is not permitted to call {tool}")
        if len(self.trace) >= self.max_tool_calls:
            raise ExecutionLimitReached(f"{self.name} reached the limit of {self.max_tool_calls} tool calls")
        start = time.perf_counter()
        try:
            result = self._tools[tool](**kwargs)
        except Exception as exc:  # a failing tool must not crash the workflow
            self.log.error(f"tool error tool={tool} error={exc!r}")
            result = ToolResult(tool, "error", warnings=[f"{type(exc).__name__}: {exc}"])
        result.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        self.trace.append({**result.to_dict(), "step": len(self.trace) + 1})
        if self.observer:                     # live progress (e.g. the voice assistant shows each step as it happens)
            try:
                self.observer(tool, result)
            except Exception:                 # an observer must never break the workflow
                pass
        self.log.info("tool " + kv(name=tool, status=result.status, evidence=len(result.evidence),
                                   duration_ms=result.duration_ms))
        return result

    def describe(self) -> dict:
        return {"name": self.name, "objectives": self.objectives, "tools": self.tools,
                "permissions": sorted(self.permissions), "data_scopes": self.data_scopes,
                "locked_actions": LOCKED_ACTIONS}


def tool(name: str):
    """Decorator that marks a method as an agent tool."""
    def wrap(fn):
        fn._is_tool, fn._tool_name = True, name
        return fn
    return wrap
