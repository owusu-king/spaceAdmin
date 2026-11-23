"""Agent coordination: builds the system, routes work and relays agent-to-agent messages.

The agents never call each other's tools. The Outage Agent may *ask* for temporary space
through `request_space_support`; the coordinator validates the request, runs the Space
Administrator Agent and returns a structured reply. Every message is logged.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

from app.agents.base import AgentMessage
from app.agents.outage_agent import OutageAgent
from app.agents.schemas import OutageRequest, SpaceRequest
from app.agents.space_agent import SpaceAdministratorAgent
from app.data.buildings import BuildingResolver
from app.data.loader import Dataset, load_dataset
from app.data.notices import Corpus, load_corpus
from app.logging_setup import get_logger, kv, run_context, timed
from app.ml.inference import SpacePredictor
from app.services.events import EventStore

log = get_logger("coordinator")
ALLOWED_INTENTS = {("outage", "space_admin", "temporary_space_request")}


@dataclass
class Coordinator:
    ds: Dataset
    predictor: SpacePredictor
    corpus: Corpus
    space: SpaceAdministratorAgent
    outage: OutageAgent
    messages: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.outage.support_channel = self._relay

    # --------------------------------------------------------------------------- relay
    def _relay(self, msg: AgentMessage) -> AgentMessage:
        self.messages.append(vars(msg))
        log.info("message " + kv(sender=msg.sender, recipient=msg.recipient, intent=msg.intent))
        if (msg.sender, msg.recipient, msg.intent) not in ALLOWED_INTENTS:
            reply = AgentMessage(msg.recipient, msg.sender, "rejected", {"reason": "intent not permitted"})
        else:
            p = msg.payload
            req = SpaceRequest(request_type="temporary_relocation", department=p.get("department"),
                               function=p.get("function", "Workstation"), exclude_buildings=p.get("exclude_buildings", []),
                               notes=p.get("reason", ""))
            saved_trace = self.space.trace
            report = self.space.run(req)
            self.space.trace = saved_trace + report["trace"]
            cands = [{k: c[k] for k in ("space_id", "building", "area", "space_class", "status", "suitability",
                                        "evidence_coverage", "allocation")} for c in report["candidates"][:5]]
            reply = AgentMessage("space_admin", msg.sender, "temporary_space_options",
                                 {"candidates": cands, "summary": report["summary"],
                                  "verification_questions": report["verification_questions"][:3],
                                  "note": "Temporary options are candidates for review, not assignments."})
        self.messages.append(vars(reply))
        log.info("message " + kv(sender=reply.sender, recipient=reply.recipient, intent=reply.intent,
                                 candidates=len(reply.payload.get("candidates", []))))
        return reply

    # --------------------------------------------------------------------------- entry points
    def run_space(self, req: SpaceRequest, use_model: bool = True, observer=None) -> dict:
        """observer(tool, result) is called after each agent tool call (live progress, e.g. for the voice assistant)."""
        with self._lock, run_context() as rid, timed(log, "space workflow", run=rid):
            self.space.observer = observer
            try:
                out = self.space.run(req, use_model=use_model)
            finally:
                self.space.observer = None
            out["run_id"] = rid
            return out

    def run_outage(self, req: OutageRequest, cooperate: bool = False) -> dict:
        with self._lock, run_context() as rid, timed(log, "outage workflow", run=rid):
            start = len(self.messages)
            out = self.outage.run(req, cooperate=cooperate)
            out["run_id"] = rid
            out["messages"] = self.messages[start:]
            return out

    def route(self, kind: str, payload: dict) -> dict:
        """Route a new case to the responsible agent."""
        if kind == "space":
            return self.run_space(SpaceRequest(**payload))
        if kind == "outage":
            return self.run_outage(OutageRequest(**payload))
        raise ValueError(f"unknown case kind '{kind}'")


def build_system(as_of: datetime | None = None, events: EventStore | None = None) -> Coordinator:
    """Load data, model and correspondence; initialize both agents.

    events: persistent EventStore (the API passes a file-backed one); default is an in-memory
    store seeded from the correspondence so notebooks and tests are self-contained."""
    with timed(log, "system initialized"):
        ds = load_dataset()
        predictor = SpacePredictor(ds.spaces)
        corpus = load_corpus()
        resolver = BuildingResolver(list(ds.spaces["building"].dropna().unique()))
        space = SpaceAdministratorAgent(ds, predictor, resolver)
        if events is None:
            events = EventStore()
        events.seed_from_corpus(corpus, resolver)
        outage = OutageAgent(ds, predictor, resolver, corpus, as_of=as_of, events=events)
    return Coordinator(ds, predictor, corpus, space, outage)
