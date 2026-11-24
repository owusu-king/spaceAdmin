"""FastAPI application: REST API under /api/v1 plus the static web client.

Run:  python -m uvicorn app.api:app --port 8000      then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from app.agents.base import LOCKED_ACTIONS
from app.agents.coordinator import Coordinator, build_system
from app.agents.schemas import OutageRequest, SpaceRequest
from app.config import BUILDINGS, DATA_DIR, EVENTS_DB_PATH, MODEL_DIR, NATURAL_SCIENCES_DEPARTMENTS, WEB_DIR
from app.data.buildings import floor_plan_files
from app.data.notices import rtf_file_text
from app.integrations.email import EmailIntake, build_source
from app.integrations.mailer import SmtpMailer
from app.data.records import RecordsPipeline, UploadSource, build_records_source, records_config
from app.integrations.msgraph import GraphAuth, GraphError
from app.integrations.triage import ModelClassifier
from app.logging_setup import get_logger, run_context
from app.scenarios import message_text, outage_sources, resolve_payload, scenarios
from app.services.cases import CaseStore, dumps
from app.services.events import EventStore
from app.services.outbox import Outbox, SendRefused
from app.services.scheduling import forecast_backtest, recurring_maintenance_forecast
from app.services.settings import AUTO_OFF_MINUTES, SettingsStore, unlock_allowed
from app.voice.llm import OllamaChat
from app.voice.service import CHOICES, VoiceService

REVIEWER_NAME = "James Delgado (Facilities Administrator)"

log = get_logger("api")
STATE: dict = {}


def coord() -> Coordinator:
    return STATE["coord"]


def store() -> CaseStore:
    return STATE["store"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _startup()
    # Live intake: with Microsoft 365 the mailbox is checked every minute unless FAI_EMAIL_POLL_SECONDS says otherwise.
    default = "60" if os.environ.get("FAI_EMAIL_SOURCE", "").lower() == "graph" else "0"
    interval = int(os.environ.get("FAI_EMAIL_POLL_SECONDS") or default)
    STATE["poll_seconds"] = interval
    task = asyncio.create_task(_poll_forever(interval)) if interval > 0 else None
    rtask = asyncio.create_task(_records_scheduler())
    yield
    for t in (task, rtask):
        if t:
            t.cancel()


async def _poll_forever(interval: int) -> None:
    """Optional background email polling (FAI_EMAIL_POLL_SECONDS > 0)."""
    while True:
        await asyncio.sleep(interval)
        try:
            await run_in_threadpool(STATE["intake"].poll, 25)
        except Exception as exc:  # keep polling; the error is logged
            log.error(f"background email poll failed error={exc!r}")


async def _records_scheduler() -> None:
    """Checks the records source on the schedule chosen in Settings (or FAI_RECORDS_REFRESH_MINUTES). The setting is
    read every minute, so a change takes effect without a restart. Uploads are never checked automatically."""
    last = time.monotonic()
    while True:
        await asyncio.sleep(60)
        cfg = records_config(STATE.get("settings"))
        if cfg["source"] == "upload" or cfg["refresh_minutes"] <= 0 or time.monotonic() - last < cfg["refresh_minutes"] * 60:
            continue
        last = time.monotonic()
        try:
            await run_in_threadpool(STATE["records"].refresh)
        except Exception as exc:
            log.error(f"scheduled records refresh failed error={exc!r}")


_SWAP = threading.Lock()


def _use_dataset(ds) -> dict:
    """Switch the running app to new records: rebuild the agents around them and swap atomically. Cases, events,
    settings and the email inbox are untouched; open voice conversations finish on the previous data."""
    with _SWAP, run_context("records"):
        new = build_system(events=STATE["events"], ds=ds)
        STATE["coord"] = new
        STATE["intake"].coord = new
        STATE["voice"].coord, STATE["voice"]._conv = new, None
        store().audit("system", "records_updated", detail=f"{len(ds.spaces)} spaces · {len(ds.people)} people")
        log.info("app switched to new records")
        return {"without_model_history": new.predictor.without_oof}


def _startup() -> None:
    with run_context("boot"):
        log.info("application starting")
        STATE["events"] = EventStore(EVENTS_DB_PATH)
        STATE["coord"] = build_system(events=STATE["events"])
        STATE["store"] = CaseStore()
        STATE["plans"] = {(p["code"], p["floor"]): p for p in floor_plan_files()}
        STATE["graph"] = GraphAuth()
        STATE["intake"] = EmailIntake(build_source(coord().corpus, STATE["graph"]), coord(), store(),
                                      classifier_factory=triage_classifier)
        STATE["mailer"] = SmtpMailer()
        attach_switches(store())
        STATE["records"] = RecordsPipeline(build_records_source(STATE["graph"], STATE["settings"]), lambda: coord().ds, _use_dataset)
        if STATE["settings"].is_on("voice_assistant"):
            STATE["voice"].prewarm()
        log.info(f"email intake ready mode={STATE['intake'].source.mode}")
        store().audit("system", "application_started", detail="data, model and agents loaded")
        log.info("application ready url=http://localhost:8000")


_TRIAGE = {"at": -1e9, "clf": None}


def triage_classifier():
    """The local model for email triage (the language model chosen in Settings); None when Ollama has no
    suitable model, in which case triage falls back to the keyword rule. Re-checked at most once a minute."""
    now = time.monotonic()
    if now - _TRIAGE["at"] > 60:
        model = STATE["voice"].selection()["voice_llm"] if "voice" in STATE else None
        _TRIAGE.update(at=now, clf=ModelClassifier(OllamaChat(model)) if model else None)
    return _TRIAGE["clf"]


def attach_switches(case_store: CaseStore) -> None:
    """Switch state and the outbox live in the given case database (tests pass a temporary one)."""
    STATE["settings"] = SettingsStore(case_store, checks={"send_email": lambda: STATE["mailer"].readiness(),
                                                          "voice_assistant": lambda: STATE["voice"].readiness()})
    STATE["outbox"] = Outbox(case_store)
    STATE["voice"] = VoiceService(STATE["coord"], STATE["settings"])


app = FastAPI(title="Campus Facilities AI", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def no_stale_assets(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _json(obj) -> JSONResponse:
    return JSONResponse(json.loads(dumps(obj)))


def _clean(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


# --------------------------------------------------------------------------- overview
@app.get("/api/v1/overview")
def overview():
    c, s = coord(), store()
    now = datetime.now()
    cases = s.list()
    hist = c.corpus.history
    upcoming = [h for h in hist if h["start"] and now - timedelta(days=1) <= datetime.fromisoformat(h["start"]) <= now + timedelta(days=45)]
    forecast = [f for f in recurring_maintenance_forecast(hist, now)
                if datetime.fromisoformat(f["estimated_next"]) <= now + timedelta(days=45)]
    sp = c.ds.spaces
    ns_mask = sp["allocation"].fillna("").apply(lambda a: any(d in a for d in NATURAL_SCIENCES_DEPARTMENTS))
    attention = [x for x in cases if x["status"] in ("ReadyForReview", "NeedsRevision")]
    # Natural Sciences seating (workpoints only) and where the model most often disagrees with the record
    seating = sp[ns_mask]["seat_status"].fillna("Not recorded").value_counts().to_dict()
    t = c.predictor.table
    flagged = t[t["verify_flag"]]
    recorded = sp.set_index("space_id")["class_group"].astype(str).reindex(flagged.index)
    pairs = (pd.DataFrame({"recorded": recorded.values, "predicted": flagged["predicted_group"].values})
             .value_counts().head(5).reset_index(name="n").to_dict("records"))
    # agenda: outages on record plus calendar events and blocked times, next 30 days
    soon = now + timedelta(days=30)
    agenda = [{"kind": "outage", "title": ", ".join(h["systems"][:2]) or "Outage", "where": ", ".join(h["buildings"]) or None,
               "start": h["start"], "ref": h["sr"]} for h in upcoming if datetime.fromisoformat(h["start"]) <= soon]
    agenda += [{"kind": "blocked" if e["kind"] == "blocked" else "event", "title": e["title"].split(" (per email")[0],
                "where": e["building"], "start": e["start"], "ref": None}
               for e in STATE["events"].list(now) if datetime.fromisoformat(e["start"]) <= soon]
    agenda.sort(key=lambda a: a["start"])
    return _json({
        "seating": {k: int(v) for k, v in seating.items()},
        "flag_pairs": [{"recorded": r["recorded"], "predicted": r["predicted"], "n": int(r["n"])} for r in pairs],
        "flagged_total": int(len(flagged)),
        "agenda": agenda[:8],
        "as_of": now.isoformat(timespec="minutes"),
        "kpis": {"awaiting_review": len(attention), "upcoming_outages": len(upcoming),
                 "flagged_records": int(c.predictor.table["verify_flag"].sum()),
                 "ns_recorded_vacancies": int((ns_mask & sp["seat_status"].isin(["Vacant", "Under Capacity"])).sum())},
        "attention": attention[:8], "recent_cases": cases[:6],
        "horizon": [{"sr": h["sr"], "start": h["start"], "end": h["end"], "buildings": h["buildings"],
                     "systems": h["systems"], "priority": h["priority"], "kind": "confirmed" if "planned_notification" in h["kinds"]
                     or "unscheduled_alert" in h["kinds"] else "planned"} for h in upcoming]
                   + [{"sr": None, "start": f["estimated_next"], "end": None, "buildings": [f["building"]],
                       "systems": f["systems"], "priority": "Prediction", "kind": "predicted",
                       "window_days": f["window_days"]} for f in forecast],
        "data_health": {**c.ds.report, "outage_requests": len(hist), "messages": len(c.corpus.messages),
                        "model_test_macro_f1": round(c.predictor.metrics["test"]["macro_f1"], 3)},
    })


# --------------------------------------------------------------------------- spaces
@app.get("/api/v1/buildings")
def buildings():
    sp, t = coord().ds.spaces, coord().predictor.table
    rows = []
    for b in BUILDINGS:
        s = sp[sp["building"] == b["export"]]
        rows.append({"code": b["code"], "name": b["name"], "export_name": b["export"], "spaces": int(len(s)),
                     "labs": int(s["class_group"].isin(["Research Lab", "Teaching Lab"]).sum()),
                     "vacant": int(s["seat_status"].isin(["Vacant", "Under Capacity"]).sum()),
                     "flagged": int(t.reindex(s["space_id"])["verify_flag"].sum())})
    others = sorted(set(sp["building"].dropna()) - {b["export"] for b in BUILDINGS})
    return {"crosswalk": rows, "other_buildings": others}


@app.get("/api/v1/spaces")
def spaces(building: str | None = None, group: str | None = None, seat: str | None = None,
           department: str | None = None, q: str | None = None, flagged: bool = False,
           vacancy: bool = False, ns: bool = False,
           limit: int = Query(100, le=500), offset: int = Query(0, ge=0)):
    sp, t = coord().ds.spaces, coord().predictor.table
    m = pd.Series(True, index=sp.index)
    if building:
        m &= sp["building"] == building
    if group:
        m &= sp["class_group"] == group
    if seat:
        m &= sp["seat_status"] == seat
    if department:
        m &= sp["allocation"].fillna("").str.contains(department, case=False, regex=False)
    if q:
        m &= (sp["room_key"].str.contains(q, case=False, regex=False)
              | sp["space_standard"].fillna("").str.contains(q, case=False, regex=False)
              | sp["allocation"].fillna("").str.contains(q, case=False, regex=False))
    if flagged:
        m &= t.reindex(sp["space_id"])["verify_flag"].values
    if vacancy:  # recorded vacancy = open seats, as counted on the Overview
        m &= sp["seat_status"].isin(["Vacant", "Under Capacity"])
    if ns:
        m &= sp["allocation"].fillna("").apply(lambda a: any(d in a for d in NATURAL_SCIENCES_DEPARTMENTS))
    hits = sp[m]
    page = hits.iloc[offset: offset + limit]
    pred = t.reindex(page["space_id"])
    rows = []
    for (_, r), (_, p) in zip(page.iterrows(), pred.iterrows()):
        rows.append({"space_id": r.space_id, "building": r.building, "floor": r.floor, "room": r.room,
                     "area": _clean(round(r.area, 1)) if pd.notna(r.area) else None, "space_class": _clean(r.space_class),
                     "class_group": r.class_group, "space_standard": _clean(r.space_standard),
                     "seat_status": _clean(r.seat_status), "allocation": _clean(r.allocation),
                     "plan_reservation": bool(r.plan_reservation), "occupants": int(r.occ_total),
                     "predicted_group": p.predicted_group, "confidence": round(float(p.confidence), 3),
                     "verify_flag": bool(p.verify_flag)})
    groups = sorted(sp["class_group"].astype(str).unique())
    return _json({"total": int(len(hits)), "rows": rows, "groups": groups,
                  "seat_statuses": sorted(sp["seat_status"].dropna().unique())})


@app.get("/api/v1/spaces/{space_id:path}")
def space_detail(space_id: str):
    c = coord()
    row = c.ds.space(space_id)
    if row is None:
        raise HTTPException(404, f"No matching record found in space records: {space_id}")
    d = c.space.get_space_details(space_id).data
    occ = c.ds.occupants(row.room_key)[["name", "netid", "department", "role", "title"]]
    code = c.space.resolver.info(row.building)["code"]
    plan = STATE["plans"].get((code, row.floor))
    return _json({"space": d, "occupants": occ.fillna("").to_dict("records"),
                  "model": c.predictor.assessment(space_id), "building": c.space.resolver.info(row.building),
                  "floor_plan": {"url": f"/api/v1/floorplans/{code}/{row.floor}", "file": plan["file"],
                                 "snapshot": plan["snapshot"]} if plan else None})


@app.get("/api/v1/floorplans/{code}/{floor}")
def floor_plan(code: str, floor: str):
    plan = STATE["plans"].get((code.upper(), floor.upper()))
    if not plan:
        raise HTTPException(404, "No floor plan on file for this building floor")
    return FileResponse(plan["path"], media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{plan["file"]}"'})


# --------------------------------------------------------------------------- outages
@app.get("/api/v1/outages")
def outages():
    c = coord()
    return _json({"history": list(reversed(c.corpus.history)),
                  "forecast": recurring_maintenance_forecast(c.corpus.history, datetime.now()),
                  "forecast_accuracy": {"systems": forecast_backtest(c.corpus.history, "systems"),
                                        "building": forecast_backtest(c.corpus.history, "building")}})


@app.get("/api/v1/sources")
def sources():
    files = sorted(p.name for p in (Path(__file__).resolve().parent.parent / "datares").glob("*.rtf"))
    return {"messages": outage_sources(coord().corpus)[:250], "threads": files,
            "scenarios": [{"id": s.id, "title": s.title, "kind": s.kind, "description": s.description} for s in scenarios()]}


class CaseIn(BaseModel):
    kind: Literal["space", "outage"]
    input: dict = Field(default_factory=dict)
    title: str | None = Field(None, max_length=160)


def _resolve_outage_input(inp: dict) -> dict:
    c = coord()
    if "message_id" in inp:
        i = int(inp["message_id"])
        if not 0 <= i < len(c.corpus.messages):
            raise HTTPException(404, "Unknown message")
        m = c.corpus.messages[i]
        return {"text": message_text(m), "source": f"{m.thread} · {m.subject}"[:200]}
    if "thread" in inp:
        path = (Path(__file__).resolve().parent.parent / "datares" / Path(inp["thread"]).name)
        if not path.exists() or path.suffix != ".rtf":
            raise HTTPException(404, "Unknown thread")
        return {"text": rtf_file_text(path), "source": path.name}
    if "scenario" in inp:
        sc = next((s for s in scenarios() if s.id == inp["scenario"]), None)
        if not sc:
            raise HTTPException(404, "Unknown scenario")
        return resolve_payload(sc, c.corpus)
    return inp


def _run(kind: str, inp: dict) -> tuple[dict, dict]:
    try:
        if kind == "space":
            if "scenario" in inp:
                sc = next(s for s in scenarios() if s.id == inp["scenario"])
                inp = sc.payload
            req = SpaceRequest(**inp)
            return req.model_dump(), coord().run_space(req)
        resolved = _resolve_outage_input(inp)
        req = OutageRequest(**resolved)
        return {"source": req.source, "chars": len(req.text), **{k: v for k, v in inp.items() if k != "text"},
                "text": req.text if len(req.text) < 20000 else None}, coord().run_outage(req, cooperate=bool(inp.get("temporary_space")))
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False))


def _known_outage_case(inp: dict) -> dict | None:
    """Before running the agent: is this notice (or its request number) already a case?"""
    source = {k: inp[k] for k in ("message_id", "thread", "scenario") if k in inp}
    reference = None
    if "message_id" in inp:
        msgs = coord().corpus.messages
        i = int(inp["message_id"]) if str(inp["message_id"]).lstrip("-").isdigit() else -1
        if 0 <= i < len(msgs) and len(msgs[i].sr_ids) == 1:
            reference = msgs[i].sr_ids[0]
    return store().find_outage(reference=reference) or (store().find_outage(source=source) if source else None)


@app.post("/api/v1/cases")
def create_case(body: CaseIn):
    """Run the agent and open a case. An outage request that already has a case gets that case back
    (existing: true) instead of a duplicate; re-running the existing case makes a new revision."""
    with store().outage_lock:             # one case per request, even with double clicks or the email poll
        if body.kind == "outage":
            known = _known_outage_case(body.input)
            if known:
                return _json({**known, "existing": True})
        inp, out = _run(body.kind, body.input)
        if body.kind == "outage":
            ref = (out.get("outage") or {}).get("request_reference")
            known = store().find_outage(reference=ref) if ref else None
            if known:
                return _json({**known, "existing": True})
        return _json(store().create(body.kind, None, inp, out))


@app.get("/api/v1/cases")
def list_cases():
    return _json(store().list())


@app.delete("/api/v1/cases/{cid}")
def delete_case(cid: str):
    try:
        return _json(store().delete(cid, REVIEWER_NAME))
    except KeyError:
        raise HTTPException(404, "Case not found")


@app.get("/api/v1/cases/{cid}")
def get_case(cid: str):
    case = store().get(cid)
    if not case:
        raise HTTPException(404, "Case not found")
    can_send = STATE["settings"].is_on("send_email")
    return _json({**case, "sent": STATE["outbox"].sent(cid),
                  "sending": {"enabled": can_send, "live": can_send and STATE["settings"].is_on("email_live_recipients"),
                              "redirect_to": STATE["mailer"].redirect_to or None}})


class RerunIn(BaseModel):
    input: dict | None = None


@app.post("/api/v1/cases/{cid}/rerun")
def rerun_case(cid: str, body: RerunIn):
    case = store().get(cid)
    if not case:
        raise HTTPException(404, "Case not found")
    inp = body.input or case["input"]
    if case["kind"] == "outage" and not inp.get("text"):
        keep = {k: v for k, v in case["input"].items() if k in ("message_id", "thread", "scenario")}
        if keep:
            inp = {**keep, "temporary_space": bool((body.input or {}).get("temporary_space"))}
    _, out = _run(case["kind"], inp)
    return _json(store().rerun(cid, case["input"] if body.input is None else inp, out))


class ReviewIn(BaseModel):
    decision: Literal["approve", "return", "decline"]
    rationale: str = Field("", max_length=2000)
    rev: int


@app.post("/api/v1/cases/{cid}/review")
def review_case(cid: str, body: ReviewIn):
    try:
        return _json(store().review(cid, body.decision, body.rationale, body.rev))
    except KeyError:
        raise HTTPException(404, "Case not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/v1/cases/{cid}/recipients.csv")
def recipients_csv(cid: str, scope: Literal["natural_sciences", "all"] = "natural_sciences"):
    """Affected people for an outage case, for James to send from his own mailbox."""
    case = store().get(cid)
    if not case or case["kind"] != "outage":
        raise HTTPException(404, "Outage case not found")
    people = (case["output"].get("affected_people") or {}).get("people", [])
    if scope == "natural_sciences":
        people = [p for p in people if p["natural_sciences"]]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["name", "email", "netid", "department", "role", "room", "reason", "natural_sciences"])
    for p in people:
        w.writerow([p["name"], p["email"], p["netid"], p["department"], p["role"], p["room"], p["reason"], p["natural_sciences"]])
    store().audit("user", "recipients_exported", cid, f"{len(people)} people · {scope}")
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{cid}-recipients.csv"'})


class SendIn(BaseModel):
    draft: Literal["coordination", "notification"]
    rev: int
    confirm: bool = False


_SEND_LOCK = threading.Lock()


@app.post("/api/v1/cases/{cid}/send")
def send_draft(cid: str, body: SendIn):
    """Send one approved draft. Needs the 'send_email' switch, an approved case and explicit confirmation.
    One send at a time, so a double click or a second tab can never deliver the same draft twice."""
    with _SEND_LOCK:
        return _send_draft(cid, body)


def _send_draft(cid: str, body: SendIn):
    settings = STATE["settings"]
    if not settings.is_on("send_email"):
        store().audit(REVIEWER_NAME, "locked_action_attempt", cid, "send_email (switch off)")
        raise HTTPException(403, "Sending is switched off. Turn on 'Send email' in Settings.")
    if not body.confirm:
        raise HTTPException(422, "Sending needs explicit confirmation")
    try:
        mail, meta = STATE["outbox"].prepare(store().get(cid), body.draft, body.rev,
                                             settings.is_on("email_live_recipients"), STATE["mailer"])
    except SendRefused as e:
        raise HTTPException(e.status, e.detail)
    try:
        message_id = STATE["mailer"].send(mail)
    except Exception as exc:
        store().audit(REVIEWER_NAME, "email_send_failed", cid, f"{body.draft}: {type(exc).__name__}")
        raise HTTPException(502, f"The mail server refused the message: {type(exc).__name__}: {exc}")
    return _json(STATE["outbox"].record(store().get(cid), body.draft, meta, message_id, REVIEWER_NAME))


# --------------------------------------------------------------------------- settings (switches)
@app.get("/api/v1/settings")
def get_settings():
    return _json({"switches": STATE["settings"].list(), "unlock_allowed": unlock_allowed(),
                  "auto_off_minutes": AUTO_OFF_MINUTES, "email": STATE["mailer"].describe()})


class SwitchIn(BaseModel):
    on: bool
    minutes: int | None = None


@app.put("/api/v1/settings/{key}")
def set_switch(key: str, body: SwitchIn):
    try:
        STATE["settings"].set(key, body.on, REVIEWER_NAME, body.minutes)
        if key == "voice_assistant" and body.on:
            STATE["voice"].prewarm()
    except KeyError:
        raise HTTPException(404, f"Unknown setting '{key}'")
    except PermissionError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return get_settings()


# --------------------------------------------------------------------------- voice assistant
@app.get("/api/v1/voice/options")
def voice_options():
    """Installed local models (only these can be chosen), the current choice and what is missing."""
    return _json({**STATE["voice"].options(), "enabled": STATE["settings"].is_on("voice_assistant")})


@app.get("/api/v1/voice/status")
def voice_status():
    """Is the voice assistant switched on? Fast: decides whether the space request dialog offers a conversation."""
    return {"enabled": STATE["settings"].switched_on("voice_assistant")}


@app.post("/api/v1/voice/warm")
def voice_warm():
    """Called when the app opens: if voice is on, load the models in the background (rate-limited)."""
    on = STATE["settings"].is_on("voice_assistant")
    if on:
        STATE["voice"].prewarm()
    return {"warming": on}


class VoiceChoiceIn(BaseModel):
    voice_llm: str | None = Field(None, max_length=120)
    voice_stt: str | None = Field(None, max_length=120)
    voice_voice: str | None = Field(None, max_length=60)


@app.put("/api/v1/voice/options")
def set_voice_options(body: VoiceChoiceIn):
    try:
        for key in CHOICES:
            value = getattr(body, key)
            if value:
                STATE["voice"].choose(key, value, REVIEWER_NAME)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return voice_options()


def _save_voice_case(state: dict) -> str:
    """Save the conversation's latest agent result as a space case (Rev A, ready for review)."""
    full = state.get("full_report")
    if not full:
        raise ValueError("Nothing to save yet: ask for space first.")
    req = SpaceRequest(**{k: v for k, v in (state.get("slots") or {}).items() if k in SpaceRequest.model_fields})
    case = store().create("space", None, {**req.model_dump(), "source": "voice"}, full)
    store().audit(REVIEWER_NAME, "case_created_by_voice", case["id"], case["title"])
    return case["id"]


@app.websocket("/api/v1/voice/ws")
async def voice_socket(ws: WebSocket):
    await ws.accept()
    if not STATE["settings"].is_on("voice_assistant"):
        await ws.send_json({"type": "error", "message": "The voice assistant is switched off. Turn it on in Settings."})
        await ws.close()
        return
    await ws.send_json({"type": "status", "state": "loading"})
    try:
        session = await run_in_threadpool(STATE["voice"].build_session, _save_voice_case)
    except Exception as exc:
        log.error(f"voice session could not start error={exc!r}")
        await ws.send_json({"type": "error", "message": f"Could not start the voice models: {exc}"})
        await ws.close()
        return
    store().audit(REVIEWER_NAME, "voice_session_started", detail=session.thread_id)
    await session.run(ws)


# --------------------------------------------------------------------------- events & schedule checks
class EventIn(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    start: datetime
    end: datetime
    building: str | None = Field(None, max_length=120)
    kind: Literal["event", "blocked"] = "event"
    note: str = Field("", max_length=500)


@app.get("/api/v1/events")
def list_events(upcoming: bool = True):
    since = datetime.now() - timedelta(days=1) if upcoming else None
    return _json({"events": STATE["events"].list(since), "counts": STATE["events"].count()})


@app.post("/api/v1/events")
def add_event(body: EventIn):
    building = None
    if body.building:
        m = coord().space.resolver.resolve(body.building)
        if m.status != "resolved":
            raise HTTPException(422, f"Building '{body.building}' is {m.status.replace('_', ' ')}")
        building = m.export_name
    try:
        ev = STATE["events"].add(body.title, body.start, body.end, building, body.kind,
                                 "blocked" if body.kind == "blocked" else "manual", body.note)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not ev.get("duplicate"):
        store().audit(REVIEWER_NAME, "time_blocked" if body.kind == "blocked" else "event_added", detail=f"{body.title} · {body.start:%Y-%m-%d %H:%M}")
    return _json(ev)


@app.delete("/api/v1/events/{eid}")
def delete_event(eid: int):
    if not STATE["events"].delete(eid):
        raise HTTPException(404, "Event not found or not removable (correspondence events are read-only)")
    store().audit(REVIEWER_NAME, "event_removed", detail=str(eid))
    return {"deleted": eid}


class IcsIn(BaseModel):
    ics: str = Field(..., min_length=20, max_length=5_000_000)


@app.post("/api/v1/events/import")
def import_events(body: IcsIn):
    try:
        res = STATE["events"].import_ics(body.ics, coord().space.resolver)
    except Exception as e:
        raise HTTPException(422, f"Could not read the calendar file: {e}")
    store().audit(REVIEWER_NAME, "calendar_imported", detail=f"{res['imported']} events, {res['skipped']} already present")
    return res


class WindowIn(BaseModel):
    start: datetime
    end: datetime
    buildings: list[str] = Field(default_factory=list, max_length=20)
    sr: str | None = None


@app.post("/api/v1/schedule/check")
def check_window(body: WindowIn):
    """Review any proposed outage time, e.g. when James or Facilities proposes another time."""
    if body.end <= body.start:
        raise HTTPException(422, "End must be after start")
    with run_context():
        res = coord().outage.review_window(body.buildings, body.start, body.end, body.sr)
    return _json(res)


@app.get("/api/v1/inbox")
def inbox():
    return _json(STATE["intake"].status())


class TriageIn(BaseModel):
    message_id: str = Field(..., max_length=500)
    label: Literal["outage", "not_outage"]


@app.post("/api/v1/inbox/triage")
def triage_email(body: TriageIn):
    """James decides an email the triage agent held for review; the triage graph then files it."""
    try:
        with run_context():
            out = STATE["intake"].decide(body.message_id, body.label, REVIEWER_NAME)
    except KeyError:
        raise HTTPException(404, "No email waiting for review with that id")
    return _json({**out, "status": STATE["intake"].status()})


@app.get("/api/v1/inbox/evaluation")
def triage_evaluation():
    """Results of python -m app.triage_eval (keyword rule vs local model vs triage agent), if it has been run."""
    f = DATA_DIR / "triage_eval.json"
    return _json(json.loads(f.read_text(encoding="utf-8")) if f.exists() else None)


# --------------------------------------------------------------------------- connections (live data)
@app.get("/api/v1/connections")
def connections():
    src = STATE["intake"].source
    return _json({"microsoft": STATE["graph"].status(),
                  "email": {**src.health_check(), "mode": src.mode, "poll_seconds": STATE.get("poll_seconds", 0)},
                  "records": {**STATE["records"].status(), "refresh_minutes": records_config(STATE["settings"])["refresh_minutes"]}})


@app.post("/api/v1/connections/microsoft/connect")
def microsoft_connect():
    """Start Microsoft 365 sign-in: returns a code that James enters at the Microsoft page himself."""
    try:
        st = STATE["graph"].start_device_flow()
    except GraphError as e:
        raise HTTPException(409, str(e))
    store().audit(REVIEWER_NAME, "microsoft_sign_in_started")
    return _json(st)


@app.post("/api/v1/connections/microsoft/disconnect")
def microsoft_disconnect():
    STATE["graph"].sign_out()
    store().audit(REVIEWER_NAME, "microsoft_signed_out")
    return connections()


class RecordsConfigIn(BaseModel):
    source: Literal["microsoft", "folder", "upload"]
    dir: str | None = Field(None, max_length=500)
    space_url: str | None = Field(None, max_length=2000)
    people_url: str | None = Field(None, max_length=2000)
    refresh_minutes: int = Field(0, ge=0, le=10080)


@app.get("/api/v1/records/config")
def get_records_config():
    return {**records_config(STATE["settings"]), "microsoft_connected": STATE["graph"].status()["status"] == "connected"}


@app.put("/api/v1/records/config")
def set_records_config(body: RecordsConfigIn):
    """James connects the records from the app: SharePoint/OneDrive links, a folder, or uploads."""
    if body.source == "microsoft" and not (body.space_url and body.people_url):
        raise HTTPException(422, "Paste the links to both reports")
    if body.source == "folder" and not (body.dir and Path(body.dir).is_dir()):
        raise HTTPException(422, "That folder does not exist on this computer")
    s = STATE["settings"]
    for key, val in (("records_source", body.source), ("records_dir", body.dir), ("records_space_url", body.space_url),
                     ("records_people_url", body.people_url), ("records_refresh_minutes", str(body.refresh_minutes))):
        if val is not None:
            s.set_choice(key, val.strip() if isinstance(val, str) else val, REVIEWER_NAME)
    STATE["records"].source = build_records_source(STATE["graph"], s)
    return get_records_config()


class UploadIn(BaseModel):
    space_b64: str = Field(..., max_length=60_000_000)
    people_b64: str = Field(..., max_length=60_000_000)


@app.post("/api/v1/records/upload")
def records_upload(body: UploadIn):
    """Two Excel reports uploaded in the app: validated, compared and switched to like any other new version."""
    import base64
    try:
        src = UploadSource(base64.b64decode(body.space_b64), base64.b64decode(body.people_b64))
    except ValueError:
        raise HTTPException(422, "The files could not be read")
    res = STATE["records"].refresh(force=True, source=src)
    if res.get("changed"):
        STATE["settings"].set_choice("records_source", "upload", REVIEWER_NAME)
        STATE["records"].source = build_records_source(STATE["graph"], STATE["settings"])
        store().audit(REVIEWER_NAME, "records_uploaded")
    return _json({**res, "status": STATE["records"].status()})


@app.post("/api/v1/records/refresh")
def records_refresh(force: bool = False):
    """Check the records source now; take in a new version if there is one (validated first)."""
    res = STATE["records"].refresh(force=force)
    if res.get("changed"):
        store().audit(REVIEWER_NAME, "records_refreshed", detail=json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in res["diff"].items() if k != "unchanged"}))
    return _json({**res, "status": STATE["records"].status()})


@app.post("/api/v1/inbox/poll")
def poll_inbox(limit: int = Query(10, ge=1, le=100)):
    with run_context():
        res = STATE["intake"].poll(limit)
    return _json({**res, "status": STATE["intake"].status()})


@app.post("/api/v1/actions/{action}")
def locked_action(action: str):
    store().audit("user", "locked_action_attempt", detail=action)
    hint = " Turn on Send email in Settings." if action == "send_email" else ""
    raise HTTPException(403, LOCKED_ACTIONS.get(action, "Not available.") + hint)


# --------------------------------------------------------------------------- analysis views
@app.get("/api/v1/model")
def model_info():
    p = coord().predictor
    t = p.table
    flagged = t[t["verify_flag"]].sort_values("confidence", ascending=False).head(25)
    return _json({"metrics": p.metrics, "flagged_examples": [
        {"space_id": i, "recorded": r.recorded_group, "predicted": r.predicted_group, "confidence": round(float(r.confidence), 3)}
        for i, r in flagged.iterrows()], "flagged_total": int(t["verify_flag"].sum()),
        "agreement_rate": round(float((t["predicted_group"] == t["recorded_group"]).mean()), 4)})


@app.get("/api/v1/ablation")
def ablation():
    f = MODEL_DIR / "ablation.json"
    return _json(json.loads(f.read_text(encoding="utf-8")) if f.exists() else None)


@app.get("/api/v1/agents")
def agents():
    c = coord()
    return _json({"agents": [c.space.describe(), c.outage.describe()], "messages": c.messages[-60:]})


@app.get("/api/v1/activity")
def activity(limit: int = Query(150, le=500)):
    return _json({"audit": store().audit_events(limit), "messages": coord().messages[-60:]})


@app.get("/api/v1/evaluation")
def evaluation_results():
    f = DATA_DIR / "evaluation.json"
    return _json(json.loads(f.read_text(encoding="utf-8")) if f.exists() else None)


@app.post("/api/v1/evaluation/run")
def run_evaluation():
    from app.evaluation import evaluate_agents
    with run_context():
        res = evaluate_agents(coord())
    res["generated"] = datetime.now().isoformat(timespec="seconds")
    (DATA_DIR / "evaluation.json").write_text(dumps(res), encoding="utf-8")
    store().audit("system", "evaluation_run", detail=f"{res['passed']}/{res['total']} checks passed")
    return _json(res)


@app.get("/api/v1/experiments")
def experiments():
    f = DATA_DIR / "experiments.json"
    return _json(json.loads(f.read_text(encoding="utf-8")) if f.exists() else None)


@app.post("/api/v1/experiments/run")
def run_experiments():
    from app.experiments import run_all
    with run_context():
        res = run_all(coord())
    store().audit("system", "experiments_run", detail=f"competition winner {res['competition']['winner']}")
    return _json(res)


# --------------------------------------------------------------------------- static client
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
