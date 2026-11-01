"""HTTP API: start runs, stream their events over SSE, take the VP's decision.

uv run uvicorn invoice_agents.api.app:app --reload
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import aiosqlite
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from invoice_agents.api.runs import RunManager, RunNotFoundError, RunStateError
from invoice_agents.config import Settings, get_settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.inventory import SEED, Inventory
from invoice_agents.llm import build_agents
from invoice_agents.models import HumanDecision, SourceFormat
from invoice_agents.replay import Recording
from invoice_agents.usage import Pricing, openrouter_pricing

log = logging.getLogger("invoice_agents.api")
SUFFIXES = {f".{f.value}" for f in SourceFormat}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if not settings.vp_tokens:
        log.warning(
            "VP_TOKENS is unset: anyone who can reach this API can approve"
            " payments. Fine for a local demo, not for anything else."
        )
    inventory = Inventory(settings.db_path)
    inventory.seed()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.checkpoint_path) as conn:
        checkpointer = AsyncSqliteSaver(conn, serde=checkpoint_serde())
        deps = Deps(
            agents=build_agents(settings),
            inventory=inventory,
            settings=settings,
            pricing=await asyncio.to_thread(load_pricing, settings),
        )
        manager = RunManager(build_graph(deps, checkpointer=checkpointer))
        app.state.settings = settings
        app.state.manager = manager
        try:
            yield
        finally:
            await manager.close()


def load_pricing(settings: Settings) -> Pricing:
    """OpenRouter's model prices, for each step's cost. Without them the
    console still shows tokens, so a failed fetch is not fatal."""
    if settings.llm_provider == "replay":
        # The prices when the recording was made.
        return Recording.shared(settings.replay_path).pricing
    if settings.llm_provider != "openrouter":
        return {}
    try:
        pricing = openrouter_pricing()
    except (OSError, ValueError):
        return {}
    if settings.record_to is not None:
        # Recording through the API keeps the prices with it, as the eval does.
        Recording.shared(settings.record_to).pricing = pricing
    return pricing


app = FastAPI(title="Acme Invoice Agents", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_manager(request: Request) -> RunManager:
    return request.app.state.manager


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


ManagerDep = Annotated[RunManager, Depends(get_manager)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


class StartRun(BaseModel):
    sample: str


def _sample_path(settings: Settings, name: str) -> Path:
    # Only bare file names from the samples folder: no paths, no traversal.
    path = settings.samples_dir / Path(name).name
    if path.name != name or path.suffix.lower() not in SUFFIXES or not path.is_file():
        raise HTTPException(404, f"No sample named {name!r}")
    return path


def _samples(settings: Settings) -> list[Path]:
    return sorted(
        p
        for p in settings.samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )


@app.get("/api/health")
async def health(settings: SettingsDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        # The console asks for a VP token up front when this is true.
        "vp_auth": bool(settings.vp_tokens),
    }


@app.get("/api/samples")
async def samples(settings: SettingsDep) -> list[dict[str, Any]]:
    return [
        {
            "name": p.name,
            "format": p.suffix.lstrip(".").lower(),
            "bytes": p.stat().st_size,
        }
        for p in _samples(settings)
    ]


@app.get("/api/inventory")
async def inventory() -> list[dict[str, Any]]:
    return [{"item": i, "stock": s, "unit_price": p} for i, s, p in SEED]


@app.get("/api/runs")
async def list_runs(manager: ManagerDep) -> list[dict[str, Any]]:
    return [run.overview() for run in manager.list()]


@app.post("/api/runs", status_code=202)
async def start_run(
    body: StartRun, manager: ManagerDep, settings: SettingsDep
) -> dict[str, Any]:
    run = manager.start(_sample_path(settings, body.sample), source=body.sample)
    return run.overview()


@app.post("/api/runs/upload", status_code=202)
async def upload_run(
    file: Annotated[UploadFile, File()], manager: ManagerDep, settings: SettingsDep
) -> dict[str, Any]:
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in SUFFIXES:
        raise HTTPException(415, f"Unsupported file type {suffix or '(none)'}")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "File too large")
    # Stored under a random name; the original name is only a label.
    path = settings.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return manager.start(path, source=name).overview()


@app.post("/api/runs/batch", status_code=202)
async def start_batch(
    manager: ManagerDep, settings: SettingsDep
) -> list[dict[str, Any]]:
    runs = await manager.start_batch(_samples(settings))
    return [run.overview() for run in runs]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    run = _run_or_404(manager, run_id)
    return {**run.overview(), "summary": run.summary, "events": run.events}


def _vp(settings: Settings, authorization: str | None) -> str | None:
    """The VP a bearer token belongs to; None when no tokens are configured."""
    if not settings.vp_tokens:
        return None
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() == "bearer" and token:
        for name, secret in settings.vp_tokens.items():
            # Constant time, so a wrong token can't be guessed byte by byte.
            if hmac.compare_digest(token.encode(), secret.get_secret_value().encode()):
                return name
    raise HTTPException(
        401,
        "Only a VP can approve or reject: send a VP token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.post("/api/runs/{run_id}/decision", status_code=202)
async def decide(
    run_id: str,
    decision: HumanDecision,
    manager: ManagerDep,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """The VP's answer. With VP_TOKENS set, it needs a VP's bearer token, and
    the run records that VP's name, not anything the client claims."""
    by = _vp(settings, authorization)
    _run_or_404(manager, run_id)
    if not decision.approved and not decision.note.strip():
        # The reason goes to the vendor and the audit log.
        raise HTTPException(422, "Say why it is rejected: a rejection needs a reason.")
    decision = decision.model_copy(update={"by": by})
    try:
        return manager.decide(run_id, decision).overview()
    except RunStateError as exc:
        raise HTTPException(409, str(exc)) from exc


def _known_run(run_id: str, manager: ManagerDep) -> str:
    """A dependency, so an unknown run is a 404 before the stream opens."""
    _run_or_404(manager, run_id)
    return run_id


@app.get("/api/runs/{run_id}/events", response_class=EventSourceResponse)
async def stream_events(
    run_id: Annotated[str, Depends(_known_run)],
    manager: ManagerDep,
    last_event_id: Annotated[str | None, Header()] = None,
) -> AsyncIterator[ServerSentEvent]:
    """Server-sent events. Replays history, then streams live until the run ends.

    The stream stays open while the run waits for the VP; the decision's events
    arrive on the same connection. FastAPI frames the events and sends a
    keep-alive comment while the stream is idle.
    """
    async for event in manager.events(run_id, _after(last_event_id)):
        yield _sse(event, "id")


@app.get("/api/events", response_class=EventSourceResponse)
async def stream_all_events(
    manager: ManagerDep,
    last_event_id: Annotated[str | None, Header()] = None,
) -> AsyncIterator[ServerSentEvent]:
    """Every run's events on one connection, for the console.

    Browsers allow only six HTTP/1.1 connections per host, so one stream per
    run would stall a batch. The SSE id here is the global `seq`.
    """
    async for event in manager.all_events(_after(last_event_id)):
        yield _sse(event, "seq")


def _after(last_event_id: str | None) -> int:
    return int(last_event_id) if last_event_id and last_event_id.isdigit() else -1


def _sse(event: dict[str, Any], id_key: str) -> ServerSentEvent:
    return ServerSentEvent(
        id=str(event[id_key]), event=event["type"], raw_data=json.dumps(event)
    )


def _run_or_404(manager: RunManager, run_id: str):
    try:
        return manager.get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(404, f"No run {run_id!r}") from exc
