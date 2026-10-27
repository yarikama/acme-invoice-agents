"""HTTP API: start runs, stream their events over SSE, take the VP's decision.

uv run uvicorn invoice_agents.api.app:app --reload
"""

from __future__ import annotations

import asyncio
import json
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
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from invoice_agents.api.runs import RunManager, RunNotFoundError, RunStateError
from invoice_agents.config import Settings, get_settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.inventory import SEED, Inventory
from invoice_agents.llm import build_agents
from invoice_agents.models import HumanDecision, SourceFormat

KEEPALIVE_SECONDS = 15
SUFFIXES = {f".{f.value}" for f in SourceFormat}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    inventory = Inventory(settings.db_path)
    inventory.seed()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.checkpoint_path) as conn:
        checkpointer = AsyncSqliteSaver(conn, serde=checkpoint_serde())
        deps = Deps(
            agents=build_agents(settings), inventory=inventory, settings=settings
        )
        manager = RunManager(build_graph(deps, checkpointer=checkpointer))
        app.state.settings = settings
        app.state.manager = manager
        try:
            yield
        finally:
            await manager.close()


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
async def health(settings: SettingsDep) -> dict[str, str]:
    return {"status": "ok", "llm_provider": settings.llm_provider}


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


@app.post("/api/runs/{run_id}/decision", status_code=202)
async def decide(
    run_id: str, decision: HumanDecision, manager: ManagerDep
) -> dict[str, Any]:
    _run_or_404(manager, run_id)
    try:
        return manager.decide(run_id, decision).overview()
    except RunStateError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    manager: ManagerDep,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Server-sent events. Replays history, then streams live until the run ends.

    The stream stays open while the run waits for the VP; the decision's events
    arrive on the same connection.
    """
    _run_or_404(manager, run_id)
    after = int(last_event_id) if last_event_id and last_event_id.isdigit() else -1
    return StreamingResponse(
        _sse(manager, run_id, after),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(manager: RunManager, run_id: str, after: int) -> AsyncIterator[str]:
    events = manager.events(run_id, after).__aiter__()
    next_event = asyncio.ensure_future(anext(events))
    try:
        while True:
            done, _ = await asyncio.wait({next_event}, timeout=KEEPALIVE_SECONDS)
            if not done:
                yield ": keepalive\n\n"
                continue
            try:
                event = next_event.result()
            except StopAsyncIteration:
                return
            yield (
                f"id: {event['id']}\nevent: {event['type']}\n"
                f"data: {json.dumps(event)}\n\n"
            )
            next_event = asyncio.ensure_future(anext(events))
    finally:
        next_event.cancel()


def _run_or_404(manager: RunManager, run_id: str):
    try:
        return manager.get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(404, f"No run {run_id!r}") from exc
