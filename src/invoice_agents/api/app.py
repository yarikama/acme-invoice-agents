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
from typing import Annotated, Any, Literal

import aiosqlite
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from invoice_agents import logs
from invoice_agents.api.chat import ChatService, ChatUnavailableError
from invoice_agents.api.runs import RunManager, RunNotFoundError, RunStateError
from invoice_agents.config import Settings, get_settings
from invoice_agents.graph import build_graph, checkpoint_serde, stored_deps
from invoice_agents.inventory import Inventory
from invoice_agents.llm import chat_model, load_pricing
from invoice_agents.models import HumanDecision, SourceFormat

log = logging.getLogger("invoice_agents.api")
SUFFIXES = {f".{f.value}" for f in SourceFormat}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app_log = logging.getLogger("invoice_agents")
    if not app_log.handlers:
        # Under uvicorn nothing shows this app's INFO lines otherwise.
        app_log.addHandler(logging.StreamHandler())
        app_log.setLevel(logging.INFO)
    if settings.log_json is not None:
        logs.to_json_file(settings.log_json)
    if not settings.vp_tokens:
        log.warning(
            "VP_TOKENS is unset: anyone who can reach this API can approve"
            " payments. Fine for a local demo, not for anything else."
        )
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.checkpoint_path) as conn:
        checkpointer = AsyncSqliteSaver(conn, serde=checkpoint_serde())
        deps = stored_deps(settings, await asyncio.to_thread(load_pricing, settings))
        manager = RunManager(
            build_graph(deps, checkpointer=checkpointer), deps.seen, deps.bank
        )
        chat = ChatService(
            settings,
            manager,
            lambda: chat_model(settings, settings.extract_model, "low"),
        )
        restored = await manager.restore()
        if restored:
            log.info("Restored %d run(s) waiting for a VP.", len(restored))
        app.state.settings = settings
        app.state.manager = manager
        app.state.chat = chat
        try:
            yield
        finally:
            await chat.close()
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


# What reads a scanned PDF or photo; unset: the configured default.
Parser = Literal["pymupdf", "docling", "grok"]


class StartRun(BaseModel):
    sample: str
    parser: Parser | None = None


class StartBatch(BaseModel):
    parser: Parser | None = None


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
async def health(settings: SettingsDep, request: Request) -> dict[str, Any]:
    chat: ChatService = request.app.state.chat
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        # "auto" found no API key: the console says it shows a recording.
        "no_key": settings.replay_for_lack_of_key,
        # The console asks for a VP token up front when this is true.
        "vp_auth": bool(settings.vp_tokens),
        # The assistant needs a live model; "shared" means Redis holds the
        # conversation, so the terminal sees it too.
        "chat": {"available": chat.available, "shared": chat.shared},
        # The parser scans get unless a run picks another.
        "parser": settings.ocr_backend,
    }


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def get_chat(request: Request) -> ChatService:
    return request.app.state.chat


ChatDep = Annotated[ChatService, Depends(get_chat)]


@app.get("/api/chat")
async def chat_messages(chat: ChatDep, after: int = -1) -> dict[str, Any]:
    """The shared conversation, from the terminal and the web console."""
    messages = await asyncio.to_thread(chat.messages, after)
    return {"messages": messages, "available": chat.available, "shared": chat.shared}


@app.post("/api/chat")
async def chat_say(body: ChatRequest, chat: ChatDep) -> dict[str, Any]:
    """Ask the assistant; it may start runs. Returns the messages it added."""
    try:
        return {"messages": await chat.say(body.text.strip())}
    except ChatUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # the provider, the network or a bad answer
        log.warning("The assistant couldn't answer: %s", exc)
        raise HTTPException(502, f"The assistant couldn't answer: {exc}") from exc


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
async def inventory(settings: SettingsDep) -> list[dict[str, Any]]:
    """What the validator checks against: the database, not the seed."""
    records = await asyncio.to_thread(Inventory(settings.db_path).records)
    return [
        {"item": r.item, "stock": r.stock, "unit_price": r.unit_price} for r in records
    ]


@app.get("/api/runs")
async def list_runs(manager: ManagerDep) -> list[dict[str, Any]]:
    return [run.overview() for run in manager.list()]


@app.post("/api/runs", status_code=202)
async def start_run(
    body: StartRun, manager: ManagerDep, settings: SettingsDep
) -> dict[str, Any]:
    run = manager.start(
        _sample_path(settings, body.sample), source=body.sample, parser=body.parser
    )
    return run.overview()


@app.post("/api/runs/upload", status_code=202)
async def upload_run(
    file: Annotated[UploadFile, File()],
    manager: ManagerDep,
    settings: SettingsDep,
    parser: Annotated[Parser | None, Form()] = None,
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
    return manager.start(path, source=name, parser=parser).overview()


@app.post("/api/runs/batch", status_code=202)
async def start_batch(
    manager: ManagerDep, settings: SettingsDep, body: StartBatch | None = None
) -> list[dict[str, Any]]:
    runs = await manager.start_batch(
        _samples(settings), parser=body.parser if body else None
    )
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


@app.post("/api/reset", status_code=204)
async def reset(
    manager: ManagerDep,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Start over: forget every run and every invoice received, so the
    samples can run again without being flagged as duplicates. With
    VP_TOKENS set, only a VP may do it."""
    _vp(settings, authorization)
    try:
        await manager.reset()
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


def mount_console(target: FastAPI, dist: Path) -> bool:
    """Serve the built console at /, when there is one. Mounted last, so every
    /api route above still wins. One server then does it all, without Node."""
    if not (dist / "index.html").is_file():
        return False
    target.mount("/", StaticFiles(directory=dist, html=True), name="console")
    return True


mount_console(app, get_settings().web_dist)
