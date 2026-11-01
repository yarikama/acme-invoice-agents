"""Run lifecycle for the API: start a graph run, record its events, resume it.

Every event a run emits is kept in order, so a client that connects late, or
reconnects with Last-Event-ID, gets the full history before live events.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from invoice_agents.agents import ExtractionUnavailableError
from invoice_agents.graph import InvoiceState
from invoice_agents.ingestion.ocr import OcrUnavailableError
from invoice_agents.models import HumanDecision
from invoice_agents.summary import Status, run_status, summarize

log = logging.getLogger(__name__)

TERMINAL: frozenset[Status] = frozenset({"paid", "rejected", "failed"})


class RunNotFoundError(KeyError):
    pass


class RunStateError(RuntimeError):
    """The run is not in a state that allows this action."""


@dataclass
class Run:
    id: str
    source: str
    path: Path
    created_at: str
    status: Status = "running"
    summary: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    # Set whenever the run stops moving: paused for the VP or finished.
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    _changed: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def config(self) -> RunnableConfig:
        return {"configurable": {"thread_id": self.id}}

    def overview(self) -> dict[str, Any]:
        invoice = (self.summary or {}).get("invoice") or {}
        return {
            "id": self.id,
            "source": self.source,
            "status": self.status,
            "created_at": self.created_at,
            "invoice_number": invoice.get("invoice_number"),
            "vendor": invoice.get("vendor"),
            "total": invoice.get("total"),
            "currency": invoice.get("currency"),
        }


class RunManager:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph
        self.runs: dict[str, Run] = {}
        # Every event from every run, in order: one stream for the whole console.
        self.log: list[dict[str, Any]] = []
        self._log_changed = asyncio.Condition()
        self._tasks: set[asyncio.Task[None]] = set()

    def get(self, run_id: str) -> Run:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise RunNotFoundError(run_id) from exc

    def list(self) -> list[Run]:
        return sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)

    def start(self, path: Path, source: str) -> Run:
        run = Run(
            id=uuid.uuid4().hex[:12],
            source=source,
            path=path,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.runs[run.id] = run
        self._spawn(self._drive(run, {"path": str(path)}))
        return run

    def decide(self, run_id: str, decision: HumanDecision) -> Run:
        run = self.get(run_id)
        if run.status != "awaiting_vp":
            raise RunStateError(f"Run {run_id} is {run.status}, not awaiting_vp")
        run.status = "running"
        run.settled.clear()
        self._spawn(
            self._drive(
                run,
                Command(resume=decision.model_dump()),
                first={"type": "decision", **decision.model_dump()},
            )
        )
        return run

    async def start_batch(self, paths: list[Path]) -> list[Run]:
        """Run samples one after another, so duplicate detection sees them in order.

        Returns as soon as the batch is queued; each run starts once the one
        before it settles (paid, rejected, failed, or paused for the VP).
        """
        runs = [
            Run(
                id=uuid.uuid4().hex[:12],
                source=p.name,
                path=p,
                created_at=datetime.now(UTC).isoformat(),
                status="queued",
            )
            for p in paths
        ]
        for run in runs:
            self.runs[run.id] = run

        async def drive_all() -> None:
            for run in runs:
                await self._drive(run, {"path": str(run.path)})

        self._spawn(drive_all())
        return runs

    async def events(
        self, run_id: str, after: int = -1
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield events with index > `after`, then live ones until the run ends."""
        run = self.get(run_id)
        cursor = after + 1
        while True:
            async with run._changed:
                await run._changed.wait_for(
                    lambda c=cursor: c < len(run.events) or run.status in TERMINAL
                )
                pending = run.events[cursor:]
            for event in pending:
                yield event
            cursor += len(pending)
            if run.status in TERMINAL and cursor >= len(run.events):
                return

    async def all_events(self, after: int = -1) -> AsyncIterator[dict[str, Any]]:
        """Every run's events with seq > `after`, then live ones. Never ends."""
        cursor = after + 1
        while True:
            async with self._log_changed:
                await self._log_changed.wait_for(lambda c=cursor: c < len(self.log))
                pending = self.log[cursor:]
            for event in pending:
                yield event
            cursor += len(pending)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # --- internals --------------------------------------------------------

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _emit(self, run: Run, event: dict[str, Any]) -> None:
        async with run._changed, self._log_changed:
            full = {
                "id": len(run.events),
                "seq": len(self.log),
                "run_id": run.id,
                "source": run.source,
                "created_at": run.created_at,
                # When this event happened, so the console can time each step.
                "at": datetime.now(UTC).isoformat(),
                "status": run.status,
                **event,
            }
            run.events.append(full)
            self.log.append(full)
            run._changed.notify_all()
            self._log_changed.notify_all()

    async def _drive(
        self, run: Run, payload: Any, first: dict[str, Any] | None = None
    ) -> None:
        run.status = "running"
        if first is None:
            await self._emit(run, {"type": "started", "source": run.source})
        else:
            await self._emit(run, first)

        request: dict[str, Any] | None = None
        try:
            async for mode, chunk in self.graph.astream(
                payload, run.config, stream_mode=["updates", "custom"]
            ):
                # With a list of stream modes, LangGraph yields (mode, chunk).
                chunk = cast(dict[str, Any], chunk)
                if mode == "custom":
                    # Renamed so it can't clobber the run's own `status`.
                    await self._emit(
                        run,
                        {
                            "type": "progress",
                            "stage": chunk["stage"],
                            "step": chunk["status"],
                        },
                    )
                    continue
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        request = update[0].value
                        continue
                    for item in (update or {}).get("trace", []):
                        await self._emit(run, {"type": "trace", "node": node, **item})
            snapshot = await self.graph.aget_state(run.config)
            state = cast(InvoiceState, snapshot.values)
            run.status = run_status(state, request is not None)
            run.summary = summarize(state, run.status)
            # Sent after the status is final, so clients can trust `status`.
            if request is not None:
                await self._emit(
                    run,
                    {"type": "awaiting_vp", "request": request, "summary": run.summary},
                )
            elif run.status in TERMINAL:
                await self._emit(run, {"type": "done", "summary": run.summary})
        except (ExtractionUnavailableError, OcrUnavailableError) as exc:
            await self._fail(run, str(exc))
        except Exception as exc:
            log.exception("Run %s failed", run.id)
            await self._fail(run, f"{type(exc).__name__}: {exc}")
        finally:
            run.settled.set()
            async with run._changed:
                run._changed.notify_all()

    async def _fail(self, run: Run, message: str) -> None:
        run.status = "failed"
        # Not "error": EventSource reserves that name for connection errors.
        await self._emit(run, {"type": "run_error", "message": message})
