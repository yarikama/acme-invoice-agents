"""The assistant in the web console, sharing one conversation with the terminal.

A request typed in the console is planned by the same assistant as in the
terminal, with the same shared history (memory.py), and its commands run
here in the API: runs start through the RunManager, so they appear in the
inbox like any other. The commands the assistant may use are the terminal's,
narrowed to what a browser should reach: files only from the sample folders,
no eval, nothing that approves, rejects or pays.

When the runs a request started settle, what came of them joins the
conversation, so "why was that one rejected?" has an answer in either
console.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from invoice_agents import assistant, memory
from invoice_agents.api.runs import Run, RunManager
from invoice_agents.config import Settings
from invoice_agents.inventory import Inventory
from invoice_agents.models import SourceFormat

log = logging.getLogger(__name__)

SUFFIXES = {f".{f.value}" for f in SourceFormat}
LIVE = ("openrouter", "xai")


class ChatUnavailableError(RuntimeError):
    pass


class ChatService:
    def __init__(
        self,
        settings: Settings,
        manager: RunManager,
        llm: Callable[[], BaseChatModel],
    ) -> None:
        self.settings = settings
        self.manager = manager
        self._llm_factory = llm
        self._llm: BaseChatModel | None = None
        self.history, self.shared = memory.history(settings)
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def available(self) -> bool:
        return self.settings.llm_provider in LIVE

    def messages(self, after: int = -1) -> list[dict[str, Any]]:
        all_ = self.history.messages
        start = max(after + 1, len(all_) - memory.SHOWN)
        return [memory.as_json(i, m) for i, m in enumerate(all_) if i >= start]

    async def say(self, text: str) -> list[dict[str, Any]]:
        """Plan the request with the conversation so far, run what code lets
        through, and return the messages it added."""
        if not self.available:
            raise ChatUnavailableError(
                "The assistant needs a live model: set LLM_PROVIDER=openrouter"
                " (or xai) and its API key."
            )
        # One request at a time, so its plan and results stay together.
        async with self._lock:
            before = len(self.history.messages)
            if self._llm is None:
                self._llm = self._llm_factory()
            plan, _ = await assistant.plan(self._llm, text, self.history, "web")
            steps, dropped = assistant.checked(plan)
            notes = [*dropped]
            started: list[Run] = []
            for step in steps:
                note, runs = self._run(step)
                notes.append(note)
                started += runs
            if notes:
                self.history.add_message(memory.answered(" ".join(notes), "web"))
            if started:
                self._follow(started)
            return self.messages(before - 1)

    # --- commands -----------------------------------------------------------

    def _run(self, step: assistant.Step) -> tuple[str, list[Run]]:
        name, args = step.command, step.args
        if name == "run":
            paths = [p for a in args if (p := self._find(a)) is not None]
            missing = [a for a in args if self._find(a) is None]
            runs = [self.manager.start(p, source=p.name) for p in paths]
            note = f"Started {_count(len(runs), 'run')}."
            if missing:
                note += f" Not found in the sample folders: {', '.join(missing)}."
            return note, runs
        if name == "dir":
            folder = self._folder(args[0] if args else "invoices")
            if folder is None:
                return f"{args[0]} isn't one of the sample folders.", []
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES)
            runs = [self.manager.start(p, source=p.name) for p in files]
            return f"Started {_count(len(runs), 'run')} from {folder.name}/.", runs
        if name == "runs":
            return self._runs_note(), []
        if name == "inventory":
            records = Inventory(self.settings.db_path).records()
            listed = ", ".join(
                f"{r.item} {r.stock} in stock at ${r.unit_price:,.2f}" for r in records
            )
            return f"Inventory: {listed}.", []
        if name == "status":
            s = self.settings
            return (
                f"Provider {s.llm_provider}; extraction {s.extract_model}, approver"
                f" {s.review_model}, critic {s.critic_model or s.review_model}.",
                [],
            )
        if name == "help":
            return (
                "Ask in plain words to run invoices from the sample folders, list"
                " the runs, or show the inventory or status. Approving and"
                " rejecting are for a VP, in the decision panel.",
                [],
            )
        # eval and open belong to the terminal.
        return f"/{name} runs in the terminal, not from the web console.", []

    def _folders(self) -> list[Path]:
        root = self.settings.samples_dir.parent
        return [root / Path(f).name for f in assistant.FOLDERS]

    def _folder(self, name: str) -> Path | None:
        wanted = Path(name).name or "invoices"
        return next(
            (f for f in self._folders() if f.name == wanted and f.is_dir()), None
        )

    def _find(self, arg: str) -> Path | None:
        """A file from the sample folders only, by name or folder/name."""
        given = Path(arg)
        for folder in self._folders():
            if given.parent.name and given.parent.name != folder.name:
                continue
            path = folder / given.name
            if path.is_file() and path.suffix.lower() in SUFFIXES:
                return path
        return None

    def _runs_note(self) -> str:
        runs = self.manager.list()
        if not runs:
            return "No runs yet."
        by_status: dict[str, int] = {}
        for r in runs:
            by_status[r.status] = by_status.get(r.status, 0) + 1
        counts = ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in by_status.items())
        waiting = [
            (r.summary or {}).get("invoice", {}) or {}
            for r in runs
            if r.status == "awaiting_vp"
        ]
        names = [w.get("invoice_number") or "an invoice" for w in waiting]
        note = f"{_count(len(runs), 'run')}: {counts}."
        if names:
            note += f" Waiting for a VP: {', '.join(names)}."
        return note

    # --- outcomes -------------------------------------------------------------

    def _follow(self, runs: list[Run]) -> None:
        async def wait_then_tell() -> None:
            await asyncio.gather(*(r.settled.wait() for r in runs))
            results = [
                {**(r.summary or {}), "status": r.status, "file": r.source}
                for r in runs
            ]
            if text := memory.outcomes(results):
                self.history.add_message(memory.answered(text, "web"))

        task = asyncio.create_task(wait_then_tell())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"
