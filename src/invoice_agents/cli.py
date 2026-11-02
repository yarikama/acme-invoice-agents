"""Command line entry point: process one invoice or a folder of them.

    python main.py                    # the interactive console (see shell.py)
    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --dir data/invoices
    python main.py --serve            # start the API and the web console

A file or folder runs, prints its result and exits, as the brief's command
expects; add -i to stay in the interactive console afterwards, as if it had
been typed as /run. With no file or folder, the console opens.

If the API is running, invoices go through it, so each run also appears in
the web console; otherwise the graph runs here. Either way the terminal shows
the run live: the pipeline lights, each agent's turn and tool calls, and the
VP's decision when one is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape

from invoice_agents import logs, remote
from invoice_agents.agents import ExtractionUnavailableError
from invoice_agents.config import Settings, get_settings
from invoice_agents.graph import (
    InvoiceState,
    build_graph,
    checkpoint_serde,
    run_events,
    stored_deps,
)
from invoice_agents.inbox import Inbox, watch
from invoice_agents.ingestion import UnsupportedFormatError
from invoice_agents.ingestion.ocr import PARSERS, OcrUnavailableError
from invoice_agents.ledger import Ledger
from invoice_agents.llm import load_pricing
from invoice_agents.payment import MockBank
from invoice_agents.summary import run_status, summarize
from invoice_agents.terminal import RunView, banner, batch_summary, result_card, vp_card

console = Console(stderr=True)

# Where the API and the console are; the Docker setup points these elsewhere.
DEFAULT_API = os.environ.get("ACME_API_URL", "http://localhost:8000")
DEFAULT_WEB = os.environ.get("ACME_WEB_URL", "http://localhost:5173")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process invoices end to end.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--invoice_path", "--invoice-path", type=Path, help="One invoice file"
    )
    target.add_argument("--dir", type=Path, help="Process every invoice in a folder")
    target.add_argument(
        "--watch",
        type=Path,
        nargs="?",
        const=Path("data/inbox"),
        metavar="FOLDER",
        help="Process invoices as they arrive in a folder, like a mailbox"
        " (default data/inbox); handled files move to processed/",
    )
    parser.add_argument(
        "--vp",
        choices=["ask", "approve", "reject", "wait"],
        default=None,
        help="How to answer when an invoice needs a human VP: ask here, approve,"
        " reject, or wait for a VP in the web console (default: ask; wait with"
        " --watch)",
    )
    parser.add_argument(
        "--parser",
        choices=PARSERS,
        default=None,
        help="What reads a scanned PDF or photo: pymupdf (Tesseract on the whole"
        " page, fast), docling (layout model, Markdown tables) or grok (Grok"
        " reads the image; needs a key). Default: OCR_BACKEND, else pymupdf",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print results as JSON to stdout"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Start the API and the web console"
    )
    parser.add_argument(
        "--local", action="store_true", help="Run here even if the API is up"
    )
    parser.add_argument(
        "--api", default=DEFAULT_API, help=f"API address (default {DEFAULT_API})"
    )
    parser.add_argument(
        "--web", default=DEFAULT_WEB, help=f"Console address (default {DEFAULT_WEB})"
    )
    stay = parser.add_mutually_exclusive_group()
    stay.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="After running the file or folder, stay in the interactive console",
    )
    stay.add_argument(
        "--once",
        action="store_true",
        help="Run the file or folder and exit (the default; kept for scripts)",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="In the interactive console, don't start the API and web console",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Forget every invoice received so far, then run the target if any",
    )
    parser.add_argument(
        "--log-json",
        type=Path,
        metavar="FILE",
        help="Also write each step to FILE as JSON lines (run id, stage, usage)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _interactive(args: argparse.Namespace) -> bool:
    """Whether to open the console, before or after running a file.

    A file or folder runs and exits, like the brief's command, unless -i asks
    to stay. With no target, a person gets the console; `--serve` alone keeps
    serving without a prompt off a terminal."""
    if args.interactive:
        return True
    if args.once or args.json or args.watch or args.invoice_path or args.dir:
        return False
    return not args.serve or sys.stdin.isatty()


@contextlib.contextmanager
def _signals_end_the_cli() -> Iterator[None]:
    """Closing the terminal (SIGHUP) or `kill` (SIGTERM) ends the CLI the way
    Ctrl-C does, so the cleanup runs and stops the servers it started: they
    run in their own session (see remote._spawn) and don't get the signal."""

    def stop(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    wanted = [signal.SIGTERM, getattr(signal, "SIGHUP", None)]
    previous = {sig: signal.signal(sig, stop) for sig in wanted if sig is not None}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# Said whenever "auto" found no key: what the viewer sees is a recording.
NO_KEY = (
    " No API key found, so this replays a recorded live run of the samples."
    " Set XAI_API_KEY (or OPENROUTER_API_KEY) in .env to run it live."
)

# A run in this process ends with it, so it can't wait for a VP; one in
# the API can (checkpoints), so it waits for the web console.
CANT_WAIT = (
    "Not decided: this run lives in this process, so it can't wait for a VP."
    " Run it again with --vp approve or --vp reject, or with the API up"
    " (--serve) to decide it in the web console."
)
WAITS = "Waiting for a VP: decide in the web console, or /approve or /reject."


def _decide(
    summary: dict[str, Any], mode: str, can_wait: bool = True
) -> tuple[bool, str] | None:
    """Show the VP card and get an answer: from the flag, or by asking.

    None leaves the invoice for a VP: when told to wait, when input isn't a
    terminal (a script, CI), or when the person closed the input.
    """
    console.line()
    console.print(vp_card(summary))
    unanswered = f"  {WAITS if can_wait else CANT_WAIT}\n"
    if mode == "wait":
        console.print(unanswered)
        return None
    if mode != "ask":
        console.print(f"  Answered by --vp {mode}.\n")
        return mode == "approve", f"--vp {mode}"
    if not sys.stdin.isatty():
        console.print("  No one to ask: input isn't a terminal.")
        console.print(unanswered)
        return None
    try:
        answer = console.input("  Approve and pay? [y/N] ").strip().lower()
        note = (
            console.input("  Note for the audit log (optional): ").strip()
            if answer
            else ""
        )
    except EOFError:
        console.print("\n  No answer.")
        console.print(unanswered)
        return None
    console.print()
    return answer in ("y", "yes"), note or "decided in the CLI"


def _print_finished(view: RunView) -> None:
    """Print turns as they finish. While a Live view is up, Rich puts them
    above it, so only the short status below is redrawn."""
    for turn in view.take():
        console.print(turn)


class Runner(Protocol):
    # The parser for scans, or None for the configured default.
    parser: str | None

    async def process(self, path: Path, vp_mode: str) -> dict[str, Any]: ...


class LocalRunner:
    """Runs the graph in this process."""

    def __init__(self, settings: Settings) -> None:
        self.parser: str | None = None
        # Prices for each step's cost, as the API shows them.
        deps = stored_deps(settings, load_pricing(settings))
        self.ledger = deps.seen
        self.graph = build_graph(
            deps, checkpointer=InMemorySaver(serde=checkpoint_serde())
        )

    async def process(self, path: Path, vp_mode: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        try:
            result = await self._process(run_id, path, vp_mode)
        except Exception as exc:
            self.ledger.settle(run_id, "failed", f"{type(exc).__name__}: {exc}")
            raise
        except BaseException:
            # Ctrl-C or a closed terminal: the run stops here, so free the
            # invoice number rather than leave it in flight as a duplicate.
            self.ledger.settle(run_id, "dropped", "Stopped before it finished.")
            raise
        if result["status"] == "awaiting_vp":
            # This process's checkpoints end with it, so nothing can resume
            # the run: free the invoice number for a later run, and say so
            # rather than claim it waits.
            self.ledger.settle(run_id, "dropped", "Left waiting for a VP in the CLI.")
            result = {**result, "status": "skipped", "reason": CANT_WAIT}
        return result

    async def _process(self, run_id: str, path: Path, vp_mode: str) -> dict[str, Any]:
        config: Any = {"configurable": {"thread_id": run_id}}
        payload: Any = {"path": str(path)}
        if self.parser:
            payload["parser"] = self.parser
        view = RunView(path.name)
        while True:
            paused = None
            with Live(view, console=console, refresh_per_second=12, transient=False):
                async for event in run_events(self.graph, payload, config):
                    if event["type"] == "interrupt":
                        paused = event["request"]
                    else:
                        view.add(event)
                    _print_finished(view)
                state = cast(InvoiceState, (await self.graph.aget_state(config)).values)
                view.status = run_status(state, paused is not None)
                _print_finished(view)
            if paused is None:
                break
            answer = _decide(summarize(state, "awaiting_vp"), vp_mode, can_wait=False)
            if answer is None:
                return {"file": path.name, **summarize(state, "awaiting_vp")}
            approved, note = answer
            payload = Command(resume={"approved": approved, "note": note})
            view = view.resumed()
        return {"file": path.name, **summarize(state, run_status(state, False))}


class RemoteRunner:
    """Runs invoices through the API, so they show in the web console too."""

    def __init__(self, api: remote.Api) -> None:
        self.api = api
        self.parser: str | None = None

    async def process(self, path: Path, vp_mode: str) -> dict[str, Any]:
        run_id = await self.api.start(path, self.parser)
        view = RunView(path.name)
        live = Live(view, console=console, refresh_per_second=12, transient=False)
        live.start()
        try:
            async for event in self.api.events(run_id):
                if event["type"] == "started" and event.get("at"):
                    view.started = event["at"]
                if event["type"] in ("trace", "progress"):
                    view.add(event)
                    _print_finished(view)
                if event["type"] == "awaiting_vp":
                    view.status = "awaiting_vp"
                    _print_finished(view)
                    live.stop()
                    answer = _decide(event["summary"], vp_mode)
                    if answer is None:
                        # The run stays paused in the API, for the console.
                        break
                    approved, note = answer
                    view = view.resumed()
                    live = Live(
                        view, console=console, refresh_per_second=12, transient=False
                    )
                    live.start()
                    await self.api.decide(run_id, approved, note)
                if event["type"] in ("done", "run_error"):
                    view.status = event.get("status", "failed")
                    _print_finished(view)
        finally:
            live.stop()
        return {"file": path.name, **(await self.api.result(run_id))}


async def _process_all(
    runner: Runner, paths: list[Path], vp_mode: str
) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        console.rule(f"[bold]{escape(path.name)}", style="#3a3a37")
        started = time.perf_counter()
        result: dict[str, Any]
        try:
            result = await runner.process(path, vp_mode)
        except (
            ExtractionUnavailableError,
            OcrUnavailableError,
            UnsupportedFormatError,
        ) as exc:
            # Not an invoice this system reads, or not without a model: no
            # outcome, but nothing broke either.
            result = {"file": path.name, "status": "skipped", "reason": str(exc)}
        except Exception as exc:
            # One bad invoice or API error must not stop the batch.
            logging.getLogger(__name__).debug("Run failed", exc_info=True)
            result = {
                "file": path.name,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        result["seconds"] = round(time.perf_counter() - started, 2)
        # A live view can end mid-line when output isn't a terminal.
        console.line()
        console.print(result_card(path.name, result))
        console.print()
        results.append(result)
    return results


@dataclass
class Connection:
    """Where runs go: through the API when it is up, otherwise in this process."""

    runner: Runner
    api_up: bool
    web_up: bool
    mode: str

    def banner(self, web: str, start: str = "python main.py --serve") -> Any:
        return banner(
            web, live=self.web_up, mirrored=self.api_up, mode=self.mode, start=start
        )


async def connect(
    args: argparse.Namespace,
    client: httpx.AsyncClient,
    local: LocalRunner | None = None,
) -> Connection:
    settings = get_settings()
    api_up = not args.local and await remote.api_up(args.api, client)
    web_up = await remote.reachable(args.web, client)
    if api_up:
        mode = (
            f"Runs go through the API at {args.api},"
            " so each one also appears in the console."
        )
        runner: Runner = RemoteRunner(remote.Api(args.api, client))
    else:
        mode = f"Running here with the {settings.llm_provider} provider."
        if settings.replay_for_lack_of_key:
            mode += NO_KEY
        runner = local or LocalRunner(settings)
    runner.parser = getattr(args, "parser", None)
    return Connection(runner, api_up, web_up, mode)


def invoice_files(target: Path) -> list[Path]:
    """The file itself, or a folder's visible files in name order."""
    paths = sorted(target.iterdir()) if target.is_dir() else [target]
    return [p for p in paths if p.is_file() and not p.name.startswith(".")]


async def process(
    runner: Runner, paths: list[Path], vp_mode: str
) -> list[dict[str, Any]]:
    """Run the files and, for more than one, print the batch summary."""
    t0 = time.perf_counter()
    results = await _process_all(runner, paths, vp_mode)
    if len(results) > 1:
        replayed = get_settings().llm_provider == "replay"
        console.print(batch_summary(results, time.perf_counter() - t0, replayed))
    return results


async def _watch(runner: Runner, folder: Path, vp_mode: str, conn: Connection) -> int:
    """Process invoices as they land in the folder, until Ctrl-C."""
    box = Inbox(folder)
    where = "the API, so the web console shows them" if conn.api_up else "this process"
    console.print(
        f"  Watching {escape(str(folder))} for invoices; they run through {where}."
        f" Handled files move to {escape(str(box.done))}. Ctrl-C to stop."
    )
    if vp_mode == "wait" and not conn.api_up:
        console.print(
            "  [bold]No API running:[/] invoices that need a VP can't wait for one"
            " here. Start with --serve to decide them in the web console."
        )

    async def handle(batch: list[Path]) -> set[Path]:
        results = await process(runner, batch, vp_mode)
        # Not decided (no VP could wait): keep it out of processed/.
        return {
            p for p, r in zip(batch, results, strict=True) if r["status"] == "skipped"
        }

    await watch(box, handle)
    return 0


async def _run(args: argparse.Namespace) -> int:
    started: list[Any] = []
    try:
        if args.serve:
            with console.status("Starting the API and the web console…"):
                started = await remote.serve(args.api, args.web, Path(".acme-logs"))
        async with httpx.AsyncClient() as client:
            conn = await connect(args, client)
            console.print(conn.banner(args.web))
            runner = conn.runner

            if args.watch:
                return await _watch(runner, args.watch, args.vp or "wait", conn)
            target = args.invoice_path or args.dir
            if target is None:
                console.print("  Serving. Press Ctrl-C to stop.")
                await asyncio.Event().wait()
                return 0
            paths = invoice_files(target) if target.exists() else []
            if not paths:
                console.print(f"  [bold]No invoice files at {escape(str(target))}.[/]")
                return 2

            results = await process(runner, paths, args.vp or "ask")
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            # A rejected invoice is a result; a run that broke is not.
            return 1 if any(r.get("status") == "failed" for r in results) else 0
    finally:
        remote.stop(started)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    if args.reset:
        settings = get_settings()
        Ledger(settings.db_path, settings.decisions_log).clear()
        MockBank(settings.db_path).clear()
        console.print(f"  Started over: {settings.db_path} holds no received invoices.")
        if not (
            args.invoice_path
            or args.dir
            or args.watch
            or args.serve
            or args.interactive
        ):
            return 0
    if args.verbose:
        # This app's steps in detail; HTTP clients stay at warnings, or their
        # connection chatter buries them.
        logging.getLogger("invoice_agents").setLevel(logging.DEBUG)
    if args.log_json:
        # The JSON file gets every step; the terminal keeps to warnings
        # (or debug with -v).
        logs.to_json_file(args.log_json)
        for handler in logging.getLogger().handlers:
            if not args.verbose:
                handler.setLevel(logging.WARNING)
    with _signals_end_the_cli():
        if _interactive(args):
            from invoice_agents.shell import run_shell

            return run_shell(args)
        try:
            return asyncio.run(_run(args))
        except KeyboardInterrupt:
            console.print("\n  Stopped.")
            return 130


if __name__ == "__main__":
    sys.exit(main())
