"""Command line entry point: process one invoice or a folder of them.

    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --dir data/invoices
    python main.py --serve            # start the API and the web console

If the API is running, invoices go through it, so each run also appears in
the web console; otherwise the graph runs here. Either way the terminal shows
the run live: the pipeline lights, each agent's turn and tool calls, and the
VP's decision when one is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler

from invoice_agents import remote
from invoice_agents.agents import ExtractionUnavailableError
from invoice_agents.config import Settings, get_settings
from invoice_agents.graph import Deps, InvoiceState, build_graph, checkpoint_serde
from invoice_agents.ingestion.ocr import OcrUnavailableError
from invoice_agents.inventory import Inventory
from invoice_agents.llm import build_agents
from invoice_agents.summary import run_status, summarize
from invoice_agents.terminal import RunView, banner, batch_summary, result_line, vp_card

console = Console(stderr=True)

DEFAULT_API = "http://localhost:8000"
DEFAULT_WEB = "http://localhost:5173"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process invoices end to end.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--invoice_path", "--invoice-path", type=Path, help="One invoice file"
    )
    target.add_argument("--dir", type=Path, help="Process every invoice in a folder")
    parser.add_argument(
        "--vp",
        choices=["ask", "approve", "reject"],
        default="ask",
        help="How to answer when an invoice needs a human VP (default: ask)",
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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not (args.invoice_path or args.dir or args.serve):
        parser.error("give --invoice_path, --dir or --serve")
    return args


def _decide(summary: dict[str, Any], mode: str) -> tuple[bool, str]:
    """Show the VP card and get an answer: from the flag, or by asking."""
    console.line()
    console.print(vp_card(summary))
    if mode != "ask":
        console.print(f"  Answered by --vp {mode}.\n")
        return mode == "approve", f"--vp {mode}"
    answer = console.input("  Approve and pay? [y/N] ").strip().lower()
    note = (
        console.input("  Note for the audit log (optional): ").strip() if answer else ""
    )
    console.print()
    return answer in ("y", "yes"), note or "decided in the CLI"


class Runner(Protocol):
    async def process(self, path: Path, vp_mode: str) -> dict[str, Any]: ...


class LocalRunner:
    """Runs the graph in this process."""

    def __init__(self, settings: Settings) -> None:
        inventory = Inventory(settings.db_path)
        inventory.seed()
        deps = Deps(
            agents=build_agents(settings), inventory=inventory, settings=settings
        )
        self.graph = build_graph(
            deps, checkpointer=InMemorySaver(serde=checkpoint_serde())
        )

    async def process(self, path: Path, vp_mode: str) -> dict[str, Any]:
        config: Any = {"configurable": {"thread_id": str(uuid.uuid4())}}
        payload: Any = {"path": str(path)}
        view = RunView(path.name)
        while True:
            paused = None
            with Live(view, console=console, refresh_per_second=12, transient=False):
                async for mode, raw in self.graph.astream(
                    payload, config, stream_mode=["updates", "custom"]
                ):
                    chunk = cast(dict[str, Any], raw)
                    if mode == "custom":
                        view.add(
                            {
                                "type": "progress",
                                "stage": chunk["stage"],
                                "step": chunk["status"],
                            }
                        )
                        continue
                    for node, update in chunk.items():
                        if node == "__interrupt__":
                            paused = update[0].value
                            continue
                        for t in (update or {}).get("trace", []):
                            view.add(
                                {
                                    "type": "trace",
                                    "node": node,
                                    "message": t["message"],
                                    "tool": t.get("tool"),
                                    "usage": t.get("usage"),
                                }
                            )
                state = cast(InvoiceState, (await self.graph.aget_state(config)).values)
                view.status = run_status(state, paused is not None)
            if paused is None:
                break
            approved, note = _decide(summarize(state, "awaiting_vp"), vp_mode)
            payload = Command(resume={"approved": approved, "note": note})
            view = view.resumed()
        return {"file": path.name, **summarize(state, run_status(state, False))}


class RemoteRunner:
    """Runs invoices through the API, so they show in the web console too."""

    def __init__(self, api: remote.Api) -> None:
        self.api = api

    async def process(self, path: Path, vp_mode: str) -> dict[str, Any]:
        run_id = await self.api.start(path)
        view = RunView(path.name)
        live = Live(view, console=console, refresh_per_second=12, transient=False)
        live.start()
        try:
            async for event in self.api.events(run_id):
                if event["type"] == "started" and event.get("at"):
                    view.started = event["at"]
                if event["type"] in ("trace", "progress"):
                    view.add(event)
                if event["type"] == "awaiting_vp":
                    view.status = "awaiting_vp"
                    live.stop()
                    approved, note = _decide(event["summary"], vp_mode)
                    view = view.resumed()
                    live = Live(
                        view, console=console, refresh_per_second=12, transient=False
                    )
                    live.start()
                    await self.api.decide(run_id, approved, note)
                if event["type"] in ("done", "run_error"):
                    view.status = event.get("status", "failed")
        finally:
            live.stop()
        return {"file": path.name, **(await self.api.result(run_id))}


async def _process_all(
    runner: Runner, paths: list[Path], vp_mode: str
) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        console.rule(f"[bold]{path.name}", style="#3a3a37")
        try:
            result = await runner.process(path, vp_mode)
        except (ExtractionUnavailableError, OcrUnavailableError) as exc:
            result = {"file": path.name, "status": "skipped", "reason": str(exc)}
        except Exception as exc:
            # One bad invoice or API error must not stop the batch.
            logging.getLogger(__name__).debug("Run failed", exc_info=True)
            result = {
                "file": path.name,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        # A live view can end mid-line when output isn't a terminal.
        console.line()
        console.print(result_line(path.name, result))
        console.print()
        results.append(result)
    return results


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    started: list[Any] = []
    try:
        if args.serve:
            with console.status("Starting the API and the web console…"):
                started = await remote.serve(args.api, args.web, Path(".acme-logs"))
        async with httpx.AsyncClient() as client:
            api_up = not args.local and await remote.reachable(
                f"{args.api}/api/health", client
            )
            web_up = await remote.reachable(args.web, client)
            if api_up:
                mode = (
                    f"Runs go through the API at {args.api},"
                    " so each one also appears in the console."
                )
                runner: Runner = RemoteRunner(remote.Api(args.api, client))
            else:
                mode = f"Running here with the {settings.llm_provider} provider."
                runner = LocalRunner(settings)
            console.print(banner(args.web, live=web_up, mirrored=api_up, mode=mode))

            paths = (
                [args.invoice_path]
                if args.invoice_path
                else sorted(args.dir.iterdir())
                if args.dir
                else []
            )
            paths = [p for p in paths if p.is_file() and not p.name.startswith(".")]
            if not paths:
                console.print("  Serving. Press Ctrl-C to stop.")
                await asyncio.Event().wait()
                return 0

            t0 = time.perf_counter()
            results = await _process_all(runner, paths, args.vp)
            if len(results) > 1:
                console.print(batch_summary(results, time.perf_counter() - t0))
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            return 0
    finally:
        remote.stop(started)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        console.print("\n  Stopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
