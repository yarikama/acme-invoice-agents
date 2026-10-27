"""Command line entry point: process one invoice or a folder of them."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.logging import RichHandler

from invoice_agents.agents import ExtractionUnavailableError
from invoice_agents.config import get_settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.inventory import Inventory
from invoice_agents.llm import build_agents

console = Console(stderr=True)

STAGE_STYLE = {
    "ingest": "cyan",
    "validate": "blue",
    "approve": "magenta",
    "pay": "green",
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process invoices end to end.")
    target = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _ask_vp(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode != "ask":
        return {"approved": mode == "approve", "note": f"--vp {mode}"}
    invoice = payload["invoice"]
    console.print(
        f"\n[bold yellow]VP review needed[/] {invoice['invoice_number']}"
        f" · {invoice['vendor']} · {invoice['currency']} {invoice['total']:,.2f}"
    )
    console.print(f"  Agent recommends: [bold]{payload['draft']['decision']}[/]")
    console.print(f"  {payload['draft']['rationale']}")
    answer = console.input("  Approve? [y/N] ").strip().lower()
    return {"approved": answer in ("y", "yes"), "note": "decided in the CLI"}


async def _process(graph: Any, path: Path, vp_mode: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload: Any = {"path": str(path)}
    console.rule(f"[bold]{path.name}")
    while True:
        interrupted = None
        async for mode, chunk in graph.astream(
            payload, config, stream_mode=["updates", "custom"]
        ):
            if mode != "updates":
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupted = update[0].value
                    continue
                for event in (update or {}).get("trace", []):
                    style = STAGE_STYLE.get(event["stage"], "white")
                    console.print(
                        f"  [{style}]{event['stage']:>8}[/] {event['message']}"
                    )
        if interrupted is None:
            break
        payload = Command(resume=_ask_vp(interrupted, vp_mode))

    state = (await graph.aget_state(config)).values
    return _summary(path, state)


def _summary(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    invoice = state.get("invoice")
    report = state.get("report")
    return {
        "file": path.name,
        "invoice_number": invoice.invoice_number if invoice else None,
        "vendor": invoice.vendor if invoice else None,
        "total": invoice.total if invoice else None,
        "currency": invoice.currency if invoice else None,
        "status": "paid" if state.get("payment") else "rejected",
        "reason": state.get("rejection_reason"),
        "findings": [f.model_dump(mode="json") for f in report.findings]
        if report
        else [],
        "trace": state.get("trace", []),
    }


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    inventory = Inventory(settings.db_path)
    inventory.seed()
    deps = Deps(agents=build_agents(settings), inventory=inventory, settings=settings)
    graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))

    paths = [args.invoice_path] if args.invoice_path else sorted(args.dir.iterdir())
    results = []
    for path in paths:
        if path.name.startswith(".") or not path.is_file():
            continue
        try:
            results.append(await _process(graph, path, args.vp))
        except ExtractionUnavailableError as exc:
            console.print(f"  [yellow]skipped[/] {exc}")
            results.append({"file": path.name, "status": "skipped", "reason": str(exc)})

    for r in results:
        color = {"paid": "green", "rejected": "red"}.get(r["status"], "yellow")
        console.print(
            f"[{color}]{r['status']:>8}[/] {r['file']}  {r.get('reason') or ''}"
        )
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
