"""Compare decision models as the triage step, on every sample invoice.

    make triage-eval
    make triage-eval ARGS="--models typesafe/jev-1.14,cloudflare/clef-flash"

Each case runs the pipeline up to the approver once (the extractor still
reads text and PDF invoices), then every model answers the same triage
questions. Two numbers matter:
- agreement: the model's decision matches the expected outcome;
- unsafe fast paths: triage settled an invoice on its own and got it wrong.
  The first is a quality signal; the second must be zero, since nothing
  checks a fast path after it is taken.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.table import Table

from invoice_agents.agents import ExtractionUnavailableError
from invoice_agents.approval import Route, resolve
from invoice_agents.config import Settings, get_settings
from invoice_agents.evaluation import CASES, Case, Outcome
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.inventory import Inventory
from invoice_agents.llm import build_agents
from invoice_agents.models import Decision, Invoice, ValidationReport
from invoice_agents.triage import DecisionTriager, TriageError, Triager, fast_path

MODELS = [
    "typesafe/jev-1.14",
    "cloudflare/clef-flash",
    "cloudflare/clef",
    "perplexity/pplx-decider-flash-v1.2",
    "liquid/d1",
    "microsoft/microsoft-decision-1",
]

AS_OUTCOME: dict[Decision, Outcome] = {
    "approve": "pay",
    "reject": "reject",
    "escalate": "human",
}


@dataclass(frozen=True)
class Prepared:
    case: Case
    invoice: Invoice
    report: ValidationReport
    route: Route


@dataclass
class Vote:
    case: str
    model: str
    expected: Outcome
    decision: Decision | None
    agrees: bool | None
    fast_path: Decision | None
    unsafe: bool
    confidence: float | None = None
    suspicion: float | None = None
    seconds: float | None = None
    cost_usd: float | None = None
    error: str = ""


def acceptable(case: Case, outcome: Outcome) -> bool:
    return outcome == case.expected or outcome in case.also_ok


async def prepare(case: Case, settings: Settings) -> Prepared | None:
    """Run the case up to the approver; None if it never gets there."""
    plain = settings.model_copy(update={"triage_model": None})
    with tempfile.TemporaryDirectory() as tmp:
        inventory = Inventory(Path(tmp) / "inventory.db")
        inventory.seed()
        deps = Deps(agents=build_agents(plain), inventory=inventory, settings=plain)
        graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))
        state: dict[str, Any] = {}
        try:
            last = case.files[-1]
            for name in case.files:
                config: RunnableConfig = {
                    "configurable": {"thread_id": uuid.uuid4().hex}
                }
                # Earlier files run to the end, so the ledger knows whether
                # they were paid; the last one stops before the approver.
                state = await graph.ainvoke(
                    {"path": str(settings.samples_dir / name)},
                    config,
                    interrupt_before=["review"] if name == last else None,
                )
        except ExtractionUnavailableError:
            return None
    if "report" not in state or state.get("invoice") is None:
        return None
    return Prepared(case, state["invoice"], state["report"], state["route"])


async def vote(p: Prepared, triager: Triager, settings: Settings) -> Vote:
    case = p.case
    try:
        answer = await triager.ask(p.invoice, p.report)
    except TriageError as exc:
        return Vote(
            case=case.name,
            model=triager.model,
            expected=case.expected,
            decision=None,
            agrees=None,
            fast_path=None,
            unsafe=False,
            error=str(exc),
        )
    quick = fast_path(
        answer, p.report, settings.triage_min_confidence, settings.triage_max_suspicion
    )
    fast = quick[0] if quick else None
    unsafe = fast is not None and not acceptable(case, resolve(fast, p.route))
    return Vote(
        case=case.name,
        model=triager.model,
        expected=case.expected,
        decision=answer.decision,
        agrees=acceptable(case, AS_OUTCOME[answer.decision]),
        fast_path=fast,
        unsafe=unsafe,
        confidence=answer.confidence,
        suspicion=answer.suspicion,
        seconds=answer.seconds,
        cost_usd=answer.cost_usd,
    )


def scoreboard(votes: list[Vote], models: list[str]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        mine = [v for v in votes if v.model == model]
        answered = [v for v in mine if v.decision is not None]
        seconds = [v.seconds for v in answered if v.seconds is not None]
        costs = [v.cost_usd for v in answered if v.cost_usd is not None]
        rows.append(
            {
                "model": model,
                "answered": len(answered),
                "failed": len(mine) - len(answered),
                "agreement": round(
                    sum(1 for v in answered if v.agrees) / len(answered), 3
                )
                if answered
                else None,
                "fast_paths": sum(1 for v in answered if v.fast_path),
                "unsafe_fast_paths": sum(1 for v in answered if v.unsafe),
                "mean_seconds": round(sum(seconds) / len(seconds), 3)
                if seconds
                else None,
                "cost_per_invoice_usd": sum(costs) / len(costs) if costs else None,
            }
        )
    return rows


def render(rows: list[dict[str, Any]], votes: list[Vote], console: Console) -> None:
    table = Table(pad_edge=False)
    for col in ("model", "agree", "fast", "unsafe", "seconds", "per invoice"):
        table.add_column(col, justify="left" if col == "model" else "right")
    for r in rows:
        agree = "-" if r["agreement"] is None else f"{r['agreement']:.0%}"
        if r["failed"]:
            agree += f" ({r['failed']} failed)"
        table.add_row(
            r["model"],
            agree,
            str(r["fast_paths"]),
            f"[red]{r['unsafe_fast_paths']}[/]" if r["unsafe_fast_paths"] else "0",
            "-" if r["mean_seconds"] is None else f"{r['mean_seconds']:.2f}",
            "-"
            if r["cost_per_invoice_usd"] is None
            else f"${r['cost_per_invoice_usd']:.6f}",
        )
    console.print(table)
    for v in votes:
        if v.unsafe:
            console.print(
                f"[red]UNSAFE[/] {v.model} on {v.case}: fast {v.fast_path},"
                f" expected {v.expected}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare decision models.")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--only", help="Run cases whose file names contain this")
    parser.add_argument("--out", type=Path, default=Path("eval-results"))
    args = parser.parse_args(argv)

    # Scoring replays answers at once, not at the pace they were recorded.
    settings = get_settings().model_copy(update={"replay_pace": 0.0})
    if settings.openrouter_api_key is None:
        print("triage-eval needs OPENROUTER_API_KEY", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = [c for c in CASES if not args.only or args.only in c.name]
    console = Console()
    console.print(f"{len(models)} models x {len(cases)} cases.")

    async def go() -> list[Vote]:
        gate = asyncio.Semaphore(2)  # extraction shares the LLM rate limit

        async def prep(case: Case) -> Prepared | None:
            async with gate:
                return await prepare(case, settings)

        prepared = [p for p in await asyncio.gather(*map(prep, cases)) if p]
        key = settings.openrouter_api_key
        assert key is not None
        triagers = [DecisionTriager(m, key) for m in models]
        calls = asyncio.Semaphore(8)

        async def one(p: Prepared, t: Triager) -> Vote:
            async with calls:
                return await vote(p, t, settings)

        return list(
            await asyncio.gather(*(one(p, t) for p in prepared for t in triagers))
        )

    votes = asyncio.run(go())
    rows = scoreboard(votes, models)
    render(rows, votes, console)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"triage-{stamp}.json"
    path.write_text(
        json.dumps({"models": rows, "votes": [asdict(v) for v in votes]}, indent=2)
    )
    console.print(f"Saved {path}")
    return 1 if any(r["unsafe_fast_paths"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
