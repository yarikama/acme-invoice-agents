"""Score the pipeline against the expected outcome of every sample invoice.

    make eval                         # provider from .env
    LLM_PROVIDER=offline make eval    # rule-based agents, no network
    make eval ARGS="--triage typesafe/jev-1.14"   # with a System 1 triage step

Each case runs with a fresh inventory and invoice history, so cases don't
affect each other. A case lists several files when an invoice only makes
sense after an earlier one (a revision); only the last file is scored.
Cost comes from the token counts the provider reports and OpenRouter's
published prices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.table import Table

from invoice_agents.agents import ExtractionUnavailableError
from invoice_agents.config import Settings, get_settings
from invoice_agents.graph import Deps, InvoiceState, build_graph, checkpoint_serde
from invoice_agents.inventory import Inventory
from invoice_agents.llm import build_agents

Outcome = Literal["pay", "reject", "human", "skipped", "error"]


@dataclass(frozen=True)
class Case:
    files: tuple[str, ...]
    expected: Outcome
    why: str
    # Other outcomes that are defensible, e.g. a judgment call the policy allows.
    also_ok: tuple[Outcome, ...] = ()

    @property
    def name(self) -> str:
        """The scored file; a + marks one that ran after earlier files."""
        return self.files[-1] + (" +" if len(self.files) > 1 else "")


CASES: list[Case] = [
    Case(("invoice_1001.txt",), "pay", "Clean, within stock."),
    Case(("invoice_1002.txt",), "reject", "20 GadgetX against 5 in stock."),
    Case(("invoice_1003.txt",), "reject", "FakeItem has no stock; pay-now pressure."),
    Case(
        ("invoice_1004.json",),
        "pay",
        "Clean, though the vendor address is the Simpsons' house.",
        # Escalating that address to a person is a fair call; rejecting is not.
        also_ok=("human",),
    ),
    Case(
        ("invoice_1004.json", "invoice_1004_revised.json"),
        "human",
        "A revision of an invoice already received; only one may be paid.",
    ),
    Case(("invoice_1005.json",), "reject", "8 GadgetX against 5 in stock."),
    Case(("invoice_1006.csv",), "pay", "Clean."),
    Case(("invoice_1007.csv",), "reject", "Total does not add up; stock exceeded."),
    Case(("invoice_1008.txt",), "reject", "Neither item is in the inventory."),
    Case(("invoice_1009.json",), "reject", "Negative quantity; missing vendor."),
    Case(
        ("invoice_1010.txt",),
        "pay",
        "The rush line is WidgetA at a premium; stock holds once lines are summed.",
        also_ok=("human",),
    ),
    Case(("invoice_1011.txt",), "pay", "Clean."),
    Case(("invoice_1011.pdf",), "pay", "Clean, from a PDF."),
    Case(
        ("invoice_1012.txt",),
        "human",
        "OCR typos, and $9,975 sits just under the $10,000 VP threshold.",
    ),
    Case(("invoice_1012.pdf",), "human", "The same invoice as a PDF."),
    Case(("invoice_1013.json",), "reject", "Total does not add up; stock exceeded."),
    Case(("invoice_1013.pdf",), "reject", "The same invoice as a PDF."),
    Case(
        ("invoice_1014.xml",),
        "human",
        "Billed in EUR; policy sends foreign currency to a VP.",
    ),
    Case(("invoice_1015.csv",), "pay", "Clean."),
    Case(("invoice_1016.json",), "reject", "WidgetC is not in the inventory."),
]


@dataclass
class Result:
    case: str
    expected: Outcome
    outcome: Outcome
    correct: bool | None  # None when the case could not run (skipped)
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    detail: str = ""
    findings: list[str] = field(default_factory=list)
    # What triage decided on its own (approve/reject), or None if it passed
    # the invoice on to the reasoning approver or did not run.
    fast_path: str | None = None


Pricing = dict[str, tuple[float, float]]  # model -> (USD per input, per output token)


def openrouter_pricing() -> Pricing:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        models = json.load(r)["data"]
    return {
        m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
        for m in models
        if "pricing" in m
    }


def cost(usage: dict[str, Any], pricing: Pricing) -> float | None:
    """Total USD for a {model: usage_metadata} map; None if a model has no price."""
    total = 0.0
    for model, u in usage.items():
        price = pricing.get(model) or pricing.get(f"x-ai/{model}")
        if price is None:
            return None
        total += u["input_tokens"] * price[0] + u["output_tokens"] * price[1]
    return total


def outcome_of(state: InvoiceState | dict[str, Any]) -> tuple[Outcome, str]:
    if "__interrupt__" in state:
        return "human", state["draft"].rationale if "draft" in state else ""
    if state.get("payment"):
        return "pay", ""
    if reason := state.get("rejection_reason"):
        return "reject", reason
    return "error", "The run ended without an outcome."


async def run_case(case: Case, settings: Settings, pricing: Pricing) -> Result:
    with tempfile.TemporaryDirectory() as tmp:
        inventory = Inventory(Path(tmp) / "inventory.db")
        inventory.seed()
        deps = Deps(
            agents=build_agents(settings), inventory=inventory, settings=settings
        )
        graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))
        start = time.perf_counter()
        findings: list[str] = []
        triage_cost = 0.0
        state: dict[str, Any] = {}
        with get_usage_metadata_callback() as usage:
            try:
                for name in case.files:
                    config: RunnableConfig = {
                        "configurable": {"thread_id": uuid.uuid4().hex}
                    }
                    path = str(settings.samples_dir / name)
                    state = await graph.ainvoke({"path": path}, config)
                    if "triage" in state:
                        triage_cost += state["triage"].cost_usd or 0.0
                outcome, detail = outcome_of(state)
                if "report" in state:
                    findings = [f.code for f in state["report"].findings]
            except ExtractionUnavailableError as exc:
                outcome, detail = "skipped", str(exc)
            except Exception as exc:
                outcome, detail = "error", f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start

    used = dict(usage.usage_metadata)
    llm_cost = cost(used, pricing) if used else None
    total_cost = (
        None
        if llm_cost is None and not triage_cost
        else (llm_cost or 0.0) + triage_cost
    )
    fast = None
    if "triage" in state and "review_rounds" not in state and "draft" in state:
        fast = state["draft"].decision
    correct = (
        None
        if outcome == "skipped"
        else outcome == case.expected or outcome in case.also_ok
    )
    return Result(
        case=case.name,
        expected=case.expected,
        outcome=outcome,
        correct=correct,
        seconds=round(seconds, 2),
        input_tokens=sum(u["input_tokens"] for u in used.values()),
        output_tokens=sum(u["output_tokens"] for u in used.values()),
        cost_usd=total_cost,
        detail=detail,
        findings=findings,
        fast_path=fast,
    )


async def run_all(
    cases: list[Case], settings: Settings, pricing: Pricing, concurrency: int
) -> list[Result]:
    gate = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> Result:
        async with gate:
            return await run_case(case, settings, pricing)

    return list(await asyncio.gather(*(one(c) for c in cases)))


def summary(results: list[Result]) -> dict[str, Any]:
    scored = [r for r in results if r.correct is not None]
    right = sum(1 for r in scored if r.correct)
    costs = [r.cost_usd for r in scored if r.cost_usd is not None]
    llm_used = any(r.input_tokens for r in scored)
    return {
        "scored": len(scored),
        "correct": right,
        "skipped": len(results) - len(scored),
        "accuracy": round(right / len(scored), 3) if scored else None,
        "total_cost_usd": round(sum(costs), 4) if costs and llm_used else None,
        "cost_per_invoice_usd": round(sum(costs) / len(costs), 4)
        if costs and llm_used
        else None,
        "mean_seconds": round(sum(r.seconds for r in scored) / len(scored), 2)
        if scored
        else None,
        "fast_paths": sum(1 for r in scored if r.fast_path),
    }


def render(results: list[Result], totals: dict[str, Any], console: Console) -> None:
    table = Table(show_lines=False, pad_edge=False)
    for col in ("case", "expected", "got", "", "seconds", "tokens", "cost"):
        table.add_column(
            col,
            justify="right" if col in {"seconds", "tokens", "cost"} else "left",
            no_wrap=col in {"case", "cost"},
        )
    for r in results:
        mark = {True: "[green]ok[/]", False: "[red]MISS[/]", None: "[dim]skip[/]"}[
            r.correct
        ]
        table.add_row(
            r.case,
            r.expected,
            r.outcome,
            mark,
            f"{r.seconds:.1f}",
            f"{r.input_tokens + r.output_tokens:,}",
            "-" if r.cost_usd is None else f"${r.cost_usd:.4f}",
        )
    console.print(table)
    for r in results:
        if r.correct is False:
            console.print(f"[red]MISS[/] {r.case}: {r.detail or r.findings}")
    accuracy = totals["accuracy"]
    line = f"{totals['correct']}/{totals['scored']} correct"
    if accuracy is not None:
        line += f" ({accuracy:.0%})"
    if totals["skipped"]:
        line += f", {totals['skipped']} skipped (need an LLM)"
    if totals["total_cost_usd"] is not None:
        line += (
            f", ${totals['total_cost_usd']:.4f} total,"
            f" ${totals['cost_per_invoice_usd']:.4f} per invoice"
        )
    if totals.get("fast_paths"):
        line += f", {totals['fast_paths']} settled by triage"
    if totals["mean_seconds"] is not None:
        line += f", {totals['mean_seconds']:.1f}s mean"
    console.print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the sample invoices.")
    parser.add_argument("--only", help="Run cases whose file names contain this")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--triage",
        metavar="MODEL",
        help="Decision model for triage, e.g. typesafe/jev-1.14",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("eval-results"), help="Folder for JSON"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.triage:
        settings = settings.model_copy(update={"triage_model": args.triage})
    cases = [c for c in CASES if not args.only or args.only in c.name]
    pricing = openrouter_pricing() if settings.llm_provider == "openrouter" else {}
    console = Console()
    console.print(
        f"Provider {settings.llm_provider}: extract {settings.extract_model},"
        f" review {settings.review_model}"
        + (f", triage {settings.triage_model}" if settings.triage_model else "")
        + f". {len(cases)} cases."
    )

    results = asyncio.run(run_all(cases, settings, pricing, args.concurrency))
    totals = summary(results)
    render(results, totals, console)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = settings.llm_provider + ("-triage" if settings.triage_model else "")
    report = args.out / f"{label}-{stamp}.json"
    report.write_text(
        json.dumps(
            {
                "provider": settings.llm_provider,
                "extract_model": settings.extract_model,
                "review_model": settings.review_model,
                "triage_model": settings.triage_model,
                "summary": totals,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    console.print(f"Saved {report}")
    return 0 if all(r.correct is not False for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
