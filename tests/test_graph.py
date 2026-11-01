"""End-to-end runs of the graph with rule-based agents and a stub extractor."""

import uuid
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from invoice_agents.agents import Extraction, Review, RuleApprover, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.llm import Agents
from invoice_agents.models import Critique, Invoice, LineItem, ToolCall
from tests.conftest import INVOICES


class StubExtractor:
    """Returns a fixed invoice; the first call can be made to come back empty."""

    def __init__(self, invoice: Invoice, fail_first: bool = False) -> None:
        self.invoice = invoice
        self.fail_first = fail_first
        self.calls: list[list[str]] = []

    async def extract(self, text: str, feedback: list[str]) -> Extraction:
        self.calls.append(feedback)
        if self.fail_first and len(self.calls) == 1:
            return Extraction(
                invoice=self.invoice.model_copy(update={"line_items": []})
            )
        return Extraction(invoice=self.invoice, corrections=["read 2O26 as 2026"])


INV_1001 = Invoice(
    invoice_number="INV-1001",
    vendor="Widgets Inc.",
    invoice_date="2026-01-15",
    due_date="2026-02-01",
    line_items=[
        LineItem(item="WidgetA", quantity=10, unit_price=250),
        LineItem(item="WidgetB", quantity=5, unit_price=500),
    ],
    subtotal=5000,
    tax_amount=0,
    total=5000,
    payment_terms="Net 15",
)


def make_graph(inventory, extractor=None):
    agents = Agents(extractor or StubExtractor(INV_1001), RuleApprover(), RuleCritic())
    deps = Deps(
        agents=agents, inventory=inventory, settings=Settings(llm_provider="offline")
    )
    return build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))


def config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


async def test_clean_text_invoice_is_paid(inventory):
    graph = make_graph(inventory)
    state = await graph.ainvoke({"path": str(INVOICES / "invoice_1001.txt")}, config())
    assert state["payment"].status == "success"
    assert state["payment"].amount == 5000
    assert [e["stage"] for e in state["trace"]][-1] == "pay"


async def test_each_step_reports_what_it_produced(inventory):
    graph = make_graph(inventory)
    state = await graph.ainvoke({"path": str(INVOICES / "invoice_1001.txt")}, config())
    data = {t["stage"]: t.get("data") for t in state["trace"] if t.get("data")}
    assert data["ingest"]["invoice"]["vendor"] == "Widgets Inc."
    assert data["ingest"]["corrections"] == ["read 2O26 as 2026"]
    assert isinstance(data["validate"]["findings"], list)
    assert data["pay"]["payment"] == {
        "status": "success",
        "vendor": "Widgets Inc.",
        "amount": 5000.0,
        "currency": "USD",
    }
    # A document parsed in code reports its invoice too.
    state = await graph.ainvoke({"path": str(INVOICES / "invoice_1016.json")}, config())
    [ingest] = [t for t in state["trace"] if t["stage"] == "ingest"]
    assert ingest["data"]["invoice"]["invoice_number"] == "INV-1016"
    [validate] = [t for t in state["trace"] if t["stage"] == "validate"]
    assert {f["code"] for f in validate["data"]["findings"]} >= {"UNKNOWN_ITEM"}


async def test_extraction_retries_with_feedback(inventory):
    extractor = StubExtractor(INV_1001, fail_first=True)
    graph = make_graph(inventory, extractor)
    state = await graph.ainvoke({"path": str(INVOICES / "invoice_1001.txt")}, config())
    assert len(extractor.calls) == 2
    assert extractor.calls[1] == ["No line items were extracted."]
    assert state["extract_attempts"] == 2
    assert "payment" in state


async def test_structured_invoice_with_errors_is_rejected(inventory):
    graph = make_graph(inventory)
    state = await graph.ainvoke({"path": str(INVOICES / "invoice_1007.csv")}, config())
    assert "payment" not in state
    assert "Total does not add up" in state["rejection_reason"]


async def test_validate_trace_counts_in_plain_english(inventory):
    graph = make_graph(inventory)
    state = await graph.ainvoke({"path": str(INVOICES / "invoice_1016.json")}, config())
    [line] = [t["message"] for t in state["trace"] if t["stage"] == "validate"]
    assert line == "1 error, 0 warnings; policy route: reject."


@pytest.mark.parametrize(("approved", "paid"), [(True, True), (False, False)])
async def test_warnings_pause_for_the_vp(inventory, approved, paid):
    graph = make_graph(inventory)
    cfg = config()
    first = await graph.ainvoke({"path": str(INVOICES / "invoice_1014.xml")}, cfg)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["invoice"]["currency"] == "EUR"

    state = await graph.ainvoke(
        Command(resume={"approved": approved, "note": "ok"}), cfg
    )
    assert ("payment" in state) is paid


async def test_checkpoints_restore_state_types_without_warnings(inventory, caplog):
    graph = make_graph(inventory)
    cfg = config()
    await graph.ainvoke({"path": str(INVOICES / "invoice_1014.xml")}, cfg)
    await graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert "unregistered type" not in caplog.text


class ToolUsingApprover:
    """Checks stock with a real tool call, then defers to the rule approver."""

    async def draft(self, invoice, report, critique, tools) -> Review:
        call = tools.run("check_inventory", {"item": "WidgetA"})
        ruled = await RuleApprover().draft(invoice, report, critique, tools)
        return Review(ruled.draft, [call])


class RecordingCritic:
    def __init__(self) -> None:
        self.checked: list[list[ToolCall]] = []

    async def review(self, invoice, report, draft, checked) -> Critique:
        self.checked.append(checked)
        return Critique(agrees=True)


async def test_tool_calls_reach_the_trace_and_the_critic(inventory):
    critic = RecordingCritic()
    agents = Agents(StubExtractor(INV_1001), ToolUsingApprover(), critic)
    deps = Deps(
        agents=agents, inventory=inventory, settings=Settings(llm_provider="offline")
    )
    graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))

    state = await graph.ainvoke({"path": str(INVOICES / "invoice_1001.txt")}, config())

    [tool_line] = [t for t in state["trace"] if t.get("tool")]
    assert tool_line == {
        "stage": "approve",
        "tool": "check_inventory",
        "message": "check_inventory(WidgetA): WidgetA: 15 in stock,"
        " list price $250.00.",
        "streamed": True,  # sent live, so clients skip it in the update
    }
    assert [c.name for c in state["tool_calls"]] == ["check_inventory"]
    assert critic.checked == [state["tool_calls"]]
    assert "payment" in state


async def test_tool_calls_stream_as_they_finish_before_the_draft(inventory):
    agents = Agents(StubExtractor(INV_1001), ToolUsingApprover(), RecordingCritic())
    deps = Deps(
        agents=agents, inventory=inventory, settings=Settings(llm_provider="offline")
    )
    graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))

    seen: list[str] = []
    async for mode, raw in graph.astream(
        {"path": str(INVOICES / "invoice_1001.txt")},
        config(),
        stream_mode=["updates", "custom"],
    ):
        chunk = cast(dict[str, Any], raw)
        if mode == "custom" and "trace" in chunk:
            seen.append("live: " + chunk["trace"]["message"].split(":")[0])
        elif mode == "updates" and "review" in chunk:
            seen.extend(
                ("skip: " if item.get("streamed") else "") + item["message"][:7]
                for item in chunk["review"]["trace"]
            )
    assert seen == [
        "live: check_inventory(WidgetA)",
        "skip: check_i",  # in the update too, marked as already sent
        "Draft 1",
    ]
