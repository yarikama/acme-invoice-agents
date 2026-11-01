"""Token use and cost, per step and per run."""

import asyncio
import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from invoice_agents import llm, usage
from invoice_agents.agents import Review, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.llm import Agents
from invoice_agents.models import ApprovalDraft, Invoice, ValidationReport
from invoice_agents.tools import ReviewTools
from tests.conftest import INVOICES


def reply(model: str, inp: int, out: int) -> AIMessage:
    return AIMessage(
        "",
        usage_metadata={
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
        },
        response_metadata={"model_name": model},
    )


def test_collecting_sums_replies_per_model() -> None:
    with usage.collecting() as tokens:
        usage.record(reply("grok-4.7", 100, 20))
        usage.record(reply("grok-4.7", 50, 5))
        usage.record(reply("grok-4.3", 10, 1))
        usage.record(AIMessage("no usage"))
        usage.record(None)
    assert tokens == {
        "grok-4.7": {"input_tokens": 150, "output_tokens": 25},
        "grok-4.3": {"input_tokens": 10, "output_tokens": 1},
    }
    usage.record(reply("grok-4.7", 1, 1))  # outside a step: ignored, no error


def test_step_usage_prices_known_models_only() -> None:
    pricing = {"x-ai/grok-4.7": (0.000002, 0.000006)}
    tokens = {"grok-4.7": {"input_tokens": 1000, "output_tokens": 100}}
    assert usage.step_usage(tokens, pricing) == {
        "model": "grok-4.7",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cost_usd": pytest.approx(0.0026),
    }
    assert usage.step_usage(tokens, {})["cost_usd"] is None  # type: ignore[index]
    assert usage.step_usage({}, pricing) is None


class RecordingApprover:
    """Approves, and reports the tokens a real model call would have used."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review:
        await asyncio.sleep(0)  # let overlapping runs interleave
        usage.record(reply("grok-4.7", self.tokens, 10))
        return Review(ApprovalDraft(decision="approve", rationale="ok"))


def graph(inventory: Any, tokens: int) -> Any:
    agents = Agents(None, RecordingApprover(tokens), RuleCritic())  # type: ignore[arg-type]
    deps = Deps(
        agents=agents,
        inventory=inventory,
        settings=Settings(llm_provider="offline", triage_model=None),
        pricing={"x-ai/grok-4.7": (0.000002, 0.000006)},
    )
    return build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))


def config() -> RunnableConfig:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


async def test_each_step_carries_its_own_usage(inventory: Any) -> None:
    state = await graph(inventory, 1000).ainvoke(
        {"path": str(INVOICES / "invoice_1006.csv")}, config()
    )
    draft = next(e for e in state["trace"] if e["message"].startswith("Draft 1"))
    assert draft["usage"] == {
        "model": "grok-4.7",
        "input_tokens": 1000,
        "output_tokens": 10,
        "cost_usd": pytest.approx(0.00206),
    }
    # Steps without a model call carry none.
    assert all(
        "usage" not in e for e in state["trace"] if e["stage"] in ("validate", "pay")
    )


async def test_overlapping_runs_do_not_mix_usage(inventory: Any) -> None:
    a, b = await asyncio.gather(
        graph(inventory, 111).ainvoke(
            {"path": str(INVOICES / "invoice_1006.csv")}, config()
        ),
        graph(inventory, 999).ainvoke(
            {"path": str(INVOICES / "invoice_1015.csv")}, config()
        ),
    )

    def tokens(state: Any) -> int:
        return next(e for e in state["trace"] if e["message"].startswith("Draft"))[
            "usage"
        ]["input_tokens"]

    assert (tokens(a), tokens(b)) == (111, 999)


def test_prices_load_only_for_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    assert llm.load_pricing(Settings(llm_provider="offline")) == {}
    monkeypatch.setattr(llm, "openrouter_pricing", lambda: {"m": (1.0, 2.0)})
    assert llm.load_pricing(Settings(llm_provider="openrouter")) == {"m": (1.0, 2.0)}

    def offline() -> usage.Pricing:
        raise OSError("no network")

    monkeypatch.setattr(llm, "openrouter_pricing", offline)
    assert llm.load_pricing(Settings(llm_provider="openrouter")) == {}
