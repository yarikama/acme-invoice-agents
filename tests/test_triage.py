"""System 1 triage: the Decisions API client, the fast-path rule, and the graph."""

import json
import uuid

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from invoice_agents.agents import NoExtractor, RuleApprover, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.llm import Agents, build_agents
from invoice_agents.models import (
    Decision,
    Finding,
    Invoice,
    Severity,
    ValidationReport,
)
from invoice_agents.triage import (
    DECISIONS_URL,
    DecisionTriager,
    TriageAnswer,
    TriageError,
    fast_path,
    triage_state,
)
from tests.conftest import INVOICES

INVOICE = Invoice(
    invoice_number="INV-1",
    vendor="Widgets Inc.",
    invoice_date="2026-01-15",
    due_date="2026-02-01",
    line_items=[],
    total=500,
)
CLEAN = ValidationReport()
ERROR = ValidationReport(
    findings=[
        Finding(code="OUT_OF_STOCK", severity=Severity.ERROR, title="t", detail="d")
    ]
)
WARNING = ValidationReport(
    findings=[
        Finding(code="NEAR_THRESHOLD", severity=Severity.WARNING, title="t", detail="d")
    ]
)
INFO = ValidationReport(
    findings=[
        Finding(code="OVER_THRESHOLD", severity=Severity.INFO, title="t", detail="d")
    ]
)

ANSWER = {
    "model": "typesafe/jev-1.14-20261014",
    "answers": {
        "decision": {
            "type": "choice",
            "choice": "approve",
            "probabilities": {"approve": 0.97, "reject": 0.01, "escalate": 0.02},
            "confidence": 0.97,
        },
        "suspicious": {"type": "noul", "noul": 0.05},
    },
    "usage": {"input_tokens": 420, "output_tokens": 59, "cost": 1.764e-05},
}


def answer(
    decision: Decision = "approve", confidence: float = 0.97, suspicion: float = 0.05
) -> TriageAnswer:
    return TriageAnswer(
        model="m",
        decision=decision,
        confidence=confidence,
        suspicion=suspicion,
        seconds=0.3,
    )


def triager(handler) -> DecisionTriager:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DecisionTriager(
        "typesafe/jev-1.14", SecretStr("test-key"), client=client, retries=2
    )


# --- client ---------------------------------------------------------------


async def test_ask_sends_state_and_questions_and_reads_the_answer():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ANSWER)

    got = await triager(handler).ask(INVOICE, CLEAN)

    assert str(seen[0].url) == DECISIONS_URL
    assert seen[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(seen[0].content)
    assert body["model"] == "typesafe/jev-1.14"
    assert set(body["questions"]) == {"decision", "suspicious"}
    assert body["state"]["findings"] == ["None: every check passed."]
    assert (got.decision, got.confidence, got.suspicion) == ("approve", 0.97, 0.05)
    assert (got.input_tokens, got.cost_usd) == (420, 1.764e-05)
    assert got.describe() == ("typesafe/jev-1.14 says approve (97% sure), suspicion 5%")


async def test_ask_falls_back_to_the_probability_when_confidence_is_missing():
    data = json.loads(json.dumps(ANSWER))
    del data["answers"]["decision"]["confidence"]
    del data["usage"]

    got = await triager(lambda r: httpx.Response(200, json=data)).ask(INVOICE, CLEAN)

    assert got.confidence == 0.97
    assert got.cost_usd is None


async def test_ask_retries_rate_limits(monkeypatch: pytest.MonkeyPatch):
    async def no_sleep(_: float) -> None: ...

    monkeypatch.setattr("invoice_agents.triage.asyncio.sleep", no_sleep)
    replies = iter(
        [httpx.Response(429), httpx.Response(503), httpx.Response(200, json=ANSWER)]
    )

    got = await triager(lambda r: next(replies)).ask(INVOICE, CLEAN)

    assert got.decision == "approve"


async def test_ask_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch):
    async def no_sleep(_: float) -> None: ...

    monkeypatch.setattr("invoice_agents.triage.asyncio.sleep", no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(TriageError, match="ConnectError"):
        await triager(handler).ask(INVOICE, CLEAN)


async def test_ask_does_not_retry_a_bad_request():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, text="bad model")

    with pytest.raises(TriageError, match="HTTP 400: bad model"):
        await triager(handler).ask(INVOICE, CLEAN)
    assert len(calls) == 1


async def test_ask_rejects_an_unexpected_answer():
    with pytest.raises(TriageError, match="Unexpected answer"):
        await triager(lambda r: httpx.Response(200, json={"answers": {}})).ask(
            INVOICE, CLEAN
        )


async def test_ask_opens_its_own_client_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
):
    real = httpx.AsyncClient

    def mocked(**kwargs):
        return real(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ANSWER))
        )

    monkeypatch.setattr("invoice_agents.triage.httpx.AsyncClient", mocked)

    got = await DecisionTriager("m", SecretStr("k")).ask(INVOICE, CLEAN)

    assert got.decision == "approve"


def test_state_lists_findings():
    assert triage_state(INVOICE, WARNING)["findings"] == [
        "[warning] NEAR_THRESHOLD: t. d"
    ]


# --- fast path ------------------------------------------------------------


def test_fast_path_approves_only_a_clean_unsuspicious_invoice():
    assert fast_path(answer(), CLEAN, 0.9, 0.2) == ("approve", "every check passed")
    assert fast_path(answer(), INFO, 0.9, 0.2) == ("approve", "every check passed")
    assert fast_path(answer(), WARNING, 0.9, 0.2) is None
    assert fast_path(answer(), ERROR, 0.9, 0.2) is None
    assert fast_path(answer(suspicion=0.5), CLEAN, 0.9, 0.2) is None
    assert fast_path(answer(confidence=0.6), CLEAN, 0.9, 0.2) is None


def test_fast_path_rejects_only_when_code_found_an_error():
    assert fast_path(answer("reject"), ERROR, 0.9, 0.2) == (
        "reject",
        "validation found errors (OUT_OF_STOCK)",
    )
    # A decision model can be talked into "reject"; without an error it must not stick.
    assert fast_path(answer("reject"), CLEAN, 0.9, 0.2) is None


def test_fast_path_never_settles_an_escalation():
    assert fast_path(answer("escalate"), WARNING, 0.9, 0.2) is None


# --- graph ----------------------------------------------------------------


class StubTriager:
    model = "stub"

    def __init__(self, reply: TriageAnswer | Exception) -> None:
        self.reply = reply
        self.calls = 0

    async def ask(self, invoice, report) -> TriageAnswer:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class CountingCritic(RuleCritic):
    def __init__(self) -> None:
        self.calls = 0

    async def review(self, *args, **kwargs):
        self.calls += 1
        return await super().review(*args, **kwargs)


def run(inventory, triage, name: str):
    critic = CountingCritic()
    agents = Agents(NoExtractor(), RuleApprover(), critic, triage)
    deps = Deps(
        agents=agents, inventory=inventory, settings=Settings(llm_provider="offline")
    )
    graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))
    config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}
    return graph.ainvoke({"path": str(INVOICES / name)}, config), critic


async def test_triage_pays_a_clean_invoice_without_the_approver(inventory):
    pending, critic = run(inventory, StubTriager(answer()), "invoice_1006.csv")
    state = await pending

    assert state["payment"].status == "success"
    assert critic.calls == 0
    assert "review_rounds" not in state
    assert state["draft"].rationale.startswith("Triage: m says approve")
    line = next(e for e in state["trace"] if e["stage"] == "triage")
    assert "reasoning approver was skipped. Outcome: pay." in line["message"]
    # The decision model's own reported use rides on its line.
    assert line["usage"] == {
        "model": "m",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": None,
    }


async def test_triage_rejects_an_invoice_with_errors(inventory):
    pending, critic = run(inventory, StubTriager(answer("reject")), "invoice_1005.json")
    state = await pending

    assert "payment" not in state
    assert state["rejection_reason"].startswith("Triage:")
    assert critic.calls == 0


async def test_triage_sends_judgment_calls_to_the_approver(inventory):
    pending, critic = run(
        inventory, StubTriager(answer("escalate")), "invoice_1006.csv"
    )
    state = await pending

    assert state["triage"].decision == "escalate"
    assert critic.calls == 1
    assert any("Needs the full review" in e["message"] for e in state["trace"])


async def test_triage_failure_falls_back_to_the_approver(inventory):
    pending, critic = run(
        inventory, StubTriager(TriageError("m failed: HTTP 503.")), "invoice_1006.csv"
    )
    state = await pending

    assert "payment" in state
    assert critic.calls == 1
    assert "triage" not in state
    assert any("Sending it to the approver" in e["message"] for e in state["trace"])


def test_build_agents_adds_a_triager_only_when_configured():
    assert (
        build_agents(Settings(llm_provider="offline", triage_model=None)).triager
        is None
    )
    agents = build_agents(
        Settings(
            llm_provider="offline",
            triage_model="typesafe/jev-1.14",
            openrouter_api_key=SecretStr("k"),
        )
    )
    assert agents.triager is not None
    assert agents.triager.model == "typesafe/jev-1.14"
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_agents(
            Settings(llm_provider="offline", triage_model="m", openrouter_api_key=None)
        )
