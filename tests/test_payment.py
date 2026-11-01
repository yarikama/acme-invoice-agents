"""The mock bank pays once per idempotency key."""

import uuid
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from invoice_agents import payment
from invoice_agents.agents import RuleApprover, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.llm import Agents
from invoice_agents.payment import MockBank, payment_key
from tests.conftest import INVOICES
from tests.test_graph import INV_1001, StubExtractor


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Each time the brief's mock_payment actually ran: money moved."""
    moved: list[float] = []
    real = payment.mock_payment

    def counting(vendor: str, amount: float, currency: str = "USD"):
        moved.append(amount)
        return real(vendor, amount, currency)

    monkeypatch.setattr(payment, "mock_payment", counting)
    return moved


def test_the_same_invoice_gets_the_same_key():
    spelled = INV_1001.model_copy(
        update={"vendor": "WIDGETS INC", "invoice_number": "inv 1001"}
    )
    assert payment_key(INV_1001) == payment_key(spelled) == "widgetsinc:inv1001:"
    revised = INV_1001.model_copy(update={"revision": "R1"})
    assert payment_key(revised) == "widgetsinc:inv1001:r1"


def test_a_key_pays_once_even_across_a_restart(tmp_path: Path, calls: list[float]):
    first = MockBank(tmp_path / "bank.db").pay(INV_1001)
    assert (first.status, first.duplicate) == ("success", False)
    again = MockBank(tmp_path / "bank.db").pay(INV_1001)  # a new process
    assert again.duplicate and again.amount == 5000
    assert calls == [5000]

    bank = MockBank(tmp_path / "bank.db")
    bank.clear()
    assert not bank.pay(INV_1001).duplicate
    assert calls == [5000, 5000]


def test_in_memory_bank(calls: list[float]):
    bank = MockBank()
    bank.pay(INV_1001)
    assert bank.pay(INV_1001).duplicate
    bank.clear()
    assert not bank.pay(INV_1001).duplicate
    assert calls == [5000, 5000]


async def test_a_second_pay_of_the_same_invoice_moves_no_money(
    inventory, calls: list[float]
):
    # The ledger is what normally stops this; here each run has its own, as
    # if the record of the first payment were lost (a retry after a crash).
    bank = MockBank()

    async def run():
        deps = Deps(
            agents=Agents(StubExtractor(INV_1001), RuleApprover(), RuleCritic()),
            inventory=inventory,
            settings=Settings(llm_provider="offline"),
            bank=bank,
        )
        graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))
        config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}
        return await graph.ainvoke({"path": str(INVOICES / "invoice_1001.txt")}, config)

    first, second = await run(), await run()
    assert not first["payment"].duplicate
    assert second["payment"].duplicate
    assert calls == [5000]
    assert second["trace"][-1]["message"] == (
        "Already paid under key widgetsinc:inv1001:; no money moved this time."
    )
