"""The ledger keeps received invoices and their outcomes across restarts."""

import json
import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from invoice_agents.agents import RuleApprover, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.ledger import Ledger
from invoice_agents.llm import Agents
from invoice_agents.validation import validate
from tests.conftest import INVOICES
from tests.test_graph import INV_1001, StubExtractor


def config() -> RunnableConfig:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def test_entries_and_outcomes_survive_a_new_instance(tmp_path: Path):
    db, log = tmp_path / "inventory.db", tmp_path / "decisions.jsonl"
    first = Ledger(db, log)
    first.add(INV_1001, run_id="r1", source="invoice_1001.txt")
    first.settle("r1", "paid")
    first.settle("unknown", "paid")  # nothing to settle: no error, no log line

    again = Ledger(db, log)
    [entry] = again.entries
    assert entry.invoice == INV_1001
    assert (entry.run_id, entry.source, entry.outcome) == (
        "r1",
        "invoice_1001.txt",
        "paid",
    )
    assert again.waiting() == []
    [line] = log.read_text().splitlines()
    record = json.loads(line)
    assert record["outcome"] == "paid" and record["invoice_number"] == "INV-1001"

    again.clear()
    assert Ledger(db, log).entries == []


def test_only_standing_invoices_count_as_duplicates(tmp_path: Path, inventory):
    ledger = Ledger(tmp_path / "inventory.db")
    ledger.add(INV_1001, run_id="r1")
    # In progress or waiting for a VP: received.
    [dup] = validate(INV_1001, inventory, ledger).findings
    assert dup.code == "DUPLICATE" and dup.detail == "INV-1001 was already received."
    ledger.settle("r1", "paid")
    codes = {f.code: f.detail for f in validate(INV_1001, inventory, ledger).findings}
    assert codes["DUPLICATE"] == "INV-1001 was already paid."
    # A rejected invoice may be sent again once fixed.
    ledger.settle("r1", "rejected", "Wrong total.")
    assert "DUPLICATE" not in {
        f.code for f in validate(INV_1001, inventory, ledger).findings
    }


async def test_a_restart_does_not_forget_a_paid_invoice(tmp_path: Path, inventory):
    # Before: a new process paid INV-1001 a second time.
    def graph():
        deps = Deps(
            agents=Agents(StubExtractor(INV_1001), RuleApprover(), RuleCritic()),
            inventory=inventory,
            settings=Settings(llm_provider="offline"),
            seen=Ledger(tmp_path / "inventory.db"),
        )
        return build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))

    path = {"path": str(INVOICES / "invoice_1001.txt")}
    first = await graph().ainvoke(path, config())
    assert first["payment"].status == "success"
    second = await graph().ainvoke(path, config())
    assert "payment" not in second
    assert "DUPLICATE" in {f.code for f in second["report"].findings}
    [paid, rejected] = Ledger(tmp_path / "inventory.db").entries
    assert (paid.outcome, rejected.outcome) == ("paid", "rejected")
    assert paid.source == "invoice_1001.txt"
