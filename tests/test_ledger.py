"""The ledger keeps received invoices and their outcomes across restarts."""

import json
import sqlite3
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
    # A rejected invoice may be sent again once fixed.
    ledger.settle("r1", "rejected", "Wrong total.")
    assert "DUPLICATE" not in {
        f.code for f in validate(INV_1001, inventory, ledger).findings
    }
    ledger.add(INV_1001, run_id="r2")
    ledger.settle("r2", "paid")
    codes = {f.code: f.detail for f in validate(INV_1001, inventory, ledger).findings}
    assert codes["DUPLICATE"] == "INV-1001 was already paid."
    # A failure after paying (a checkpoint write, say) never undoes the payment.
    ledger.settle("r2", "failed", "Checkpoint write failed.")
    codes = {f.code: f.detail for f in validate(INV_1001, inventory, ledger).findings}
    assert codes["DUPLICATE"] == "INV-1001 was already paid."


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


def test_a_bad_row_is_skipped_not_fatal(tmp_path: Path):
    # Before: one row the model could not read broke every later run.
    ledger = Ledger(tmp_path / "inventory.db")
    ledger.add(INV_1001, run_id="good")
    with sqlite3.connect(tmp_path / "inventory.db") as conn:
        conn.execute(
            "INSERT INTO ledger (run_id, source, invoice, received_at)"
            " VALUES ('bad', 'x', ?, 'z')",
            ('{"line_items": [{"unit_price": null}]}',),
        )
    assert [e.run_id for e in ledger.entries] == ["good"]


async def test_a_run_stopped_after_paying_is_restored_as_paid(tmp_path: Path):
    """The process died between the bank's payment and the ledger's record:
    on restart the bank's key says it was paid, so it stays a duplicate."""
    from types import SimpleNamespace

    from invoice_agents.api.runs import RunManager
    from invoice_agents.payment import MockBank

    class Stopped:
        async def aget_state(self, _config):
            return SimpleNamespace(tasks=[], values={})

    db = tmp_path / "inventory.db"
    ledger, bank = Ledger(db), MockBank(db)
    ledger.add(INV_1001, run_id="paid-run")
    bank.pay(INV_1001)
    ledger.add(
        INV_1001.model_copy(update={"invoice_number": "INV-2002"}), run_id="lost"
    )
    await RunManager(Stopped(), ledger, bank).restore()  # pyright: ignore[reportArgumentType]
    outcome = {e.run_id: e.outcome for e in ledger.entries}
    assert outcome == {"paid-run": "paid", "lost": "failed"}
