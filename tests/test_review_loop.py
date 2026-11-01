"""The approver-critic loop: an objection changes the decision, and a dispute
that never settles goes to a person instead of letting the draft through."""

import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from invoice_agents.agents import Review, RuleApprover, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.llm import Agents
from invoice_agents.models import ApprovalDraft, Critique, Decision, LineItem
from tests.conftest import INVOICES
from tests.test_graph import INV_1001, StubExtractor


def config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


class ScriptedApprover:
    """Drafts the given decisions in turn, and keeps the objections it heard."""

    def __init__(self, *decisions: Decision) -> None:
        self.decisions = list(decisions)
        self.heard: list[list[str]] = []

    async def draft(self, invoice, report, critique, tools) -> Review:
        self.heard.append(list(critique))
        decision = self.decisions.pop(0)
        codes = [f.code for f in report.findings]
        return Review(
            ApprovalDraft(decision=decision, rationale="Scripted.", addressed=codes)
        )


class ScriptedCritic:
    def __init__(self, *verdicts: bool) -> None:
        self.verdicts = list(verdicts)

    async def review(self, invoice, report, draft, checked) -> Critique:
        if self.verdicts.pop(0):
            return Critique(agrees=True)
        return Critique(agrees=False, issues=["Vendor history was not checked."])


async def run(inventory, approver, critic, invoice=INV_1001):
    deps = Deps(
        agents=Agents(StubExtractor(invoice), approver, critic),
        inventory=inventory,
        settings=Settings(llm_provider="offline"),
    )
    graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))
    cfg = config()
    await graph.ainvoke({"path": str(INVOICES / "invoice_1001.txt")}, cfg)
    return (await graph.aget_state(cfg)).values


def messages(state) -> list[str]:
    return [t["message"] for t in state["trace"] if t["stage"] == "approve"]


async def test_an_objection_turns_an_approval_into_a_rejection(inventory):
    approver = ScriptedApprover("approve", "reject")
    state = await run(inventory, approver, ScriptedCritic(False, True))
    assert approver.heard == [[], ["Vendor history was not checked."]]
    assert state["outcome"] == "reject"
    assert "payment" not in state
    assert [m.split(".")[0] for m in messages(state)] == [
        "Draft 1: approve",
        "Critic objects: Vendor history was not checked",
        "Draft 2: reject",
        "Critic agrees",
    ]


async def test_a_dispute_that_never_settles_goes_to_a_person(inventory):
    # Before: after the last round the approver's draft went through, paid.
    state = await run(
        inventory, ScriptedApprover("approve", "approve"), ScriptedCritic(False, False)
    )
    assert state["outcome"] == "human"
    assert "payment" not in state
    assert "Still disputed after 2 rounds. Outcome: human." in messages(state)[-1]


async def test_a_dispute_over_an_invoice_the_rules_reject_stays_rejected(inventory):
    bad = INV_1001.model_copy(
        update={"line_items": [LineItem(item="FakeItem", quantity=1, unit_price=5)]}
    )
    state = await run(
        inventory,
        ScriptedApprover("escalate", "escalate"),
        ScriptedCritic(False, False),
        bad,
    )
    assert state["outcome"] == "reject"


async def test_rule_agents_go_through_a_round_of_review(inventory):
    # Paperwork warnings only: the rule approver lets them through, its
    # critic objects, and the second draft escalates.
    late = INV_1001.model_copy(
        update={"due_date": "2026-01-16", "payment_terms": "Net 30"}
    )
    approver = RuleApprover()
    state = await run(inventory, approver, RuleCritic(), late)
    lines = messages(state)
    assert lines[0].startswith("Draft 1: approve. Only paperwork warnings")
    assert lines[1].startswith("Critic objects: The draft approves with TERMS_MISMATCH")
    assert lines[2].startswith("Draft 2: escalate. The critic objected")
    assert state["outcome"] == "human"
