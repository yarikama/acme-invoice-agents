"""The agents, with a fake chat model standing in for Grok."""

from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from invoice_agents.agents import (
    SUBMIT,
    Extraction,
    ExtractionUnavailableError,
    LLMApprover,
    LLMCritic,
    LLMExtractor,
    NoExtractor,
    RuleApprover,
    RuleCritic,
)
from invoice_agents.inventory import Inventory
from invoice_agents.models import (
    ApprovalDraft,
    Critique,
    Finding,
    Invoice,
    Severity,
    ToolCall,
    ValidationReport,
)
from invoice_agents.tools import ReviewTools
from invoice_agents.validation import SeenInvoices

INVOICE = Invoice(
    invoice_number="INV-1",
    vendor="Widgets Inc.",
    invoice_date="2026-01-15",
    due_date="2026-02-01",
    line_items=[],
    total=500,
)


class FakeChatModel:
    """Records each prompt and returns the next canned structured output."""

    def __init__(self, *outputs: dict[str, Any] | Exception) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.schemas: list[type[BaseModel]] = []

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Any:
        assert kwargs == {"method": "json_schema", "include_raw": True}
        self.schemas.append(schema)
        return self

    async def ainvoke(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def as_llm(self) -> BaseChatModel:
        return cast(BaseChatModel, self)


def parsed(value: BaseModel) -> dict[str, Any]:
    return {"raw": None, "parsed": value, "parsing_error": None}


def report(*findings: Finding) -> ValidationReport:
    return ValidationReport(findings=list(findings))


ERROR = Finding(
    code="OUT_OF_STOCK", severity=Severity.ERROR, title="FakeItem out", detail="0 left"
)
WARNING = Finding(
    code="FOREIGN_CURRENCY", severity=Severity.WARNING, title="EUR", detail="FX needed"
)


async def test_extractor_sends_text_and_feedback() -> None:
    fake = FakeChatModel(parsed(Extraction(invoice=INVOICE, corrections=["O→0"])))

    out = await LLMExtractor(fake.as_llm()).extract(
        "INV-1 Widgets", ["Line items are empty"]
    )

    assert out.invoice == INVOICE
    assert out.corrections == ["O→0"]
    assert fake.schemas == [Extraction]
    [prompt] = fake.prompts
    assert "INV-1 Widgets" in prompt
    assert "- Line items are empty" in prompt


async def test_extractor_first_attempt_has_no_feedback_section() -> None:
    fake = FakeChatModel(parsed(Extraction(invoice=INVOICE)))

    await LLMExtractor(fake.as_llm()).extract("text", [])

    assert "previous attempt" not in fake.prompts[0]


@pytest.mark.parametrize(
    "output",
    [
        {"raw": None, "parsed": None, "parsing_error": None},
        {"raw": None, "parsed": None, "parsing_error": "bad json"},
    ],
)
async def test_unparseable_output_raises(output: dict[str, Any]) -> None:
    fake = FakeChatModel(output)

    with pytest.raises(ValueError, match="did not match Extraction"):
        await LLMExtractor(fake.as_llm()).extract("text", [])


class FakeToolModel:
    """Plays back scripted AI turns and records what the agent sent."""

    def __init__(self, *turns: AIMessage | Exception) -> None:
        self.turns = list(turns)
        self.requests: list[list[BaseMessage]] = []
        self.bindings: list[tuple[list[str], Any]] = []

    def bind_tools(self, tools: list[Any], tool_choice: Any = None) -> Any:
        names = [getattr(t, "name", None) or t.__name__ for t in tools]
        self.bindings.append((names, tool_choice))
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.requests.append(list(messages))
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def as_llm(self) -> BaseChatModel:
        return cast(BaseChatModel, self)


def call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def submit(
    decision: str, rationale: str, call_id: str = "s", addressed: tuple[str, ...] = ()
) -> AIMessage:
    return call(
        SUBMIT,
        call_id,
        decision=decision,
        rationale=rationale,
        addressed=list(addressed),
    )


@pytest.fixture
def tools(inventory: Inventory) -> ReviewTools:
    return ReviewTools(inventory, SeenInvoices(), INVOICE)


async def test_approver_looks_things_up_before_deciding(tools: ReviewTools) -> None:
    fake = FakeToolModel(
        call("check_inventory", "c1", item="GadgetX"),
        submit("reject", "Only 5 GadgetX in stock.", addressed=("OUT_OF_STOCK",)),
    )

    out = await LLMApprover(fake.as_llm()).draft(INVOICE, report(ERROR), [], tools)

    assert out.draft == ApprovalDraft(
        decision="reject",
        rationale="Only 5 GadgetX in stock.",
        addressed=["OUT_OF_STOCK"],
    )
    assert [c.describe() for c in out.tool_calls] == [
        "check_inventory(GadgetX): GadgetX: 5 in stock, list price $750.00."
    ]
    names, choice = fake.bindings[0]
    assert names == ["check_inventory", "vendor_history", "convert_to_usd", SUBMIT]
    assert choice == "any"
    # The tool's answer went back to the model with the matching call id.
    reply = fake.requests[1][-1]
    assert isinstance(reply, ToolMessage)
    assert reply.tool_call_id == "c1"
    assert "5 in stock" in reply.content


async def test_approver_prompt_has_findings_and_critique(tools: ReviewTools) -> None:
    fake = FakeToolModel(
        call("check_inventory", "c1", item="FakeItem"),
        submit("reject", "Out of stock.", addressed=("OUT_OF_STOCK",)),
    )

    await LLMApprover(fake.as_llm()).draft(
        INVOICE, report(ERROR), ["You ignored the stock error"], tools
    )

    prompt = str(fake.requests[0][0].content)
    assert "[error] OUT_OF_STOCK: FakeItem out. 0 left" in prompt
    assert "A reviewer rejected your previous draft" in prompt
    assert "- You ignored the stock error" in prompt
    assert '"invoice_number": "INV-1"' in prompt


async def test_approver_with_clean_report(tools: ReviewTools) -> None:
    fake = FakeToolModel(submit("approve", "ok"))

    out = await LLMApprover(fake.as_llm()).draft(INVOICE, report(), [], tools)

    assert out.tool_calls == []
    prompt = str(fake.requests[0][0].content)
    assert "No findings: every check passed." in prompt
    assert "reviewer rejected" not in prompt


async def test_approver_is_nudged_when_it_answers_in_prose(tools: ReviewTools) -> None:
    fake = FakeToolModel(
        AIMessage("I think we should approve."), submit("approve", "ok")
    )

    out = await LLMApprover(fake.as_llm()).draft(INVOICE, report(), [], tools)

    assert out.draft.decision == "approve"
    assert "call SubmitDecision" in str(fake.requests[1][-1].content)


async def test_approver_retries_an_invalid_decision(tools: ReviewTools) -> None:
    fake = FakeToolModel(submit("pay it", "sure", "bad"), submit("approve", "ok"))

    out = await LLMApprover(fake.as_llm()).draft(INVOICE, report(), [], tools)

    assert out.draft.decision == "approve"
    feedback = fake.requests[1][-1]
    assert isinstance(feedback, ToolMessage)
    assert "Invalid decision" in str(feedback.content)


async def test_guardrail_sends_back_a_decision_that_skips_a_finding(
    tools: ReviewTools,
) -> None:
    fake = FakeToolModel(
        submit("approve", "Looks fine.", "s1"),
        call("check_inventory", "c1", item="FakeItem"),
        submit("reject", "No stock.", "s2", addressed=("OUT_OF_STOCK",)),
    )

    out = await LLMApprover(fake.as_llm()).draft(INVOICE, report(ERROR), [], tools)

    assert out.draft.decision == "reject"
    [guard, lookup] = out.steps
    assert guard.name == "guardrail"
    assert "Approve is not allowed with error findings (OUT_OF_STOCK)." in guard.result
    assert "Missing: OUT_OF_STOCK." in guard.result
    assert "call check_inventory" in guard.result
    assert out.tool_calls == [lookup]
    reply = fake.requests[1][-1]
    assert isinstance(reply, ToolMessage)
    assert str(reply.content).startswith("Decision not accepted.")


async def test_approver_must_decide_when_out_of_steps(tools: ReviewTools) -> None:
    fake = FakeToolModel(
        call("convert_to_usd", "c1", amount=100, currency="EUR"),
        call("convert_to_usd", "c2", amount=100, currency="EUR"),
        submit("escalate", "Still unsure."),
    )

    out = await LLMApprover(fake.as_llm(), max_steps=2).draft(
        INVOICE, report(WARNING), [], tools
    )

    assert out.draft.decision == "escalate"
    assert len(out.tool_calls) == 2
    assert fake.bindings[-1] == ([SUBMIT], SUBMIT)


async def test_approver_that_never_decides_is_an_error(tools: ReviewTools) -> None:
    fake = FakeToolModel(call("vendor_history", "c1", vendor="X"), AIMessage("no"))

    with pytest.raises(ValueError, match="did not submit"):
        await LLMApprover(fake.as_llm(), max_steps=1).draft(
            INVOICE, report(), [], tools
        )


async def test_critic_sees_the_draft_and_the_lookups() -> None:
    critique = Critique(agrees=False, issues=["EUR needs FX"])
    fake = FakeChatModel(parsed(critique))
    draft = ApprovalDraft(decision="approve", rationale="Looks fine")
    checked = [ToolCall(name="convert_to_usd", args={"a": 1}, result="$1.08")]

    out = await LLMCritic(fake.as_llm()).review(
        INVOICE, report(WARNING), draft, checked
    )

    assert out == critique
    prompt = fake.prompts[0]
    assert "Draft decision: approve" in prompt
    assert "Draft rationale: Looks fine" in prompt
    assert "FOREIGN_CURRENCY" in prompt
    assert "- convert_to_usd(1): $1.08" in prompt


async def test_critic_is_told_when_nothing_was_checked() -> None:
    fake = FakeChatModel(parsed(Critique(agrees=True)))
    draft = ApprovalDraft(decision="approve", rationale="ok")

    await LLMCritic(fake.as_llm()).review(INVOICE, report(), draft, [])

    assert "What the approver looked up:\nNothing." in fake.prompts[0]


async def test_no_extractor_explains_itself() -> None:
    with pytest.raises(ExtractionUnavailableError, match="needs an LLM"):
        await NoExtractor().extract("text", [])


async def test_rule_approver_dedupes_titles(tools: ReviewTools) -> None:
    out = await RuleApprover().draft(INVOICE, report(ERROR, ERROR), [], tools)

    assert out.draft.decision == "reject"
    assert out.draft.rationale == "Hard failures: FakeItem out."
    assert out.tool_calls == []


async def test_rule_approver_escalates_warnings_and_approves_clean(
    tools: ReviewTools,
) -> None:
    warned = await RuleApprover().draft(INVOICE, report(WARNING), [], tools)
    clean = await RuleApprover().draft(INVOICE, report(), [], tools)

    assert warned.draft.decision == "escalate"
    assert clean.draft.decision == "approve"


async def test_rule_critic_catches_ignored_errors() -> None:
    lenient = ApprovalDraft(decision="approve", rationale="fine")
    strict = ApprovalDraft(decision="reject", rationale="error")

    assert not (await RuleCritic().review(INVOICE, report(ERROR), lenient, [])).agrees
    assert (await RuleCritic().review(INVOICE, report(ERROR), strict, [])).agrees
