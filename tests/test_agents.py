"""The agents, with a fake chat model standing in for Grok."""

from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from invoice_agents.agents import (
    Extraction,
    ExtractionUnavailableError,
    LLMApprover,
    LLMCritic,
    LLMExtractor,
    NoExtractor,
    RuleApprover,
    RuleCritic,
)
from invoice_agents.models import (
    ApprovalDraft,
    Critique,
    Finding,
    Invoice,
    Severity,
    ValidationReport,
)

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

    def __init__(self, *outputs: dict[str, Any]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.schemas: list[type[BaseModel]] = []

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Any:
        assert kwargs == {"method": "json_schema", "include_raw": True}
        self.schemas.append(schema)
        return self

    async def ainvoke(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        return self.outputs.pop(0)

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


async def test_approver_includes_findings_and_critique() -> None:
    draft = ApprovalDraft(decision="reject", rationale="Out of stock.")
    fake = FakeChatModel(parsed(draft))

    out = await LLMApprover(fake.as_llm()).draft(
        INVOICE, report(ERROR), ["You ignored the stock error"]
    )

    assert out == draft
    [prompt] = fake.prompts
    assert "[error] OUT_OF_STOCK: FakeItem out. 0 left" in prompt
    assert "A reviewer rejected your previous draft" in prompt
    assert "- You ignored the stock error" in prompt
    assert '"invoice_number": "INV-1"' in prompt


async def test_approver_with_clean_report() -> None:
    fake = FakeChatModel(parsed(ApprovalDraft(decision="approve", rationale="ok")))

    await LLMApprover(fake.as_llm()).draft(INVOICE, report(), [])

    assert "No findings: every check passed." in fake.prompts[0]
    assert "reviewer rejected" not in fake.prompts[0]


async def test_critic_sees_the_draft() -> None:
    critique = Critique(agrees=False, issues=["EUR needs FX"])
    fake = FakeChatModel(parsed(critique))
    draft = ApprovalDraft(decision="approve", rationale="Looks fine")

    out = await LLMCritic(fake.as_llm()).review(INVOICE, report(WARNING), draft)

    assert out == critique
    assert "Draft decision: approve" in fake.prompts[0]
    assert "Draft rationale: Looks fine" in fake.prompts[0]
    assert "FOREIGN_CURRENCY" in fake.prompts[0]


async def test_no_extractor_explains_itself() -> None:
    with pytest.raises(ExtractionUnavailableError, match="needs an LLM"):
        await NoExtractor().extract("text", [])


async def test_rule_approver_dedupes_titles() -> None:
    draft = await RuleApprover().draft(INVOICE, report(ERROR, ERROR), [])

    assert draft.decision == "reject"
    assert draft.rationale == "Hard failures: FakeItem out."


async def test_rule_approver_escalates_warnings_and_approves_clean() -> None:
    assert (await RuleApprover().draft(INVOICE, report(WARNING), [])).decision == (
        "escalate"
    )
    assert (await RuleApprover().draft(INVOICE, report(), [])).decision == "approve"


async def test_rule_critic_catches_ignored_errors() -> None:
    lenient = ApprovalDraft(decision="approve", rationale="fine")
    strict = ApprovalDraft(decision="reject", rationale="error")

    assert not (await RuleCritic().review(INVOICE, report(ERROR), lenient)).agrees
    assert (await RuleCritic().review(INVOICE, report(ERROR), strict)).agrees
