"""The agents: an extractor, an approver and a critic.

Each has an LLM version and a rule-based one. The rule-based ones run with
no network, so tests are deterministic and the CLI works without a key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from invoice_agents.models import (
    ApprovalDraft,
    Critique,
    Invoice,
    ToolCall,
    ValidationReport,
)
from invoice_agents.tools import ReviewTools


class Extraction(BaseModel):
    invoice: Invoice
    corrections: list[str] = Field(
        default_factory=list,
        description="Each place where the text was ambiguous or misread and what "
        "you chose, e.g. 'read 2O26 as 2026 (letter O for zero)'",
    )


@dataclass(frozen=True)
class Review:
    """An approval draft and the tool calls the approver made to reach it."""

    draft: ApprovalDraft
    tool_calls: list[ToolCall] = field(default_factory=list)


class SubmitDecision(ApprovalDraft):
    """Submit your final decision. Call this once, after checking what you need."""


SUBMIT = SubmitDecision.__name__


class ExtractionUnavailableError(RuntimeError):
    """Free-text invoices need an LLM, and none is configured."""


class Extractor(Protocol):
    async def extract(self, text: str, feedback: list[str]) -> Extraction: ...


class Approver(Protocol):
    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review: ...


class Critic(Protocol):
    async def review(
        self,
        invoice: Invoice,
        report: ValidationReport,
        draft: ApprovalDraft,
        checked: list[ToolCall],
    ) -> Critique: ...


def _findings_text(report: ValidationReport) -> str:
    if not report.findings:
        return "No findings: every check passed."
    return "\n".join(
        f"- [{f.severity.value}] {f.code}: {f.title}. {f.detail}"
        for f in report.findings
    )


# --- LLM agents -----------------------------------------------------------

EXTRACT_PROMPT = """You extract invoice data for an accounts payable team.

Rules:
- Copy values as printed. Do not fix arithmetic, fill in missing fields, or
  guess. A wrong total must stay wrong; a missing vendor stays null.
- Fix obvious OCR confusions inside numbers and dates (letter O for zero,
  letter l for one) and list each fix in `corrections`.
- Dates in ISO 8601 when you can read them; otherwise the raw text.
- Amounts as plain numbers without currency symbols or thousands separators.
- Put shipping in `shipping`, tax in `tax_amount`, and any free-text notes or
  message to the payer in `notes`.

Invoice text:
<<<
{text}
>>>"""

APPROVE_PROMPT = """You are the VP of Finance reviewing an invoice before payment.

Policy:
- Any [error] finding means reject.
- Totals over $10,000 always need a human VP; you may still recommend.
- Warnings need judgment: approve only if each one is explained and harmless,
  escalate when unsure, reject when they point to fraud or a bad invoice.

Tools:
- check_inventory: stock and list price of an item
- vendor_history: this vendor's earlier invoices in this batch
- convert_to_usd: the USD value of a foreign-currency amount
Use them to confirm what your decision depends on, for example whether a
foreign total crosses $10,000, or whether a price is far above list. The
findings were computed in code and are correct; do not re-check arithmetic.

Invoice:
{invoice}

Validation findings:
{findings}
{critique}
Decide approve, reject or escalate. Then call SubmitDecision once, with a
rationale that cites the findings and what your tool calls showed."""

CRITIQUE_PROMPT = """You audit approval decisions. Find what the draft got wrong.

Check the draft against the policy (errors mean reject; over $10,000 needs a
human) and against every finding. Did it ignore a warning, accept a weak
explanation, or reject something harmless? Agree only if it holds up.

Invoice:
{invoice}

Findings:
{findings}

What the approver looked up:
{checked}

Draft decision: {decision}
Draft rationale: {rationale}"""


async def _structured[T: BaseModel](
    llm: BaseChatModel, schema: type[T], prompt: str
) -> T:
    runnable = llm.with_structured_output(
        schema, method="json_schema", include_raw=True
    )
    # include_raw=True always returns {raw, parsed, parsing_error}.
    out = cast(dict[str, Any], await runnable.ainvoke(prompt))
    if out.get("parsing_error") or out.get("parsed") is None:
        raise ValueError(
            f"Model output did not match {schema.__name__}: {out.get('parsing_error')}"
        )
    return out["parsed"]


class LLMExtractor:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def extract(self, text: str, feedback: list[str]) -> Extraction:
        prompt = EXTRACT_PROMPT.format(text=text)
        if feedback:
            prompt += (
                "\n\nYour previous attempt had these problems; fix them:\n"
                + "\n".join(f"- {item}" for item in feedback)
            )
        return await _structured(self.llm, Extraction, prompt)


class LLMApprover:
    """A tool-calling agent: looks things up, then submits a decision."""

    def __init__(self, llm: BaseChatModel, max_steps: int = 6) -> None:
        self.llm = llm
        self.max_steps = max_steps

    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review:
        critique_text = (
            "\nA reviewer rejected your previous draft:\n"
            + "\n".join(f"- {c}" for c in critique)
            + "\nRevise your decision.\n"
            if critique
            else ""
        )
        prompt = APPROVE_PROMPT.format(
            invoice=invoice.model_dump_json(indent=2),
            findings=_findings_text(report),
            critique=critique_text,
        )
        messages: list[BaseMessage] = [HumanMessage(prompt)]
        calls: list[ToolCall] = []
        agent = self.llm.bind_tools(
            [*tools.as_langchain(), SubmitDecision], tool_choice="any"
        )
        for _ in range(self.max_steps):
            reply = cast(AIMessage, await agent.ainvoke(messages))
            messages.append(reply)
            if not reply.tool_calls:
                messages.append(HumanMessage("Call a tool, or call SubmitDecision."))
                continue
            draft: ApprovalDraft | None = None
            for tc in reply.tool_calls:
                # Every call gets an answer, or the next request is invalid.
                if tc["name"] == SUBMIT:
                    try:
                        draft = ApprovalDraft.model_validate(tc["args"])
                        result = "Decision recorded."
                    except ValidationError as exc:
                        result = f"Invalid decision, try again: {exc}"
                else:
                    call = tools.run(tc["name"], tc["args"])
                    calls.append(call)
                    result = call.result
                messages.append(ToolMessage(result, tool_call_id=tc["id"] or ""))
            if draft is not None:
                return Review(draft, calls)

        # Out of steps: decide with what has been gathered so far.
        final = self.llm.bind_tools([SubmitDecision], tool_choice=SUBMIT)
        reply = cast(AIMessage, await final.ainvoke(messages))
        for tc in reply.tool_calls:
            if tc["name"] == SUBMIT:
                return Review(ApprovalDraft.model_validate(tc["args"]), calls)
        raise ValueError("The approver did not submit a decision.")


class LLMCritic:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def review(
        self,
        invoice: Invoice,
        report: ValidationReport,
        draft: ApprovalDraft,
        checked: list[ToolCall],
    ) -> Critique:
        prompt = CRITIQUE_PROMPT.format(
            invoice=invoice.model_dump_json(indent=2),
            findings=_findings_text(report),
            checked="\n".join(f"- {c.describe()}" for c in checked) or "Nothing.",
            decision=draft.decision,
            rationale=draft.rationale,
        )
        return await _structured(self.llm, Critique, prompt)


# --- Rule-based agents ----------------------------------------------------


class NoExtractor:
    async def extract(self, text: str, feedback: list[str]) -> Extraction:
        raise ExtractionUnavailableError(
            "This invoice is free text and needs an LLM. Set LLM_PROVIDER and a key."
        )


class RuleApprover:
    """Decides from the findings alone; it has no use for tools."""

    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review:
        if report.errors:
            titles = "; ".join(dict.fromkeys(f.title for f in report.errors))
            draft = ApprovalDraft(
                decision="reject", rationale=f"Hard failures: {titles}."
            )
        elif report.warnings:
            titles = "; ".join(dict.fromkeys(f.title for f in report.warnings))
            draft = ApprovalDraft(
                decision="escalate", rationale=f"Needs a human look: {titles}."
            )
        else:
            draft = ApprovalDraft(decision="approve", rationale="Every check passed.")
        return Review(draft)


class RuleCritic:
    async def review(
        self,
        invoice: Invoice,
        report: ValidationReport,
        draft: ApprovalDraft,
        checked: list[ToolCall],
    ) -> Critique:
        if report.errors and draft.decision != "reject":
            return Critique(
                agrees=False, issues=["The draft ignores an error finding."]
            )
        return Critique(agrees=True)
