"""The agents: an extractor, an approver and a critic.

Each has an LLM version and a rule-based one. The rule-based ones run with
no network, so tests are deterministic and the CLI works without a key.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from invoice_agents.models import ApprovalDraft, Critique, Invoice, ValidationReport


class Extraction(BaseModel):
    invoice: Invoice
    corrections: list[str] = Field(
        default_factory=list,
        description="Each place where the text was ambiguous or misread and what "
        "you chose, e.g. 'read 2O26 as 2026 (letter O for zero)'",
    )


class ExtractionUnavailableError(RuntimeError):
    """Free-text invoices need an LLM, and none is configured."""


class Extractor(Protocol):
    async def extract(self, text: str, feedback: list[str]) -> Extraction: ...


class Approver(Protocol):
    async def draft(
        self, invoice: Invoice, report: ValidationReport, critique: list[str]
    ) -> ApprovalDraft: ...


class Critic(Protocol):
    async def review(
        self, invoice: Invoice, report: ValidationReport, draft: ApprovalDraft
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

Invoice:
{invoice}

Validation findings:
{findings}
{critique}
Decide approve, reject or escalate, and give the rationale citing findings."""

CRITIQUE_PROMPT = """You audit approval decisions. Find what the draft got wrong.

Check the draft against the policy (errors mean reject; over $10,000 needs a
human) and against every finding. Did it ignore a warning, accept a weak
explanation, or reject something harmless? Agree only if it holds up.

Invoice:
{invoice}

Findings:
{findings}

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
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def draft(
        self, invoice: Invoice, report: ValidationReport, critique: list[str]
    ) -> ApprovalDraft:
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
        return await _structured(self.llm, ApprovalDraft, prompt)


class LLMCritic:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def review(
        self, invoice: Invoice, report: ValidationReport, draft: ApprovalDraft
    ) -> Critique:
        prompt = CRITIQUE_PROMPT.format(
            invoice=invoice.model_dump_json(indent=2),
            findings=_findings_text(report),
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
    async def draft(
        self, invoice: Invoice, report: ValidationReport, critique: list[str]
    ) -> ApprovalDraft:
        if report.errors:
            titles = "; ".join(dict.fromkeys(f.title for f in report.errors))
            return ApprovalDraft(
                decision="reject", rationale=f"Hard failures: {titles}."
            )
        if report.warnings:
            titles = "; ".join(dict.fromkeys(f.title for f in report.warnings))
            return ApprovalDraft(
                decision="escalate", rationale=f"Needs a human look: {titles}."
            )
        return ApprovalDraft(decision="approve", rationale="Every check passed.")


class RuleCritic:
    async def review(
        self, invoice: Invoice, report: ValidationReport, draft: ApprovalDraft
    ) -> Critique:
        if report.errors and draft.decision != "reject":
            return Critique(
                agrees=False, issues=["The draft ignores an error finding."]
            )
        return Critique(agrees=True)
