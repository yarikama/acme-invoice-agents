"""The agents: an extractor, an approver and a critic.

Each has an LLM version and a rule-based one. The rule-based ones run with
no network, so tests are deterministic and the CLI works without a key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from invoice_agents.guardrails import lint_decision
from invoice_agents.models import (
    ApprovalDraft,
    Critique,
    Invoice,
    ToolCall,
    ValidationReport,
)
from invoice_agents.tools import ReviewTools
from invoice_agents.usage import record


class Extraction(BaseModel):
    invoice: Invoice
    corrections: list[str] = Field(
        default_factory=list,
        description="Each place where the text was ambiguous or misread and what "
        "you chose, e.g. 'read 2O26 as 2026 (letter O for zero)'",
    )


GUARDRAIL = "guardrail"


@dataclass(frozen=True)
class Review:
    """An approval draft and the steps the approver took to reach it.

    Steps are tool calls in order, plus a `guardrail` step each time a
    submitted decision failed the linter and went back to the model.
    """

    draft: ApprovalDraft
    steps: list[ToolCall] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s for s in self.steps if s.name != GUARDRAIL]


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
rationale that cites the findings and what your tool calls showed, and with
the code of every error and warning finding in `addressed`. Code checks the
decision before accepting it and will send it back if anything is missing."""

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


# --- Provider hiccups -----------------------------------------------------

LLM_ATTEMPTS = 3
# Status codes OpenRouter and xAI use for overload and upstream failures.
TRANSIENT_CODES = {408, 429, 500, 502, 503, 504, 524, 529}
# Packages that turn a provider's response into messages. An exception raised
# inside one of them on a malformed body is the provider's hiccup, not ours.
CLIENT_PACKAGES = ("langchain_openai", "langchain_xai", "openai")


def raised_at(exc: BaseException) -> tuple[Path, int] | None:
    """File and line of the innermost frame that raised exc."""
    tb = exc.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    return Path(tb.tb_frame.f_code.co_filename), tb.tb_lineno


def is_transient(exc: BaseException) -> bool:
    """A provider failure worth retrying, as opposed to a bug or a bad answer.

    The HTTP client already retries 429s and 5xx responses. These are the
    failures it can't see: OpenRouter reports an upstream error inside a 200
    body, which langchain-openai raises as ValueError(error_dict), and a body
    with missing fields fails inside the client library.
    """
    if isinstance(exc, ValueError) and exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code") in TRANSIENT_CODES
    if isinstance(exc, (TypeError, KeyError)):
        where = raised_at(exc)
        return where is not None and any(p in where[0].parts for p in CLIENT_PACKAGES)
    return False


async def with_retries[T](
    call: Callable[[], Awaitable[T]], attempts: int = LLM_ATTEMPTS
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            await asyncio.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")  # pragma: no cover


async def _structured[T: BaseModel](
    llm: BaseChatModel, schema: type[T], prompt: str
) -> T:
    runnable = llm.with_structured_output(
        schema, method="json_schema", include_raw=True
    )
    # include_raw=True always returns {raw, parsed, parsing_error}.
    out = cast(dict[str, Any], await with_retries(lambda: runnable.ainvoke(prompt)))
    record(out.get("raw"))
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
        steps: list[ToolCall] = []
        agent = self.llm.bind_tools(
            [*tools.as_langchain(), SubmitDecision], tool_choice="any"
        )
        for _ in range(self.max_steps):
            reply = cast(AIMessage, await with_retries(lambda: agent.ainvoke(messages)))
            record(reply)
            messages.append(reply)
            if not reply.tool_calls:
                messages.append(HumanMessage("Call a tool, or call SubmitDecision."))
                continue
            draft: ApprovalDraft | None = None
            for tc in reply.tool_calls:
                # Every call gets an answer, or the next request is invalid.
                if tc["name"] == SUBMIT:
                    before = len(steps)
                    result, draft = self._check(invoice, report, tc["args"], steps)
                    for sent_back in steps[before:]:
                        tools.note(sent_back)
                else:
                    call = tools.run(tc["name"], tc["args"])
                    steps.append(call)
                    result = call.result
                messages.append(ToolMessage(result, tool_call_id=tc["id"] or ""))
            if draft is not None:
                return Review(draft, steps)

        # Out of steps: decide with what has been gathered so far.
        final = self.llm.bind_tools([SubmitDecision], tool_choice=SUBMIT)
        reply = cast(AIMessage, await with_retries(lambda: final.ainvoke(messages)))
        record(reply)
        for tc in reply.tool_calls:
            if tc["name"] == SUBMIT:
                try:
                    # Accepted as is: the policy route still overrides it in code.
                    return Review(ApprovalDraft.model_validate(tc["args"]), steps)
                except ValidationError:
                    break
        # Models occasionally ignore even a forced tool choice. No decision is
        # not a reason to fail the run or to pay: hand it to a person.
        return Review(
            ApprovalDraft(
                decision="escalate",
                rationale=f"The approver did not reach a valid decision in"
                f" {self.max_steps + 1} turns, so a person should decide.",
                addressed=[f.code for f in report.findings],
            ),
            steps,
        )

    @staticmethod
    def _check(
        invoice: Invoice,
        report: ValidationReport,
        args: dict[str, Any],
        steps: list[ToolCall],
    ) -> tuple[str, ApprovalDraft | None]:
        """Validate and lint a submitted decision: (reply to the model, draft)."""
        try:
            draft = ApprovalDraft.model_validate(args)
        except ValidationError as exc:
            return f"Invalid decision, try again: {exc}", None
        calls = [s for s in steps if s.name != GUARDRAIL]
        problems = lint_decision(invoice, report, draft, calls)
        if not problems:
            return "Decision recorded.", draft
        text = " ".join(problems)
        steps.append(ToolCall(name=GUARDRAIL, args={}, result=text))
        return f"Decision not accepted. Fix this and submit again: {text}", None


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


# Paperwork warnings the rule-based approver lets through on a first draft.
# Its critic doesn't, so offline runs show the review loop at work.
PAPERWORK = frozenset(
    {"TERMS_MISMATCH", "DUE_BEFORE_ISSUE", "UNPARSEABLE_DATE", "STOCK_DEPLETED"}
)


class RuleApprover:
    """Decides from the findings alone; it has no use for tools.

    A first draft approves past paperwork warnings, the way a busy clerk
    might. When the critic objects, the next draft escalates instead.
    """

    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review:
        codes = list(dict.fromkeys(f.code for f in report.findings))
        warnings = {f.code for f in report.warnings}
        if report.errors:
            titles = "; ".join(dict.fromkeys(f.title for f in report.errors))
            draft = ApprovalDraft(
                decision="reject",
                rationale=f"Hard failures: {titles}.",
                addressed=codes,
            )
        elif critique:
            draft = ApprovalDraft(
                decision="escalate",
                rationale=f"The critic objected: {' '.join(critique)}"
                " A person should decide.",
                addressed=codes,
            )
        elif warnings and not warnings <= PAPERWORK:
            titles = "; ".join(dict.fromkeys(f.title for f in report.warnings))
            draft = ApprovalDraft(
                decision="escalate",
                rationale=f"Needs a human look: {titles}.",
                addressed=codes,
            )
        elif warnings:
            draft = ApprovalDraft(
                decision="approve",
                rationale=f"Only paperwork warnings ({', '.join(sorted(warnings))});"
                " the goods and amounts check out.",
                addressed=codes,
            )
        else:
            draft = ApprovalDraft(
                decision="approve", rationale="Every check passed.", addressed=codes
            )
        return Review(draft)


class RuleCritic:
    """Objects to any draft that lets an error or a warning through."""

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
        if report.warnings and draft.decision == "approve":
            codes = ", ".join(sorted({f.code for f in report.warnings}))
            return Critique(
                agrees=False,
                issues=[
                    f"The draft approves with {codes} unresolved; nothing in the"
                    " record clears them."
                ],
            )
        return Critique(agrees=True)
