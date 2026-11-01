"""The LangGraph workflow: ingest -> validate -> [triage] -> review <-> critique
-> pay/reject.

Two loops are edges in the graph: extraction retries with feedback, and the
approver revises its draft until the critic agrees. Invoices that need a
human pause in `human_review` until a decision is sent with Command(resume=...).
"""

from __future__ import annotations

import asyncio
import logging
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_config, get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from invoice_agents.agents import GUARDRAIL, Extraction, ExtractionUnavailableError
from invoice_agents.approval import (
    Outcome,
    Route,
    policy_route,
    resolve,
    unresolved,
    vp_reason,
)
from invoice_agents.config import Settings
from invoice_agents.ingestion import load_document
from invoice_agents.ingestion.ocr import BACKENDS as OCR_BACKENDS
from invoice_agents.inventory import Inventory
from invoice_agents.llm import Agents
from invoice_agents.models import (
    ApprovalDraft,
    Critique,
    Finding,
    HumanDecision,
    Invoice,
    LineItem,
    PaymentResult,
    Severity,
    ToolCall,
    ValidationReport,
)
from invoice_agents.payment import MockBank
from invoice_agents.tools import ReviewTools
from invoice_agents.triage import TriageAnswer, TriageError, fast_path
from invoice_agents.usage import Pricing, StepUsage, Tokens, collecting, step_usage
from invoice_agents.validation import (
    Policy,
    SeenInvoices,
    renamed_from,
    ungrounded,
    validate,
)

STATE_TYPES = (
    Invoice,
    LineItem,
    Finding,
    Severity,
    ValidationReport,
    ApprovalDraft,
    Critique,
    HumanDecision,
    PaymentResult,
    ToolCall,
    TriageAnswer,
)


def checkpoint_serde() -> JsonPlusSerializer:
    """A serializer that rebuilds only our own state types from a checkpoint."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in STATE_TYPES]
    )


log = logging.getLogger(__name__)


class TraceEvent(TypedDict):
    stage: str
    message: str
    # Set on lines that record a tool call, to the tool's name.
    tool: NotRequired[str]
    # Tokens and cost of the model calls this step made, on its last line.
    usage: NotRequired[StepUsage]
    # Already sent live while the node ran (see `review`): clients reading
    # the node's update skip it; the state keeps it for summaries.
    streamed: NotRequired[bool]
    # What the step produced, for clients to show: the extracted invoice,
    # the validation findings, the payment.
    data: NotRequired[dict[str, Any]]


class InvoiceState(TypedDict, total=False):
    path: str
    # What to call the file: an upload is stored under a random name.
    source: str
    source_format: str
    invoice: Invoice | None
    corrections: list[str]
    # What the extraction model reported that the document doesn't say.
    ungrounded: list[str]
    # A "formerly ..." line in the document: the old name, for validation.
    renamed_from: str | None
    extract_feedback: list[str]
    extract_attempts: int
    report: ValidationReport
    route: Route
    route_reason: str | None
    triage: TriageAnswer
    draft: ApprovalDraft
    tool_calls: list[ToolCall]
    critique: Critique
    review_rounds: int
    outcome: Outcome
    human: HumanDecision
    payment: PaymentResult
    rejection_reason: str
    trace: Annotated[list[TraceEvent], operator.add]


@dataclass
class Deps:
    agents: Agents
    inventory: Inventory
    settings: Settings
    seen: SeenInvoices = field(default_factory=SeenInvoices)
    # The mock bank: pays once per idempotency key.
    bank: MockBank = field(default_factory=MockBank)
    # Model prices, for the cost of each step; empty means tokens only.
    pricing: Pricing = field(default_factory=dict)

    @property
    def policy(self) -> Policy:
        return Policy(
            approval_threshold=self.settings.approval_threshold,
            near_threshold_ratio=self.settings.near_threshold_ratio,
            vp_required=frozenset(self.settings.vp_required_findings),
        )


def _current_run() -> str | None:
    """The run (thread) this node is executing in, for the logs."""
    try:
        return _run_id(get_config()) or None
    except RuntimeError:  # called outside a graph run
        return None


def _run_id(config: RunnableConfig) -> str:
    """The run's thread id: its key in the ledger and the checkpointer."""
    return str(config.get("configurable", {}).get("thread_id", ""))


def _tool_line(step: ToolCall) -> TraceEvent:
    return {
        "stage": "approve",
        "message": f"Sent the decision back. {step.result}"
        if step.name == GUARDRAIL
        else step.describe(),
        "tool": step.name,
    }


def _emit(stage: str, status: str, **data: Any) -> None:
    get_stream_writer()({"stage": stage, "status": status, **data})


def _event(
    stage: str,
    message: str,
    usage: StepUsage | None = None,
    data: dict[str, Any] | None = None,
) -> list[TraceEvent]:
    event = TraceEvent(stage=stage, message=message)
    log.info(
        "%s: %s",
        stage,
        message,
        extra={"stage": stage, "run_id": _current_run(), "usage": usage},
    )
    if usage:
        event["usage"] = usage
    if data:
        event["data"] = data
    return [event]


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _invoice(state: InvoiceState) -> Invoice:
    invoice = state.get("invoice")
    if invoice is None:
        raise RuntimeError("No invoice in state; ingest must run first")
    return invoice


def _extraction_feedback(extraction: Extraction) -> list[str]:
    """Problems that point to a bad extraction rather than a bad invoice."""
    inv = extraction.invoice
    problems = []
    if not inv.line_items:
        problems.append("No line items were extracted.")
    if inv.total is None:
        problems.append("No total was extracted; look for TOTAL, Amt or Total Amount.")
    return problems


def build_graph(
    deps: Deps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    settings = deps.settings

    async def ingest(state: InvoiceState) -> InvoiceState:
        _emit("ingest", "start")
        # Disk, PDF parsing and OCR all block: keep them off the event loop.
        doc = await asyncio.to_thread(
            load_document, state["path"], OCR_BACKENDS[settings.ocr_backend]
        )
        if doc.invoice is not None:
            return {
                "source_format": doc.format.value,
                "invoice": doc.invoice,
                "renamed_from": renamed_from(doc.text),
                "extract_feedback": [],
                "trace": _event(
                    "ingest",
                    f"Parsed {doc.format.value.upper()} in code.",
                    data={"invoice": doc.invoice.model_dump(mode="json")},
                ),
            }
        if not doc.text.strip():
            # Nothing to extract from, so no attempt would help: reject it.
            problem = doc.problem or "The file has no text to read."
            return {
                "source_format": doc.format.value,
                "invoice": None,
                "extract_attempts": settings.max_extract_attempts,
                "extract_feedback": [problem],
                "trace": _event("ingest", f"Unreadable. {problem}"),
            }

        attempt = state.get("extract_attempts", 0) + 1
        tokens: Tokens = {}
        try:
            with collecting() as tokens:
                extraction = await deps.agents.extractor.extract(
                    doc.text, state.get("extract_feedback", [])
                )
        except ExtractionUnavailableError as exc:
            if not doc.problem:
                raise  # Free text with no LLM: skipped, not rejected.
            # A structured file code couldn't read, and no agent to read it.
            problem = f"{doc.problem} Reading it as text needs an LLM ({exc})"
            return {
                "source_format": doc.format.value,
                "invoice": None,
                "extract_attempts": settings.max_extract_attempts,
                "extract_feedback": [problem],
                "trace": _event("ingest", f"Unreadable. {problem}"),
            }
        except ValueError as exc:
            return {
                "source_format": doc.format.value,
                "invoice": None,
                "extract_attempts": attempt,
                "extract_feedback": [str(exc)],
                "trace": _event(
                    "ingest",
                    f"Attempt {attempt}: output rejected ({exc}).",
                    step_usage(tokens, deps.pricing),
                ),
            }
        feedback = _extraction_feedback(extraction)
        message = f"Attempt {attempt}: extracted with the LLM"
        if doc.problem:
            message = f"{doc.problem} The extraction agent read its text. " + message
        if doc.ocr:
            message = f"Read the scan with OCR ({settings.ocr_backend}). " + message
        if feedback:
            message += f", retrying: {' '.join(feedback)}"
        return {
            "source_format": doc.format.value,
            "invoice": extraction.invoice,
            "renamed_from": renamed_from(doc.text),
            "corrections": extraction.corrections,
            "ungrounded": ungrounded(
                extraction.invoice, doc.text, extraction.corrections
            ),
            "extract_attempts": attempt,
            "extract_feedback": feedback,
            "trace": _event(
                "ingest",
                message + ".",
                step_usage(tokens, deps.pricing),
                data={
                    "invoice": extraction.invoice.model_dump(mode="json"),
                    "corrections": extraction.corrections,
                },
            ),
        }

    def after_ingest(state: InvoiceState) -> Literal["ingest", "validate", "reject"]:
        if state.get("extract_feedback"):
            if state.get("extract_attempts", 0) < settings.max_extract_attempts:
                return "ingest"
            return "reject" if state.get("invoice") is None else "validate"
        return "validate"

    async def validate_node(
        state: InvoiceState, config: RunnableConfig
    ) -> InvoiceState:
        _emit("validate", "start")
        invoice = _invoice(state)
        # Checking and recording the invoice stay one step on the event loop,
        # with no await between them, so runs that overlap see each other for
        # duplicates and splits. The checks are fast SQLite lookups.
        old = state.get("renamed_from")
        document = f"formerly {old}" if old else None
        report = validate(invoice, deps.inventory, deps.seen, deps.policy, document)
        report.findings += [
            Finding(
                code="EXTRACTION_CORRECTION",
                severity=Severity.INFO,
                title="Extraction correction",
                detail=correction,
            )
            for correction in state.get("corrections", [])
        ]
        if missing := state.get("ungrounded"):
            report.findings.append(
                Finding(
                    code="NOT_IN_DOCUMENT",
                    severity=Severity.WARNING,
                    title="Extracted values the document doesn't show",
                    detail="The extraction reported "
                    + ", ".join(missing)
                    + ", which the document's text does not contain.",
                )
            )
        deps.seen.add(
            invoice,
            run_id=_run_id(config),
            source=state.get("source") or Path(state["path"]).name,
        )
        route = policy_route(invoice, report, deps.policy)
        reason = vp_reason(invoice, report, deps.policy) if route == "human" else None
        return {
            "report": report,
            "route": route,
            "route_reason": reason,
            "trace": _event(
                "validate",
                f"{_count(len(report.errors), 'error')},"
                f" {_count(len(report.warnings), 'warning')};"
                f" policy route: {route}" + (f" ({reason})." if reason else "."),
                data={"findings": [f.model_dump(mode="json") for f in report.findings]},
            ),
        }

    def after_validate(state: InvoiceState) -> Literal["triage", "review"]:
        return "triage" if deps.agents.triager else "review"

    async def triage(state: InvoiceState) -> InvoiceState:
        """System 1: settle the easy cases without the reasoning approver."""
        triager = deps.agents.triager
        assert triager is not None
        _emit("approve", "triage")
        report = state["report"]
        try:
            answer = await triager.ask(_invoice(state), report)
        except TriageError as exc:
            return {"trace": _event("triage", f"{exc} Sending it to the approver.")}
        update: InvoiceState = {"triage": answer}
        # The Decisions API reports its own cost; decision models bill input only.
        use = StepUsage(
            model=answer.model,
            input_tokens=answer.input_tokens,
            output_tokens=0,
            cost_usd=answer.cost_usd,
        )
        quick = fast_path(
            answer,
            report,
            settings.triage_min_confidence,
            settings.triage_max_suspicion,
        )
        if quick is None:
            update["trace"] = _event(
                "triage", f"{answer.describe()}. Needs the full review.", use
            )
            return update
        decision, confirmed = quick
        update["draft"] = ApprovalDraft(
            decision=decision,
            rationale=f"Triage: {answer.describe()}; code confirmed {confirmed}.",
            addressed=list(dict.fromkeys(f.code for f in report.findings)),
        )
        update["outcome"] = resolve(decision, state["route"])
        update["trace"] = _event(
            "triage",
            f"{answer.describe()}. Code confirmed {confirmed}, so the reasoning"
            f" approver was skipped. Outcome: {update['outcome']}.",
            use,
        )
        return update

    async def review(state: InvoiceState) -> InvoiceState:
        _emit("approve", "draft")
        critique = state.get("critique")
        issues = critique.issues if critique and not critique.agrees else []
        invoice = _invoice(state)
        write = get_stream_writer()
        streamed: set[int] = set()

        def live(step: ToolCall) -> None:
            # Each tool call as it finishes, not all of them with the draft.
            write({"stage": "approve", "status": "tool", "trace": _tool_line(step)})
            streamed.add(id(step))

        tools = ReviewTools(deps.inventory, deps.seen, invoice, on_step=live)
        with collecting() as tokens:
            result = await deps.agents.approver.draft(
                invoice, state["report"], issues, tools
            )
        draft = result.draft
        rounds = state.get("review_rounds", 0) + 1
        trace: list[TraceEvent] = [
            {**_tool_line(step), "streamed": id(step) in streamed}
            for step in result.steps
        ]
        trace += _event(
            "approve",
            f"Draft {rounds}: {draft.decision}. {draft.rationale}",
            step_usage(tokens, deps.pricing),
        )
        return {
            "draft": draft,
            "tool_calls": result.tool_calls,
            "review_rounds": rounds,
            "trace": trace,
        }

    async def critique_node(state: InvoiceState) -> InvoiceState:
        _emit("approve", "critique")
        draft = state["draft"]
        with collecting() as tokens:
            critique = await deps.agents.critic.review(
                _invoice(state), state["report"], draft, state.get("tool_calls", [])
            )
        update: InvoiceState = {"critique": critique}
        if critique.agrees:
            message = "Critic agrees."
        else:
            message = f"Critic objects: {' '.join(critique.issues)}"
        if critique.agrees:
            update["outcome"] = resolve(draft.decision, state["route"])
            message += f" Outcome: {update['outcome']}."
        elif state["review_rounds"] >= settings.max_review_rounds:
            # Out of rounds and still disputed: neither agent gets the last
            # word. A person decides, unless the rules already reject it.
            update["outcome"] = unresolved(state["route"])
            message += (
                f" Still disputed after {state['review_rounds']} rounds."
                f" Outcome: {update['outcome']}."
            )
        update["trace"] = _event("approve", message, step_usage(tokens, deps.pricing))
        return update

    def after_critique(
        state: InvoiceState,
    ) -> Literal["review", "human_review", "pay", "reject"]:
        outcome = state.get("outcome")
        if outcome is None:
            return "review"
        return "human_review" if outcome == "human" else outcome

    after_triage = after_critique

    def human_review(state: InvoiceState) -> InvoiceState:
        # Nothing with side effects before interrupt(): on resume this node
        # runs again from the top.
        answer = interrupt(
            {
                "invoice": _invoice(state).model_dump(),
                "findings": [f.model_dump() for f in state["report"].findings],
                "draft": state["draft"].model_dump(),
            }
        )
        decision = HumanDecision.model_validate(answer)
        verdict = "approved" if decision.approved else "rejected"
        who = f" ({decision.by})" if decision.by else ""
        note = f" Note: {decision.note}" if decision.note else ""
        return {
            "human": decision,
            "trace": _event("approve", f"VP {verdict}{who}.{note}"),
        }

    def after_human(state: InvoiceState) -> Literal["pay", "reject"]:
        return "pay" if state["human"].approved else "reject"

    async def pay(state: InvoiceState, config: RunnableConfig) -> InvoiceState:
        _emit("pay", "start")
        invoice = _invoice(state)
        result = deps.bank.pay(invoice)
        paid = result.status == "success"
        deps.seen.settle(
            _run_id(config),
            "paid" if paid else "failed",
            None if paid else f"mock_payment returned {result.status}",
        )
        return {
            "payment": result,
            "trace": _event(
                "pay",
                f"Already paid under key {result.idempotency_key}; no money moved"
                " this time."
                if result.duplicate
                else f"mock_payment returned {result.status}.",
                data={"payment": result.model_dump(mode="json")},
            ),
        }

    async def reject(state: InvoiceState, config: RunnableConfig) -> InvoiceState:
        _emit("pay", "skipped")
        if state.get("human") and not state["human"].approved:
            who = f" ({state['human'].by})" if state["human"].by else ""
            reason = f"Rejected by the VP{who}. {state['human'].note}".strip()
        elif "draft" in state:
            reason = state["draft"].rationale
        else:
            reason = "Could not extract the invoice: " + " ".join(
                state.get("extract_feedback", [])
            )
        deps.seen.settle(_run_id(config), "rejected", reason)
        return {
            "rejection_reason": reason,
            "trace": _event("pay", f"Not paid. {reason}"),
        }

    graph = StateGraph(InvoiceState)
    graph.add_node("ingest", ingest)
    graph.add_node("validate", validate_node)
    graph.add_node("triage", triage)
    graph.add_node("review", review)
    graph.add_node("critique", critique_node)
    graph.add_node("human_review", human_review)
    graph.add_node("pay", pay)
    graph.add_node("reject", reject)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", after_ingest)
    graph.add_conditional_edges("validate", after_validate)
    graph.add_conditional_edges("triage", after_triage)
    graph.add_edge("review", "critique")
    graph.add_conditional_edges("critique", after_critique)
    graph.add_conditional_edges("human_review", after_human)
    graph.add_edge("pay", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=checkpointer)
