"""The LangGraph workflow: ingest -> validate -> review <-> critique -> pay/reject.

Two loops are edges in the graph: extraction retries with feedback, and the
approver revises its draft until the critic agrees. Invoices that need a
human pause in `human_review` until a decision is sent with Command(resume=...).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from invoice_agents.agents import Extraction
from invoice_agents.approval import Outcome, Route, policy_route, resolve
from invoice_agents.config import Settings
from invoice_agents.ingestion import load_document
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
    ValidationReport,
)
from invoice_agents.payment import mock_payment
from invoice_agents.validation import Policy, SeenInvoices, validate

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
)


def checkpoint_serde() -> JsonPlusSerializer:
    """A serializer that rebuilds only our own state types from a checkpoint."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in STATE_TYPES]
    )


class TraceEvent(TypedDict):
    stage: str
    message: str


class InvoiceState(TypedDict, total=False):
    path: str
    source_format: str
    invoice: Invoice | None
    corrections: list[str]
    extract_feedback: list[str]
    extract_attempts: int
    report: ValidationReport
    route: Route
    draft: ApprovalDraft
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

    @property
    def policy(self) -> Policy:
        return Policy(
            approval_threshold=self.settings.approval_threshold,
            near_threshold_ratio=self.settings.near_threshold_ratio,
        )


def _emit(stage: str, status: str, **data: Any) -> None:
    get_stream_writer()({"stage": stage, "status": status, **data})


def _event(stage: str, message: str) -> list[TraceEvent]:
    return [TraceEvent(stage=stage, message=message)]


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
        doc = load_document(state["path"])
        if doc.invoice is not None:
            return {
                "source_format": doc.format.value,
                "invoice": doc.invoice,
                "extract_feedback": [],
                "trace": _event(
                    "ingest", f"Parsed {doc.format.value.upper()} in code."
                ),
            }

        attempt = state.get("extract_attempts", 0) + 1
        try:
            extraction = await deps.agents.extractor.extract(
                doc.text, state.get("extract_feedback", [])
            )
        except ValueError as exc:
            return {
                "source_format": doc.format.value,
                "invoice": None,
                "extract_attempts": attempt,
                "extract_feedback": [str(exc)],
                "trace": _event(
                    "ingest", f"Attempt {attempt}: output rejected ({exc})."
                ),
            }
        feedback = _extraction_feedback(extraction)
        message = f"Attempt {attempt}: extracted with the LLM"
        if feedback:
            message += f", retrying: {' '.join(feedback)}"
        return {
            "source_format": doc.format.value,
            "invoice": extraction.invoice,
            "corrections": extraction.corrections,
            "extract_attempts": attempt,
            "extract_feedback": feedback,
            "trace": _event("ingest", message + "."),
        }

    def after_ingest(state: InvoiceState) -> Literal["ingest", "validate", "reject"]:
        if state.get("extract_feedback"):
            if state.get("extract_attempts", 0) < settings.max_extract_attempts:
                return "ingest"
            return "reject" if state.get("invoice") is None else "validate"
        return "validate"

    async def validate_node(state: InvoiceState) -> InvoiceState:
        _emit("validate", "start")
        invoice = _invoice(state)
        report = validate(invoice, deps.inventory, deps.seen, deps.policy)
        report.findings += [
            Finding(
                code="EXTRACTION_CORRECTION",
                severity=Severity.INFO,
                title="Extraction correction",
                detail=correction,
            )
            for correction in state.get("corrections", [])
        ]
        deps.seen.add(invoice)
        route = policy_route(invoice, report, deps.policy)
        return {
            "report": report,
            "route": route,
            "trace": _event(
                "validate",
                f"{len(report.errors)} errors, {len(report.warnings)} warnings;"
                f" policy route: {route}.",
            ),
        }

    async def review(state: InvoiceState) -> InvoiceState:
        _emit("approve", "draft")
        critique = state.get("critique")
        issues = critique.issues if critique and not critique.agrees else []
        draft = await deps.agents.approver.draft(
            _invoice(state), state["report"], issues
        )
        rounds = state.get("review_rounds", 0) + 1
        return {
            "draft": draft,
            "review_rounds": rounds,
            "trace": _event(
                "approve", f"Draft {rounds}: {draft.decision}. {draft.rationale}"
            ),
        }

    async def critique_node(state: InvoiceState) -> InvoiceState:
        _emit("approve", "critique")
        draft = state["draft"]
        critique = await deps.agents.critic.review(
            _invoice(state), state["report"], draft
        )
        update: InvoiceState = {"critique": critique}
        if critique.agrees:
            message = "Critic agrees."
        else:
            message = f"Critic objects: {' '.join(critique.issues)}"
        if critique.agrees or state["review_rounds"] >= settings.max_review_rounds:
            update["outcome"] = resolve(draft.decision, state["route"])
            message += f" Outcome: {update['outcome']}."
        update["trace"] = _event("approve", message)
        return update

    def after_critique(
        state: InvoiceState,
    ) -> Literal["review", "human_review", "pay", "reject"]:
        outcome = state.get("outcome")
        if outcome is None:
            return "review"
        return "human_review" if outcome == "human" else outcome

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
        note = f" Note: {decision.note}" if decision.note else ""
        return {"human": decision, "trace": _event("approve", f"VP {verdict}.{note}")}

    def after_human(state: InvoiceState) -> Literal["pay", "reject"]:
        return "pay" if state["human"].approved else "reject"

    async def pay(state: InvoiceState) -> InvoiceState:
        _emit("pay", "start")
        invoice = _invoice(state)
        assert invoice.vendor and invoice.total is not None
        result = mock_payment(invoice.vendor, invoice.total, invoice.currency)
        return {
            "payment": result,
            "trace": _event("pay", f"mock_payment returned {result.status}."),
        }

    async def reject(state: InvoiceState) -> InvoiceState:
        _emit("pay", "skipped")
        if state.get("human") and not state["human"].approved:
            reason = f"Rejected by the VP. {state['human'].note}".strip()
        elif "draft" in state:
            reason = state["draft"].rationale
        else:
            reason = "Could not extract the invoice: " + " ".join(
                state.get("extract_feedback", [])
            )
        return {
            "rejection_reason": reason,
            "trace": _event("pay", f"Not paid. {reason}"),
        }

    graph = StateGraph(InvoiceState)
    graph.add_node("ingest", ingest)
    graph.add_node("validate", validate_node)
    graph.add_node("review", review)
    graph.add_node("critique", critique_node)
    graph.add_node("human_review", human_review)
    graph.add_node("pay", pay)
    graph.add_node("reject", reject)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", after_ingest)
    graph.add_edge("validate", "review")
    graph.add_edge("review", "critique")
    graph.add_conditional_edges("critique", after_critique)
    graph.add_conditional_edges("human_review", after_human)
    graph.add_edge("pay", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=checkpointer)
