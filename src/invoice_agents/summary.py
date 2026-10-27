"""A JSON-ready view of a finished (or paused) run, shared by the CLI and API."""

from __future__ import annotations

from typing import Any, Literal

from invoice_agents.graph import InvoiceState

Status = Literal["queued", "running", "awaiting_vp", "paid", "rejected", "failed"]


def run_status(state: InvoiceState, interrupted: bool) -> Status:
    if interrupted:
        return "awaiting_vp"
    if state.get("payment"):
        return "paid"
    if state.get("rejection_reason"):
        return "rejected"
    return "running"


def summarize(state: InvoiceState, status: Status) -> dict[str, Any]:
    invoice = state.get("invoice")
    report = state.get("report")
    draft = state.get("draft")
    payment = state.get("payment")
    return {
        "status": status,
        "source_format": state.get("source_format"),
        "invoice": invoice.model_dump(mode="json") if invoice else None,
        "findings": [f.model_dump(mode="json") for f in report.findings]
        if report
        else [],
        "route": state.get("route"),
        "draft": draft.model_dump(mode="json") if draft else None,
        "payment": payment.model_dump(mode="json") if payment else None,
        "reason": state.get("rejection_reason"),
        "trace": state.get("trace", []),
    }
