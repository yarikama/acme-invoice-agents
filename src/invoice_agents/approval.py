"""Approval policy: what the agents may decide, and what overrides them."""

from __future__ import annotations

from typing import Literal

from invoice_agents.models import Decision, Invoice, ValidationReport
from invoice_agents.validation import Policy

Route = Literal["reject", "human", "auto"]
Outcome = Literal["pay", "reject", "human"]


def policy_route(invoice: Invoice, report: ValidationReport, policy: Policy) -> Route:
    """The floor set by rules, before any agent weighs in."""
    if not report.passed:
        return "reject"
    if invoice.total is not None and invoice.total > policy.approval_threshold:
        return "human"
    return "auto"


def resolve(decision: Decision, route: Route) -> Outcome:
    """Combine the agents' decision with the rules. Agents can only be stricter.

    An LLM that approves an invoice with a hard error, or one over the
    threshold, is overruled here in code, not by asking it nicely.
    """
    if route == "reject" or decision == "reject":
        return "reject"
    if decision == "escalate" or route == "human":
        return "human"
    return "pay"
