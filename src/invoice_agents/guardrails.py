"""A linter for the approver's decision, run in code before it is accepted.

The approver submits its decision as a tool call. These checks run on that
call; if any fails, the decision goes back to the model with the reasons and
it tries again. They catch the mistakes a prompt alone does not prevent:
approving past an error, ignoring a finding, or deciding without looking up
the fact the finding depends on.
"""

from __future__ import annotations

from invoice_agents.models import ApprovalDraft, Invoice, ToolCall, ValidationReport

STOCK_CODES = {"UNKNOWN_ITEM", "OUT_OF_STOCK", "STOCK_EXCEEDED", "STOCK_DEPLETED"}
HISTORY_CODES = {"DUPLICATE", "SUPERSEDES"}


def lint_decision(
    invoice: Invoice,
    report: ValidationReport,
    draft: ApprovalDraft,
    calls: list[ToolCall],
) -> list[str]:
    """Problems with a decision; an empty list means it is accepted."""
    problems: list[str] = []
    codes = {f.code for f in report.findings}
    called = {c.name for c in calls}

    if draft.decision == "approve" and report.errors:
        errors = ", ".join(sorted({f.code for f in report.errors}))
        problems.append(f"Approve is not allowed with error findings ({errors}).")
    if draft.decision == "reject" and not report.errors and not report.warnings:
        # A suspicion code could not confirm is for a person to weigh.
        problems.append(
            "Every check passed, so a rejection would rest on suspicion alone:"
            " escalate instead and state your concern in the rationale."
        )

    # Info findings are notes; errors and warnings each need an answer.
    must_address = {f.code for f in [*report.errors, *report.warnings]}
    missing = sorted(must_address - set(draft.addressed))
    if missing:
        problems.append(
            "List every error and warning code in `addressed` and explain each"
            f" in the rationale. Missing: {', '.join(missing)}."
        )

    if invoice.currency.upper() != "USD" and "convert_to_usd" not in called:
        problems.append(
            f"The invoice is in {invoice.currency}: call convert_to_usd on the"
            " total before deciding."
        )
    if codes & STOCK_CODES and "check_inventory" not in called:
        problems.append(
            "There are stock findings: call check_inventory for those items"
            " before deciding."
        )
    if codes & HISTORY_CODES and "vendor_history" not in called:
        problems.append(
            "This invoice repeats or revises an earlier one: call vendor_history"
            " before deciding."
        )
    return problems
