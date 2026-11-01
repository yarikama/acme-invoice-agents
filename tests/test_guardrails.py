"""The decision linter and the policy rules that override the agents."""

import pytest

from invoice_agents.approval import policy_route, resolve, vp_reason
from invoice_agents.guardrails import lint_decision
from invoice_agents.models import (
    ApprovalDraft,
    Finding,
    Invoice,
    Severity,
    ToolCall,
    ValidationReport,
)
from invoice_agents.validation import Policy


def finding(code: str, severity: Severity = Severity.WARNING) -> Finding:
    return Finding(code=code, severity=severity, title=code.title(), detail="")


def invoice(currency: str = "USD", total: float = 500) -> Invoice:
    return Invoice(
        invoice_number="INV-1",
        vendor="Widgets Inc.",
        invoice_date="2026-01-15",
        due_date="2026-02-01",
        line_items=[],
        total=total,
        currency=currency,
    )


def draft(decision: str = "approve", *addressed: str) -> ApprovalDraft:
    return ApprovalDraft.model_validate(
        {"decision": decision, "rationale": "r", "addressed": list(addressed)}
    )


def called(*names: str) -> list[ToolCall]:
    return [ToolCall(name=n, args={}, result="") for n in names]


def test_a_clean_decision_passes() -> None:
    report = ValidationReport(findings=[finding("TERMS_MISMATCH")])

    assert (
        lint_decision(invoice(), report, draft("approve", "TERMS_MISMATCH"), []) == []
    )


def test_info_findings_need_no_answer() -> None:
    report = ValidationReport(findings=[finding("OVER_THRESHOLD", Severity.INFO)])

    assert lint_decision(invoice(), report, draft("approve"), []) == []


def test_approving_past_an_error_is_refused() -> None:
    report = ValidationReport(findings=[finding("DUPLICATE", Severity.ERROR)])

    problems = lint_decision(
        invoice(), report, draft("approve", "DUPLICATE"), called("vendor_history")
    )

    assert problems == ["Approve is not allowed with error findings (DUPLICATE)."]


def test_a_rejection_needs_a_finding_behind_it() -> None:
    clean = ValidationReport(findings=[])

    [problem] = lint_decision(invoice(), clean, draft("reject"), [])
    assert "escalate instead" in problem
    assert lint_decision(invoice(), clean, draft("escalate"), []) == []


def test_every_finding_must_be_addressed() -> None:
    report = ValidationReport(
        findings=[finding("PRESSURE_LANGUAGE"), finding("TERMS_MISMATCH")]
    )

    [problem] = lint_decision(
        invoice(), report, draft("escalate", "TERMS_MISMATCH"), []
    )

    assert problem.endswith("Missing: PRESSURE_LANGUAGE.")


@pytest.mark.parametrize(
    ("currency", "code", "tool"),
    [
        ("EUR", "FOREIGN_CURRENCY", "convert_to_usd"),
        ("USD", "STOCK_EXCEEDED", "check_inventory"),
        ("USD", "SUPERSEDES", "vendor_history"),
    ],
)
def test_the_lookup_a_finding_depends_on_is_required(
    currency: str, code: str, tool: str
) -> None:
    report = ValidationReport(findings=[finding(code)])
    decision = draft("escalate", code)

    [problem] = lint_decision(invoice(currency), report, decision, [])
    assert f"call {tool}" in problem
    assert lint_decision(invoice(currency), report, decision, called(tool)) == []


def test_policy_sends_listed_warnings_to_a_vp() -> None:
    report = ValidationReport(findings=[finding("NEAR_THRESHOLD")])

    assert policy_route(invoice(), report, Policy()) == "human"
    assert vp_reason(invoice(), report, Policy()) == "Near_Threshold"
    # An empty list hands the call back to the approver.
    assert policy_route(invoice(), report, Policy(vp_required=frozenset())) == "auto"


def test_the_threshold_reason() -> None:
    report = ValidationReport(findings=[])

    assert vp_reason(invoice(total=12_000), report, Policy()) == (
        "the total is over $10,000"
    )
    assert vp_reason(invoice(), report, Policy()) is None


def test_agents_can_only_make_it_stricter() -> None:
    assert resolve("approve", "human") == "human"
    assert resolve("approve", "reject") == "reject"
    assert resolve("escalate", "auto") == "human"
    assert resolve("approve", "auto") == "pay"


def test_a_vp_is_needed_over_the_threshold_in_dollars():
    euro = invoice(currency="EUR", total=9_500)
    assert vp_reason(euro, ValidationReport(), Policy()) == "the total is over $10,000"
    yen = invoice(currency="JPY", total=20_000)
    assert vp_reason(yen, ValidationReport(), Policy()) is None
