"""Deterministic checks. Arithmetic and stock are never left to an LLM."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from invoice_agents.ingestion.dates import parse_date
from invoice_agents.inventory import Inventory, item_key
from invoice_agents.models import Finding, Invoice, Severity, ValidationReport

TOLERANCE = 0.01
PRESSURE_WORDS = re.compile(
    r"\b(urgent|immediately|wire transfer|avoid penalties|act now)\b", re.IGNORECASE
)


@dataclass
class SeenInvoices:
    """Invoices already processed in this run, for duplicates and vendor history."""

    revisions: dict[str, set[str | None]] = field(default_factory=dict)
    invoices: list[Invoice] = field(default_factory=list)

    def add(self, invoice: Invoice) -> None:
        self.invoices.append(invoice)
        if invoice.invoice_number:
            key = _number_key(invoice.invoice_number)
            self.revisions.setdefault(key, set()).add(invoice.revision)


@dataclass(frozen=True)
class Policy:
    approval_threshold: float = 10_000
    near_threshold_ratio: float = 0.95


def validate(
    invoice: Invoice,
    inventory: Inventory,
    seen: SeenInvoices | None = None,
    policy: Policy | None = None,
) -> ValidationReport:
    policy = policy or Policy()
    findings: list[Finding] = []
    for check in (
        _required_fields,
        _quantities,
        _line_amounts,
        _totals,
        _currency,
        _dates,
        _pressure_language,
    ):
        findings += check(invoice)
    findings += _stock(invoice, inventory)
    findings += _threshold(invoice, policy)
    if seen is not None:
        findings += _duplicates(invoice, seen)
    return ValidationReport(findings=findings)


def _f(code: str, severity: Severity, title: str, detail: str) -> Finding:
    return Finding(code=code, severity=severity, title=title, detail=detail)


def _usd(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _number_key(number: str) -> str:
    return re.sub(r"[^0-9]", "", number)


def _required_fields(inv: Invoice) -> list[Finding]:
    missing = [
        name
        for name, value in [
            ("vendor", inv.vendor),
            ("invoice number", inv.invoice_number),
            ("due date", inv.due_date),
            ("total", inv.total),
        ]
        if value is None
    ]
    if not inv.line_items:
        missing.append("line items")
    if not missing:
        return []
    return [
        _f(
            "MISSING_FIELDS",
            Severity.ERROR,
            "Missing required fields",
            ", ".join(missing),
        )
    ]


def _quantities(inv: Invoice) -> list[Finding]:
    return [
        _f(
            "INVALID_QUANTITY",
            Severity.ERROR,
            "Quantity must be positive",
            f"{line.item} has quantity {line.quantity:g}.",
        )
        for line in inv.line_items
        if line.quantity <= 0
    ]


def _line_amounts(inv: Invoice) -> list[Finding]:
    findings = []
    for line in inv.line_items:
        expected = line.quantity * line.unit_price
        if line.amount is not None and abs(line.amount - expected) > TOLERANCE:
            findings.append(
                _f(
                    "LINE_AMOUNT_MISMATCH",
                    Severity.ERROR,
                    "Line amount does not add up",
                    f"{line.item}: {line.quantity:g} x {_usd(line.unit_price)}"
                    f" = {_usd(expected)}, invoice shows {_usd(line.amount)}.",
                )
            )
    return findings


def _totals(inv: Invoice) -> list[Finding]:
    findings = []
    lines_sum = sum(line.quantity * line.unit_price for line in inv.line_items)
    if inv.subtotal is not None and abs(inv.subtotal - lines_sum) > TOLERANCE:
        findings.append(
            _f(
                "SUBTOTAL_MISMATCH",
                Severity.ERROR,
                "Subtotal does not match the line items",
                f"Line items sum to {_usd(lines_sum)}, subtotal states"
                f" {_usd(inv.subtotal)}.",
            )
        )
    if inv.total is not None:
        expected = lines_sum + (inv.tax_amount or 0) + (inv.shipping or 0)
        if abs(inv.total - expected) > TOLERANCE:
            parts = [f"line items {_usd(lines_sum)}"]
            if inv.tax_amount:
                parts.append(f"tax {_usd(inv.tax_amount)}")
            if inv.shipping:
                parts.append(f"shipping {_usd(inv.shipping)}")
            findings.append(
                _f(
                    "TOTAL_MISMATCH",
                    Severity.ERROR,
                    "Total does not add up",
                    f"{' + '.join(parts)} = {_usd(expected)}; invoice states"
                    f" {_usd(inv.total)}, off by {_usd(inv.total - expected)}.",
                )
            )
    return findings


def _stock(inv: Invoice, inventory: Inventory) -> list[Finding]:
    """Sum quantities per item first, so splitting an item over lines can't hide it."""
    billed: dict[str, float] = defaultdict(float)
    names: dict[str, str] = {}
    for line in inv.line_items:
        if line.quantity > 0:
            billed[item_key(line.item)] += line.quantity
            names.setdefault(item_key(line.item), line.item)

    findings = []
    for key, qty in billed.items():
        name = names[key]
        record = inventory.lookup(name)
        if record is None:
            findings.append(
                _f(
                    "UNKNOWN_ITEM",
                    Severity.ERROR,
                    "Item not in inventory",
                    f"{name} is not a known product.",
                )
            )
        elif record.stock == 0:
            findings.append(
                _f(
                    "OUT_OF_STOCK",
                    Severity.ERROR,
                    "Item has zero stock",
                    f"{record.item} billed {qty:g}, inventory holds 0.",
                )
            )
        elif qty > record.stock:
            findings.append(
                _f(
                    "STOCK_EXCEEDED",
                    Severity.ERROR,
                    "Quantity exceeds stock",
                    f"{record.item} billed {qty:g} in total, {record.stock} in stock.",
                )
            )
        elif qty == record.stock:
            findings.append(
                _f(
                    "STOCK_DEPLETED",
                    Severity.WARNING,
                    "Order takes all remaining stock",
                    f"{record.item}: {qty:g} of {record.stock}.",
                )
            )
    return findings


def _currency(inv: Invoice) -> list[Finding]:
    if inv.currency.upper() == "USD":
        return []
    return [
        _f(
            "FOREIGN_CURRENCY",
            Severity.WARNING,
            f"Billed in {inv.currency.upper()}",
            "Payments run in USD; an FX rate is needed before paying.",
        )
    ]


def _dates(inv: Invoice) -> list[Finding]:
    findings = []
    due = parse_date(inv.due_date)
    issued = parse_date(inv.invoice_date)
    if inv.due_date and due is None:
        findings.append(
            _f(
                "UNPARSEABLE_DATE",
                Severity.WARNING,
                "Due date is not a date",
                f'The due date reads "{inv.due_date}".',
            )
        )
    if due and issued:
        if due < issued:
            findings.append(
                _f(
                    "DUE_BEFORE_ISSUE",
                    Severity.WARNING,
                    "Due before the invoice date",
                    f"Issued {issued}, due {due}.",
                )
            )
        terms = re.search(r"net\s*(\d+)", inv.payment_terms or "", re.IGNORECASE)
        if terms and (due - issued).days < int(terms.group(1)) - 1:
            findings.append(
                _f(
                    "TERMS_MISMATCH",
                    Severity.WARNING,
                    "Due date contradicts payment terms",
                    f"{inv.payment_terms} from {issued} is not {due}.",
                )
            )
    return findings


def _pressure_language(inv: Invoice) -> list[Finding]:
    hits = sorted({m.lower() for m in PRESSURE_WORDS.findall(inv.notes or "")})
    if not hits:
        return []
    return [
        _f(
            "PRESSURE_LANGUAGE",
            Severity.WARNING,
            "Payment pressure language",
            f"Notes contain: {', '.join(hits)}.",
        )
    ]


def _threshold(inv: Invoice, policy: Policy) -> list[Finding]:
    if inv.total is None:
        return []
    limit = policy.approval_threshold
    if inv.total > limit:
        return [
            _f(
                "OVER_THRESHOLD",
                Severity.INFO,
                f"Over {_usd(limit)}",
                "VP review required by policy.",
            )
        ]
    if inv.total >= limit * policy.near_threshold_ratio:
        return [
            _f(
                "NEAR_THRESHOLD",
                Severity.WARNING,
                f"Just under the {_usd(limit)} threshold",
                f"{_usd(inv.total)} is {_usd(limit - inv.total)} below VP review.",
            )
        ]
    return []


def _duplicates(inv: Invoice, seen: SeenInvoices) -> list[Finding]:
    if not inv.invoice_number:
        return []
    earlier = seen.revisions.get(_number_key(inv.invoice_number))
    if not earlier:
        return []
    if inv.revision and inv.revision not in earlier:
        return [
            _f(
                "SUPERSEDES",
                Severity.WARNING,
                f"Revision {inv.revision} of an earlier invoice",
                f"{inv.invoice_number} was already received; only one may be paid.",
            )
        ]
    return [
        _f(
            "DUPLICATE",
            Severity.ERROR,
            "Duplicate invoice",
            f"{inv.invoice_number} was already processed.",
        )
    ]
