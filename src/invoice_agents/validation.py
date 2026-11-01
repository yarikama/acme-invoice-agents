"""Deterministic checks. Arithmetic and stock are never left to an LLM."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from invoice_agents.ingestion.dates import parse_date
from invoice_agents.inventory import Inventory, item_key
from invoice_agents.models import Finding, Invoice, Severity, ValidationReport

TOLERANCE = 0.01
PRESSURE_WORDS = re.compile(
    r"\b(urgent|immediately|wire transfer|avoid penalties|act now)\b", re.IGNORECASE
)
# Text in an invoice that addresses the system reviewing it, not the payer.
INSTRUCTION_WORDS = re.compile(
    r"\b(?:ignore|disregard)\b[^.]{0,60}\b(?:instructions?|warnings?|rules?|checks?"
    r"|validation)\b"
    r"|\bpre-?approved\b"
    r"|\b(?:automated|automatic|ai)\s+(?:invoice\s+)?(?:reviewer|review|system"
    r"|approver|agent)\b"
    r"|\bsubmit\b[^.]{0,40}\bapprov",
    re.IGNORECASE,
)
# A request to pay somewhere other than the account on file.
BANK_CHANGE_WORDS = re.compile(
    r"\bbank(?:ing)?\s+(?:details|information|account)\s+(?:have\s+|has\s+)?changed\b"
    r"|\bnew\s+(?:bank\s+)?account\b"
    r"|\bchange\s+of\s+bank\b"
    r"|\bdo\s+not\s+use\s+the\s+account\b",
    re.IGNORECASE,
)
# A unit price this far above the inventory's list price needs a person.
PRICE_RATIO = 1.25
# Invoices from one vendor this many days apart count as one purchase when
# looking for a total split to stay under the approval threshold.
SPLIT_WINDOW_DAYS = 3
# A vendor name this close to a known one, but not the same, may be an
# impostor. The closest pair among the sample vendors is 0.74; a one-letter
# typo or swap of a known name scores above 0.9.
LOOKALIKE_RATIO = 0.88
# Legal-form words that don't make two vendors different.
COMPANY_SUFFIXES = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|co|company|corp|corporation)\b\.?", re.I
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


# Warnings an agent may not clear on its own: each one goes to a VP.
VP_REQUIRED_DEFAULT = frozenset(
    {
        "NEAR_THRESHOLD",
        "FOREIGN_CURRENCY",
        "SUPERSEDES",
        "SPLIT_BELOW_THRESHOLD",
        "PRICE_ABOVE_LIST",
        "INSTRUCTIONS_IN_DOCUMENT",
        "BANK_DETAILS_CHANGED",
        "LOOKALIKE_VENDOR",
    }
)


@dataclass(frozen=True)
class Policy:
    approval_threshold: float = 10_000
    near_threshold_ratio: float = 0.95
    vp_required: frozenset[str] = VP_REQUIRED_DEFAULT


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
        _instructions,
        _bank_change,
    ):
        findings += check(invoice)
    findings += _stock(invoice, inventory)
    findings += _prices(invoice, inventory)
    findings += _threshold(invoice, policy)
    if seen is not None:
        findings += _duplicates(invoice, seen)
        findings += _split(invoice, seen, policy)
        findings += _lookalike_vendor(invoice, seen)
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


def _free_text(inv: Invoice) -> str:
    return " ".join(t for t in (inv.notes, inv.payment_terms) if t)


def _instructions(inv: Invoice) -> list[Finding]:
    hits = [m.group(0) for m in INSTRUCTION_WORDS.finditer(_free_text(inv))]
    if not hits:
        return []
    quoted = "; ".join(f'"{h}"' for h in dict.fromkeys(hits))
    return [
        _f(
            "INSTRUCTIONS_IN_DOCUMENT",
            Severity.WARNING,
            "Instructions aimed at the reviewer",
            f"The invoice tells its reviewer what to do: {quoted}.",
        )
    ]


def _bank_change(inv: Invoice) -> list[Finding]:
    if not BANK_CHANGE_WORDS.search(_free_text(inv)):
        return []
    return [
        _f(
            "BANK_DETAILS_CHANGED",
            Severity.WARNING,
            "Asks to pay a new bank account",
            "Confirm the change with the vendor on a known number before paying.",
        )
    ]


def _prices(inv: Invoice, inventory: Inventory) -> list[Finding]:
    over = []
    for line in inv.line_items:
        record = inventory.lookup(line.item)
        # Items without a list price (FakeItem) have nothing to compare to.
        if record and 0 < record.unit_price * PRICE_RATIO < line.unit_price:
            ratio = line.unit_price / record.unit_price
            over.append(
                f"{line.item} at {_usd(line.unit_price)}, {ratio:.1f}x the"
                f" {_usd(record.unit_price)} list price"
            )
    if not over:
        return []
    return [
        _f(
            "PRICE_ABOVE_LIST",
            Severity.WARNING,
            "Priced far above list",
            "; ".join(over) + ".",
        )
    ]


def _split(inv: Invoice, seen: SeenInvoices, policy: Policy) -> list[Finding]:
    """Invoices from one vendor, days apart, each under the threshold, that
    together cross it: a purchase split to avoid VP review."""
    limit = policy.approval_threshold
    if inv.total is None or inv.total > limit or not inv.vendor:
        return []
    vendor = item_key(inv.vendor)
    number = _number_key(inv.invoice_number or "")
    issued = parse_date(inv.invoice_date)

    def near(other: Invoice) -> bool:
        when = parse_date(other.invoice_date)
        if issued and when:
            return abs((issued - when).days) <= SPLIT_WINDOW_DAYS
        return bool(inv.invoice_date) and other.invoice_date == inv.invoice_date

    others = [
        o
        for o in seen.invoices
        if o.vendor
        and item_key(o.vendor) == vendor
        and o.total is not None
        and o.total <= limit
        and _number_key(o.invoice_number or "") != number
        and near(o)
    ]
    combined = inv.total + sum(o.total or 0 for o in others)
    if not others or combined <= limit:
        return []
    numbers = ", ".join(o.invoice_number or "unnumbered" for o in others)
    return [
        _f(
            "SPLIT_BELOW_THRESHOLD",
            Severity.WARNING,
            "May be split to stay under the threshold",
            f"With {numbers} from the same vendor within {SPLIT_WINDOW_DAYS} days,"
            f" the total is {_usd(combined)}, over {_usd(limit)}.",
        )
    ]


def _vendor_key(name: str) -> str:
    return item_key(COMPANY_SUFFIXES.sub("", name))


def _closest(key: str, vendors: list[str]) -> str | None:
    for vendor in vendors:
        if SequenceMatcher(None, key, _vendor_key(vendor)).ratio() >= LOOKALIKE_RATIO:
            return vendor
    return None


def _lookalike_vendor(inv: Invoice, seen: SeenInvoices) -> list[Finding]:
    """A name a letter or two off a vendor already invoiced: "Summit
    Manufactruing Co." for "Summit Manufacturing Co." is how impostor invoices
    get paid.

    Vendors count in the order they were first seen, and a name close to an
    earlier vendor is a suspect, not a vendor: an impostor's second invoice is
    flagged like its first. Order is all there is to go on here, so an impostor
    seen before the real vendor would flag the real one instead; a vendor
    master file would settle that.
    """
    if not inv.vendor:
        return []
    vendors: list[str] = []
    suspects: dict[str, str] = {}  # suspect name key -> the vendor it resembles
    for other in seen.invoices:
        if not other.vendor or other is inv:
            continue
        key = _vendor_key(other.vendor)
        if key in suspects or any(_vendor_key(v) == key for v in vendors):
            continue
        close = _closest(key, vendors)
        if close:
            suspects[key] = close
        else:
            vendors.append(other.vendor)

    key = _vendor_key(inv.vendor)
    if key in suspects:
        close: str | None = suspects[key]
    elif any(_vendor_key(v) == key for v in vendors):
        return []
    else:
        close = _closest(key, vendors)
    if not close:
        return []
    return [
        _f(
            "LOOKALIKE_VENDOR",
            Severity.WARNING,
            "Vendor name is close to a known vendor's",
            f"{inv.vendor!r} is not {close!r}, a vendor already invoiced, but"
            " differs by a letter or two. Confirm it is a real, separate vendor.",
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
