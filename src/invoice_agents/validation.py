"""Deterministic checks. Arithmetic and stock are never left to an LLM."""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Literal

from invoice_agents.fx import to_usd
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


# What became of an invoice: None while it is in progress or waiting for a
# VP. Paid and in-flight invoices count as received for duplicates; one that
# was rejected, failed or dropped may be sent again.
Outcome = Literal["paid", "rejected", "failed", "dropped"]
STANDING: frozenset[Outcome | None] = frozenset({None, "paid"})


@dataclass
class Received:
    """One invoice that reached validation, and what became of it."""

    invoice: Invoice
    run_id: str
    source: str = ""
    outcome: Outcome | None = None
    reason: str | None = None
    received_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class SeenInvoices:
    """Invoices already received, for duplicates, splits and vendor history.

    Kept in memory for one process; `Ledger` keeps them in SQLite, so they
    survive a restart.
    """

    def __init__(self) -> None:
        self._entries: list[Received] = []

    @property
    def entries(self) -> list[Received]:
        return self._entries

    @property
    def invoices(self) -> list[Invoice]:
        return [e.invoice for e in self.entries]

    def add(
        self, invoice: Invoice, run_id: str | None = None, source: str = ""
    ) -> None:
        self._entries.append(Received(invoice, run_id or uuid.uuid4().hex, source))

    def settle(self, run_id: str, outcome: Outcome, reason: str | None = None) -> None:
        for entry in self._entries:
            # A payment is never undone by a later failure: duplicate checks
            # must keep seeing it.
            if entry.run_id == run_id and entry.outcome != "paid":
                entry.outcome, entry.reason = outcome, reason

    def clear(self) -> None:
        self._entries.clear()


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
        "POSSIBLE_DUPLICATE",
        "VENDOR_RENAMED",
        "NOT_IN_DOCUMENT",
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
    document: str | None = None,
) -> ValidationReport:
    """Every check on one invoice. `document` is the file's text, for what an
    extraction may leave out (a vendor's "formerly ..." line)."""
    policy = policy or Policy()
    findings: list[Finding] = []
    for check in (
        _required_fields,
        _quantities,
        _amounts,
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
    findings += _renamed(invoice, document)
    master = inventory.vendors()
    seen = seen if seen is not None else SeenInvoices()
    findings += _duplicates(invoice, seen) or _resubmitted(invoice, seen)
    findings += _split(invoice, seen, policy)
    findings += _lookalike_vendor(invoice, seen, master) or _new_vendor(
        invoice, seen, master
    )
    return ValidationReport(findings=findings)


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ungrounded(invoice: Invoice, text: str, corrections: list[str]) -> list[str]:
    """What an extraction model reported that the document doesn't say: an
    item name that isn't in the text, or a total that isn't. Names it says
    it corrected (an OCR misread, a typo) are its to explain."""
    plain = _plain(text)
    fixed = _plain(" ".join(corrections))
    missing = [
        f"item {line.item!r}"
        for line in invoice.line_items
        if _plain(line.item)
        and _plain(line.item) not in plain
        and _plain(line.item) not in fixed
    ]
    if invoice.total is not None:
        figures = re.sub(r"[,\s$€£¥]", "", text)
        total = abs(invoice.total)
        forms = {f"{total:.2f}", f"{total:.1f}", f"{total:g}"}
        if total.is_integer():
            forms.add(str(int(total)))
        if not any(f in figures for f in forms) and "total" not in fixed:
            missing.append(f"total {total:,.2f}")
    return missing


def _f(code: str, severity: Severity, title: str, detail: str) -> Finding:
    return Finding(code=code, severity=severity, title=title, detail=detail)


def _usd(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


# A revision written into the invoice number: "INV-1004-R1", "INV-1004 Rev. 2".
_REVISION_SUFFIX = re.compile(
    r"^(?P<base>.*\d)[\s\-_/.]*(?:r|rev\.?|revision)\s*-?\s*(?P<rev>\d+)$", re.I
)


def _number_key(number: str) -> str:
    """The digits that identify an invoice, without a revision suffix."""
    match = _REVISION_SUFFIX.match(number.strip())
    return re.sub(r"[^0-9]", "", match["base"] if match else number)


def _revision_of(inv: Invoice) -> str | None:
    """The revision, from its own field or a suffix on the number."""
    if inv.revision:
        return inv.revision
    match = _REVISION_SUFFIX.match((inv.invoice_number or "").strip())
    return f"R{match['rev']}" if match else None


def paid_already(inv: Invoice, seen: SeenInvoices, run_id: str) -> str | None:
    """Asked again just before paying: whether another run paid this invoice,
    or another version of it, while this one waited (for a VP, say). The
    checks at validation saw the ledger as it was then."""
    if not inv.invoice_number:
        return None
    key = _number_key(inv.invoice_number)
    vendor = _vendor_key(inv.vendor or "")
    for e in seen.entries:
        if (
            e.run_id != run_id
            and e.outcome == "paid"
            and e.invoice.invoice_number
            and _number_key(e.invoice.invoice_number) == key
            and (
                not vendor
                or not e.invoice.vendor
                or _vendor_key(e.invoice.vendor) == vendor
            )
        ):
            return (
                f"{e.invoice.invoice_number} was paid while this run waited;"
                f" {inv.invoice_number} can't be paid on top of it."
            )
    return None


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


def _amounts(inv: Invoice) -> list[Finding]:
    """Money only flows to the vendor: a negative price or a total of zero
    or less is a credit note or a bad invoice, never a payment."""
    problems = [
        f"{line.item} is priced at {_usd(line.unit_price)}"
        for line in inv.line_items
        if line.unit_price < 0
    ]
    if inv.total is not None and inv.total <= 0:
        problems.append(f"the total is {_usd(inv.total)}")
    findings = []
    if problems:
        findings.append(
            _f(
                "NON_POSITIVE_AMOUNT",
                Severity.ERROR,
                "Amount is not positive",
                _sentence("; ".join(problems)),
            )
        )
    fractional = [
        f"{line.item} x {line.quantity:g}"
        for line in inv.line_items
        if line.quantity > 0 and not float(line.quantity).is_integer()
    ]
    if fractional:
        findings.append(
            _f(
                "FRACTIONAL_QUANTITY",
                Severity.WARNING,
                "Part of a unit billed",
                f"Items are counted in whole units: {', '.join(fractional)}.",
            )
        )
    return findings


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
    """Sum quantities per item first, so splitting an item over lines can't hide it.

    Each invoice is checked against stock on hand, and paying one doesn't
    reserve stock for the next: the brief's samples are each meant to be
    checked against the seed, and an AP payment isn't a stock movement.
    """
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
    total = usd_total(inv)
    if total is None or total > limit or not inv.vendor:
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
        and (usd_total(o) or 0) > 0
        and (usd_total(o) or 0) <= limit
        and _number_key(o.invoice_number or "") != number
        and near(o)
    ]
    # In dollars, like the threshold: a ¥500,000 and a $8,750 invoice add up.
    combined = total + sum(usd_total(o) or 0 for o in others)
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


RENAMED = re.compile(r"\b(?:formerly|previously|f/k/a|fka)\b[\s:]*([^)\n;]+)", re.I)


def renamed_from(text: str) -> str | None:
    """The old name in a "formerly ..." line, if the text has one."""
    match = RENAMED.search(text)
    return match.group(1).strip(" .") if match else None


def _renamed(inv: Invoice, document: str | None = None) -> list[Finding]:
    """A vendor announcing a new name: how an impostor explains away a name
    that doesn't match the one on file. Read from the extracted fields, and
    from the document itself, which an extraction may have trimmed."""
    fields = " ".join(t for t in (inv.vendor, inv.vendor_address, inv.notes) if t)
    old = renamed_from(fields) or (renamed_from(document) if document else None)
    if not old:
        return []
    return [
        _f(
            "VENDOR_RENAMED",
            Severity.WARNING,
            "Vendor says it changed its name",
            f"{inv.vendor or 'The vendor'} was formerly {old}. Confirm the change,"
            " and the bank account, with the vendor on a number already on file.",
        )
    ]


def _new_vendor(inv: Invoice, seen: SeenInvoices, master: list[str]) -> list[Finding]:
    """A first invoice from a vendor neither in the master nor seen before:
    a note for the reviewer, not a reason to refuse."""
    if not inv.vendor or not master:
        return []
    key = _vendor_key(inv.vendor)
    known = [*master, *(i.vendor for i in seen.invoices if i.vendor)]
    if any(_vendor_key(v) == key for v in known):
        return []
    return [
        _f(
            "NEW_VENDOR",
            Severity.INFO,
            "First invoice from a new vendor",
            f"{inv.vendor} is not in the vendor master and has not invoiced before.",
        )
    ]


def _lookalike_vendor(
    inv: Invoice, seen: SeenInvoices, master: list[str] | None = None
) -> list[Finding]:
    """A name a letter or two off a vendor already invoiced: "Summit
    Manufactruing Co." for "Summit Manufacturing Co." is how impostor invoices
    get paid.

    The vendor master comes first, then vendors in the order they were first
    seen; a name close to an earlier vendor is a suspect, not a vendor, so an
    impostor's second invoice is flagged like its first. For a vendor not in
    the master, order is all there is to go on: an impostor seen before the
    real vendor would flag the real one instead.
    """
    if not inv.vendor:
        return []
    vendors: list[str] = list(master or [])
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
            f"{inv.vendor!r} is not {close!r}, a known vendor, but"
            " differs by a letter or two. Confirm it is a real, separate vendor.",
        )
    ]


def usd_total(inv: Invoice) -> float | None:
    """The total in US dollars, which the approval threshold is set in. A
    currency with no rate counts at face value; it is flagged anyway."""
    if inv.total is None:
        return None
    converted = to_usd(inv.total, inv.currency)
    return inv.total if converted is None else converted


def _threshold(inv: Invoice, policy: Policy) -> list[Finding]:
    total = usd_total(inv)
    if total is None:
        return []
    limit = policy.approval_threshold
    if total > limit:
        return [
            _f(
                "OVER_THRESHOLD",
                Severity.INFO,
                f"Over {_usd(limit)}",
                "VP review required by policy.",
            )
        ]
    if total >= limit * policy.near_threshold_ratio:
        return [
            _f(
                "NEAR_THRESHOLD",
                Severity.WARNING,
                f"Just under the {_usd(limit)} threshold",
                f"{_usd(total)} is {_usd(limit - total)} below VP review.",
            )
        ]
    return []


def _resubmitted(inv: Invoice, seen: SeenInvoices) -> list[Finding]:
    """The same vendor, total and date as an invoice still standing, under
    another number: sent again with a new number, or the same invoice with its
    number misread (OCR read a scanned INV-1011 as INV-10114). Code can't tell
    which, so a person decides; nothing pays it on its own."""
    if inv.total is None or not inv.vendor:
        return []
    vendor = _vendor_key(inv.vendor)
    number = _number_key(inv.invoice_number or "")
    issued = parse_date(inv.invoice_date)

    def same_day(other: Invoice) -> bool:
        when = parse_date(other.invoice_date)
        # A missing date can't rule a match out.
        return issued is None or when is None or issued == when

    earlier = [
        e.invoice
        for e in seen.entries
        if e.outcome in STANDING
        and e.invoice.vendor
        and _vendor_key(e.invoice.vendor) == vendor
        and e.invoice.total is not None
        and abs(e.invoice.total - inv.total) < 0.005
        and e.invoice.currency.upper() == inv.currency.upper()
        and _number_key(e.invoice.invoice_number or "") != number
        and same_day(e.invoice)
    ]
    if not earlier:
        return []
    first = earlier[0]
    return [
        _f(
            "POSSIBLE_DUPLICATE",
            Severity.WARNING,
            "Same vendor, date and total as an earlier invoice",
            f"{first.invoice_number or 'An unnumbered invoice'} from {first.vendor}"
            f" for {inv.currency} {inv.total:,.2f} was already received. This may"
            " be the same invoice sent again under a new number, or its number"
            " misread.",
        )
    ]


def _duplicates(inv: Invoice, seen: SeenInvoices) -> list[Finding]:
    """The same invoice number again. Only earlier copies still standing
    count: one that was paid, or is in progress or waiting for a VP. A
    rejected invoice may be sent again once it is fixed."""
    if not inv.invoice_number:
        return []
    key = _number_key(inv.invoice_number)
    vendor = _vendor_key(inv.vendor or "")
    # Numbers are unique per vendor: Gadgets Co.'s GC-1001 is not a second
    # INV-1001 from Widgets Inc. A missing vendor matches any, to be safe.
    standing = [
        e
        for e in seen.entries
        if e.outcome in STANDING
        and e.invoice.invoice_number
        and _number_key(e.invoice.invoice_number) == key
        and (
            not vendor
            or not e.invoice.vendor
            or _vendor_key(e.invoice.vendor) == vendor
        )
    ]
    if not standing:
        return []
    paid = [e for e in standing if e.outcome == "paid"]
    revision = _revision_of(inv)
    if revision and all(_revision_of(e.invoice) != revision for e in standing):
        if paid:
            first = paid[0].invoice
            return [
                _f(
                    "ALREADY_PAID",
                    Severity.ERROR,
                    "An earlier version was already paid",
                    f"{inv.invoice_number} was paid at"
                    f" {first.currency} {first.total or 0:,.2f}; revision"
                    f" {revision} cannot be paid on top of it.",
                )
            ]
        return [
            _f(
                "SUPERSEDES",
                Severity.WARNING,
                f"Revision {revision} of an earlier invoice",
                f"{inv.invoice_number} was already received; only one may be paid.",
            )
        ]
    return [
        _f(
            "DUPLICATE",
            Severity.ERROR,
            "Duplicate invoice",
            f"{inv.invoice_number} was already {'paid' if paid else 'received'}.",
        )
    ]
