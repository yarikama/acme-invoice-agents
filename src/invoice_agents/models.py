"""Data shapes shared by every stage of the pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceFormat(StrEnum):
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    PDF = "pdf"
    # Photos and scans of an invoice: read with OCR (`uv sync --extra ocr`).
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"


class LineItem(BaseModel):
    item: str = Field(description="Product name exactly as written on the invoice")
    quantity: float = Field(description="Units billed; keep negative values as-is")
    unit_price: float = Field(description="Price per unit, without currency symbols")
    amount: float | None = Field(
        default=None, description="Line total as printed, if the invoice shows one"
    )
    note: str | None = Field(default=None, description="Per-line note, if any")


class Invoice(BaseModel):
    """One invoice as extracted, before any validation.

    Fields stay as close to the document as possible: a wrong total or a
    missing vendor is kept, so the validator can report it.
    """

    invoice_number: str | None = Field(description="Invoice number, e.g. INV-1001")
    revision: str | None = Field(default=None, description="Revision label, e.g. R1")
    vendor: str | None = Field(description="Vendor (seller) name")
    vendor_address: str | None = None
    invoice_date: str | None = Field(
        description="Issue date: ISO 8601 if readable, otherwise the raw text"
    )
    due_date: str | None = Field(
        description="Due date: ISO 8601 if readable, otherwise the raw text"
    )
    currency: str = Field(default="USD", description="ISO 4217 code")
    line_items: list[LineItem]
    subtotal: float | None = None
    tax_amount: float | None = None
    shipping: float | None = None
    total: float | None = Field(description="Total amount due as printed")
    payment_terms: str | None = None
    notes: str | None = Field(
        default=None, description="Free-text notes on the invoice"
    )


class Severity(StrEnum):
    ERROR = "error"  # blocks payment
    WARNING = "warning"  # needs a judgment call
    INFO = "info"


class Finding(BaseModel):
    code: str
    severity: Severity
    title: str
    detail: str


class ValidationReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors


Decision = Literal["approve", "reject", "escalate"]


class ApprovalDraft(BaseModel):
    decision: Decision
    rationale: str = Field(description="Why, citing the findings that drove it")
    addressed: list[str] = Field(
        default_factory=list,
        description="The code of every finding this decision accounts for,"
        " e.g. NEAR_THRESHOLD",
    )


class ToolCall(BaseModel):
    """One tool the approver called while drafting, and what it returned."""

    name: str
    args: dict[str, Any]
    result: str

    def describe(self) -> str:
        args = ", ".join(str(v) for v in self.args.values())
        return f"{self.name}({args}): {self.result}"


class Critique(BaseModel):
    agrees: bool = Field(description="True if the draft decision should stand")
    issues: list[str] = Field(
        default_factory=list,
        description="What the draft got wrong or ignored; empty when it agrees",
    )


class HumanDecision(BaseModel):
    approved: bool
    note: str = ""
    # Who decided, as the server authenticated them; None when VP_TOKENS is
    # unset (a local demo). Anything a client sends here is overwritten.
    by: str | None = None


class PaymentResult(BaseModel):
    status: Literal["success", "failed"]
    vendor: str
    amount: float
    currency: str
    # The bank's idempotency key: one payment per key, however often asked.
    idempotency_key: str | None = None
    # True when this key was already paid: the earlier payment, returned
    # again, with no money moved this time.
    duplicate: bool = False
