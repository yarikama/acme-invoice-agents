"""Read an invoice file into a Document.

JSON, CSV and XML have a fixed structure, so they are mapped to an Invoice
in code. TXT and PDF are free text: the Document carries the text and the
extraction agent turns it into an Invoice.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from defusedxml import ElementTree
from pydantic import ValidationError

from invoice_agents.ingestion.dates import normalize_date
from invoice_agents.ingestion.ocr import Ocr, docling_text
from invoice_agents.models import Invoice, LineItem, SourceFormat


class UnsupportedFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Document:
    path: Path
    format: SourceFormat
    text: str
    # Set for structured formats; None means an LLM has to extract it.
    invoice: Invoice | None = None
    # The text came from OCR: a scan or a photo rather than a text layer.
    ocr: bool = False
    # Why code could not read it: a structured file that didn't parse (the
    # extraction agent reads its text instead), or a file with no text.
    problem: str | None = None


IMAGES = {SourceFormat.PNG, SourceFormat.JPG, SourceFormat.JPEG}
# A PDF with less text than this has no real text layer: it is a scan.
MIN_PDF_TEXT = 20


def load_document(path: str | Path, ocr: Ocr = docling_text) -> Document:
    """Read an invoice file. Blocking (disk, PDF parsing, OCR): callers in
    async code run it in a thread."""
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    try:
        fmt = SourceFormat(suffix)
    except ValueError as exc:
        raise UnsupportedFormatError(f"Unsupported file type: .{suffix}") from exc

    if fmt in IMAGES:
        return Document(path, fmt, ocr(path), ocr=True)
    if fmt is SourceFormat.PDF:
        try:
            text = _pdf_text(path)
        except Exception as exc:
            # Not a PDF after all, or a damaged one: nothing to read or OCR.
            return Document(
                path, fmt, "", problem=f"The PDF could not be opened ({exc})."
            )
        if len(text) < MIN_PDF_TEXT:
            return Document(path, fmt, ocr(path), ocr=True)
        return Document(path, fmt, text)

    text = _decode(path.read_bytes())
    if not text.strip():
        return Document(path, fmt, text, problem="The file is empty.")
    parsers = {
        SourceFormat.JSON: _from_json,
        SourceFormat.CSV: _from_csv,
        SourceFormat.XML: _from_xml,
    }
    parser = parsers.get(fmt)
    if parser is None:
        return Document(path, fmt, text)
    try:
        return Document(path, fmt, text, parser(text))
    except Exception as exc:
        # A structured file in a layout the parser doesn't know. Its text
        # still says what the invoice is, so the extraction agent reads it.
        name = fmt.value.upper()
        return Document(
            path, fmt, text, problem=f"Could not read the {name} in code: {_why(exc)}."
        )


def _decode(raw: bytes) -> str:
    """UTF-8, or Windows-1252 for files saved by older tools; never fails."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _why(exc: Exception) -> str:
    """What was wrong with a structured file, in a sentence fragment."""
    if isinstance(exc, json.JSONDecodeError):
        return f"not valid JSON ({exc.msg}, line {exc.lineno})"
    if isinstance(exc, ElementTree.ParseError):
        return f"not valid XML ({exc})"
    if isinstance(exc, KeyError):
        return f"no {exc.args[0]!r} field"
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return f"{where}: {first['msg'].lower()}"
    if isinstance(exc, ValueError) and str(exc):
        return str(exc)
    return "the layout is not one this reader knows"


def _pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text(layout=True) or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _money(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(str(value).replace("$", "").replace(",", "").strip())


# --- JSON -----------------------------------------------------------------


def _from_json(text: str) -> Invoice:
    data = json.loads(text)
    vendor = data.get("vendor")
    if isinstance(vendor, dict):
        vendor_name = vendor.get("name")
        vendor_address = vendor.get("address")
    else:
        vendor_name, vendor_address = vendor, None

    return Invoice(
        invoice_number=_blank_to_none(data.get("invoice_number")),
        revision=_blank_to_none(data.get("revision")),
        vendor=_blank_to_none(vendor_name),
        vendor_address=_blank_to_none(vendor_address),
        invoice_date=normalize_date(data.get("date")),
        due_date=normalize_date(data.get("due_date")),
        currency=data.get("currency") or "USD",
        line_items=[
            LineItem(
                item=line["item"],
                quantity=line["quantity"],
                unit_price=line["unit_price"],
                amount=line.get("amount"),
                note=line.get("note"),
            )
            for line in data.get("line_items", [])
        ],
        subtotal=_money(data.get("subtotal")),
        tax_amount=_money(data.get("tax_amount")),
        shipping=_money(data.get("shipping")),
        total=_money(data.get("total")),
        payment_terms=_blank_to_none(data.get("payment_terms")),
        notes=_blank_to_none(data.get("notes")),
    )


# --- CSV ------------------------------------------------------------------


def _from_csv(text: str) -> Invoice:
    rows = [row for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]
    header = [c.strip().lower() for c in rows[0]]
    if header == ["field", "value"]:
        return _from_key_value_csv(rows[1:])
    return _from_table_csv(header, rows[1:])


def _from_key_value_csv(rows: list[list[str]]) -> Invoice:
    """One `field,value` pair per row; item, quantity, unit_price repeat per line."""
    fields: dict[str, str] = {}
    lines: list[dict[str, str]] = []
    for key, value, *_ in rows:
        key = key.strip()
        if key == "item":
            lines.append({"item": value})
        elif key in ("quantity", "unit_price") and lines:
            lines[-1][key] = value
        else:
            fields[key] = value

    return Invoice(
        invoice_number=fields.get("invoice_number"),
        vendor=fields.get("vendor"),
        invoice_date=normalize_date(fields.get("date")),
        due_date=normalize_date(fields.get("due_date")),
        line_items=[
            LineItem(
                item=line["item"],
                quantity=float(line["quantity"]),
                unit_price=float(line["unit_price"]),
            )
            for line in lines
        ],
        subtotal=_money(fields.get("subtotal")),
        tax_amount=_money(fields.get("tax")),
        total=_money(fields.get("total")),
        payment_terms=fields.get("payment_terms"),
    )


def _from_table_csv(header: list[str], rows: list[list[str]]) -> Invoice:
    """One row per line item, then summary rows labelled in the price column."""
    col = {name: i for i, name in enumerate(header)}
    items: list[list[str]] = []
    summary: dict[str, float | None] = {}
    for row in rows:
        if row[col["invoice number"]].strip():
            items.append(row)
            continue
        label = row[col["unit price"]].strip().lower()
        value = _money(row[col["line total"]])
        for key in ("subtotal", "tax", "total"):
            if label.startswith(key):
                summary[key] = value

    first = items[0]
    return Invoice(
        invoice_number=first[col["invoice number"]],
        vendor=first[col["vendor"]],
        invoice_date=normalize_date(first[col["date"]]),
        due_date=normalize_date(first[col["due date"]]),
        line_items=[
            LineItem(
                item=row[col["item"]],
                quantity=float(row[col["qty"]]),
                unit_price=float(row[col["unit price"]]),
                amount=_money(row[col["line total"]]),
            )
            for row in items
        ],
        subtotal=summary.get("subtotal"),
        tax_amount=summary.get("tax"),
        total=summary.get("total"),
    )


# --- XML ------------------------------------------------------------------


def _from_xml(text: str) -> Invoice:
    root = ElementTree.fromstring(text)

    def find(path: str) -> str | None:
        node = root.find(path)
        return node.text.strip() if node is not None and node.text else None

    return Invoice(
        invoice_number=find("header/invoice_number"),
        vendor=find("header/vendor"),
        invoice_date=normalize_date(find("header/date")),
        due_date=normalize_date(find("header/due_date")),
        currency=find("header/currency") or "USD",
        line_items=[
            LineItem(
                item=(item.findtext("name") or "").strip(),
                quantity=float(item.findtext("quantity") or 0),
                unit_price=float(item.findtext("unit_price") or 0),
            )
            for item in root.findall("line_items/item")
        ],
        subtotal=_money(find("totals/subtotal")),
        tax_amount=_money(find("totals/tax_amount")),
        total=_money(find("totals/total")),
        payment_terms=find("payment_terms"),
    )
