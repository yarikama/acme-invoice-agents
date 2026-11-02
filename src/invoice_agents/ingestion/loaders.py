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
        if any(len(page) < MIN_PDF_TEXT for page in text):
            # A scan, or a scanned page among typed ones (the line items, say):
            # OCR reads typed pages too, so read the whole file that way.
            return Document(path, fmt, ocr(path), ocr=True)
        return Document(path, fmt, "\n".join(text).strip())

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


def _pdf_text(path: Path) -> list[str]:
    """Each page's text layer, stripped."""
    with pdfplumber.open(path) as pdf:
        return [
            (page.extract_text(layout=True) or "").strip() for page in pdf.pages
        ] or [""]


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _money(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(str(value).replace("$", "").replace(",", "").strip())


def _check_known(fields: dict[str, Any], known: set[str], what: str) -> None:
    """A field this reader doesn't map could say anything (new bank details,
    a currency, a note to the reviewer): rather than drop it, give up, so the
    extraction agent reads the whole file."""
    unknown = sorted(
        k for k, v in fields.items() if k not in known and v not in (None, "", [], {})
    )
    if unknown:
        raise ValueError(
            f"{what} {', '.join(repr(k) for k in unknown)} not read in code"
        )


def _check_found(inv: Invoice) -> Invoice:
    """A layout this reader doesn't know parses to nothing: give up rather
    than reject an empty invoice, so the extraction agent reads the file."""
    if not (inv.invoice_number or inv.vendor or inv.line_items):
        raise ValueError("none of the fields this reader knows were found")
    return inv


# --- JSON -----------------------------------------------------------------

JSON_FIELDS = {
    "invoice_number", "revision", "vendor", "date", "due_date", "currency",
    "line_items", "subtotal", "tax_amount", "tax_rate", "shipping", "total",
    "payment_terms", "notes",
}  # fmt: skip
JSON_LINE_FIELDS = {"item", "quantity", "unit_price", "amount", "note"}


def _from_json(text: str) -> Invoice:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("the top level is not an object")
    _check_known(data, JSON_FIELDS, "field")
    for line in data.get("line_items") or []:
        _check_known(line, JSON_LINE_FIELDS, "line item field")
    if isinstance(data.get("vendor"), dict):
        _check_known(data["vendor"], {"name", "address"}, "vendor field")
    vendor = data.get("vendor")
    if isinstance(vendor, dict):
        vendor_name = vendor.get("name")
        vendor_address = vendor.get("address")
    else:
        vendor_name, vendor_address = vendor, None

    return _check_found(
        Invoice(
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
    )


# --- CSV ------------------------------------------------------------------


def _from_csv(text: str) -> Invoice:
    rows = [row for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]
    header = [c.strip().lower() for c in rows[0]]
    if header == ["field", "value"]:
        return _from_key_value_csv(rows[1:])
    return _from_table_csv(header, rows[1:])


KV_FIELDS = {
    "invoice_number", "revision", "vendor", "vendor_address", "date", "due_date",
    "currency", "subtotal", "tax", "tax_rate", "shipping", "total",
    "payment_terms", "notes",
}  # fmt: skip


def _from_key_value_csv(rows: list[list[str]]) -> Invoice:
    """One `field,value` pair per row; item, quantity, unit_price repeat per line."""
    fields: dict[str, str] = {}
    lines: list[dict[str, str]] = []
    for key, value, *_ in rows:
        key = key.strip().lower()
        if key == "item":
            lines.append({"item": value})
        elif key in ("quantity", "unit_price") and lines:
            lines[-1][key] = value
        else:
            fields[key] = value
    _check_known(fields, KV_FIELDS, "field")

    return Invoice(
        invoice_number=_blank_to_none(fields.get("invoice_number")),
        revision=_blank_to_none(fields.get("revision")),
        vendor=_blank_to_none(fields.get("vendor")),
        vendor_address=_blank_to_none(fields.get("vendor_address")),
        invoice_date=normalize_date(fields.get("date")),
        due_date=normalize_date(fields.get("due_date")),
        currency=(fields.get("currency") or "").strip() or "USD",
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
        shipping=_money(fields.get("shipping")),
        total=_money(fields.get("total")),
        payment_terms=_blank_to_none(fields.get("payment_terms")),
        notes=_blank_to_none(fields.get("notes")),
    )


TABLE_COLUMNS = (
    "invoice number", "vendor", "date", "due date", "item", "qty", "unit price",
    "line total", "currency", "notes",
)  # fmt: skip


def _from_table_csv(header: list[str], rows: list[list[str]]) -> Invoice:
    """One row per line item, then summary rows labelled in the price column."""
    col = {name: i for i, name in enumerate(header)}
    known: set[str] = set(TABLE_COLUMNS)
    items: list[list[str]] = []
    summary: dict[str, float | None] = {}
    for row in rows:
        # A value in a column this reader doesn't map: let the agent read it.
        _check_known(
            {name: row[i].strip() for name, i in col.items() if i < len(row)},
            known,
            "column",
        )
        if row[col["invoice number"]].strip():
            items.append(row)
            continue
        label = row[col["unit price"]].strip().lower()
        value = _money(row[col["line total"]])
        for key in ("subtotal", "tax", "shipping", "total"):
            if label.startswith(key):
                summary[key] = value
                break
        else:
            if label or value is not None:
                raise ValueError(f"summary row {label!r} not read in code")

    def cell(row: list[str], name: str) -> str | None:
        return (
            row[col[name]].strip() or None
            if name in col and col[name] < len(row)
            else None
        )

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
        currency=cell(first, "currency") or "USD",
        subtotal=summary.get("subtotal"),
        tax_amount=summary.get("tax"),
        shipping=summary.get("shipping"),
        total=summary.get("total"),
        notes=cell(first, "notes"),
    )


# --- XML ------------------------------------------------------------------


XML_FIELDS = {
    "header/invoice_number", "header/revision", "header/vendor",
    "header/vendor_address", "header/date", "header/due_date", "header/currency",
    "line_items/item/name", "line_items/item/quantity", "line_items/item/unit_price",
    "line_items/item/amount", "line_items/item/note", "totals/subtotal",
    "totals/tax_amount", "totals/tax_rate", "totals/shipping", "totals/total",
    "payment_terms", "notes",
}  # fmt: skip


def _xml_leaves(node: Any, path: str = "") -> dict[str, str]:
    """Every element with text, by its path below the root."""
    out: dict[str, str] = {}
    for child in node:
        where = f"{path}{child.tag}"
        if len(child):
            out.update(_xml_leaves(child, where + "/"))
        elif child.text and child.text.strip():
            out[where] = child.text.strip()
    return out


def _from_xml(text: str) -> Invoice:
    root = ElementTree.fromstring(text)
    # A namespaced or unknown layout, or an element this reader doesn't map
    # (a note, new bank details): the extraction agent reads the file.
    _check_known(_xml_leaves(root), XML_FIELDS, "element")

    def find(path: str) -> str | None:
        node = root.find(path)
        return node.text.strip() if node is not None and node.text else None

    return _check_found(
        Invoice(
            invoice_number=find("header/invoice_number"),
            revision=find("header/revision"),
            vendor=find("header/vendor"),
            vendor_address=find("header/vendor_address"),
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
            shipping=_money(find("totals/shipping")),
            total=_money(find("totals/total")),
            payment_terms=find("payment_terms"),
            notes=find("notes"),
        )
    )
