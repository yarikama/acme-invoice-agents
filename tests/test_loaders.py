from pathlib import Path

import pytest

from invoice_agents.ingestion import UnsupportedFormatError, load_document
from invoice_agents.ingestion.dates import normalize_date
from invoice_agents.models import SourceFormat
from tests.conftest import INVOICES


def test_json_with_nested_vendor():
    inv = load_document(INVOICES / "invoice_1004.json").invoice
    assert inv is not None
    assert inv.vendor == "Precision Parts Ltd."
    assert inv.vendor_address == "742 Evergreen Terrace, Springfield, IL 62704"
    assert [li.item for li in inv.line_items] == ["WidgetA", "WidgetB"]
    assert inv.total == 1890.00


def test_json_blank_fields_become_none():
    inv = load_document(INVOICES / "invoice_1009.json").invoice
    assert inv is not None
    assert inv.vendor is None
    assert inv.due_date is None
    assert inv.payment_terms is None
    assert inv.line_items[0].quantity == -5


def test_json_revision():
    inv = load_document(INVOICES / "invoice_1004_revised.json").invoice
    assert inv is not None
    assert inv.revision == "R1"


def test_key_value_csv_groups_repeated_keys():
    inv = load_document(INVOICES / "invoice_1006.csv").invoice
    assert inv is not None
    assert [(li.item, li.quantity, li.unit_price) for li in inv.line_items] == [
        ("WidgetA", 5, 250.0),
        ("WidgetB", 3, 500.0),
    ]
    assert inv.total == 2750.00


def test_table_csv_reads_summary_rows_and_us_dates():
    inv = load_document(INVOICES / "invoice_1007.csv").invoice
    assert inv is not None
    assert inv.invoice_date == "2026-01-28"
    assert len(inv.line_items) == 3
    assert (inv.subtotal, inv.tax_amount, inv.total) == (14750.00, 885.00, 15525.00)


def test_xml_keeps_currency():
    inv = load_document(INVOICES / "invoice_1014.xml").invoice
    assert inv is not None
    assert inv.currency == "EUR"
    assert inv.total == 4125.00


@pytest.mark.parametrize("name", ["invoice_1001.txt", "invoice_1011.pdf"])
def test_free_text_needs_extraction(name):
    doc = load_document(INVOICES / name)
    assert doc.invoice is None
    assert "INV-1001" in doc.text or "INV-1011" in doc.text


def test_pdf_text_is_extracted():
    doc = load_document(INVOICES / "invoice_1013.pdf")
    assert doc.format is SourceFormat.PDF
    assert "Grand Total" in doc.text


def test_unsupported_format(tmp_path):
    path = tmp_path / "invoice.docx"
    path.write_text("x")
    with pytest.raises(UnsupportedFormatError):
        load_document(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-01-15", "2026-01-15"),
        ("01/28/2026", "2026-01-28"),
        ("Jan 30 2026", "2026-01-30"),
        ("January 27, 2026", "2026-01-27"),
        ("25-Feb-2026", "2026-02-25"),
        ("yesterday", "yesterday"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    ("name", "content", "why"),
    [
        ("weird.csv", b"a,b\n1\n", "Could not read the CSV in code"),
        ("list.json", b"[]", "Could not read the JSON in code"),
        ("broken.json", b'{"vendor": ', "not valid JSON"),
        (
            "types.json",
            b'{"line_items": [{"item": "W", "quantity": "lots", "unit_price": 1}]}',
            "quantity: input should be a valid number",
        ),
        ("broken.xml", b"<invoice><total>", "not valid XML"),
    ],
)
def test_a_structured_file_code_cannot_read_goes_to_the_agent_as_text(
    tmp_path: Path, name: str, content: bytes, why: str
) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    doc = load_document(path)
    assert doc.invoice is None
    assert doc.text == content.decode()
    assert doc.problem is not None and why in doc.problem


def test_text_in_an_older_encoding_is_still_read(tmp_path: Path) -> None:
    path = tmp_path / "old.txt"
    path.write_bytes("Café Supplies, total €1,000".encode("cp1252"))
    assert load_document(path).text == "Café Supplies, total €1,000"


def test_files_with_nothing_to_read_say_why(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n")
    assert load_document(empty).problem == "The file is empty."
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not a pdf")
    doc = load_document(fake)
    assert doc.text == ""
    assert doc.problem is not None
    assert doc.problem.startswith("The PDF could not be opened")


def test_nan_in_a_json_invoice_goes_to_the_agent_as_text(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text(
        '{"line_items": [{"item": "W", "quantity": 1, "unit_price": "NaN"}],'
        ' "total": "Infinity"}'
    )
    doc = load_document(path)
    assert doc.invoice is None and doc.problem is not None
    assert doc.problem.startswith("Could not read the JSON in code")
