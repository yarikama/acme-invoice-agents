from invoice_agents.ingestion import load_document
from invoice_agents.models import Invoice, LineItem, Severity
from invoice_agents.validation import SeenInvoices, validate
from tests.conftest import INVOICES


def codes(report, severity=None):
    return {
        f.code for f in report.findings if severity is None or f.severity is severity
    }


def load(name) -> Invoice:
    inv = load_document(INVOICES / name).invoice
    assert inv is not None
    return inv


def make(items, total, **kw) -> Invoice:
    fields = {
        "invoice_number": "INV-9000",
        "vendor": "Test Vendor",
        "invoice_date": "2026-01-01",
        "due_date": "2026-02-01",
        "line_items": [LineItem(item=i, quantity=q, unit_price=p) for i, q, p in items],
        "total": total,
    }
    return Invoice(**(fields | kw))


def test_clean_invoice_passes(inventory):
    report = validate(load("invoice_1015.csv"), inventory)
    assert report.passed
    assert not report.warnings


def test_total_mismatch_1007(inventory):
    report = validate(load("invoice_1007.csv"), inventory)
    total = next(f for f in report.findings if f.code == "TOTAL_MISMATCH")
    assert "$15,635.00" in total.detail
    assert "-$110.00" in total.detail
    assert "STOCK_EXCEEDED" in codes(report)


def test_split_lines_are_summed_before_the_stock_check_1013(inventory):
    report = validate(load("invoice_1013.json"), inventory)
    stock = [f.detail for f in report.findings if f.code == "STOCK_EXCEEDED"]
    assert len(stock) == 3
    assert any("WidgetA billed 22" in d for d in stock)
    assert "TOTAL_MISMATCH" in codes(report)


def test_negative_quantity_and_missing_fields_1009(inventory):
    report = validate(load("invoice_1009.json"), inventory)
    assert {"INVALID_QUANTITY", "MISSING_FIELDS", "SUBTOTAL_MISMATCH"} <= codes(
        report, Severity.ERROR
    )


def test_unknown_item_1016(inventory):
    assert "UNKNOWN_ITEM" in codes(validate(load("invoice_1016.json"), inventory))


def test_zero_stock_item(inventory):
    report = validate(make([("FakeItem", 100, 1000)], 100_000), inventory)
    assert "OUT_OF_STOCK" in codes(report, Severity.ERROR)


def test_item_names_match_loosely(inventory):
    report = validate(
        make([("Widget A", 2, 250), ("gadget-x", 1, 750)], 1250), inventory
    )
    assert report.passed


def test_stock_depleted_is_a_warning(inventory):
    report = validate(make([("GadgetX", 5, 750)], 3750), inventory)
    assert codes(report, Severity.WARNING) == {"STOCK_DEPLETED"}


def test_foreign_currency_1014(inventory):
    assert "FOREIGN_CURRENCY" in codes(validate(load("invoice_1014.xml"), inventory))


def test_due_date_contradicts_terms(inventory):
    inv = make(
        [("GadgetX", 1, 750)],
        750,
        invoice_date="2026-01-30",
        due_date="2026-01-30",
        payment_terms="Net 30",
    )
    assert "TERMS_MISMATCH" in codes(validate(inv, inventory))


def test_unparseable_due_date_and_pressure_language(inventory):
    inv = make(
        [("WidgetA", 1, 250)],
        250,
        due_date="yesterday",
        notes="URGENT - Pay immediately to avoid penalties!!! Wire transfer preferred.",
    )
    assert {"UNPARSEABLE_DATE", "PRESSURE_LANGUAGE"} <= codes(validate(inv, inventory))


def test_near_and_over_threshold(inventory):
    near = make([("WidgetA", 1, 250)], 9975, line_items=[])
    assert "NEAR_THRESHOLD" in codes(validate(near, inventory))
    over = load("invoice_1005.json")
    assert "OVER_THRESHOLD" in codes(validate(over, inventory), Severity.INFO)


def test_revision_supersedes_and_exact_duplicate(inventory):
    seen = SeenInvoices()
    original = load("invoice_1004.json")
    assert not codes(validate(original, inventory, seen)) & {"DUPLICATE", "SUPERSEDES"}
    seen.add(original)

    revised = validate(load("invoice_1004_revised.json"), inventory, seen)
    assert "SUPERSEDES" in codes(revised, Severity.WARNING)

    again = validate(original, inventory, seen)
    assert "DUPLICATE" in codes(again, Severity.ERROR)
