"""The approver's tools: plain code over the inventory and invoice history."""

import pytest

from invoice_agents.inventory import Inventory
from invoice_agents.models import Invoice
from invoice_agents.tools import ReviewTools
from invoice_agents.validation import SeenInvoices


def invoice(number: str, vendor: str = "Widgets Inc.", **extra: object) -> Invoice:
    fields: dict[str, object] = {
        "invoice_number": number,
        "vendor": vendor,
        "invoice_date": "2026-01-15",
        "due_date": "2026-02-01",
        "line_items": [],
        "total": 500,
    }
    return Invoice.model_validate(fields | extra)


CURRENT = invoice("INV-9")


@pytest.fixture
def seen() -> SeenInvoices:
    return SeenInvoices()


@pytest.fixture
def tools(inventory: Inventory, seen: SeenInvoices) -> ReviewTools:
    return ReviewTools(inventory, seen, CURRENT)


def test_check_inventory_matches_names_loosely(tools: ReviewTools) -> None:
    assert tools.check_inventory("gadget x") == (
        "GadgetX: 5 in stock, list price $750.00."
    )
    assert tools.check_inventory("SuperGizmo") == (
        "SuperGizmo is not in the inventory database."
    )


def test_vendor_history_lists_earlier_invoices_but_not_this_one(
    tools: ReviewTools, seen: SeenInvoices
) -> None:
    seen.add(invoice("INV-1", vendor="widgets inc", total=1200.5))
    seen.add(invoice("INV-2", revision="2", total=None, invoice_date=None))
    seen.add(invoice("INV-3", vendor="Someone Else"))
    seen.add(CURRENT)  # recorded by validation before the approver runs

    assert tools.vendor_history("Widgets Inc.") == (
        "2 earlier from Widgets Inc.: INV-1, 2026-01-15, USD 1200.5;"
        " INV-2 rev 2, undated, USD no total."
    )


def test_vendor_history_with_none(tools: ReviewTools, seen: SeenInvoices) -> None:
    seen.add(CURRENT)

    assert tools.vendor_history("Widgets Inc.") == (
        "No earlier invoices from this vendor."
    )


def test_convert_to_usd(tools: ReviewTools) -> None:
    assert tools.convert_to_usd(4125, "eur") == (
        "EUR 4,125.00 is $4,455.00 at the mock rate 1.08 USD per EUR."
    )
    assert (
        tools.convert_to_usd(10, "XYZ") == "No rate for XYZ; a human must convert it."
    )


def test_run_records_the_call(tools: ReviewTools) -> None:
    call = tools.run("convert_to_usd", {"amount": 100, "currency": "GBP"})

    assert call.name == "convert_to_usd"
    assert call.args == {"amount": 100, "currency": "GBP"}
    assert call.describe() == (
        "convert_to_usd(100, GBP): GBP 100.00 is $127.00 at the mock rate 1.27"
        " USD per GBP."
    )


def test_run_reports_bad_calls_instead_of_raising(tools: ReviewTools) -> None:
    assert tools.run("pay_now", {}).result == "Unknown tool 'pay_now'."
    assert tools.run("check_inventory", {}).result.startswith(
        "Bad arguments for check_inventory:"
    )


def test_langchain_tools_have_schemas(tools: ReviewTools) -> None:
    by_name = {t.name: t for t in tools.as_langchain()}

    assert list(by_name) == ["check_inventory", "vendor_history", "convert_to_usd"]
    assert "stock and list price" in by_name["check_inventory"].description
    assert set(by_name["convert_to_usd"].args) == {"amount", "currency"}
