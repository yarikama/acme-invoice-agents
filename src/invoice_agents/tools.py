"""Tools the approver can call while it decides.

Each tool is plain code over local data, so the facts it returns are
exact. The model chooses what to look up; it never computes the answer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from invoice_agents.inventory import Inventory, item_key
from invoice_agents.models import Invoice, ToolCall
from invoice_agents.validation import SeenInvoices

# Fixed mock rates: the system runs offline, so there is no live FX feed.
USD_PER_UNIT: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CAD": 0.73,
    "MXN": 0.055,
    "JPY": 0.0067,
}


def _usd(amount: float) -> str:
    return f"${amount:,.2f}"


class ReviewTools:
    """The approver's tools, bound to this run's inventory and history."""

    def __init__(
        self, inventory: Inventory, seen: SeenInvoices, current: Invoice
    ) -> None:
        self.inventory = inventory
        self.seen = seen
        self.current = current
        # Built on first use, once: schemas come from the signatures.
        self._tools: dict[str, BaseTool] | None = None
        self._funcs: dict[str, Callable[..., str]] = {
            "check_inventory": self.check_inventory,
            "vendor_history": self.vendor_history,
            "convert_to_usd": self.convert_to_usd,
        }

    def check_inventory(self, item: str) -> str:
        """Look up an item's stock and list price in the inventory database."""
        record = self.inventory.lookup(item)
        if record is None:
            return f"{item} is not in the inventory database."
        return (
            f"{record.item}: {record.stock} in stock,"
            f" list price {_usd(record.unit_price)}."
        )

    def vendor_history(self, vendor: str) -> str:
        """List earlier invoices from this vendor that were already processed."""
        key = item_key(vendor)
        earlier = [inv for inv in self._others() if item_key(inv.vendor or "") == key]
        if not earlier:
            return "No earlier invoices from this vendor."
        lines = [
            f"{inv.invoice_number or 'no number'}"
            f"{f' rev {inv.revision}' if inv.revision else ''},"
            f" {inv.invoice_date or 'undated'},"
            f" {inv.currency} {inv.total if inv.total is not None else 'no total'}"
            for inv in earlier
        ]
        return f"{len(earlier)} earlier from {vendor}: " + "; ".join(lines) + "."

    def convert_to_usd(self, amount: float, currency: str) -> str:
        """Convert an amount to US dollars at the fixed mock rate."""
        rate = USD_PER_UNIT.get(currency.upper())
        if rate is None:
            return f"No rate for {currency}; a human must convert it."
        return (
            f"{currency.upper()} {amount:,.2f} is {_usd(amount * rate)}"
            f" at the mock rate {rate} USD per {currency.upper()}."
        )

    def as_langchain(self) -> list[BaseTool]:
        if self._tools is None:
            self._tools = {
                name: StructuredTool.from_function(func=f, name=name)
                for name, f in self._funcs.items()
            }
        return list(self._tools.values())

    def run(self, name: str, args: dict[str, Any]) -> ToolCall:
        """Run a tool the model called. LangChain validates and coerces the
        arguments against the signature, so "100" arrives as 100.0."""
        tool = next((t for t in self.as_langchain() if t.name == name), None)
        extra = sorted(set(args) - set(tool.args)) if tool else []
        if tool is None:
            result = f"Unknown tool {name!r}."
        elif extra:
            # The schema would drop them silently; tell the model instead.
            result = f"Bad arguments for {name}: unexpected {', '.join(extra)}."
        else:
            try:
                result = str(tool.invoke(args))
            except (TypeError, ValueError) as exc:
                result = f"Bad arguments for {name}: {exc}"
        return ToolCall(name=name, args=args, result=result)

    def _others(self) -> list[Invoice]:
        """Processed invoices except this one (it is recorded before review)."""
        others = list(self.seen.invoices)
        if self.current in others:
            others.remove(self.current)
        return others
