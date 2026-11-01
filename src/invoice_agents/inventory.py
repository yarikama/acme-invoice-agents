"""The mock inventory database (SQLite) the validator checks against: the
stock and list price of each item, and the vendor master."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# The brief's seed rows, plus list prices so validation can compare them.
SEED: list[tuple[str, int, float]] = [
    ("WidgetA", 15, 250.00),
    ("WidgetB", 10, 500.00),
    ("GadgetX", 5, 750.00),
    ("FakeItem", 0, 1000.00),
]


# Vendors Acme has bought from before: the vendor master. An invoice from
# anyone else is a first invoice from a new vendor, and a name a letter off
# one of these is a possible impostor.
VENDORS: list[str] = [
    "Widgets Inc.",
    "Gadgets Co.",
    "Precision Parts Ltd.",
    "Global Supply Chain Partners",
    "Acme Industrial Supplies",
    "MegaWidgets Corp",
    "Consolidated Materials Group",
    "Summit Manufacturing Co.",
    "Atlas Industrial Supply",
    "TechParts International",
    "Reliable Components Inc.",
]


def item_key(name: str) -> str:
    """Compare names loosely: "Widget A", "widget-a" and "WidgetA" match.

    Notes in parentheses are dropped, so "WidgetA (rush order)" is WidgetA
    and its quantity counts against WidgetA's stock.
    """
    without_notes = re.sub(r"\([^)]*\)", "", name)
    return re.sub(r"[^a-z0-9]", "", without_notes.lower())


@dataclass(frozen=True)
class StockRecord:
    item: str
    stock: int
    unit_price: float


class Inventory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def seed(self) -> None:
        """Create the table and (re)load the seed rows. Safe to run twice."""
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS inventory ("
                " item TEXT PRIMARY KEY,"
                " stock INTEGER NOT NULL,"
                " unit_price REAL NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO inventory (item, stock, unit_price) VALUES (?, ?, ?)"
                " ON CONFLICT(item) DO UPDATE SET"
                " stock = excluded.stock, unit_price = excluded.unit_price",
                SEED,
            )
            conn.execute("CREATE TABLE IF NOT EXISTS vendors (name TEXT PRIMARY KEY)")
            conn.executemany(
                "INSERT OR IGNORE INTO vendors (name) VALUES (?)",
                [(v,) for v in VENDORS],
            )

    def records(self) -> list[StockRecord]:
        """Every item, as the database holds it now."""
        return sorted(self._all().values(), key=lambda r: r.item)

    def vendors(self) -> list[str]:
        """The vendor master; empty for a database without the table."""
        with sqlite3.connect(self.path) as conn:
            try:
                rows = conn.execute("SELECT name FROM vendors ORDER BY name")
            except sqlite3.OperationalError:
                return []
            return [r[0] for r in rows]

    def lookup(self, name: str) -> StockRecord | None:
        key = item_key(name)
        return self._all().get(key)

    def _all(self) -> dict[str, StockRecord]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT item, stock, unit_price FROM inventory")
            return {item_key(r[0]): StockRecord(*r) for r in rows}
