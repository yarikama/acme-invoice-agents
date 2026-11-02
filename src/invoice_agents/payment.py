"""The mock banking API from the brief, behind an idempotency key.

A graph node can run again: after a crash between paying and checkpointing,
LangGraph resumes from the last checkpoint and the pay node runs a second
time. Banks answer that with an idempotency key, and so does this mock: the
first call with a key pays, any later call with the same key returns that
first payment and moves no money. The duplicate checks in validation stop a
second invoice; the key stops a second payment of the same one.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from invoice_agents.models import Invoice, PaymentResult

log = logging.getLogger(__name__)


def mock_payment(vendor: str, amount: float, currency: str = "USD") -> PaymentResult:
    """The brief's mock payment call: always succeeds."""
    log.info("Paid %s %.2f to %s", currency, amount, vendor)
    return PaymentResult(
        status="success", vendor=vendor, amount=amount, currency=currency
    )


def payment_key(invoice: Invoice) -> str:
    """One key per vendor, invoice number and revision: the same invoice
    always gets the same key, however its fields were spelled."""

    def plain(text: str | None) -> str:
        return re.sub(r"[^a-z0-9]", "", (text or "").lower())

    return ":".join(
        [plain(invoice.vendor), plain(invoice.invoice_number), plain(invoice.revision)]
    )


class MockBank:
    """mock_payment with idempotency keys, kept in SQLite when given a path
    (so a restart remembers them) or in memory for one process."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path
        self._memory: dict[str, PaymentResult] = {}
        if db_path is not None:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS payments ("
                    " key TEXT PRIMARY KEY,"
                    " result TEXT NOT NULL,"
                    " paid_at TEXT NOT NULL)"
                )

    def pay(self, invoice: Invoice) -> PaymentResult:
        assert invoice.vendor and invoice.total is not None
        key = payment_key(invoice)
        earlier = self._find(key)
        if earlier is not None:
            log.warning("Payment %s was already made; not paying again.", key)
            return earlier.model_copy(update={"duplicate": True})
        result = mock_payment(invoice.vendor, invoice.total, invoice.currency)
        result = result.model_copy(update={"idempotency_key": key})
        if result.status == "success":
            self._store(key, result)
        return result

    def paid(self, invoice: Invoice) -> bool:
        """Whether this invoice's payment went through."""
        return self._find(payment_key(invoice)) is not None

    def _find(self, key: str) -> PaymentResult | None:
        if self.db_path is None:
            return self._memory.get(key)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT result FROM payments WHERE key = ?", (key,)
            ).fetchone()
        return PaymentResult.model_validate_json(row[0]) if row else None

    def _store(self, key: str, result: PaymentResult) -> None:
        if self.db_path is None:
            self._memory[key] = result
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO payments (key, result, paid_at)"
                " VALUES (?, ?, ?)",
                (key, result.model_dump_json(), datetime.now(UTC).isoformat()),
            )

    def clear(self) -> None:
        """Forget every payment: part of starting a demo over."""
        self._memory.clear()
        if self.db_path is not None:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM payments")
