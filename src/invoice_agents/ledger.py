"""The ledger: every invoice that reached validation, and its outcome.

It is what duplicate, split and lookalike checks compare against, and what
the approver's vendor_history tool reads, so a restart doesn't forget an
invoice that was already paid. It is also the decision log: each outcome is
stored with its reason, and appended as one JSON line to a log file that
other tools can read.

The table lives in the inventory database. Every read goes to SQLite, so the
API and a CLI run on the same machine see each other's invoices.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from invoice_agents.models import Invoice
from invoice_agents.validation import Outcome, Received, SeenInvoices

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    run_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    invoice_number TEXT,
    vendor TEXT,
    total REAL,
    currency TEXT,
    invoice TEXT NOT NULL,
    outcome TEXT,
    reason TEXT,
    received_at TEXT NOT NULL,
    settled_at TEXT
)
"""


class Ledger(SeenInvoices):
    def __init__(self, db_path: Path, log_path: Path | None = None) -> None:
        super().__init__()
        self.db_path = db_path
        self.log_path = log_path
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @property
    def entries(self) -> list[Received]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT invoice, run_id, source, outcome, reason, received_at"
                " FROM ledger ORDER BY received_at, rowid"
            ).fetchall()
        return [Received(Invoice.model_validate_json(row[0]), *row[1:]) for row in rows]

    def add(
        self, invoice: Invoice, run_id: str | None = None, source: str = ""
    ) -> None:
        entry = Received(invoice, run_id or uuid.uuid4().hex, source)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ledger (run_id, source, invoice_number,"
                " vendor, total, currency, invoice, received_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.run_id,
                    entry.source,
                    invoice.invoice_number,
                    invoice.vendor,
                    invoice.total,
                    invoice.currency,
                    invoice.model_dump_json(),
                    entry.received_at,
                ),
            )

    def settle(self, run_id: str, outcome: Outcome, reason: str | None = None) -> None:
        settled_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT source, invoice_number, vendor, total, currency"
                " FROM ledger WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return
            conn.execute(
                "UPDATE ledger SET outcome = ?, reason = ?, settled_at = ?"
                " WHERE run_id = ?",
                (outcome, reason, settled_at, run_id),
            )
        source, number, vendor, total, currency = row
        self._log(
            {
                "at": settled_at,
                "run_id": run_id,
                "source": source,
                "invoice_number": number,
                "vendor": vendor,
                "total": total,
                "currency": currency,
                "outcome": outcome,
                "reason": reason,
            }
        )

    def waiting(self) -> list[Received]:
        """Invoices not settled yet: in progress, or paused for a VP."""
        return [e for e in self.entries if e.outcome is None]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ledger")

    def _log(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
