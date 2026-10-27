"""The mock banking API from the brief."""

from __future__ import annotations

import logging

from invoice_agents.models import PaymentResult

log = logging.getLogger(__name__)


def mock_payment(vendor: str, amount: float, currency: str = "USD") -> PaymentResult:
    log.info("Paid %s %.2f to %s", currency, amount, vendor)
    return PaymentResult(
        status="success", vendor=vendor, amount=amount, currency=currency
    )
