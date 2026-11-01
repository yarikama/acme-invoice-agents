"""Fixed mock exchange rates: the system runs offline, with no live FX feed."""

from __future__ import annotations

USD_PER_UNIT: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CAD": 0.73,
    "MXN": 0.055,
    "JPY": 0.0067,
}


def to_usd(amount: float, currency: str) -> float | None:
    """The amount in US dollars, or None when there is no rate for it."""
    rate = USD_PER_UNIT.get(currency.upper())
    return None if rate is None else amount * rate
