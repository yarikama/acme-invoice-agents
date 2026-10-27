"""Turn the date formats seen on invoices into ISO 8601."""

from __future__ import annotations

from datetime import date, datetime

_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%b %d %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%d-%b-%Y",
    "%d %b %Y",
)


def parse_date(raw: str | None) -> date | None:
    """Return the date, or None when the text is not a date ("yesterday")."""
    if not raw:
        return None
    text = raw.strip()
    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).date()  # noqa: DTZ007 - a date, no time
        except ValueError:
            continue
    return None


def normalize_date(raw: str | None) -> str | None:
    """ISO 8601 when parseable; otherwise the raw text, so validation can flag it."""
    parsed = parse_date(raw)
    if parsed:
        return parsed.isoformat()
    return raw.strip() if raw and raw.strip() else None
