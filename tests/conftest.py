from collections.abc import Iterator
from pathlib import Path

import pytest

from invoice_agents.inventory import Inventory

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"


@pytest.fixture
def inventory(tmp_path: Path) -> Inventory:
    inv = Inventory(tmp_path / "inventory.db")
    inv.seed()
    return inv


@pytest.fixture(autouse=True)
def fresh_memory(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts with no conversation kept in the process, and never
    touches (or starts) the developer's Redis; test_memory uses its own."""
    from invoice_agents import memory
    from invoice_agents.config import get_settings

    monkeypatch.setattr(memory, "_local", {})
    monkeypatch.setenv("REDIS_URL", "")
    # Replayed answers come back at once in tests, not at the recorded pace.
    monkeypatch.setenv("REPLAY_PACE", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
