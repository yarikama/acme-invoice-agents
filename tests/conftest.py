from pathlib import Path

import pytest

from invoice_agents.inventory import Inventory

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"


@pytest.fixture
def inventory(tmp_path: Path) -> Inventory:
    inv = Inventory(tmp_path / "inventory.db")
    inv.seed()
    return inv
