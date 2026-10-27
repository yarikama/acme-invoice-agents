.PHONY: install lint format typecheck test check db run run-all clean

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

test:
	uv run pytest

# Everything CI runs.
check: lint typecheck test

db:
	uv run python -c "from invoice_agents.inventory import Inventory; Inventory('inventory.db').seed()"

run:
	uv run python main.py --invoice_path=$(INVOICE)

run-all:
	uv run python main.py --dir data/invoices

clean:
	rm -rf .pytest_cache .ruff_cache inventory.db
