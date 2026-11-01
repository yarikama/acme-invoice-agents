.PHONY: install lint format typecheck test check db run run-all api web web-check clean

install:
	uv sync
	uv run pre-commit install
	cd web && npm ci

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

test:
	uv run pytest --cov

# Everything CI runs.
check: lint typecheck test web-check

db:
	uv run python -c "from invoice_agents.inventory import Inventory; Inventory('inventory.db').seed()"

run:
	uv run python main.py --invoice_path=$(INVOICE)

run-all:
	uv run python main.py --dir data/invoices

web:
	cd web && npm run dev

web-check:
	cd web && npm run lint && npm run typecheck && npm test && npm run build

api:
	uv run uvicorn invoice_agents.api.app:app --reload --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache inventory.db checkpoints.db uploads
