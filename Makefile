.PHONY: docker-up docker-down docker-run install lint format typecheck test check db eval triage-eval run run-all api web web-check clean

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
	cd web && npm run format

typecheck:
	uv run pyright

test:
	uv run pytest --cov

# Everything CI runs.
check: lint typecheck test web-check

db:
	uv run python -c "from invoice_agents.inventory import Inventory; Inventory('inventory.db').seed()"

eval:
	uv run python -m invoice_agents.evaluation $(ARGS)

triage-eval:
	uv run python -m invoice_agents.triage_eval $(ARGS)

run:
	uv run python main.py --invoice_path=$(INVOICE)

run-all:
	uv run python main.py --dir data/invoices

web:
	cd web && npm run dev

web-check:
	cd web && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build

api:
	uv run uvicorn invoice_agents.api.app:app --reload --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache inventory.db checkpoints.db uploads

# The whole system in Docker: API and console on :8000, Redis for the chat.
docker-up:
	docker compose up --build -d
	@echo "Console and API: http://localhost:8000"

docker-down:
	docker compose down

# e.g. make docker-run ARGS="--invoice_path=data/invoices/invoice_1001.txt"
docker-run:
	docker compose run --rm cli $(ARGS)
