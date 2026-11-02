# One image for the API (which also serves the built web console) and the CLI.
# Live Grok when .env has a key (LLM_PROVIDER=auto); with none, it replays a
# recorded live run of the samples and says so.

# --- The web console, built once -------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- Python app --------------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Dependencies first, so code changes don't reinstall them. OCR (docling) is
# left out: it is large, and only scanned PDFs need it.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src/ src/
RUN uv sync --locked --no-dev

COPY main.py ./
COPY data/ data/
COPY --from=web /web/dist web/dist

# Runtime state lives on a volume: the inventory and ledger database,
# checkpoints, uploads and the decision log.
ENV LLM_PROVIDER=auto \
    DB_PATH=/state/inventory.db \
    CHECKPOINT_PATH=/state/checkpoints.db \
    UPLOADS_DIR=/state/uploads \
    DECISIONS_LOG=/state/logs/decisions.jsonl
RUN useradd --create-home acme && mkdir -p /state data/inbox \
    && chown -R acme /state data/inbox
USER acme
VOLUME /state

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "invoice_agents.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
