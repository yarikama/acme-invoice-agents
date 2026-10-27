# Acme Invoice Agents

A multi-agent system that automates Acme Corp's invoice processing end to end: ingestion, validation, approval, and payment.

> 🚧 Work in progress.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install                # dependencies and the pre-commit hooks
cp .env.example .env        # set LLM_PROVIDER and a key; offline works without one
python main.py --invoice_path=data/invoices/invoice_1013.json
python main.py --dir data/invoices --vp ask
```

`--vp ask|approve|reject` answers invoices that pause for a human VP. `--json` prints the results to stdout.

In `offline` mode the agents are rule-based and nothing leaves the machine. JSON, CSV and XML invoices are fully processed; TXT and PDF invoices need an LLM to extract them and are skipped.

## How it works

```
ingest ──▶ validate ──▶ review ──▶ critique ──▶ pay
  ▲  │                    ▲           │  │
  └──┘ retry with         └───────────┘  └──▶ human_review ──▶ pay / reject
       feedback            revise until
                           the critic agrees
```

- **Ingest.** JSON, CSV and XML are parsed in code. TXT and PDF go to the extraction agent, which returns a typed `Invoice`; an empty or partial result is sent back with the problems listed.
- **Validate.** Deterministic checks, never an LLM: totals and line amounts, quantities summed per item against SQLite stock, unknown items, duplicates and revisions, currency, dates against payment terms, and amounts just under the approval threshold.
- **Approve.** An approver drafts approve, reject or escalate; a critic checks the draft against the findings and the policy, and the approver revises. Code has the last word: an invoice with an error is rejected and one over $10,000 goes to a human, whatever the agents say.
- **Pay.** `mock_payment` for approved invoices; rejections are logged with the reasoning.

## API

```bash
make api        # http://localhost:8000, docs at /docs
```

| Method | Path | |
|---|---|---|
| `POST` | `/api/runs` | Start a run on a sample: `{"sample": "invoice_1013.json"}` |
| `POST` | `/api/runs/upload` | Start a run on an uploaded file |
| `POST` | `/api/runs/batch` | Run every sample, in order |
| `GET` | `/api/runs`, `/api/runs/{id}` | Run list; one run's state and events |
| `GET` | `/api/runs/{id}/events` | Server-sent events |
| `POST` | `/api/runs/{id}/decision` | The VP's answer: `{"approved": true, "note": "..."}` |

The event stream replays the run's history, then stays open for live events, including while the run waits for the VP; the decision's events arrive on the same connection. Reconnecting with `Last-Event-ID` skips what the client already has. Checkpoints are stored in SQLite, so the graph state of a paused run survives a restart.

## Development

```bash
make check      # ruff, ruff format --check, pyright, pytest: what CI runs
make format
```

## Credits

The case brief (`docs/case-brief.md`) and the sample invoices in `data/` come from [galatiq-ai/galatiq-case-invoices](https://github.com/galatiq-ai/galatiq-case-invoices).
