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
ingest ──▶ validate ──▶ [triage] ──▶ review ──▶ critique ──▶ pay
  ▲  │                     │          ▲           │  │
  └──┘ retry with          │          └───────────┘  └──▶ human_review ──▶ pay / reject
       feedback            │           revise until
                           │           the critic agrees
                           └──▶ pay / reject / human_review   (clear-cut cases only)
```

- **Ingest.** JSON, CSV and XML are parsed in code. TXT and PDF go to the extraction agent, which returns a typed `Invoice`; an empty or partial result is sent back with the problems listed.
- **Validate.** Deterministic checks, never an LLM: totals and line amounts, quantities summed per item against SQLite stock, unknown items, duplicates and revisions, currency, dates against payment terms, and amounts just under the approval threshold.
- **Triage (optional).** With `TRIAGE_MODEL` set, a decision model on OpenRouter's Decisions API reads the invoice and its findings first and answers two typed questions: approve, reject or escalate, and whether anything looks off. It takes a fraction of a second and costs about $0.00001. Its answer settles the invoice only where code confirms it: a reject when validation found an error, or an approve when every check passed and the model sees nothing suspicious, each above a confidence bar. Everything else goes to the reasoning approver. Text planted in an invoice can flip a decision model's answer, so triage can only take a path the rules already allow; it can never pay an invoice with a warning.
- **Approve.** The approver is a tool-calling agent. Before it decides, it can look up an item's stock and list price (`check_inventory`), the vendor's earlier invoices (`vendor_history`), and the USD value of a foreign total (`convert_to_usd`, at fixed mock rates). It submits approve, reject or escalate through a `SubmitDecision` tool, citing what it found. Code lints that decision before accepting it, and sends it back with the reasons if it approves past an error, rejects an invoice that passed every check (it must escalate its suspicion instead), leaves a finding unaddressed, or skips the lookup a finding depends on. A critic then checks the accepted draft against the findings, the policy and the lookups, and the approver revises if needed. Every tool call and send-back shows in the trace.

  Code has the last word: an invoice with an error is rejected, and one over $10,000 goes to a VP, whatever the agents say. So do invoices billed in a foreign currency, totals just under the threshold, and revisions of an invoice already received (`VP_REQUIRED_FINDINGS`; set it to `[]` to let the approver clear those warnings itself).
- **Pay.** `mock_payment` for approved invoices; rejections are logged with the reasoning.

## Evaluation

`make eval` runs all 20 sample cases (19 files, plus a revision that only makes sense after the original) against expected outcomes, with a fresh inventory per case, and reports accuracy, latency and cost from the provider's token counts. `make triage-eval` asks several decision models the triage questions on the same cases.

Live runs on OpenRouter, extraction on `grok-4.3`, review on `grok-4.7`:

| Setup | Correct | Settled by triage | Mean time | Cost per invoice |
|---|---|---|---|---|
| Grok only | 19/20 | — | 31.6 s | $0.0257 |
| + triage, `typesafe/jev-1.14` | 18/20 ¹ | 8 | 20.3 s | $0.0152 |
| + triage, `perplexity/pplx-decider-flash-v1.2` | 20/20 | 16 | 12.9 s | $0.0087 |

¹ Both misses were provider errors on the Grok side (a 502 and an empty response); both cases pass on a rerun.

Triage alone, six decision models, same 20 cases:

| Model | Agrees with expected | Would settle alone | Wrong when settling alone | Mean latency | Cost per invoice |
|---|---|---|---|---|---|
| `typesafe/jev-1.14` | 100% | 7 | 0 | 0.31 s | $0.000032 |
| `cloudflare/clef-flash` | 90% | 2 | 0 | 0.68 s | $0.000011 |
| `cloudflare/clef` | 100% | 4 | 0 | 0.42 s | $0.000107 |
| `perplexity/pplx-decider-flash-v1.2` | 100% | 16 | 0 | 0.27 s | $0.000008 |
| `liquid/d1` | 100% | 10 | 0 | 0.34 s | $0.000009 |
| `microsoft/microsoft-decision-1` | 100% | 16 | 0 | 0.49 s | $0.000016 |

What this shows, and what it doesn't:

- The decision models see the findings code computed, so agreement mostly measures whether they read those findings correctly. Twenty invoices is a smoke test, not a benchmark.
- The useful differences are how often a model is sure enough to settle an invoice and how well it spots something odd. Jev was the only model to give INV-1004, whose vendor address is the Simpsons' house, enough suspicion (0.30) to send it on to Grok; Perplexity's model approved it on the fast path. Jev's lower confidence on rejects is why it settles fewer invoices.
- No model was ever wrong when it settled an invoice alone, because code only lets it settle the two kinds of case the rules already decide.

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

The event stream replays the run's history, then stays open for live events, including while the run waits for the VP; the decision's events arrive on the same connection. Reconnecting with `Last-Event-ID` skips what the client already has. Graph checkpoints are stored in SQLite. The list of runs is kept in memory for now, so a restart forgets them.

## Web console

```bash
make api        # terminal 1: the API on :8000
make web        # terminal 2: the console on http://localhost:5173
```

React, Vite and Tailwind. One server-sent event stream (`/api/events`) feeds the whole console: the inbox, the pipeline lights, the agent trace and the VP decision panel all update as events arrive.

## Development

```bash
make check      # ruff, ruff format --check, pyright, pytest: what CI runs
make format
```

## Credits

The case brief (`docs/case-brief.md`) and the sample invoices in `data/` come from [galatiq-ai/galatiq-case-invoices](https://github.com/galatiq-ai/galatiq-case-invoices).
