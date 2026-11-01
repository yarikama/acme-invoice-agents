"""Record what the agents answered on a live run, and replay it with no key.

`LLM_PROVIDER=replay` runs the whole pipeline (validation, policy, the VP
pause, payment) for real, but the extraction agent, the approver and the
critic answer from a recording of a live run instead of calling a model. A
reviewer without an API key sees the same agent conversation, tool calls and
token counts the live run produced.

What is recorded is each agent's answer, keyed by what it was given: the
document text for extraction; the invoice, the validation findings and the
critic's objections for the approver; the invoice, the findings and the draft
for the critic. The findings are in the key so that an invoice replayed in a
different situation (a duplicate, say) never shows a draft written for a
clean one. Prompts are not the key:
the approver's tool results depend on which invoices ran earlier, and a
reviewer may run the samples in any order. On replay the approver's tool
calls run again in code, so their results match the current state.

Anything not in the recording falls back to the rule-based agents, and says
so in its rationale; free text that was never recorded is skipped, as in
offline mode.

Record with `RECORD_TO=data/replay/recording.json` on a live provider.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from invoice_agents import usage
from invoice_agents.agents import (
    GUARDRAIL,
    Approver,
    Critic,
    Extraction,
    ExtractionUnavailableError,
    Extractor,
    Review,
    RuleApprover,
    RuleCritic,
)
from invoice_agents.models import (
    ApprovalDraft,
    Critique,
    Invoice,
    ToolCall,
    ValidationReport,
)
from invoice_agents.tools import ReviewTools

FALLBACK = "Not in the recording, so the rule-based approver decided: "


class NotRecordedError(ExtractionUnavailableError):
    """Replay has no answer for this document."""


def _key(kind: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]
    return f"{kind}:{digest}"


def _invoice(invoice: Invoice) -> str:
    return invoice.model_dump_json()


def draft_key(invoice: Invoice, report: ValidationReport, critique: list[str]) -> str:
    return _key("draft", _invoice(invoice), report.model_dump_json(), *critique)


def critique_key(
    invoice: Invoice, report: ValidationReport, draft: ApprovalDraft
) -> str:
    return _key(
        "critique", _invoice(invoice), report.model_dump_json(), draft.model_dump_json()
    )


class Recording:
    """Agent answers and the prices at the time, in one JSON file."""

    def __init__(
        self,
        path: Path,
        entries: dict[str, Any] | None = None,
        pricing: usage.Pricing | None = None,
    ) -> None:
        self.path = path
        self.entries: dict[str, Any] = entries or {}
        self.pricing: usage.Pricing = pricing or {}

    @classmethod
    def shared(cls, path: Path) -> Recording:
        """One instance per file, so concurrent runs add to the same entries."""
        key = path.resolve()
        if key not in _open:
            _open[key] = cls.load(path)
        return _open[key]

    @classmethod
    def load(cls, path: Path) -> Recording:
        if not path.is_file():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        pricing = {m: (p[0], p[1]) for m, p in data.get("pricing", {}).items()}
        return cls(path, data.get("entries", {}), pricing)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.entries[key] = value
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        used = {m for e in self.entries.values() for m in e.get("tokens", {})}
        data = {
            # Only the models the recording used.
            "pricing": {
                m: list(p) for m, p in sorted(self.pricing.items()) if m in used
            },
            "entries": dict(sorted(self.entries.items())),
        }
        self.path.write_text(
            json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )


_open: dict[Path, Recording] = {}


@contextmanager
def _spent() -> Iterator[usage.Tokens]:
    """The tokens recorded inside the block, still counted for the step."""
    outer = usage._current.get()  # pyright: ignore[reportPrivateUsage]
    with usage.collecting() as inner:
        yield inner
    if outer is not None:
        usage.add(outer, inner)


def _replayed(tokens: usage.Tokens) -> None:
    """Count a recorded answer's tokens for the step, marked as replayed."""
    current = usage._current.get()  # pyright: ignore[reportPrivateUsage]
    if current is not None:
        usage.add(current, {m: {**u, "replayed": 1} for m, u in tokens.items()})


# --- Recording -----------------------------------------------------------


class RecordingExtractor:
    def __init__(self, inner: Extractor, recording: Recording) -> None:
        self.inner, self.recording = inner, recording

    async def extract(self, text: str, feedback: list[str]) -> Extraction:
        with _spent() as tokens:
            out = await self.inner.extract(text, feedback)
        self.recording.put(
            _key("extract", text, *feedback),
            {"extraction": out.model_dump(mode="json"), "tokens": tokens},
        )
        return out


class RecordingApprover:
    def __init__(self, inner: Approver, recording: Recording) -> None:
        self.inner, self.recording = inner, recording

    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review:
        with _spent() as tokens:
            out = await self.inner.draft(invoice, report, critique, tools)
        self.recording.put(
            draft_key(invoice, report, critique),
            {
                "draft": out.draft.model_dump(mode="json"),
                "steps": [s.model_dump(mode="json") for s in out.steps],
                "tokens": tokens,
            },
        )
        return out


class RecordingCritic:
    def __init__(self, inner: Critic, recording: Recording) -> None:
        self.inner, self.recording = inner, recording

    async def review(
        self,
        invoice: Invoice,
        report: ValidationReport,
        draft: ApprovalDraft,
        checked: list[ToolCall],
    ) -> Critique:
        with _spent() as tokens:
            out = await self.inner.review(invoice, report, draft, checked)
        self.recording.put(
            critique_key(invoice, report, draft),
            {"critique": out.model_dump(mode="json"), "tokens": tokens},
        )
        return out


# --- Replay --------------------------------------------------------------


class ReplayExtractor:
    def __init__(self, recording: Recording) -> None:
        self.recording = recording

    async def extract(self, text: str, feedback: list[str]) -> Extraction:
        hit = self.recording.get(_key("extract", text, *feedback))
        if hit is None:
            raise NotRecordedError(
                "Replay has no recorded extraction for this document; it covers"
                " the sample invoices. Set LLM_PROVIDER and an API key to read"
                " new free-text files."
            )
        _replayed(hit["tokens"])
        return Extraction.model_validate(hit["extraction"])


class ReplayApprover:
    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.fallback = RuleApprover()

    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review:
        hit = self.recording.get(draft_key(invoice, report, critique))
        if hit is None:
            review = await self.fallback.draft(invoice, report, critique, tools)
            draft = review.draft.model_copy(
                update={"rationale": FALLBACK + review.draft.rationale}
            )
            return Review(draft, review.steps)
        _replayed(hit["tokens"])
        steps: list[ToolCall] = []
        for raw in hit["steps"]:
            step = ToolCall.model_validate(raw)
            # The model's choice of call is replayed; code answers it again.
            steps.append(
                tools.note(step)
                if step.name == GUARDRAIL
                else tools.run(step.name, step.args)
            )
        return Review(ApprovalDraft.model_validate(hit["draft"]), steps)


class ReplayCritic:
    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.fallback = RuleCritic()

    async def review(
        self,
        invoice: Invoice,
        report: ValidationReport,
        draft: ApprovalDraft,
        checked: list[ToolCall],
    ) -> Critique:
        hit = self.recording.get(critique_key(invoice, report, draft))
        if hit is None:
            return await self.fallback.review(invoice, report, draft, checked)
        _replayed(hit["tokens"])
        return Critique.model_validate(hit["critique"])
