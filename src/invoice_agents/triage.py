"""System 1: a decision model's quick read, before the reasoning approver.

Decision models (OpenRouter's Decisions API) answer typed questions in one
pass: no text, just a choice and its probabilities, in well under a second
and for a tiny fraction of a cent. Triage asks one what to do with an
invoice and whether anything about it looks off.

Its answer only counts where code can confirm it:
- reject, when validation found an error (the rules reject it anyway);
- approve, when every check passed and the model sees nothing suspicious.
Anything else, or any answer below the confidence bar, goes to the reasoning
approver. Text planted in an invoice can flip a decision model's answer, so
triage never decides alone: it can only take a path the rules already allow.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, SecretStr

from invoice_agents.models import Decision, Invoice, ValidationReport

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

QUESTIONS: dict[str, Any] = {
    "decision": {
        "type": "choice",
        "instructions": (
            "You review invoices for accounts payable. Given the invoice and"
            " the findings that code computed from it, what should happen?"
        ),
        "criteria": {
            "approve": "Every check passed and nothing about the invoice is odd.",
            "reject": "An error finding means the invoice cannot be paid.",
            "escalate": "A warning, or something odd, needs a person to judge.",
        },
    },
    "suspicious": {
        "type": "noul",
        "instructions": (
            "Does anything suggest fraud or a fake invoice: pressure to pay"
            " now, an unlikely vendor or address, or items that make no sense?"
        ),
        "criteria": {
            "true": "Something about the invoice looks wrong or made up.",
            "false": "It reads like an ordinary invoice from a real supplier.",
        },
    },
}


class TriageAnswer(BaseModel):
    """What the decision model said, and what asking it cost."""

    model: str
    decision: Decision
    confidence: float
    # Probability that something about the invoice looks off.
    suspicion: float
    seconds: float
    input_tokens: int = 0
    cost_usd: float | None = None

    def describe(self) -> str:
        return (
            f"{self.model} says {self.decision} ({self.confidence:.0%} sure),"
            f" suspicion {self.suspicion:.0%}"
        )


class Triager(Protocol):
    model: str

    async def ask(self, invoice: Invoice, report: ValidationReport) -> TriageAnswer: ...


def triage_state(invoice: Invoice, report: ValidationReport) -> dict[str, Any]:
    return {
        "invoice": invoice.model_dump(mode="json", exclude_none=True),
        "findings": [
            f"[{f.severity.value}] {f.code}: {f.title}. {f.detail}"
            for f in report.findings
        ]
        or ["None: every check passed."],
    }


class TriageError(RuntimeError):
    """The decision model could not be reached or gave an unusable answer."""


class DecisionTriager:
    """Asks a model on OpenRouter's Decisions API."""

    def __init__(
        self,
        model: str,
        api_key: SecretStr,
        client: httpx.AsyncClient | None = None,
        retries: int = 4,
        timeout: float = 30,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.client = client
        self.retries = retries
        self.timeout = timeout

    async def ask(self, invoice: Invoice, report: ValidationReport) -> TriageAnswer:
        body = {
            "model": self.model,
            "state": triage_state(invoice, report),
            "questions": QUESTIONS,
        }
        start = time.perf_counter()
        data = await self._post(body)
        seconds = time.perf_counter() - start
        try:
            choice = data["answers"]["decision"]
            usage = data.get("usage") or {}
            return TriageAnswer(
                model=self.model,
                decision=choice["choice"],
                confidence=float(
                    choice.get("confidence")
                    or choice.get("probabilities", {}).get(choice["choice"], 0)
                ),
                suspicion=float(data["answers"]["suspicious"]["noul"]),
                seconds=round(seconds, 3),
                input_tokens=int(usage.get("input_tokens") or 0),
                cost_usd=usage.get("cost"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TriageError(f"Unexpected answer from {self.model}: {exc}") from exc

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
        client = self.client or httpx.AsyncClient(timeout=self.timeout)
        error = "no attempt made"
        try:
            for attempt in range(self.retries + 1):
                try:
                    r = await client.post(DECISIONS_URL, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    if r.status_code == 200:
                        return r.json()
                    error = f"HTTP {r.status_code}: {r.text[:200]}"
                    if r.status_code != 429 and r.status_code < 500:
                        break
                if attempt < self.retries:
                    await asyncio.sleep(min(2**attempt, 8))
            raise TriageError(f"{self.model} failed: {error}")
        finally:
            if self.client is None:
                await client.aclose()


def fast_path(
    answer: TriageAnswer,
    report: ValidationReport,
    min_confidence: float,
    max_suspicion: float,
) -> tuple[Decision, str] | None:
    """The decision to take without the reasoning approver, and the code check
    that confirmed it; None when the invoice needs the full review."""
    if answer.confidence < min_confidence:
        return None
    if answer.decision == "reject" and report.errors:
        codes = ", ".join(dict.fromkeys(f.code for f in report.errors))
        return "reject", f"validation found errors ({codes})"
    if (
        answer.decision == "approve"
        and not report.errors
        and not report.warnings
        and answer.suspicion <= max_suspicion
    ):
        return "approve", "every check passed"
    return None
