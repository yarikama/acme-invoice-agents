"""Token use and cost, per graph step and per run.

Agents call `record()` on every model reply. A graph node wraps its agent
call in `collecting()`, so what is recorded belongs to that step of that run:
the context variable is per task, so runs that overlap don't mix.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypedDict

Pricing = dict[str, tuple[float, float]]  # model -> (USD per input, per output token)
Tokens = dict[str, dict[str, int]]  # model -> {"input_tokens", "output_tokens"}


class StepUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    # None when a model's price is unknown (e.g. direct xAI, or offline).
    cost_usd: float | None


_current: ContextVar[Tokens | None] = ContextVar("usage", default=None)


@contextmanager
def collecting() -> Iterator[Tokens]:
    """Collect the token use of every model reply recorded inside the block."""
    tokens: Tokens = {}
    reset = _current.set(tokens)
    try:
        yield tokens
    finally:
        _current.reset(reset)


def record(message: Any) -> None:
    """Add a model reply's token use to the step being collected, if any."""
    tokens = _current.get()
    meta = getattr(message, "usage_metadata", None)
    if tokens is None or not meta:
        return
    info = getattr(message, "response_metadata", None) or {}
    model = info.get("model_name") or info.get("model") or "unknown"
    counts = tokens.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
    counts["input_tokens"] += int(meta.get("input_tokens") or 0)
    counts["output_tokens"] += int(meta.get("output_tokens") or 0)


def add(into: Tokens, tokens: Tokens) -> None:
    """Add one {model: usage} map to another."""
    for model, u in tokens.items():
        counts = into.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
        counts["input_tokens"] += int(u.get("input_tokens", 0))
        counts["output_tokens"] += int(u.get("output_tokens", 0))


def cost(tokens: dict[str, Any], pricing: Pricing) -> float | None:
    """Total USD for a {model: usage} map; None if a model has no price."""
    total = 0.0
    for model, u in tokens.items():
        price = pricing.get(model) or pricing.get(f"x-ai/{model}")
        if price is None:
            return None
        total += u["input_tokens"] * price[0] + u["output_tokens"] * price[1]
    return total


def step_usage(tokens: Tokens, pricing: Pricing) -> StepUsage | None:
    """One step's totals, or None if it made no model calls."""
    if not tokens:
        return None
    return StepUsage(
        input_tokens=sum(u["input_tokens"] for u in tokens.values()),
        output_tokens=sum(u["output_tokens"] for u in tokens.values()),
        cost_usd=cost(tokens, pricing),
    )


def openrouter_pricing() -> Pricing:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        models = json.load(r)["data"]
    return {
        m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
        for m in models
        if "pricing" in m
    }
