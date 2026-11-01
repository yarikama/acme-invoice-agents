"""Retrying provider hiccups the HTTP client can't see, and nothing else."""

from typing import Any

import pytest

from invoice_agents.agents import (
    Critique,
    LLMApprover,
    LLMCritic,
    is_transient,
    raised_at,
    with_retries,
)
from invoice_agents.inventory import Inventory
from invoice_agents.models import ApprovalDraft
from invoice_agents.tools import ReviewTools
from invoice_agents.validation import SeenInvoices
from tests.test_agents import (
    INVOICE,
    FakeChatModel,
    FakeToolModel,
    parsed,
    report,
    submit,
)

UPSTREAM_502 = ValueError(
    {
        "message": "Service temporarily unavailable.",
        "code": 502,
        "metadata": {"error_type": "provider_unavailable"},
    }
)


def raise_from(package: str, exc: Exception) -> Exception:
    """Raise exc from a frame whose file sits in `package`, and return it."""
    code = compile(
        "def boom(e):\n    raise e\n", f"/site-packages/{package}/x.py", "exec"
    )
    scope: dict[str, Any] = {}
    exec(code, scope)
    try:
        scope["boom"](exc)
    except Exception as caught:
        return caught
    raise AssertionError("did not raise")


def ours(exc: Exception) -> Exception:
    try:
        raise exc
    except Exception as caught:
        return caught


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("invoice_agents.agents.asyncio.sleep", sleep)
    return slept


def test_upstream_errors_in_the_body_are_transient():
    assert is_transient(UPSTREAM_502)
    assert is_transient(ValueError({"code": 429, "message": "rate limited"}))
    assert not is_transient(ValueError({"code": 400, "message": "bad request"}))
    assert not is_transient(ValueError("Model output did not match Critique"))


def test_malformed_bodies_count_only_inside_the_client():
    assert is_transient(raise_from("langchain_openai", TypeError("null choices")))
    assert is_transient(raise_from("openai", KeyError("choices")))
    # The same error from our own code is a bug: retrying would hide it.
    assert not is_transient(ours(TypeError("'NoneType' object is not iterable")))
    assert not is_transient(TypeError("never raised"))


def test_raised_at_finds_the_innermost_frame():
    exc = raise_from("openai", TypeError("x"))
    where = raised_at(exc)
    assert where is not None
    assert where[0].parts[-2:] == ("openai", "x.py")
    assert where[1] == 2
    assert raised_at(TypeError("never raised")) is None


async def test_with_retries_retries_then_succeeds(no_sleep: list[float]):
    calls = iter([UPSTREAM_502, UPSTREAM_502, "ok"])

    async def call() -> str:
        out = next(calls)
        if isinstance(out, Exception):
            raise out
        return out

    assert await with_retries(call) == "ok"
    assert no_sleep == [2, 4]


async def test_with_retries_gives_up_after_its_attempts():
    async def call() -> str:
        raise UPSTREAM_502

    with pytest.raises(ValueError, match="502"):
        await with_retries(call, attempts=2)


async def test_with_retries_does_not_retry_a_bug(no_sleep: list[float]):
    async def call() -> str:
        raise ours(TypeError("bug"))

    with pytest.raises(TypeError, match="bug"):
        await with_retries(call)
    assert no_sleep == []


async def test_approver_survives_an_upstream_error(inventory: Inventory):
    fake = FakeToolModel(UPSTREAM_502, submit("approve", "ok"))
    tools = ReviewTools(inventory, SeenInvoices(), INVOICE)

    out = await LLMApprover(fake.as_llm()).draft(INVOICE, report(), [], tools)

    assert out.draft.decision == "approve"
    assert len(fake.requests) == 2


async def test_critic_survives_an_upstream_error():
    fake = FakeChatModel(UPSTREAM_502, parsed(Critique(agrees=True)))
    draft = ApprovalDraft(decision="approve", rationale="ok")

    out = await LLMCritic(fake.as_llm()).review(INVOICE, report(), draft, [])

    assert out.agrees
