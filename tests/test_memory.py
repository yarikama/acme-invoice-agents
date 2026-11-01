"""The assistant's conversation, shared through Redis."""

import os

import pytest

from invoice_agents import memory
from invoice_agents.config import Settings

# A Redis database tests may wipe; CI runs one as a service.
TEST_REDIS = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


def settings(url: str, session: str = "test") -> Settings:
    return Settings(_env_file=None, redis_url=url, chat_session=session)  # pyright: ignore[reportCallIssue]


@pytest.mark.skipif(not memory.redis_up(TEST_REDIS), reason="no Redis to test with")
def test_the_terminal_and_the_api_share_one_conversation() -> None:
    terminal, shared = memory.history(settings(TEST_REDIS))
    terminal.clear()
    api, _ = memory.history(settings(TEST_REDIS))  # as another process would
    assert shared

    terminal.add_message(memory.said("run 1015", "terminal"))
    api.add_message(memory.answered("Started 1 run.", "web", ["/run invoice_1015.csv"]))

    for side in (terminal, api):
        first, second = side.messages
        assert (first.text, first.additional_kwargs["via"]) == ("run 1015", "terminal")
        assert second.additional_kwargs["commands"] == ["/run invoice_1015.csv"]
    assert memory.as_json(1, api.messages[1]) == {
        "index": 1,
        "role": "assistant",
        "text": "Started 1 run.",
        "via": "web",
        "at": api.messages[1].additional_kwargs["at"],
        "commands": ["/run invoice_1015.csv"],
    }
    # Another session is another conversation.
    other, _ = memory.history(settings(TEST_REDIS, "someone-else"))
    other.clear()
    assert other.messages == []
    terminal.clear()


def test_without_redis_it_lasts_for_this_process() -> None:
    first, shared = memory.history(settings(""))
    assert not shared
    first.add_message(memory.said("hi", "terminal"))
    again, _ = memory.history(settings("redis://localhost:1/0"))  # nothing there
    assert [m.text for m in again.messages] == ["hi"]
    again.clear()
    assert first.messages == []
    assert not memory.redis_up("not a url")


def test_outcomes_say_what_came_of_each_invoice() -> None:
    assert memory.outcomes(
        [
            {
                "status": "paid",
                "invoice": {"invoice_number": "INV-1"},
                "draft": {"rationale": "Every check passed. Under the threshold."},
                "findings": [{"code": "NEW_VENDOR", "severity": "info"}],
            },
            {
                "status": "awaiting_vp",
                "file": "b.xml",
                "route_reason": "billed in EUR",
                "findings": [{"code": "FOREIGN_CURRENCY", "severity": "warning"}],
            },
            {
                "status": "rejected",
                "invoice": {"invoice_number": "INV-3"},
                "reason": "Duplicate. More.",
            },
            {"status": "rejected", "file": "c.json"},
            {"status": "failed", "file": "d.pdf"},
        ]
    ) == (
        "INV-1 was paid: Every check passed."
        " b.xml is waiting for a VP (FOREIGN_CURRENCY): billed in EUR."
        " INV-3 was rejected: Duplicate. c.json was rejected. d.pdf: failed."
    )
