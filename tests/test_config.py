"""The provider is live when a key allows it, and says so when it isn't."""

import pytest

from invoice_agents.config import Settings


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("keys", "provider", "no_key"),
    [
        ({"xai_api_key": "x", "openrouter_api_key": "o"}, "xai", False),  # xAI first
        ({"openrouter_api_key": "o"}, "openrouter", False),
        ({}, "replay", True),
        ({"xai_api_key": " ", "openrouter_api_key": ""}, "replay", True),  # blank
    ],
)
def test_auto_runs_live_when_a_key_is_set(
    monkeypatch: pytest.MonkeyPatch, keys: dict, provider: str, no_key: bool
) -> None:
    for name in ("LLM_PROVIDER", "XAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    got = settings(**keys)
    assert (got.llm_provider, got.replay_for_lack_of_key) == (provider, no_key)


def test_an_explicit_provider_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    got = settings(llm_provider="replay", xai_api_key="x")
    assert (got.llm_provider, got.replay_for_lack_of_key) == ("replay", False)
    assert settings(llm_provider="offline").llm_provider == "offline"
