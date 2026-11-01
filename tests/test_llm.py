"""Picking the chat model and agents for each provider. No network calls."""

from typing import Any

import pytest
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

from invoice_agents.agents import (
    LLMApprover,
    LLMCritic,
    LLMExtractor,
    NoExtractor,
    RuleApprover,
    RuleCritic,
)
from invoice_agents.config import Settings
from invoice_agents.llm import OPENROUTER_BASE_URL, build_agents, chat_model


def settings(**values: Any) -> Settings:
    # Ignore the developer's .env so a real key can't leak into a test.
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


def test_offline_uses_rule_agents() -> None:
    agents = build_agents(settings(llm_provider="offline"))

    assert isinstance(agents.extractor, NoExtractor)
    assert isinstance(agents.approver, RuleApprover)
    assert isinstance(agents.critic, RuleCritic)


def test_xai_model() -> None:
    model = chat_model(settings(llm_provider="xai", xai_api_key="k"), "grok-4.3", "low")

    assert isinstance(model, ChatXAI)
    assert model.model_name == "grok-4.3"
    assert model.max_tokens == 8000
    assert model.reasoning_effort == "low"


def test_openrouter_model() -> None:
    s = settings(llm_provider="openrouter", openrouter_api_key="k")
    model = chat_model(s, "grok-4.7", "medium")

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "x-ai/grok-4.7"
    assert model.openai_api_base == OPENROUTER_BASE_URL
    assert model.max_tokens == 8000
    assert model.extra_body == {"reasoning": {"effort": "medium"}}


@pytest.mark.parametrize("provider", ["xai", "openrouter"])
def test_missing_key_is_a_clear_error(provider: str) -> None:
    with pytest.raises(ValueError, match="needs"):
        chat_model(settings(llm_provider=provider), "grok-4.3", "low")


def test_unknown_provider() -> None:
    bogus = Settings.model_construct(llm_provider="bogus")

    with pytest.raises(ValueError, match="bogus"):
        chat_model(bogus, "grok-4.3", "low")


def test_llm_provider_builds_llm_agents() -> None:
    agents = build_agents(settings(llm_provider="openrouter", openrouter_api_key="k"))

    assert isinstance(agents.extractor, LLMExtractor)
    assert isinstance(agents.approver, LLMApprover)
    assert isinstance(agents.critic, LLMCritic)
    assert agents.approver.llm is agents.critic.llm
