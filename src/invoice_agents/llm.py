"""Build the agents for the configured provider."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from invoice_agents.agents import (
    Approver,
    Critic,
    Extractor,
    LLMApprover,
    LLMCritic,
    LLMExtractor,
    NoExtractor,
    RuleApprover,
    RuleCritic,
)
from invoice_agents.config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Agents:
    extractor: Extractor
    approver: Approver
    critic: Critic


def chat_model(settings: Settings, model: str) -> BaseChatModel:
    if settings.llm_provider == "xai":
        from langchain_xai import ChatXAI

        if settings.xai_api_key is None:
            raise ValueError("LLM_PROVIDER=xai needs XAI_API_KEY")
        return ChatXAI(model=model, api_key=settings.xai_api_key)

    if settings.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if settings.openrouter_api_key is None:
            raise ValueError("LLM_PROVIDER=openrouter needs OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=f"x-ai/{model}",
            base_url=OPENROUTER_BASE_URL,
            api_key=settings.openrouter_api_key,
        )

    raise ValueError(f"No chat model for provider {settings.llm_provider!r}")


def build_agents(settings: Settings) -> Agents:
    if settings.llm_provider == "offline":
        return Agents(NoExtractor(), RuleApprover(), RuleCritic())
    extract_llm = chat_model(settings, settings.extract_model)
    review_llm = chat_model(settings, settings.review_model)
    return Agents(
        LLMExtractor(extract_llm), LLMApprover(review_llm), LLMCritic(review_llm)
    )
