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
from invoice_agents.triage import DecisionTriager, Triager

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Agents:
    extractor: Extractor
    approver: Approver
    critic: Critic
    # System 1, asked before the approver; None skips triage.
    triager: Triager | None = None


def chat_model(settings: Settings, model: str, reasoning: str) -> BaseChatModel:
    if settings.llm_provider == "xai":
        from langchain_xai import ChatXAI

        if settings.xai_api_key is None:
            raise ValueError("LLM_PROVIDER=xai needs XAI_API_KEY")
        return ChatXAI(
            model=model,
            api_key=settings.xai_api_key,
            max_tokens=settings.max_output_tokens,
            max_retries=settings.llm_max_retries,
            reasoning_effort=reasoning,
        )

    if settings.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if settings.openrouter_api_key is None:
            raise ValueError("LLM_PROVIDER=openrouter needs OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=f"x-ai/{model}",
            base_url=OPENROUTER_BASE_URL,
            api_key=settings.openrouter_api_key,
            max_completion_tokens=settings.max_output_tokens,
            max_retries=settings.llm_max_retries,
            # OpenRouter's unified reasoning control.
            extra_body={"reasoning": {"effort": reasoning}},
        )

    raise ValueError(f"No chat model for provider {settings.llm_provider!r}")


def build_triager(settings: Settings) -> Triager | None:
    if settings.triage_model is None:
        return None
    if settings.openrouter_api_key is None:
        raise ValueError("TRIAGE_MODEL needs OPENROUTER_API_KEY")
    return DecisionTriager(settings.triage_model, settings.openrouter_api_key)


def build_agents(settings: Settings) -> Agents:
    if settings.llm_provider == "offline":
        return Agents(
            NoExtractor(), RuleApprover(), RuleCritic(), build_triager(settings)
        )
    extract_llm = chat_model(
        settings, settings.extract_model, settings.extract_reasoning
    )
    review_llm = chat_model(settings, settings.review_model, settings.review_reasoning)
    return Agents(
        LLMExtractor(extract_llm),
        LLMApprover(review_llm, max_steps=settings.max_tool_steps),
        LLMCritic(review_llm),
        build_triager(settings),
    )
