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
from invoice_agents.replay import (
    Recording,
    RecordingApprover,
    RecordingCritic,
    RecordingExtractor,
    ReplayApprover,
    ReplayCritic,
    ReplayExtractor,
)
from invoice_agents.triage import DecisionTriager, Triager
from invoice_agents.usage import Pricing, openrouter_pricing

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
            # A bare name is a Grok model; "vendor/model" is used as given.
            model=model if "/" in model else f"x-ai/{model}",
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
    if settings.llm_provider == "replay":
        # Triage isn't recorded: every invoice goes through the agents.
        recording = Recording.shared(settings.replay_path)
        pace = settings.replay_pace
        return Agents(
            ReplayExtractor(recording, pace),
            ReplayApprover(recording, pace),
            ReplayCritic(recording, pace),
        )
    extract_llm = chat_model(
        settings, settings.extract_model, settings.extract_reasoning
    )
    review_llm = chat_model(settings, settings.review_model, settings.review_reasoning)
    critic_llm = chat_model(
        settings,
        settings.critic_model or settings.review_model,
        settings.critic_reasoning or settings.review_reasoning,
    )
    agents = Agents(
        LLMExtractor(extract_llm),
        LLMApprover(review_llm, max_steps=settings.max_tool_steps),
        LLMCritic(critic_llm),
        build_triager(settings),
    )
    if settings.record_to is None:
        return agents
    recording = Recording.shared(settings.record_to)
    return Agents(
        RecordingExtractor(agents.extractor, recording),
        RecordingApprover(agents.approver, recording),
        RecordingCritic(agents.critic, recording),
        agents.triager,
    )


def load_pricing(settings: Settings) -> Pricing:
    """OpenRouter's model prices, for each step's cost. Without them the
    console still shows tokens, so a failed fetch is not fatal."""
    if settings.llm_provider == "replay":
        # The prices when the recording was made.
        return Recording.shared(settings.replay_path).pricing
    if settings.llm_provider != "openrouter":
        return {}
    try:
        pricing = openrouter_pricing()
    except (OSError, ValueError):
        return {}
    if settings.record_to is not None:
        # Recording through the API keeps the prices with it, as the eval does.
        Recording.shared(settings.record_to).pricing = pricing
    return pricing
