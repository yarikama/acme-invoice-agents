"""Settings, read from the environment or a .env file."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "offline" runs rule-based agents with no network, for tests and demos
    # without an API key. Text and PDF invoices need an LLM to extract.
    llm_provider: Literal["xai", "openrouter", "offline"] = "offline"
    xai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    # Who may approve or reject paused invoices: name -> bearer token, as JSON,
    # e.g. VP_TOKENS='{"Dana Whitfield": "..."}'. Unset, anyone who can reach
    # the API can decide, which is only fit for a local demo.
    # Parsed below rather than by pydantic-settings, so a blank VP_TOKENS=
    # means none instead of failing to start.
    vp_tokens: Annotated[dict[str, SecretStr], NoDecode] = {}
    # The CLI's own token, sent when it decides through the API.
    vp_token: SecretStr | None = None
    extract_model: str = "grok-4.3"
    review_model: str = "grok-4.7"
    # Cap output, reasoning included: without it the provider reserves the
    # model's maximum (64K) and small credit limits reject the request.
    max_output_tokens: int = 8000
    # Retries with backoff on 429s and 5xx; new OpenRouter accounts get 20 RPM.
    llm_max_retries: int = 8
    extract_reasoning: Literal["low", "medium", "high"] = "low"
    review_reasoning: Literal["low", "medium", "high"] = "medium"

    db_path: Path = Path("inventory.db")
    # API: graph checkpoints, so a run paused for the VP survives a restart.
    checkpoint_path: Path = Path("checkpoints.db")
    samples_dir: Path = Path("data/invoices")
    uploads_dir: Path = Path("uploads")
    max_upload_bytes: int = 5 * 1024 * 1024
    cors_origins: list[str] = ["http://localhost:5173"]
    approval_threshold: float = 10_000
    # Totals within this fraction below the threshold get a warning.
    near_threshold_ratio: float = 0.95
    # Finding codes that always send an invoice to a VP, whatever the agents
    # decide. An empty list lets the approver clear these warnings itself.
    vp_required_findings: list[str] = [
        "NEAR_THRESHOLD",
        "FOREIGN_CURRENCY",
        "SUPERSEDES",
        "SPLIT_BELOW_THRESHOLD",
        "PRICE_ABOVE_LIST",
        "INSTRUCTIONS_IN_DOCUMENT",
        "BANK_DETAILS_CHANGED",
    ]
    max_extract_attempts: int = 3
    max_review_rounds: int = 2
    # Model turns the approver gets for tool calls before it must decide.
    max_tool_steps: int = 6
    # A decision model on OpenRouter's Decisions API (e.g. typesafe/jev-1.14)
    # that reads each invoice before the reasoning approver. Unset: no triage.
    triage_model: str | None = None
    # Triage settles an invoice on its own only when it is at least this sure,
    # and (to approve) sees at most this much chance that something is off.
    triage_min_confidence: float = 0.9
    triage_max_suspicion: float = 0.2

    @field_validator("vp_tokens", mode="before")
    @classmethod
    def _tokens_from_json(cls, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value) if value.strip() else {}
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
