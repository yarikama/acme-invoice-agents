"""Settings, read from the environment or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "offline" runs rule-based agents with no network, for tests and demos
    # without an API key. Text and PDF invoices need an LLM to extract.
    llm_provider: Literal["xai", "openrouter", "offline"] = "offline"
    xai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    extract_model: str = "grok-4.3"
    review_model: str = "grok-4.7"

    db_path: Path = Path("inventory.db")
    approval_threshold: float = 10_000
    # Totals within this fraction below the threshold get a warning.
    near_threshold_ratio: float = 0.95
    max_extract_attempts: int = 3
    max_review_rounds: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
