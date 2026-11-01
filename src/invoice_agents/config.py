"""Settings, read from the environment or a .env file."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# The repository: defaults resolve against it, not the working directory, so
# `python /path/to/main.py` from anywhere finds the data, the recording, the
# database and .env. Outside a checkout (an installed package), the working
# directory.
_HERE = Path(__file__).resolve().parents[2]
ROOT = _HERE if (_HERE / "pyproject.toml").is_file() else Path.cwd()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # replay (the default): answer from a recording of a live run, with no
    # key or network, so a fresh checkout runs every sample as recorded.
    # "offline" runs rule-based agents with no network; text and PDF
    # invoices then need an LLM to extract and are skipped.
    llm_provider: Literal["xai", "openrouter", "offline", "replay"] = "replay"
    replay_path: Path = ROOT / "data/replay/recording.json"
    # Set on a live provider to record the agents' answers for replay.
    record_to: Path | None = None
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
    # The assistant's memory: one conversation, in Redis so the terminal and
    # the web console share it. Without Redis it lasts for one process.
    redis_url: str = "redis://localhost:6379/0"
    chat_session: str = "console"
    extract_model: str = "grok-4.3"
    review_model: str = "grok-4.7"
    # Cap output, reasoning included: without it the provider reserves the
    # model's maximum (64K) and small credit limits reject the request.
    max_output_tokens: int = 8000
    # Retries with backoff on 429s and 5xx; new OpenRouter accounts get 20 RPM.
    llm_max_retries: int = 8
    extract_reasoning: Literal["low", "medium", "high"] = "low"
    review_reasoning: Literal["low", "medium", "high"] = "medium"
    # The critic checks the approver's draft. Unset, it uses the review model;
    # a model from another vendor gives a second opinion with different blind
    # spots. On OpenRouter, a name with a "/" is used as given (e.g.
    # anthropic/claude-haiku-5.5); a bare name is a Grok model.
    critic_model: str | None = None
    critic_reasoning: Literal["low", "medium", "high"] | None = None

    db_path: Path = ROOT / "inventory.db"
    # One JSON line per settled invoice: outcome, reason, amount. The same
    # records are in the ledger table of the database above.
    decisions_log: Path | None = ROOT / "logs" / "decisions.jsonl"
    # The API's structured log: every step as a JSON line (see logs.py).
    log_json: Path | None = None
    # API: graph checkpoints, so a run paused for the VP survives a restart.
    checkpoint_path: Path = ROOT / "checkpoints.db"
    samples_dir: Path = ROOT / "data/invoices"
    # The folder --watch and /watch treat as the AP mailbox.
    inbox_dir: Path = ROOT / "data/inbox"
    uploads_dir: Path = ROOT / "uploads"
    # The console, built (npm run build): the API serves it when it exists,
    # so one server is enough, without Node (the Docker image does this).
    web_dist: Path = ROOT / "web/dist"
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
        "LOOKALIKE_VENDOR",
    ]
    # Reads scanned PDFs and photos (needs `uv sync --extra ocr`).
    ocr_backend: Literal["docling"] = "docling"
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

    @field_validator("triage_model", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """`TRIAGE_MODEL=` in a .env means no triage, not a model named ""."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("vp_tokens", mode="before")
    @classmethod
    def _tokens_from_json(cls, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value) if value.strip() else {}
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
