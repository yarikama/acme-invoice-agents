"""The assistant's memory: one conversation, shared by the terminal and the
web console.

Both read and write the same Redis list through LangChain's
`RedisChatMessageHistory`, so a request typed in one is context for the next
request in the other, and the web console shows what was said in the
terminal. Without Redis, the conversation is kept in this process only and
the two don't see each other.

Each message carries who sent it and from where, in `additional_kwargs`:
`via` ("terminal" or "web"), `at` (ISO time), and for the assistant's
messages the commands it ran.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from invoice_agents.config import Settings

log = logging.getLogger(__name__)

KEY_PREFIX = "acme:chat:"
# Messages /history and the web console show.
SHOWN = 50


class LocalHistory(BaseChatMessageHistory):
    """The conversation in this process, for when there is no Redis.
    (langchain-core's InMemoryChatMessageHistory is deprecated.)"""

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return list(self._messages)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self._messages.extend(messages)

    def clear(self) -> None:
        self._messages.clear()


# Conversations kept in this process when Redis isn't there, by session.
_local: dict[str, LocalHistory] = {}


def redis_up(url: str) -> bool:
    try:
        return bool(redis.Redis.from_url(url, socket_connect_timeout=0.5).ping())
    except (redis.RedisError, OSError, ValueError):
        return False


def history(settings: Settings) -> tuple[BaseChatMessageHistory, bool]:
    """The conversation for this console, and whether it is shared (Redis)."""
    if settings.redis_url and redis_up(settings.redis_url):
        with warnings.catch_warnings():
            # langchain-community announces its sunset on import; it still
            # holds the plain-Redis history (langchain-redis needs Redis
            # Stack's JSON and search modules).
            warnings.simplefilter("ignore", DeprecationWarning)
            from langchain_community.chat_message_histories import (
                RedisChatMessageHistory,
            )
        return (
            RedisChatMessageHistory(
                session_id=settings.chat_session,
                url=settings.redis_url,
                key_prefix=KEY_PREFIX,
            ),
            True,
        )
    log.info("No Redis at %s: chat memory stays in this process.", settings.redis_url)
    return _local.setdefault(settings.chat_session, LocalHistory()), False


def said(text: str, via: str) -> HumanMessage:
    return HumanMessage(text, additional_kwargs={"via": via, "at": _now()})


def answered(text: str, via: str, commands: list[str] | None = None) -> AIMessage:
    extra: dict[str, Any] = {"via": via, "at": _now()}
    if commands:
        extra["commands"] = commands
    return AIMessage(text, additional_kwargs=extra)


def outcomes(results: list[dict[str, Any]]) -> str:
    """What came of the invoices a request ran, a sentence each, with the
    flags and the first line of the reasoning, so a later "why?" has an
    answer in the conversation."""

    def first(text: str | None) -> str:
        return (text or "").split(". ")[0].strip().rstrip(".")

    lines = []
    for r in results:
        inv = r.get("invoice") or {}
        name = inv.get("invoice_number") or r.get("file") or "An invoice"
        flags = [
            f["code"] for f in r.get("findings") or [] if f.get("severity") != "info"
        ]
        tag = f" ({', '.join(flags)})" if flags else ""
        status = r.get("status")
        if status == "paid":
            why = first((r.get("draft") or {}).get("rationale"))
            lines.append(f"{name} was paid{tag}" + (f": {why}." if why else "."))
        elif status == "awaiting_vp":
            why = r.get("route_reason") or "a VP has to decide"
            lines.append(f"{name} is waiting for a VP{tag}: {why}.")
        elif status == "rejected":
            why = first(r.get("reason"))
            lines.append(f"{name} was rejected{tag}" + (f": {why}." if why else "."))
        elif status:
            lines.append(f"{name}: {status}.")
    return " ".join(lines)


def as_json(index: int, message: BaseMessage) -> dict[str, Any]:
    """One message for the web console."""
    extra = message.additional_kwargs
    return {
        "index": index,
        "role": "user" if message.type == "human" else "assistant",
        "text": message.text,
        "via": extra.get("via"),
        "at": extra.get("at"),
        "commands": extra.get("commands", []),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
