"""Structured logs: one JSON object per line, for tools rather than people.

Every trace step is logged (graph._event) with its run id and stage, so a
file written with --log-json (or LOG_JSON for the API) can be filtered by
run, stage or level with jq, or shipped to a log store as is.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Fields a log call may pass with `extra=`, copied into the JSON object.
FIELDS = ("run_id", "stage", "usage", "outcome")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "at": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        data |= {f: getattr(record, f) for f in FIELDS if getattr(record, f, None)}
        if record.exc_info:
            data["error"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def to_json_file(path: Path, level: int = logging.INFO) -> logging.Handler:
    """Write this app's logs to `path` as JSON lines, at `level` and above."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)
    logger = logging.getLogger("invoice_agents")
    logger.addHandler(handler)
    if logger.getEffectiveLevel() > level:
        logger.setLevel(level)
    return handler
