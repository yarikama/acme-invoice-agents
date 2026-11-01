import json
import logging
import sys

from invoice_agents.logs import JsonFormatter


def test_a_record_becomes_one_json_line_with_its_fields() -> None:
    record = logging.LogRecord(
        "invoice_agents.graph", logging.INFO, "f", 1, "pay: %s", ("ok",), None
    )
    record.run_id = "r1"
    record.stage = "pay"
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "pay: ok" and data["level"] == "info"
    assert (data["run_id"], data["stage"]) == ("r1", "pay")
    assert "usage" not in data  # absent fields stay out
    try:
        raise ValueError("boom")
    except ValueError:
        record.exc_info = sys.exc_info()
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["error"]
