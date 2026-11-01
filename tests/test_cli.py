"""The command line, run in-process with offline agents."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from invoice_agents import cli
from invoice_agents.agents import NoExtractor, Review, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.ledger import Ledger
from invoice_agents.llm import Agents
from invoice_agents.models import Invoice, ValidationReport
from invoice_agents.tools import ReviewTools
from tests.conftest import INVOICES


@pytest.fixture(autouse=True)
def no_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach a real API or console from a test: always run locally."""

    async def down(*_args: object) -> bool:
        return False

    monkeypatch.setattr(cli.remote, "reachable", down)

    monkeypatch.setattr(cli.remote, "api_up", down)


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        db_path=tmp_path / "inventory.db",
        decisions_log=tmp_path / "decisions.jsonl",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


def run(
    capsys: pytest.CaptureFixture[str], *argv: str, code: int = 0
) -> list[dict[str, Any]]:
    assert cli.main([*argv, "--json"]) == code
    return json.loads(capsys.readouterr().out)


def test_clean_invoice_is_paid(capsys: pytest.CaptureFixture[str]) -> None:
    [result] = run(capsys, f"--invoice_path={INVOICES / 'invoice_1015.csv'}")

    assert result["file"] == "invoice_1015.csv"
    assert result["status"] == "paid"
    assert result["payment"]["status"] == "success"


def test_dash_spelling_and_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    [result] = run(capsys, "-v", "--invoice-path", str(INVOICES / "invoice_1016.json"))

    assert result["status"] == "rejected"
    assert result["reason"] == "Hard failures: Item not in inventory."
    assert [f["code"] for f in result["findings"]] == ["UNKNOWN_ITEM"]


def test_folder_skips_hidden_files_and_text_without_llm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    for name in ("invoice_1015.csv", "invoice_1016.json", "invoice_1001.txt"):
        shutil.copy(INVOICES / name, folder / name)
    (folder / ".DS_Store").write_text("")
    (folder / "archive").mkdir()

    results = run(capsys, f"--dir={folder}")

    assert {r["file"]: r["status"] for r in results} == {
        "invoice_1001.txt": "skipped",
        "invoice_1015.csv": "paid",
        "invoice_1016.json": "rejected",
    }
    skipped = next(r for r in results if r["status"] == "skipped")
    assert "needs an LLM" in skipped["reason"]


@pytest.mark.parametrize(
    ("vp", "status"), [("approve", "paid"), ("reject", "rejected")]
)
def test_vp_flag_answers_the_pause(
    vp: str, status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    [result] = run(
        capsys, f"--invoice_path={INVOICES / 'invoice_1014.xml'}", f"--vp={vp}"
    )

    assert result["status"] == status
    assert result["draft"]["decision"] == "escalate"


@pytest.mark.parametrize(("answer", "status"), [("y", "paid"), ("", "rejected")])
def test_vp_prompt_asks_on_the_terminal(
    answer: str,
    status: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return answer

    monkeypatch.setattr(cli.console, "input", fake_input)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    [result] = run(capsys, f"--invoice_path={INVOICES / 'invoice_1014.xml'}")

    expected = ["  Approve and pay? [y/N] "]
    if answer:
        expected.append("  Note for the audit log (optional): ")
    assert prompts == expected
    assert result["status"] == status


def test_with_no_one_to_ask_the_invoice_waits_for_a_vp(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    offline: Settings,
) -> None:
    # Before: a script or CI run crashed with EOFError at the VP prompt.
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    sample = f"--invoice_path={INVOICES / 'invoice_1014.xml'}"
    [result] = run(capsys, sample)
    # Nothing can resume it once this process ends, so it says so instead of
    # claiming it waits, and the ledger lets the invoice be sent again.
    assert result["status"] == "skipped"
    assert "can't wait for a VP" in result["reason"]
    [entry] = Ledger(offline.db_path).entries
    assert entry.outcome == "dropped"
    [again] = run(capsys, sample)
    assert "DUPLICATE" not in {f["code"] for f in again["findings"]}


def test_closing_the_input_at_the_prompt_leaves_it_waiting(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def closed(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(cli.console, "input", closed)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    [result] = run(capsys, f"--invoice_path={INVOICES / 'invoice_1014.xml'}")
    assert result["status"] == "skipped"


def test_one_failure_does_not_stop_the_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenApprover:
        async def draft(
            self,
            invoice: Invoice,
            report: ValidationReport,
            critique: list[str],
            tools: ReviewTools,
        ) -> Review:
            raise RuntimeError("provider returned 402")

    monkeypatch.setattr(
        cli,
        "build_agents",
        lambda _settings: Agents(NoExtractor(), BrokenApprover(), RuleCritic()),
    )
    # A run that broke exits non-zero, so a script notices.
    [result] = run(capsys, f"--invoice_path={INVOICES / 'invoice_1015.csv'}", code=1)

    assert result["status"] == "failed"
    assert result["reason"] == "RuntimeError: provider returned 402"


def test_a_tool_call_is_shown_once_as_it_happens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.test_graph import (
        INV_1001,
        RecordingCritic,
        StubExtractor,
        ToolUsingApprover,
    )

    agents = Agents(StubExtractor(INV_1001), ToolUsingApprover(), RecordingCritic())
    monkeypatch.setattr(cli, "build_agents", lambda _settings: agents)
    path = INVOICES / "invoice_1001.txt"
    assert cli.main([f"--invoice_path={path}", "--json"]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)[0]["status"] == "paid"
    assert err.count("check_inventory(WidgetA)") == 1


def test_reset_forgets_what_was_received(
    capsys: pytest.CaptureFixture[str], offline: Settings
) -> None:
    sample = f"--invoice_path={INVOICES / 'invoice_1015.csv'}"
    assert [r["status"] for r in run(capsys, sample)] == ["paid"]
    assert [r["status"] for r in run(capsys, sample)] == ["rejected"]  # duplicate
    assert cli.main(["--reset"]) == 0
    assert Ledger(offline.db_path).entries == []
    assert [r["status"] for r in run(capsys, sample, "--reset")] == ["paid"]


def test_a_file_type_it_does_not_read_is_skipped_not_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    odd = tmp_path / "invoice.docx"
    odd.write_bytes(b"PK")
    [result] = run(capsys, f"--invoice_path={odd}")
    assert result["status"] == "skipped"
    assert "Unsupported file type: .docx" in result["reason"]


def test_log_json_writes_each_step_with_its_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import logging

    out = tmp_path / "run.jsonl"
    app_log = logging.getLogger("invoice_agents")
    before = list(app_log.handlers)
    try:
        [result] = run(
            capsys,
            f"--invoice_path={INVOICES / 'invoice_1015.csv'}",
            f"--log-json={out}",
        )
    finally:
        for handler in app_log.handlers[len(before) :]:
            handler.close()
            app_log.removeHandler(handler)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    steps = [line for line in lines if "stage" in line]
    assert [s["stage"] for s in steps][:2] == ["ingest", "validate"]
    assert steps[-1]["stage"] == "pay" and result["status"] == "paid"
    assert len({s["run_id"] for s in steps}) == 1 and steps[0]["run_id"]
    assert {"at", "level", "logger", "message"} <= set(steps[0])
