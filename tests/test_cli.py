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


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> list[dict[str, Any]]:
    assert cli.main([*argv, "--json"]) == 0
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
    [result] = run(capsys, f"--invoice_path={INVOICES / 'invoice_1014.xml'}")

    expected = ["  Approve and pay? [y/N] "]
    if answer:
        expected.append("  Note for the audit log (optional): ")
    assert prompts == expected
    assert result["status"] == status


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
    [result] = run(capsys, f"--invoice_path={INVOICES / 'invoice_1015.csv'}")

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
