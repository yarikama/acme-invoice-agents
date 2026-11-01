"""The eval harness, run offline so it needs no network or credit."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from invoice_agents import evaluation
from invoice_agents.agents import NoExtractor, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.evaluation import CASES, Case, cost, outcome_of, summary
from invoice_agents.llm import Agents
from tests.conftest import INVOICES


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        samples_dir=INVOICES,
    )
    monkeypatch.setattr(evaluation, "get_settings", lambda: settings)
    return settings


def test_every_sample_invoice_has_a_case() -> None:
    covered = {name for case in CASES for name in case.files}
    assert covered == {p.name for p in INVOICES.iterdir() if p.is_file()} - {
        "generate_pdfs.py"
    }


def test_offline_run_scores_structured_files_and_skips_text(
    offline: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert evaluation.main(["--out", str(tmp_path)]) == 0

    [report] = tmp_path.glob("offline-*.json")
    data = json.loads(report.read_text())
    by_case = {r["case"]: r for r in data["results"]}
    assert by_case["invoice_1001.txt"]["outcome"] == "skipped"
    assert by_case["invoice_1004_revised.json +"]["outcome"] == "human"
    assert by_case["invoice_1016.json"]["findings"] == ["UNKNOWN_ITEM"]
    assert data["summary"]["accuracy"] == 1.0
    assert data["summary"]["total_cost_usd"] is None
    assert "skipped (need an LLM)" in capsys.readouterr().out


def test_a_wrong_outcome_fails_the_run(
    offline: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        evaluation, "CASES", [Case(("invoice_1016.json",), "pay", "deliberately wrong")]
    )

    assert evaluation.main(["--out", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "MISS invoice_1016.json" in out
    assert "0 paid that should not have been" in out


def test_an_exception_is_recorded_as_an_error(
    offline: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        async def draft(self, *args: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        evaluation,
        "build_agents",
        lambda _s: Agents(NoExtractor(), Broken(), RuleCritic()),  # pyright: ignore[reportArgumentType]
    )
    case = Case(("invoice_1015.csv",), "pay", "")

    [result] = evaluation.asyncio.run(evaluation.run_all([case], offline, {}, 1))

    assert result.outcome == "error"
    # Where it was raised, so a miss in a live run can be traced afterwards.
    assert result.detail.startswith("RuntimeError: boom (raised at test_evaluation.py:")
    assert result.correct is False


def test_outcome_of_an_unfinished_run() -> None:
    assert outcome_of({}) == ("error", "The run ended without an outcome.")


def test_cost_uses_per_token_prices() -> None:
    usage = {"grok-4.3": {"input_tokens": 1000, "output_tokens": 100}}
    pricing = {"x-ai/grok-4.3": (0.00000125, 0.0000025)}

    assert cost(usage, pricing) == pytest.approx(0.00125 + 0.00025)
    assert cost(usage, {}) is None


def test_summary_of_nothing_scored() -> None:
    assert summary([])["accuracy"] is None


def test_openrouter_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "data": [
            {
                "id": "x-ai/grok-4.7",
                "pricing": {"prompt": "0.000002", "completion": "0.000006"},
            },
            {"id": "no-price"},
        ]
    }
    monkeypatch.setattr(
        evaluation.urllib.request,
        "urlopen",
        lambda *_a, **_k: io.BytesIO(json.dumps(body).encode()),
    )

    assert evaluation.openrouter_pricing() == {"x-ai/grok-4.7": (0.000002, 0.000006)}


def test_every_attack_has_a_case() -> None:
    attacks = INVOICES.parent / "adversarial"
    covered = {name for case in evaluation.ADVERSARIAL for name in case.files}
    assert covered == {p.name for p in attacks.iterdir() if p.is_file()}
    assert all(c.folder == "adversarial" for c in evaluation.ADVERSARIAL)


def test_repeat_reports_how_consistent_each_case_was(
    offline: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        evaluation.main(
            [
                "--set",
                "adversarial",
                "--only",
                "json",
                "--repeat",
                "2",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    [report] = tmp_path.glob("offline-adversarial-*.json")
    data = json.loads(report.read_text())
    assert data["set"] == "adversarial"
    assert data["summary"]["repeat"] == 2
    assert data["summary"]["consistent_cases"] == data["summary"]["cases"] == 2
    assert sorted({r["attempt"] for r in data["results"]}) == [1, 2]
    out = " ".join(capsys.readouterr().out.split())  # rich wraps long lines
    assert "2/2" in out and "gave the same outcome all 2 times" in out


def test_consistency_names_flaky_cases() -> None:
    def result(outcome: evaluation.Outcome, attempt: int) -> evaluation.Result:
        return evaluation.Result(
            "a.json", "pay", outcome, outcome == "pay", 1.0, attempt=attempt
        )

    assert evaluation.consistency([result("pay", 1)]) == {}
    assert (
        evaluation.summary([result("pay", 1), result("human", 2)])["wrongly_paid"] == 0
    )
    wrong = evaluation.Result("b.json", "reject", "pay", False, 1.0)
    assert evaluation.summary([wrong])["wrongly_paid"] == 1
    flaky = evaluation.consistency([result("pay", 1), result("human", 2)])
    assert flaky == {
        "repeat": 2,
        "consistent_cases": 0,
        "cases": 1,
        "flaky": ["a.json"],
    }


def test_sets() -> None:
    assert evaluation.cases_in("samples") == evaluation.CASES
    assert evaluation.cases_in("scanned") == evaluation.SCANNED
    assert evaluation.cases_in("all") == (
        evaluation.CASES + evaluation.ADVERSARIAL + evaluation.SCANNED
    )
