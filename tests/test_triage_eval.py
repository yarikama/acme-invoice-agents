"""The triage comparison and the eval's triage mode, offline with stub models."""

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from invoice_agents import evaluation, triage_eval
from invoice_agents.config import Settings
from invoice_agents.evaluation import Case
from invoice_agents.models import Invoice, ValidationReport
from invoice_agents.triage import TriageAnswer, TriageError
from invoice_agents.triage_eval import Prepared, Vote, acceptable, scoreboard, vote
from tests.conftest import INVOICES


class RuleTriager:
    """Answers the way the findings point, like a well-behaved decision model."""

    def __init__(self, model: str, api_key: SecretStr, **_: object) -> None:
        self.model = model

    async def ask(self, invoice: Invoice, report: ValidationReport) -> TriageAnswer:
        if self.model == "down":
            raise TriageError("down failed: HTTP 503.")
        decision = (
            "reject" if report.errors else "escalate" if report.warnings else "approve"
        )
        if self.model == "gullible":
            decision = "approve"
        return TriageAnswer(
            model=self.model,
            decision=decision,
            confidence=0.99,
            suspicion=0.01,
            seconds=0.2,
            input_tokens=400,
            cost_usd=0.00002,
        )


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        samples_dir=INVOICES,
        openrouter_api_key=SecretStr("test-key"),
    )
    monkeypatch.setattr(triage_eval, "get_settings", lambda: s)
    monkeypatch.setattr(evaluation, "get_settings", lambda: s)
    monkeypatch.setattr(triage_eval, "DecisionTriager", RuleTriager)
    monkeypatch.setattr("invoice_agents.llm.DecisionTriager", RuleTriager)
    return s


def test_acceptable_counts_alternatives():
    case = Case(("a",), "pay", "", also_ok=("human",))
    assert acceptable(case, "pay") and acceptable(case, "human")
    assert not acceptable(case, "reject")


async def test_prepare_stops_before_the_approver(settings: Settings):
    case = Case(("invoice_1004.json", "invoice_1004_revised.json"), "reject", "")
    prepared = await triage_eval.prepare(case, settings)

    # The original ran to the end and was paid, so the revision can't be.
    assert prepared is not None
    assert prepared.route == "reject"
    assert "ALREADY_PAID" in {f.code for f in prepared.report.findings}


async def test_prepare_skips_invoices_that_need_an_extractor(settings: Settings):
    assert (
        await triage_eval.prepare(Case(("invoice_1001.txt",), "pay", ""), settings)
        is None
    )


async def test_a_wrong_fast_path_is_unsafe(settings: Settings):
    case = Case(("invoice_1016.json",), "reject", "")
    prepared = await triage_eval.prepare(case, settings)
    assert prepared is not None

    good = await vote(prepared, RuleTriager("m", SecretStr("k")), settings)
    assert (good.agrees, good.fast_path, good.unsafe) == (True, "reject", False)

    # Approving past an error is never a fast path, so it cannot be unsafe.
    gullible = await vote(prepared, RuleTriager("gullible", SecretStr("k")), settings)
    assert (gullible.agrees, gullible.fast_path) == (False, None)

    clean = Prepared(
        Case(("x",), "human", ""), prepared.invoice, ValidationReport(), "auto"
    )
    wrong = await vote(clean, RuleTriager("m", SecretStr("k")), settings)
    assert (wrong.fast_path, wrong.unsafe) == ("approve", True)


async def test_a_failed_call_is_recorded(settings: Settings):
    prepared = await triage_eval.prepare(
        Case(("invoice_1006.csv",), "pay", ""), settings
    )
    assert prepared is not None

    got = await vote(prepared, RuleTriager("down", SecretStr("k")), settings)

    assert got.decision is None and got.error.startswith("down failed")


def test_scoreboard():
    votes = [
        Vote("a", "m", "pay", "approve", True, "approve", False, 0.99, 0.0, 0.2, 1e-5),
        Vote("b", "m", "reject", "approve", False, None, False, 0.5, 0.0, 0.4, 3e-5),
        Vote("c", "m", "pay", None, None, None, False, error="x"),
    ]

    [row] = scoreboard(votes, ["m"])

    assert row["answered"] == 2 and row["failed"] == 1
    assert row["agreement"] == 0.5
    assert row["fast_paths"] == 1 and row["unsafe_fast_paths"] == 0
    assert row["mean_seconds"] == 0.3
    assert row["cost_per_invoice_usd"] == pytest.approx(2e-5)


def test_main_compares_models(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = triage_eval.main(
        ["--models", "good,down,gullible", "--only", "json", "--out", str(tmp_path)]
    )

    assert code == 0
    [report] = tmp_path.glob("triage-*.json")
    rows = {r["model"]: r for r in json.loads(report.read_text())["models"]}
    assert rows["good"]["agreement"] == 1.0
    assert rows["down"]["failed"] == rows["good"]["answered"]
    assert rows["gullible"]["agreement"] < 1.0
    out = capsys.readouterr().out
    assert "failed" in out


def test_main_flags_unsafe_fast_paths(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        triage_eval,
        "CASES",
        [Case(("invoice_1006.csv",), "reject", "wrong on purpose")],
    )

    assert triage_eval.main(["--models", "good", "--out", str(tmp_path)]) == 1
    assert "UNSAFE good on invoice_1006.csv" in capsys.readouterr().out


def test_main_needs_a_key(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        triage_eval,
        "get_settings",
        lambda: settings.model_copy(update={"openrouter_api_key": None}),
    )
    assert triage_eval.main([]) == 2


def test_eval_with_triage_counts_fast_paths_and_their_cost(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        evaluation.main(["--triage", "good", "--only", "json", "--out", str(tmp_path)])
        == 0
    )

    [report] = tmp_path.glob("offline-triage-*.json")
    data = json.loads(report.read_text())
    by_case = {r["case"]: r for r in data["results"]}
    assert data["triage_model"] == "good"
    assert by_case["invoice_1016.json"]["fast_path"] == "reject"
    assert by_case["invoice_1016.json"]["cost_usd"] == pytest.approx(0.00002)
    # A revision of a paid invoice has an error, which code confirms.
    assert by_case["invoice_1004_revised.json +"]["fast_path"] == "reject"
    assert data["summary"]["fast_paths"] >= 3
    assert "settled by triage" in capsys.readouterr().out
