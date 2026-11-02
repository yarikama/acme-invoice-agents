"""Interfaces under hostile or careless input: the assistant's paths, writes
from another site, control characters in invoices, Ctrl-C mid-run."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from invoice_agents import cli, remote
from invoice_agents.assistant import Plan, Step, checked, within_reach
from invoice_agents.models import Invoice, LineItem


def invoice(**kw: Any) -> Invoice:
    fields: dict[str, Any] = {
        "invoice_number": "INV-1",
        "vendor": "v",
        "invoice_date": None,
        "due_date": None,
        "total": None,
        "line_items": [],
    }
    return Invoice(**(fields | kw))


def outside(args: list[str], request: str) -> list[str]:
    return within_reach(Step(command="run", args=args), request)


def test_the_assistant_touches_only_whole_paths_the_person_typed() -> None:
    sample = "data/invoices/invoice_1001.txt"
    assert outside([sample], "run the first sample") == []  # a sample folder
    assert outside(["/tmp/x.pdf"], "please run /tmp/x.pdf now") == []  # typed
    # Fragments of what was typed are not paths the person asked for.
    assert outside([""], "what is waiting for a VP?") == ['""']
    assert outside(["."], "Run it. Then stop.") == ["."]
    assert outside([".."], "wait...") == [".."]
    assert outside(["/tmp"], "please run /tmp/x.pdf now") == ["/tmp"]


def test_the_assistant_s_eval_takes_a_set_name_only() -> None:
    plan = Plan(
        reply="Scoring.",
        steps=[
            Step(command="eval", args=["samples", "--repeat", "1000", "--out", "/"])
        ],
    )
    [step], _ = checked(plan)
    assert step.args == ["samples"]


def test_control_characters_never_leave_an_invoice() -> None:
    evil = "Acme\x1b]0;X\x1b\\\x1b[2J Supply"
    inv = invoice(
        invoice_number="INV-1\x1b[1A",
        vendor=evil,
        line_items=[LineItem(item="Widget\x1b[2KA", quantity=1, unit_price=1)],
    )
    assert inv.vendor == "Acme]0;X\\[2J Supply"
    assert "\x1b" not in (inv.invoice_number or "") + inv.line_items[0].item
    # Line breaks and tabs in notes are text, not control.
    assert invoice(notes="a\nb\tc").notes == "a\nb\tc"


async def test_ctrl_c_mid_run_frees_the_invoice_number(tmp_path: Path) -> None:
    from invoice_agents.config import Settings

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        db_path=tmp_path / "inventory.db",
        decisions_log=tmp_path / "decisions.jsonl",
    )
    runner = cli.LocalRunner(settings)

    async def stopped(run_id: str, *_args: Any) -> dict[str, Any]:
        runner.ledger.add(invoice(), run_id=run_id)
        raise asyncio.CancelledError

    runner._process = stopped  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await runner.process(Path("x.txt"), "ask")
    assert [e.outcome for e in runner.ledger.entries] == ["dropped"]


async def test_ctrl_c_while_the_servers_start_stops_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stopped: list[list[Any]] = []

    async def start(_api: str, _web: str, _log: Path, started: list[Any]) -> list[Any]:
        started.append("api-process")
        raise KeyboardInterrupt

    monkeypatch.setattr(remote, "_start", start)
    monkeypatch.setattr(remote, "stop", lambda procs: stopped.append(list(procs)))
    with pytest.raises(KeyboardInterrupt):
        await remote.serve("http://localhost:1", "http://localhost:2", tmp_path)
    assert stopped == [["api-process"]]


def test_a_bad_eval_flag_does_not_close_the_console(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from invoice_agents import evaluation, shell

    def bad(_argv: list[str]) -> None:
        raise SystemExit("argument --repeat: invalid int value: 'x'")

    monkeypatch.setattr(evaluation, "main", bad)
    shell.Shell.eval(object.__new__(shell.Shell), ["samples", "--repeat", "x"])
    assert "The eval stopped" in capsys.readouterr().err


def test_last_event_id_takes_ascii_digits_only() -> None:
    from invoice_agents.api.app import _after

    assert _after("12") == 12
    assert _after("²") == -1
    assert _after("9" * 5000) == -1
    assert _after(None) == -1


async def test_a_vp_deciding_first_in_the_console_is_not_a_failed_run(
    tmp_path: Path,
) -> None:
    import httpx

    summary = {"invoice": {"total": 100.0, "currency": "USD"}, "findings": []}

    class WebDecidedFirst:
        async def start(self, path: Path, parser: str | None = None) -> str:
            return "r1"

        async def events(self, run_id: str):
            yield {"type": "awaiting_vp", "summary": summary}
            yield {"type": "done", "status": "paid"}

        async def decide(self, run_id: str, approved: bool, note: str) -> None:
            request = httpx.Request("POST", "http://api/decision")
            raise httpx.HTTPStatusError(
                "409", request=request, response=httpx.Response(409, request=request)
            )

        async def result(self, run_id: str) -> dict[str, Any]:
            return {"status": "paid"}

    runner = cli.RemoteRunner(WebDecidedFirst())  # pyright: ignore[reportArgumentType]
    result = await runner.process(tmp_path / "x.json", "approve")
    assert result["status"] == "paid"
