"""The interactive console, driven line by line with offline agents."""

import argparse
import asyncio
import io
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from invoice_agents import cli, shell
from invoice_agents.config import Settings
from tests.conftest import INVOICES


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        db_path=tmp_path / "inventory.db",
        samples_dir=tmp_path / "samples",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(shell, "get_settings", lambda: settings)

    async def down(*_args: object) -> bool:
        return False

    monkeypatch.setattr(cli.remote, "reachable", down)
    return settings


def args(*argv: str) -> argparse.Namespace:
    return cli._parse_args(["--no-serve", *argv])


@pytest.fixture
def sh() -> Iterator[shell.Shell]:
    with asyncio.Runner() as loop:
        s = shell.Shell(args(), loop)
        s.start()
        yield s
        s.close()


def said(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err


def test_a_typo_names_the_command_it_meant() -> None:
    assert shell.suggest("hlep") == "help"
    assert shell.suggest("serv") == "serve"
    assert shell.suggest("exti") == "quit"  # via the "exit" alias
    assert shell.suggest("zzzz") is None


def test_tab_completes_commands_paths_and_choices(tmp_path: Path) -> None:
    assert shell.complete("/ru", "/ru") == ["/run ", "/runs "]
    (tmp_path / "invoice_1.json").write_text("{}")
    (tmp_path / "folder").mkdir()
    prefix = str(tmp_path) + "/"
    assert shell.complete(prefix + "inv", "/run " + prefix + "inv") == [
        prefix + "invoice_1.json"
    ]
    assert shell.complete(prefix + "fo", prefix + "fo") == [prefix + "folder/"]
    assert shell.complete("ap", "/vp ap") == ["approve"]
    assert shell.complete("di", "/help di") == ["dir"]
    assert shell.complete("x", "/status x") == []


def test_unknown_input_points_to_help(
    sh: shell.Shell, capsys: pytest.CaptureFixture[str]
) -> None:
    said(capsys)
    assert sh.handle("/serv")
    assert "Did you mean /serve?" in said(capsys)
    assert sh.handle("hello")
    assert "Not a command or a file: hello" in said(capsys)
    assert sh.handle("/[x]")  # markup in what was typed is printed as is
    assert "Unknown command /[x]." in said(capsys)
    assert sh.handle('/run "unclosed')
    assert "Can't read that line" in said(capsys)
    assert sh.handle("") and not sh.handle("/quit") and not sh.handle("/exit")


def test_help_lists_every_command_and_explains_one(
    sh: shell.Shell, capsys: pytest.CaptureFixture[str]
) -> None:
    said(capsys)
    sh.handle("/help")
    out = said(capsys)
    assert all(c.usage in out for c in shell.COMMANDS)
    sh.handle("/help dir")
    out = said(capsys)
    assert "/dir [folder]" in out and "/run" not in out
    sh.handle("/help nope")
    assert "No command /nope." in said(capsys)


def test_runs_a_path_then_a_folder_and_remembers_them(
    sh: shell.Shell, offline: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    offline.samples_dir.mkdir()
    for name in ("invoice_1015.csv", "invoice_1016.json"):
        shutil.copy(INVOICES / name, offline.samples_dir / name)

    sh.handle(str(offline.samples_dir / "invoice_1015.csv"))
    sh.handle("/dir")  # the samples folder by default
    out = said(capsys)
    assert "INV-1015" in out and "2 invoices" in out
    # The session keeps one graph, so the second 1015 is a duplicate.
    assert [r["status"] for r in sh.history] == ["paid", "rejected", "rejected"]

    sh.handle("/runs")
    assert said(capsys).count("INV-1015") == 2

    sh.handle("/run")
    assert "Usage: /run <file>..." in said(capsys)
    sh.handle("/dir nowhere")
    assert "Not a folder: nowhere" in said(capsys)
    sh.handle(f"/run {offline.samples_dir / 'missing.json'}")
    assert "No such file:" in said(capsys)
    (offline.samples_dir / "empty").mkdir()
    sh.handle(f"/dir {offline.samples_dir / 'empty'}")
    assert "No invoice files there." in said(capsys)


def test_settings_commands(sh: shell.Shell, capsys: pytest.CaptureFixture[str]) -> None:
    said(capsys)
    sh.handle("/runs")
    assert "No runs yet" in said(capsys)
    sh.handle("/vp approve")
    assert sh.vp_mode == "approve"
    sh.handle("/vp")
    assert "approve" in said(capsys)
    sh.handle("/vp maybe")
    assert "Usage: /vp" in said(capsys)
    sh.handle("/status")
    out = said(capsys)
    assert "offline" in out and "not running" in out
    sh.handle("/inventory")
    assert "WidgetA" in said(capsys)
    sh.handle("/open")
    assert "isn't running" in said(capsys)
    sh.handle("/approve")
    assert "Usage: /approve" in said(capsys)
    sh.handle("/approve abc")
    assert "start it with /serve" in said(capsys)


def test_eval_runs_the_named_set(
    sh: shell.Shell, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from invoice_agents import evaluation

    calls: list[list[str]] = []
    monkeypatch.setattr(evaluation, "main", lambda argv: calls.append(argv) or 0)
    sh.handle("/eval")
    sh.handle("/eval adversarial --repeat 2")
    sh.handle("/eval bogus")
    assert calls == [["--set", "samples"], ["--set", "adversarial", "--repeat", "2"]]
    assert "No set bogus" in said(capsys)


def test_through_the_api_runs_lists_and_answers_waiting_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = [
        {
            "id": "abc123",
            "source": "a.json",
            "status": "awaiting_vp",
            "invoice_number": "INV-1",
            "vendor": "V",
            "total": 12000.0,
            "currency": "USD",
        },
        {"id": "abd999", "source": "b.json", "status": "paid"},
    ]
    decided: list[dict[str, Any]] = []

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/runs":
            return httpx.Response(200, json=runs)
        if request.url.path == "/api/runs/abc123/decision":
            body = json.loads(request.content)
            if body["note"] == "refuse":
                return httpx.Response(401, json={"detail": "VP token required"})
            decided.append(body)
            return httpx.Response(202, json={})
        return httpx.Response(404)

    async def up(*_args: object) -> bool:
        return True

    opened: list[str] = []
    monkeypatch.setattr(cli.remote, "reachable", up)
    monkeypatch.setattr(shell.webbrowser, "open", opened.append)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        shell.httpx,
        "AsyncClient",
        lambda: real(transport=httpx.MockTransport(api)),
    )
    with asyncio.Runner() as loop:
        s = shell.Shell(args(), loop)
        s.start()
        said(capsys)
        s.handle("/runs")
        out = said(capsys)
        assert "abc123" in out and "INV-1" in out and "$12,000.00" in out
        assert "/approve <run>" in out
        s.handle("/approve abc ok by me")
        assert decided == [{"approved": True, "note": "ok by me"}]
        assert "Approved INV-1" in said(capsys)
        s.handle("/reject abc refuse")
        assert "The API refused: VP token required" in said(capsys)
        s.handle("/reject zz")
        assert "No run waiting for the VP starts with zz" in said(capsys)
        s.handle("/open")
        assert opened == ["http://localhost:5173"]
        s.close()


def test_main_opens_the_shell_without_a_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("/help\n"))
    assert cli.main(["--no-serve"]) == 0  # ends at the end of input
    assert "/inventory" in said(capsys)


def test_dash_i_runs_the_file_then_stays(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("/runs\n/quit\n"))
    path = INVOICES / "invoice_1015.csv"
    assert cli.main(["--no-serve", "-i", f"--invoice_path={path}"]) == 0
    assert said(capsys).count("INV-1015") >= 2  # the run, then /runs


def test_a_console_that_will_not_start_falls_back_to_local(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(*_args: object) -> list[Any]:
        raise RuntimeError("The console needs Node.")

    monkeypatch.setattr(shell.remote, "serve", broken)
    with asyncio.Runner() as loop:
        s = shell.Shell(cli._parse_args([]), loop)
        s.start()
        s.handle("/serve")
        s.close()
    out = said(capsys)
    assert out.count("The console needs Node. Running here instead.") == 2
    assert "/serve to start it" in out


def test_when_the_shell_opens() -> None:
    def interactive(*argv: str) -> bool:
        return cli._interactive(cli._parse_args(list(argv)))

    assert interactive()
    assert interactive("-i", "--dir", "x")
    assert not interactive("--invoice_path", "x")
    assert not interactive("--dir", "x")
    # --serve alone keeps serving without a prompt when stdin isn't a terminal.
    assert not interactive("--serve")


def test_a_missing_file_exits_instead_of_waiting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--invoice_path", "nope.json"]) == 2
    assert "No invoice files at nope.json" in said(capsys)


def test_ctrl_c_stops_a_run_or_clears_the_line_without_leaving(
    sh: shell.Shell, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines: list[str | BaseException] = [KeyboardInterrupt(), "/status", EOFError()]

    def fake_input(_prompt: str) -> str:
        item = lines.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def interrupted(_args: list[str]) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)
    sh.handlers["status"] = interrupted
    assert sh.loop_forever() == 0
    assert "Stopped." in said(capsys) and not lines


def test_tab_completion_is_wired_to_readline() -> None:
    readline = pytest.importorskip("readline")
    shell._install_completion()
    completer = readline.get_completer()
    assert completer is not None
    assert completer("/zz", 0) is None
