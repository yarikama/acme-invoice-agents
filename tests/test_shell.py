"""The interactive console, driven line by line with offline agents."""

import argparse
import asyncio
import glob
import io
import json
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from invoice_agents import cli, shell
from invoice_agents.config import Settings
from tests.conftest import INVOICES
from tests.fakes import FakePlanner


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        db_path=tmp_path / "inventory.db",
        decisions_log=tmp_path / "decisions.jsonl",
        samples_dir=tmp_path / "samples",
        redis_url="",  # never the developer's Redis
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
    # Commands only, even on Linux where "/run/" is a folder at the root.
    assert shell.complete("/ru", "/ru") == ["/run ", "/runs "]
    assert shell.complete("/Use", "/Use") == sorted(
        p + "/" for p in glob.glob("/Use*")
    )  # no command matches, so it's a path
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
    assert "Plain-language requests need a live model" in said(capsys)
    assert sh.handle("/[x]")  # markup in what was typed is printed as is
    assert "Unknown command /[x]." in said(capsys)
    assert sh.handle("what's waiting?")  # an open quote: still a sentence
    assert "need a live model" in said(capsys)
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
    out = said(capsys)
    assert "WidgetA" in out and "Vendor master (11)" in out
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
        s.handle("/reject abc")  # a rejection needs a reason
        assert "Usage: /reject <run> <reason>" in said(capsys)
        s.handle("/reject zz wrong rate")
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


def test_when_the_shell_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    def interactive(*argv: str) -> bool:
        return cli._interactive(cli._parse_args(list(argv)))

    # Off a terminal (a script, CI, this test): a file runs once and exits.
    assert interactive()
    assert interactive("-i", "--dir", "x")
    assert not interactive("--invoice_path", "x")
    assert not interactive("--dir", "x")
    # --serve alone keeps serving without a prompt when stdin isn't a terminal.
    assert not interactive("--serve")

    # At a terminal, a file runs and the console stays, as if typed as /run...
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert interactive("--invoice_path", "x")
    assert interactive("--serve")
    # ...unless the output is for a program, or asked to exit.
    assert not interactive("--invoice_path", "x", "--json")
    assert not interactive("--invoice_path", "x", "--once")
    with pytest.raises(SystemExit):
        cli._parse_args(["--once", "-i"])


def test_a_hangup_or_kill_ends_the_cli_and_runs_its_cleanup() -> None:
    import os
    import signal

    before = signal.getsignal(signal.SIGTERM)
    cleaned = False
    with pytest.raises(SystemExit) as stopped, cli._signals_end_the_cli():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        finally:
            cleaned = True
    assert stopped.value.code == 128 + signal.SIGTERM and cleaned
    assert signal.getsignal(signal.SIGTERM) is before


def test_closing_stops_the_servers_even_when_the_terminal_is_gone(
    sh: shell.Shell, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped: list[Any] = []

    def gone(*_a: object, **_k: object) -> None:
        raise OSError(5, "Input/output error")

    sh.started = [cast(Any, "api")]
    monkeypatch.setattr(shell.console, "print", gone)
    monkeypatch.setattr(shell.remote, "stop", stopped.append)
    sh.close()
    sh.started = []
    assert stopped == [["api"]]


def test_a_missing_file_exits_instead_of_waiting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--invoice_path", "nope.json"]) == 2
    assert "No invoice files at nope.json" in said(capsys)


def test_ctrl_c_stops_a_run_or_clears_the_line_without_leaving(
    sh: shell.Shell, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines: list[str | BaseException] = [KeyboardInterrupt(), "/status", EOFError()]

    def fake_input(_prompt: str = "") -> str:
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


def test_the_menu_opens_for_commands_and_paths_not_sentences() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    def menu(text: str, tab: bool = False) -> list[tuple[str, str]]:
        found = shell.Completer().get_completions(
            Document(text), CompleteEvent(completion_requested=tab)
        )
        return [(c.text, c.display_meta_text) for c in found]

    assert ("/runs ", "Recent runs and where each one ended.") in menu("/r")
    assert menu("help me with") == []
    assert menu("help me with", tab=True) == []  # no file starts with "with"
    assert menu("data/invoices/invoice_101") != []


def test_the_input_box_reads_a_line_and_runs_menu_picks(tmp_path: Path) -> None:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    def read(*keys: str) -> str:
        """Type each group of keys a moment apart, as a person would: the
        menu fills in the background, so Enter right after Tab beats it."""
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=DummyOutput()),
        ):

            def type_all() -> None:
                for group in keys:
                    time.sleep(0.15)
                    pipe.send_text(group)

            threading.Thread(target=type_all, daemon=True).start()
            return shell.Prompt(lambda: "status").read()

    assert read("hello there\r") == "hello there"
    # Enter on a menu item with nothing more to type runs it at once...
    assert read("/inv", "\t", "\r") == "/inventory "
    # ...and one that needs an argument waits for it.
    assert read("/ru", "\t", "\r", "invoice.json\r") == "/run invoice.json"
    with pytest.raises(EOFError):
        read("\x04")
    with pytest.raises(KeyboardInterrupt):
        read("half typed\x03")
    assert "replay · local" in str(shell.Prompt(lambda: "replay · local")._hints())


def test_a_sentence_goes_to_the_assistant_and_code_filters_its_plan(
    sh: shell.Shell,
    offline: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from invoice_agents import assistant

    offline.llm_provider = "openrouter"
    monkeypatch.setattr(shell, "chat_model", lambda *_a: object())
    path = INVOICES / "invoice_1015.csv"
    answer = assistant.Plan(
        reply="Running it.",
        steps=[
            assistant.Step(command="/RUN", args=[str(path)]),
            assistant.Step(command="approve", args=["abc"]),
            assistant.Step(command="rm", args=["-rf"]),
        ],
    )

    async def planned(
        _llm: object, request: str, history: Any, via: str
    ) -> tuple[assistant.Plan, Any]:
        assert (request, via) == ("please ingest 1015", "terminal")
        return answer, {"grok-4.3": {"input_tokens": 1000, "output_tokens": 200}}

    monkeypatch.setattr(assistant, "plan", planned)
    said(capsys)
    sh.handle("please ingest 1015")
    out = said(capsys)
    assert "1,200 tokens" in out and "Running it." in out
    assert "Not run: /approve is for a person to type" in out
    assert "Not run: /rm isn't a command the assistant may use" in out
    assert "→ /run" in out and "invoice_1015.csv" in out and "INV-1015" in out
    assert [r["status"] for r in sh.history] == ["paid"]
    # No Redis in tests: remembered for this session, and it says so.
    assert "not shared with the web console" in out
    # What came of it joins the conversation, for the next request.
    assert sh.memory is not None
    assert sh.memory.messages[-1].text == "INV-1015 was paid: Every check passed."

    sh.handle("/history")
    assert "INV-1015 was paid: Every check passed." in said(capsys)

    async def broken(*_a: object) -> Any:
        raise ValueError("rate limited")

    monkeypatch.setattr(assistant, "plan", broken)
    sh.handle("again please")
    assert "The assistant couldn't answer: rate limited" in said(capsys)


async def test_the_assistant_asks_with_a_schema_and_counts_tokens() -> None:

    from invoice_agents import assistant, memory

    model = FakePlanner(assistant.Plan(reply="ok", steps=[]), None)
    history = memory.LocalHistory()
    answer, tokens = await assistant.plan(cast(Any, model), "hi", history, "terminal")
    assert answer.reply == "ok"
    assert tokens == {"x-ai/grok-4.3": {"input_tokens": 900, "output_tokens": 40}}
    assert model.schema == (
        assistant.Plan,
        {"method": "json_schema", "include_raw": True},
    )
    system = model.prompts[0][0].content
    assert "invoice_1015.csv" in system and "cannot approve" in system

    with pytest.raises(ValueError, match="bad json"):
        await assistant.plan(cast(Any, model), "hi", history, "terminal")
    # A failed plan leaves nothing half-said in the conversation.
    assert [m.text for m in history.messages] == ["hi", "ok"]
    assert assistant.known_files(Path("/nonexistent")) == "(none found)"


async def test_the_assistant_remembers_the_conversation_within_a_buffer() -> None:

    from invoice_agents import assistant, memory

    run = assistant.Step(command="run", args=["data/invoices/invoice_1004.json"])
    model = FakePlanner(
        assistant.Plan(reply="Running 1004.", steps=[run]),
        assistant.Plan(reply="It was paid.", steps=[]),
        assistant.Plan(reply="Sure.", steps=[]),
    )
    history = memory.LocalHistory()
    await assistant.plan(cast(Any, model), "run 1004", history, "terminal")
    history.add_message(memory.answered("INV-1004 was paid.", "terminal"))
    await assistant.plan(cast(Any, model), "what happened to it?", history, "web")

    # The second prompt carries the first request, the plan and its outcome.
    second = [m.text for m in model.prompts[1][1:]]
    assert second == [
        "run 1004",
        "Running 1004.",
        "INV-1004 was paid.",
        "what happened to it?",
    ]
    said, planned, _outcome, asked, answered = history.messages
    assert said.additional_kwargs["via"] == "terminal"
    assert planned.additional_kwargs["commands"] == [
        "/run data/invoices/invoice_1004.json"
    ]
    assert asked.additional_kwargs["via"] == "web"
    assert answered.text == "It was paid."

    # Only the last few messages go into a prompt, starting on a request.
    await assistant.plan(cast(Any, model), "thanks", history, "web", buffer=3)
    assert [m.text for m in model.prompts[2][1:]] == [
        "what happened to it?",
        "It was paid.",
        "thanks",
    ]


def test_a_dumb_terminal_gets_plain_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert shell._fancy_prompt(lambda: "") is None


def test_reset_starts_over(
    sh: shell.Shell, offline: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = str(INVOICES / "invoice_1015.csv")
    sh.handle(sample)
    sh.handle(sample)
    assert [r["status"] for r in sh.history] == ["paid", "rejected"]

    sh.handle("/reset")
    assert "Started over" in said(capsys)
    assert sh.history == []
    sh.handle(sample)  # no longer a duplicate
    assert [r["status"] for r in sh.history] == ["paid"]


def test_watch_runs_what_lands_and_waits_for_a_vp_instead_of_asking(
    sh: shell.Shell,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    shutil.copy(INVOICES / "invoice_1014.xml", inbox / "invoice_1014.xml")

    async def once(box, handle, interval: float = 2.0, stop=None) -> None:
        await handle([inbox / "invoice_1014.xml"])

    monkeypatch.setattr(shell, "watch", once)
    sh.handle(f"/watch {inbox}")
    assert "Watching" in said(capsys)
    assert [r["status"] for r in sh.history] == ["awaiting_vp"]
    sh.handle("/watch a b")
    assert "Usage: /watch [folder]" in said(capsys)
    sh.handle("/vp wait")
    assert sh.vp_mode == "wait"
