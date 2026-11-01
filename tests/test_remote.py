"""The CLI talking to the API, against a scripted HTTP transport."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from invoice_agents import cli, remote
from invoice_agents.config import get_settings
from tests.conftest import INVOICES

SUMMARY = {
    "invoice": {
        "invoice_number": "INV-1014",
        "vendor": "TechParts",
        "total": 4125.0,
        "currency": "EUR",
    },
    "draft": {"decision": "escalate", "rationale": "Billed in EUR."},
    "route_reason": "billed in EUR",
    "findings": [],
    "trace": [],
}


def sse(*events: dict[str, Any]) -> str:
    lines = [": keepalive"]
    for i, e in enumerate(events):
        lines.append(f"id: {i}\nevent: {e['type']}\ndata: {json.dumps(e)}\n")
    return "\n".join(lines) + "\n"


class FakeApi:
    """Answers the CLI's requests the way the API does; records what it got."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []
        self.uploads: list[str] = []
        self.auth: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/api/runs/upload":
            self.uploads.append(request.content.decode(errors="ignore"))
            return httpx.Response(202, json={"id": "r1"})
        if path == "/api/runs/r1/events":
            t = "2026-11-01T00:00:0{}+00:00"
            body = sse(
                {"type": "started", "at": t.format(0)},
                {
                    "type": "progress",
                    "stage": "ingest",
                    "step": "start",
                    "at": t.format(0),
                },
                {
                    "type": "trace",
                    "node": "ingest",
                    "message": "Parsed XML in code.",
                    "at": t.format(1),
                },
                {
                    "type": "trace",
                    "node": "validate",
                    "message": "0 errors, 1 warning; policy route: human.",
                    "at": t.format(2),
                },
                {"type": "awaiting_vp", "summary": SUMMARY, "at": t.format(3)},
                {
                    "type": "trace",
                    "node": "human_review",
                    "message": "VP approved.",
                    "at": t.format(4),
                },
                {
                    "type": "done",
                    "status": "paid",
                    "summary": SUMMARY,
                    "at": t.format(5),
                },
                {
                    "type": "trace",
                    "node": "ignored",
                    "message": "after done",
                    "at": t.format(6),
                },
            )
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        if path == "/api/runs/r1/decision":
            self.decisions.append(json.loads(request.content))
            self.auth.append(request.headers.get("authorization"))
            return httpx.Response(202, json={"id": "r1"})
        if path == "/api/runs/r1":
            return httpx.Response(
                200,
                json={
                    "id": "r1",
                    "status": "paid",
                    "summary": {**SUMMARY, "status": "paid"},
                    "events": [],
                },
            )
        return httpx.Response(404)


def client(fake: FakeApi) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(fake))


async def test_remote_runner_streams_the_run_and_answers_the_vp() -> None:
    fake = FakeApi()
    async with client(fake) as c:
        runner = cli.RemoteRunner(remote.Api("http://api", c))
        result = await runner.process(INVOICES / "invoice_1014.xml", "approve")

    assert 'filename="invoice_1014.xml"' in fake.uploads[0]
    assert fake.decisions == [{"approved": True, "note": "--vp approve"}]
    assert fake.auth == [None]
    assert result["status"] == "paid" and result["run_id"] == "r1"
    assert result["file"] == "invoice_1014.xml"


async def test_events_parse_times_and_stop_at_the_end() -> None:
    async with client(FakeApi()) as c:
        events = [e async for e in remote.Api("http://api/", c).events("r1")]
    assert events[-1]["type"] == "done"
    assert events[0]["at"] == 1793491200.0
    assert all(e.get("node") != "ignored" for e in events)


async def test_reachable() -> None:
    async with client(FakeApi()) as c:
        assert await remote.reachable("http://api/api/health", c)

    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as c:
        assert not await remote.reachable("http://api/api/health", c)
    assert not await remote.reachable("http://127.0.0.1:9/")


def test_seconds() -> None:
    assert remote.seconds("2026-11-01T00:00:00+00:00") == 1793491200.0
    assert remote.seconds(None) is None
    assert remote.seconds("not a time") is None


def test_cli_uses_the_api_when_it_is_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeApi()
    real = httpx.AsyncClient
    monkeypatch.setattr(
        cli.httpx, "AsyncClient", lambda: real(transport=httpx.MockTransport(fake))
    )

    assert (
        cli.main(
            [
                f"--invoice_path={INVOICES / 'invoice_1014.xml'}",
                "--vp=approve",
                "--json",
            ]
        )
        == 0
    )

    [result] = json.loads(capsys.readouterr().out)
    assert result["status"] == "paid"
    assert fake.decisions


def test_serve_with_nothing_to_run_waits_then_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []

    async def fake_serve(api: str, web: str, logs: Path) -> list[Any]:
        started.append("serve")
        return []

    async def down(*_: object) -> bool:
        return False

    class Instant:
        async def wait(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli.remote, "serve", fake_serve)
    monkeypatch.setattr(cli.remote, "reachable", down)
    monkeypatch.setattr(cli.asyncio, "Event", Instant)

    assert cli.main(["--serve"]) == 130
    assert started == ["serve"]


async def test_serve_explains_a_missing_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def down(*_: object) -> bool:
        return False

    spawned: list[list[str]] = []

    class Proc:
        def poll(self) -> int:
            return 0

        def terminate(self) -> None: ...

        def wait(self, timeout: float) -> None: ...

    def fake_spawn(cmd: list[str], cwd: Path, log: Path) -> Proc:
        spawned.append(cmd)
        return Proc()

    monkeypatch.setattr(remote, "reachable", down)
    monkeypatch.setattr(remote, "_spawn", fake_spawn)
    monkeypatch.setattr(remote.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="make install"):
        await remote.serve("http://localhost:8000", "http://localhost:5173", tmp_path)
    assert spawned and spawned[0][-1] == "8000"


async def test_serve_gives_up_when_a_server_dies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def down(*_: object) -> bool:
        return False

    class Dead:
        def poll(self) -> int:
            return 1

        def terminate(self) -> None: ...

        def wait(self, timeout: float) -> None: ...

    monkeypatch.setattr(remote, "reachable", down)
    monkeypatch.setattr(remote, "_spawn", lambda *a: Dead())
    monkeypatch.setattr(remote.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(remote, "REPO", tmp_path)
    (tmp_path / "web" / "node_modules").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="did not start"):
        await remote.serve(
            "http://localhost:8000", "http://localhost:5173", tmp_path / "logs"
        )


def test_servers_never_read_the_terminal(tmp_path: Path) -> None:
    """Vite reads shortcuts from stdin; sharing the terminal with the console's
    input box split the typed keys between them."""
    script = (
        "import os, sys; print(sys.stdin.read() == '', os.getsid(0) == os.getpid())"
    )
    log = tmp_path / "out.log"
    p = remote._spawn([sys.executable, "-c", script], tmp_path, log)
    p.wait(timeout=10)
    # No stdin (reads nothing), and its own session (Ctrl-C doesn't reach it).
    assert log.read_text().split() == ["True", "True"]


def test_spawn_and_stop_a_real_process(tmp_path: Path) -> None:
    p = remote._spawn(["sleep", "30"], tmp_path, tmp_path / "logs" / "x.log")
    assert p.poll() is None
    remote.stop([p])
    assert p.poll() is not None

    class Stubborn:
        killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None: ...

        def wait(self, timeout: float) -> None:
            raise subprocess.TimeoutExpired("x", timeout)

        def kill(self) -> None:
            Stubborn.killed = True

    remote.stop([Stubborn()])  # type: ignore[list-item]
    assert Stubborn.killed


async def test_a_vp_token_goes_with_the_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VP_TOKEN", "s3cret")
    get_settings.cache_clear()
    fake = FakeApi()
    async with client(fake) as c:
        await remote.Api("http://api", c).decide("r1", False, "no")
    assert fake.auth == ["Bearer s3cret"]
    get_settings.cache_clear()


async def test_reset_through_the_api_says_why_it_was_refused() -> None:
    answers = iter(
        [
            httpx.Response(204),
            httpx.Response(409, json={"detail": "1 run(s) still working"}),
        ]
    )
    transport = httpx.MockTransport(lambda _request: next(answers))
    async with httpx.AsyncClient(transport=transport) as c:
        api = remote.Api("http://api", c)
        assert await api.reset() is None
        assert await api.reset() == "1 run(s) still working"


def test_serve_starts_redis_only_when_it_is_missing_and_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[list[str]] = []

    class Proc:
        def poll(self) -> None:
            return None

    def spawn(cmd: list[str], _cwd: Path, _log: Path) -> Proc:
        spawned.append(cmd)
        return Proc()

    monkeypatch.setattr(remote, "_spawn", spawn)
    monkeypatch.setattr(remote.shutil, "which", lambda _name: "/usr/bin/redis-server")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6399/0")
    get_settings.cache_clear()
    calls = iter([False, True])  # down before the start, up just after
    monkeypatch.setattr(remote.memory, "redis_up", lambda _url: next(calls))
    assert remote._start_redis(tmp_path) is not None
    assert spawned[0][:3] == ["/usr/bin/redis-server", "--port", "6399"]

    # Already up, not installed, or not on this machine: nothing to start.
    monkeypatch.setattr(remote.memory, "redis_up", lambda _url: True)
    assert remote._start_redis(tmp_path) is None
    monkeypatch.setattr(remote.memory, "redis_up", lambda _url: False)
    monkeypatch.setenv("REDIS_URL", "redis://cache.example.com:6379/0")
    get_settings.cache_clear()
    assert remote._start_redis(tmp_path) is None
    get_settings.cache_clear()
