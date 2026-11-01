"""The watched folder standing in for the AP mailbox."""

import asyncio
import os
import shutil
import time
from pathlib import Path

import pytest

from invoice_agents import cli
from invoice_agents.config import Settings
from invoice_agents.inbox import SETTLE_SECONDS, Inbox, watch
from tests.conftest import INVOICES


def landed(folder: Path, name: str, age: float = 10.0) -> Path:
    path = folder / name
    path.write_text("x")
    past = time.time() - age
    os.utime(path, (past, past))
    return path


def test_only_settled_visible_files_are_ready(tmp_path: Path) -> None:
    box = Inbox(tmp_path)
    older = landed(tmp_path, "b.json", age=20)
    newer = landed(tmp_path, "a.json", age=10)
    landed(tmp_path, ".hidden")
    landed(tmp_path, "copying.pdf", age=SETTLE_SECONDS / 10)  # still being written
    assert box.arrived() == [older, newer]  # oldest first; processed/ is skipped
    assert box.done.is_dir()


def test_filing_never_overwrites(tmp_path: Path) -> None:
    box = Inbox(tmp_path)
    first = box.file_away(landed(tmp_path, "inv.json"))
    second = box.file_away(landed(tmp_path, "inv.json"))
    assert first == box.done / "inv.json"
    assert second.parent == box.done and second.name.endswith("-inv.json")
    assert first.exists() and second.exists()


async def test_watch_hands_over_what_arrives_then_files_it(tmp_path: Path) -> None:
    box = Inbox(tmp_path)
    landed(tmp_path, "one.json")
    stop = asyncio.Event()
    seen: list[list[str]] = []

    async def handle(batch: list[Path]) -> None:
        seen.append([p.name for p in batch])
        if len(seen) == 1:
            landed(tmp_path, "two.json")  # arrives while the first is handled
        else:
            stop.set()

    await asyncio.wait_for(watch(box, handle, interval=0.01, stop=stop), timeout=5)
    assert seen == [["one.json"], ["two.json"]]
    assert sorted(p.name for p in box.done.iterdir()) == ["one.json", "two.json"]
    assert box.arrived() == []


def test_cli_watch_processes_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        db_path=tmp_path / "inventory.db",
        decisions_log=tmp_path / "decisions.jsonl",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    async def down(*_args: object) -> bool:
        return False

    monkeypatch.setattr(cli.remote, "reachable", down)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("invoice_1015.csv", "invoice_1014.xml"):
        copy = Path(shutil.copy(INVOICES / name, inbox / name))
        os.utime(copy, (time.time() - 10, time.time() - 10))

    handled: list[list[dict]] = []

    async def once(box: Inbox, handle, interval: float = 2.0, stop=None) -> None:
        # One look at the folder instead of watching forever.
        batch = box.arrived()
        real = cli.process

        async def recording(runner, paths, vp_mode):
            results = await real(runner, paths, vp_mode)
            handled.append(results)
            return results

        monkeypatch.setattr(cli, "process", recording)
        await handle(batch)
        for path in batch:
            box.file_away(path)

    monkeypatch.setattr(cli, "watch", once)
    assert not cli._interactive(cli._parse_args(["--watch", str(inbox)]))
    assert cli.main(["--watch", str(inbox)]) == 0
    [results] = handled
    status = {r["file"]: r["status"] for r in results}
    # Nobody to ask while watching: the EUR invoice is left for a VP.
    assert status == {"invoice_1015.csv": "paid", "invoice_1014.xml": "awaiting_vp"}
    out = capsys.readouterr().err
    assert "Watching" in out and "Left waiting for a VP" in out
    assert sorted(p.name for p in (inbox / "processed").iterdir()) == [
        "invoice_1014.xml",
        "invoice_1015.csv",
    ]
