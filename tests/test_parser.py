"""The parser that reads scans: chosen per run, from the CLI, the console or
the API, and Grok reading the page itself, recorded and replayed."""

import asyncio
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from invoice_agents import cli, graph, shell
from invoice_agents.agents import LLMVisionReader, RuleApprover, RuleCritic
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.ingestion import ocr
from invoice_agents.ingestion.ocr import OcrUnavailableError
from invoice_agents.llm import Agents
from invoice_agents.replay import Recording, RecordingReader, ReplayReader
from invoice_agents.usage import collecting
from tests.conftest import INVOICES
from tests.test_graph import INV_1001, StubExtractor

SCANNED = INVOICES.parent / "scanned"
SCAN = SCANNED / "invoice_1011_scan.png"


@pytest.fixture(autouse=True)
def fresh_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "_read", {})


class FakeReader:
    model = "grok-test"

    def __init__(self) -> None:
        self.read_paths: list[Path] = []

    def read(self, path: Path) -> str:
        self.read_paths.append(path)
        return f"Grok read {path.name}"


def make_graph(inventory: Any, reader: Any = None, default: str = "pymupdf") -> Any:
    agents = Agents(
        StubExtractor(INV_1001), RuleApprover(), RuleCritic(), reader=reader
    )
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        ocr_backend=default,  # pyright: ignore[reportArgumentType]
    )
    deps = Deps(agents=agents, inventory=inventory, settings=settings)
    return build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))


def config() -> Any:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


async def first_step(g: Any, payload: dict[str, Any]) -> str:
    state = await g.ainvoke(payload, config())
    return state["trace"][0]["message"]


async def test_a_run_picks_its_parser_over_the_default(
    inventory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    used: list[str] = []
    for name in ("pymupdf", "docling"):
        monkeypatch.setitem(
            graph.OCR_BACKENDS, name, lambda p, n=name: used.append(n) or "text"
        )
    g = make_graph(inventory)
    assert "OCR (pymupdf)" in await first_step(g, {"path": str(SCAN)})
    assert "OCR (docling)" in await first_step(
        g, {"path": str(SCAN), "parser": "docling"}
    )
    assert used == ["pymupdf", "docling"]


async def test_the_grok_parser_has_grok_read_the_scan_once(inventory: Any) -> None:
    reader = FakeReader()
    g = make_graph(inventory, reader, default="grok")
    message = await first_step(g, {"path": str(SCAN)})
    assert message.startswith("Read the scan with Grok vision (grok-test).")
    await first_step(g, {"path": str(SCAN)})
    assert reader.read_paths == [SCAN]  # the second run reads it from cache


async def test_grok_without_a_model_fails_scans_only(inventory: Any) -> None:
    g = make_graph(inventory, reader=None)
    payload = {"path": str(SCAN), "parser": "grok"}
    with pytest.raises(OcrUnavailableError, match="XAI_API_KEY or OPENROUTER"):
        await g.ainvoke(payload, config())
    # A text invoice never needs the parser, so it runs as usual.
    text = {"path": str(INVOICES / "invoice_1001.txt"), "parser": "grok"}
    state = await g.ainvoke(text, config())
    assert state["outcome"] == "pay"


class FakeVisionLLM:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.messages.append(messages)
        return AIMessage(
            content=" INV-1011  WidgetA 6 $250.00 ",
            usage_metadata={
                "input_tokens": 1500,
                "output_tokens": 40,
                "total_tokens": 1540,
            },
            response_metadata={"model_name": "x-ai/grok-4.3"},
        )


def test_the_vision_reader_sends_each_page_as_an_image_and_counts_tokens() -> None:
    llm = FakeVisionLLM()
    reader = LLMVisionReader(llm, "grok-4.3")  # pyright: ignore[reportArgumentType]
    with collecting() as tokens:
        text = reader.read(SCANNED / "invoice_1011_scan.pdf")
    assert text == "INV-1011  WidgetA 6 $250.00"
    [[message]] = llm.messages
    prompt, image = message.content
    assert "exactly as printed" in prompt["text"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert tokens == {"x-ai/grok-4.3": {"input_tokens": 1500, "output_tokens": 40}}


def test_a_grok_reading_is_recorded_and_replayed_by_the_file_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.json"
    recorder = RecordingReader(FakeReader(), Recording(path))
    assert recorder.read(SCAN) == f"Grok read {SCAN.name}"
    # The same scan under another name replays the same reading.
    copy = tmp_path / "renamed.png"
    shutil.copy(SCAN, copy)
    replay = ReplayReader(Recording.load(path), "grok-4.3")
    assert replay.read(copy) == f"Grok read {SCAN.name}"
    with pytest.raises(OcrUnavailableError, match="pymupdf or docling"):
        replay.read(SCANNED / "invoice_1011_scan.pdf")


# --- Where a run gets its parser -------------------------------------------


@pytest.fixture
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="offline",
        db_path=tmp_path / "inventory.db",
        decisions_log=tmp_path / "decisions.jsonl",
        redis_url="",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(shell, "get_settings", lambda: settings)

    async def down(*_args: object) -> bool:
        return False

    monkeypatch.setattr(cli.remote, "reachable", down)
    monkeypatch.setattr(cli.remote, "api_up", down)
    return settings


async def test_the_cli_flag_reaches_the_runner(offline: Settings) -> None:
    import httpx

    args = cli._parse_args(["--parser", "docling", "--local"])
    async with httpx.AsyncClient() as client:
        conn = await cli.connect(args, client)
    assert conn.runner.parser == "docling"
    with pytest.raises(SystemExit):
        cli._parse_args(["--parser", "tesseract"])


@pytest.fixture
def sh(offline: Settings) -> Iterator[shell.Shell]:
    with asyncio.Runner() as loop:
        s = shell.Shell(cli._parse_args(["--no-serve"]), loop)
        s.start()
        yield s
        s.close()


def test_the_console_switches_the_parser(
    sh: shell.Shell, capsys: pytest.CaptureFixture[str]
) -> None:
    sh.handle("/parser")
    assert "read by pymupdf" in capsys.readouterr().err
    sh.handle("/parser grok")
    assert "read by grok (Grok reads the page image)" in capsys.readouterr().err
    assert sh.conn is not None and sh.conn.runner.parser == "grok"
    assert "parser: grok" in sh._status_line()  # pyright: ignore[reportPrivateUsage]
    sh.handle("/parser tesseract")
    assert "Usage: /parser" in capsys.readouterr().err
    assert shell.complete("gr", "/parser gr") == ["grok"]


def test_the_assistant_may_pick_a_parser_for_a_run_and_nothing_else() -> None:
    from invoice_agents.assistant import Plan, Step, checked, commands

    plan = Plan(
        reply="Reading the scan with Grok.",
        steps=[
            Step(
                command="run",
                args=["data/scanned/invoice_1011_scan.png"],
                parser="Grok",
            ),
            Step(command="dir", args=["data/scanned"], parser="tesseract"),
            Step(command="inventory", parser="docling"),
        ],
    )
    steps, dropped = checked(plan)
    assert [s.parser for s in steps] == ["grok", None, None]
    assert not dropped
    assert (
        commands(steps)[0] == "/run data/scanned/invoice_1011_scan.png (parser: grok)"
    )


def test_a_step_s_parser_lasts_for_the_step(sh: shell.Shell) -> None:
    assert sh.conn is not None
    runner = sh.conn.runner
    with sh._parser_for("grok"):  # pyright: ignore[reportPrivateUsage]
        assert runner.parser == "grok"
    assert runner.parser is None
