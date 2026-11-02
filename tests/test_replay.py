"""Recording a run's agent answers and replaying them with no model."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from invoice_agents import evaluation, replay, usage
from invoice_agents.agents import Review, RuleApprover, RuleCritic
from invoice_agents.api.app import load_pricing
from invoice_agents.config import Settings
from invoice_agents.graph import Deps, build_graph, checkpoint_serde
from invoice_agents.inventory import Inventory
from invoice_agents.llm import Agents, build_agents
from invoice_agents.models import (
    ApprovalDraft,
    Critique,
    Invoice,
    ToolCall,
    ValidationReport,
)
from invoice_agents.replay import (
    FALLBACK,
    NotRecordedError,
    Recording,
    RecordingApprover,
    RecordingCritic,
    RecordingExtractor,
    ReplayApprover,
    ReplayCritic,
    ReplayExtractor,
)
from invoice_agents.tools import ReviewTools
from invoice_agents.validation import SeenInvoices
from tests.conftest import INVOICES
from tests.test_graph import INV_1001, StubExtractor


class Reply:
    """What `usage.record` reads off a model reply."""

    def __init__(self, tokens: int) -> None:
        self.usage_metadata = {"input_tokens": tokens, "output_tokens": 1}
        self.response_metadata = {"model_name": "x-ai/grok-4.7"}


class LookingApprover:
    """Checks the inventory, spends tokens, then approves."""

    async def draft(
        self,
        invoice: Invoice,
        report: ValidationReport,
        critique: list[str],
        tools: ReviewTools,
    ) -> Review:
        usage.record(Reply(1000))
        call = tools.run("check_inventory", {"item": "WidgetA"})
        draft = ApprovalDraft(decision="approve", rationale="Stock covers it.")
        return Review(draft, [call])


def graph(inventory: Inventory, agents: Agents) -> Any:
    deps = Deps(
        agents=agents,
        inventory=inventory,
        settings=Settings(llm_provider="offline"),
        pricing={"x-ai/grok-4.7": (1e-6, 1e-6)},
    )
    return build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))


async def run(inventory: Inventory, agents: Agents, name: str) -> dict[str, Any]:
    config: Any = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return await graph(inventory, agents).ainvoke(
        {"path": str(INVOICES / name)}, config
    )


def lines(state: dict[str, Any]) -> list[tuple[str, str]]:
    return [(e["stage"], e["message"]) for e in state["trace"]]


async def test_a_recorded_run_replays_the_same_way(
    inventory: Inventory, tmp_path: Path
) -> None:
    path = tmp_path / "recording.json"
    recording = Recording(path)
    live = Agents(
        RecordingExtractor(StubExtractor(INV_1001), recording),
        RecordingApprover(LookingApprover(), recording),
        RecordingCritic(RuleCritic(), recording),
    )
    first = await run(inventory, live, "invoice_1001.txt")
    assert first["payment"].status == "success"
    data = json.loads(path.read_text())
    assert sorted(k.split(":")[0] for k in data["entries"]) == [
        "critique",
        "draft",
        "extract",
    ]

    loaded = Recording.load(path)
    replayed = Agents(
        ReplayExtractor(loaded), ReplayApprover(loaded), ReplayCritic(loaded)
    )
    again = await run(inventory, replayed, "invoice_1001.txt")
    assert again["payment"].status == "success"
    assert lines(again) == lines(first)
    # The approver's recorded tokens come back on its step, priced.
    draft = next(e for e in again["trace"] if e["message"].startswith("Draft 1"))
    assert draft["usage"]["input_tokens"] == 1000
    assert draft["usage"]["cost_usd"] == pytest.approx(0.001001)


async def test_replayed_tool_calls_are_answered_by_code_again(
    inventory: Inventory, tmp_path: Path
) -> None:
    recording = Recording(tmp_path / "r.json")
    stale = ToolCall(name="check_inventory", args={"item": "WidgetA"}, result="STALE")
    guard = ToolCall(name="guardrail", args={}, result="Address NEAR_THRESHOLD.")
    recording.entries[replay.draft_key(INV_1001, ValidationReport(findings=[]), [])] = {
        "draft": {"decision": "approve", "rationale": "ok", "addressed": []},
        "steps": [stale.model_dump(), guard.model_dump()],
        "tokens": {},
    }
    tools = ReviewTools(inventory, SeenInvoices(), INV_1001)
    review = await ReplayApprover(recording).draft(
        INV_1001, ValidationReport(findings=[]), [], tools
    )
    assert review.steps[0].result != "STALE"
    assert "WidgetA" in review.steps[0].result
    assert review.steps[1] == guard  # the linter's reply is part of the record


async def test_what_was_not_recorded(inventory: Inventory, tmp_path: Path) -> None:
    empty = Recording.load(tmp_path / "missing.json")
    assert empty.entries == {} and empty.pricing == {}

    with pytest.raises(NotRecordedError, match="no recorded extraction"):
        await ReplayExtractor(empty).extract("some text", [])

    agents = Agents(ReplayExtractor(empty), ReplayApprover(empty), ReplayCritic(empty))
    # A structured invoice needs no extraction; the rules stand in for the rest.
    state = await run(inventory, agents, "invoice_1004.json")
    assert state["draft"].rationale.startswith(FALLBACK)
    review = await ReplayCritic(empty).review(
        INV_1001,
        ValidationReport(findings=[]),
        ApprovalDraft(decision="approve", rationale="x"),
        [],
    )
    assert review == await RuleCritic().review(
        INV_1001,
        ValidationReport(findings=[]),
        ApprovalDraft(decision="approve", rationale="x"),
        [],
    )


def test_recording_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "r.json"
    rec = Recording(path, pricing={"m": (1.0, 2.0), "unused": (3.0, 3.0)})
    rec.put("b:1", {"x": 1, "tokens": {"m": {"input_tokens": 1, "output_tokens": 1}}})
    rec.put("a:1", {"y": "é"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data["entries"]) == ["a:1", "b:1"]  # stable diffs
    back = Recording.load(path)
    assert back.pricing == {"m": (1.0, 2.0)} and back.get("a:1") == {"y": "é"}
    assert back.get("c:1") is None


def test_settings_choose_replay_or_recording(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    Recording(
        path, {"k": {"tokens": {"x-ai/grok-4.7": {}}}}, {"x-ai/grok-4.7": (1e-6, 2e-6)}
    ).save()
    settings = Settings(_env_file=None, llm_provider="replay", replay_path=path)  # pyright: ignore[reportCallIssue]
    agents = build_agents(settings)
    assert isinstance(agents.approver, ReplayApprover) and agents.triager is None
    # One shared recording per file, and its prices price the steps.
    assert agents.approver.recording is Recording.shared(path)
    assert load_pricing(settings) == {"x-ai/grok-4.7": (1e-6, 2e-6)}

    live = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="openrouter",
        openrouter_api_key="k",
        record_to=tmp_path / "new.json",
    )
    recording = build_agents(live)
    assert isinstance(recording.extractor, RecordingExtractor)
    assert isinstance(recording.approver, RecordingApprover)
    assert isinstance(recording.critic, RecordingCritic)


async def test_recorded_tokens_still_count_for_the_live_step(tmp_path: Path) -> None:
    recording = Recording(tmp_path / "r.json")
    approver = RecordingApprover(LookingApprover(), recording)
    inventory = Inventory(tmp_path / "inv.db")
    inventory.seed()
    tools = ReviewTools(inventory, SeenInvoices(), INV_1001)
    with usage.collecting() as step:
        await approver.draft(INV_1001, ValidationReport(findings=[]), [], tools)
    assert step == {"x-ai/grok-4.7": {"input_tokens": 1000, "output_tokens": 1}}
    [entry] = recording.entries.values()
    assert entry["tokens"] == step
    # Outside a step nothing is collected, and nothing breaks.
    await approver.draft(INV_1001, ValidationReport(findings=[]), [], tools)
    await RuleApprover().draft(INV_1001, ValidationReport(findings=[]), [], tools)
    assert Critique(agrees=True).issues == []


async def test_a_draft_is_only_replayed_for_the_findings_it_was_written_for(
    inventory: Inventory, tmp_path: Path
) -> None:
    from invoice_agents.models import Finding, Severity

    clean = ValidationReport(findings=[])
    duplicate = ValidationReport(
        findings=[
            Finding(
                code="DUPLICATE",
                severity=Severity.ERROR,
                title="Duplicate invoice",
                detail="INV-1001 was already processed.",
            )
        ]
    )
    recording = Recording(tmp_path / "r.json")
    recording.entries[replay.draft_key(INV_1001, clean, [])] = {
        "draft": {"decision": "approve", "rationale": "Clean.", "addressed": []},
        "steps": [],
        "tokens": {},
    }
    tools = ReviewTools(inventory, SeenInvoices(), INV_1001)
    approver = ReplayApprover(recording)
    assert (
        await approver.draft(INV_1001, clean, [], tools)
    ).draft.rationale == "Clean."
    # The same invoice seen again as a duplicate doesn't get the clean draft.
    again = await approver.draft(INV_1001, duplicate, [], tools)
    assert again.draft.rationale.startswith(FALLBACK)
    assert again.draft.decision == "reject"


class SlowApprover:
    """Two tool calls, 0.05 s apart, then a decision 0.05 s later."""

    async def draft(self, invoice, report, critique, tools) -> Review:
        await asyncio.sleep(0.05)
        first = tools.run("check_inventory", {"item": "WidgetA"})
        await asyncio.sleep(0.05)
        second = tools.run("check_inventory", {"item": "WidgetB"})
        await asyncio.sleep(0.05)
        return Review(
            ApprovalDraft(decision="approve", rationale="Fine."), [first, second]
        )


async def test_replay_takes_the_time_the_live_run_did(
    inventory: Inventory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = Recording(tmp_path / "recording.json")
    tools = ReviewTools(inventory, SeenInvoices(), INV_1001)
    report = ValidationReport()
    await RecordingApprover(SlowApprover(), recording).draft(
        INV_1001, report, [], tools
    )
    [entry] = recording.entries.values()
    first, second = entry["step_at"]
    assert 0.04 <= first < second <= entry["seconds"]

    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(replay.asyncio, "sleep", sleep)
    seen: list[str] = []
    live = ReviewTools(
        inventory, SeenInvoices(), INV_1001, on_step=lambda s: seen.append(s.name)
    )
    await ReplayApprover(recording, pace=1.0).draft(INV_1001, report, [], live)
    # Each tool call waits for its own moment, then the rest of the call.
    assert waits == pytest.approx([first, second - first, entry["seconds"] - second])
    assert seen == ["check_inventory", "check_inventory"]

    waits.clear()
    await ReplayApprover(recording, pace=0.0).draft(INV_1001, report, [], tools)
    assert waits == []  # the eval's pace: at once


# Every case the eval scores must replay from the recording, not fall back to
# the rule-based agents: a reviewer without a key sees the real agents. Scans
# need the OCR extra and are left out. When this fails, the agents' inputs
# changed; re-record with RECORD_TO=data/replay/recording.json on a live run.
RECORDED = [c for c in evaluation.cases_in("all") if c.folder != "scanned"]


@pytest.mark.parametrize("case", RECORDED, ids=lambda c: c.name)
async def test_the_recording_covers_every_eval_case(
    case: evaluation.Case, tmp_path: Path
) -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="replay",
        replay_pace=0.0,
    )
    inventory = Inventory(tmp_path / "inventory.db")
    inventory.seed()
    deps = Deps(agents=build_agents(settings), inventory=inventory, settings=settings)
    graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))
    for name in case.files:
        config: Any = {"configurable": {"thread_id": str(uuid.uuid4())}}
        path = settings.samples_dir.parent / case.folder / name
        state = await graph.ainvoke({"path": str(path)}, config)
        assert not any(replay.FALLBACK in t["message"] for t in state["trace"]), name
