"""The HTTP API, run in-process with the offline agents."""

import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from invoice_agents.api.app import app, lifespan
from invoice_agents.api.runs import Run, RunManager
from invoice_agents.config import Settings, get_settings
from tests.conftest import INVOICES


@pytest.fixture
def vp_tokens() -> str | None:
    """VP_TOKENS for the API under test; a test overrides it to require them."""
    return None


@pytest.fixture
async def client(tmp_path, monkeypatch, vp_tokens) -> AsyncIterator[AsyncClient]:
    if vp_tokens is None:
        monkeypatch.delenv("VP_TOKENS", raising=False)
    else:
        monkeypatch.setenv("VP_TOKENS", vp_tokens)
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "inventory.db"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setenv("DECISIONS_LOG", str(tmp_path / "decisions.jsonl"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SAMPLES_DIR", str(INVOICES))
    get_settings.cache_clear()
    async with (
        lifespan(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c
    get_settings.cache_clear()


def manager() -> RunManager:
    return app.state.manager


async def settle(run_id: str) -> None:
    await manager().get(run_id).settled.wait()


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.strip().split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads(data[0]))
    return events


async def test_health_and_samples(client):
    assert (await client.get("/api/health")).json()["llm_provider"] == "offline"
    names = [s["name"] for s in (await client.get("/api/samples")).json()]
    assert "invoice_1013.json" in names
    assert len(names) == 20


async def test_rejected_run_streams_its_history(client):
    run = (await client.post("/api/runs", json={"sample": "invoice_1007.csv"})).json()
    await settle(run["id"])

    detail = (await client.get(f"/api/runs/{run['id']}")).json()
    assert detail["status"] == "rejected"
    assert detail["invoice_number"] == "INV-1007"
    assert any(f["code"] == "TOTAL_MISMATCH" for f in detail["summary"]["findings"])

    resp = await client.get(f"/api/runs/{run['id']}/events")
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "started"
    assert types[-1] == "done"
    assert "progress" in types and "trace" in types
    assert [e["id"] for e in events] == list(range(len(events)))
    # Proxies must not cache or buffer the stream; each event names its type.
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    first = resp.text.split("\n\n")[0].splitlines()
    assert "event: started" in first and "id: 0" in first


async def test_events_for_an_unknown_run_are_a_404_not_an_empty_stream(client):
    resp = await client.get("/api/runs/nope/events")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "No run 'nope'"


async def test_last_event_id_skips_what_the_client_has(client):
    run = (await client.post("/api/runs", json={"sample": "invoice_1015.csv"})).json()
    await settle(run["id"])
    total = len(manager().get(run["id"]).events)

    resp = await client.get(
        f"/api/runs/{run['id']}/events", headers={"Last-Event-ID": "2"}
    )
    events = parse_sse(resp.text)
    assert events[0]["id"] == 3
    assert len(events) == total - 3


async def test_vp_pause_and_resume(client):
    run = (await client.post("/api/runs", json={"sample": "invoice_1014.xml"})).json()
    await settle(run["id"])
    assert manager().get(run["id"]).status == "awaiting_vp"

    seen = []
    async for event in manager().events(run["id"]):
        seen.append(event["type"])
        if event["type"] == "awaiting_vp":
            assert event["request"]["invoice"]["currency"] == "EUR"
            break
    assert "done" not in seen

    resp = await client.post(
        f"/api/runs/{run['id']}/decision", json={"approved": True, "note": "FX ok"}
    )
    assert resp.status_code == 202
    await settle(run["id"])

    detail = (await client.get(f"/api/runs/{run['id']}")).json()
    assert detail["status"] == "paid"
    assert detail["summary"]["payment"]["currency"] == "EUR"
    types = [e["type"] for e in detail["events"]]
    assert types.index("awaiting_vp") < types.index("decision") < types.index("done")


async def test_decision_needs_a_paused_run(client):
    run = (await client.post("/api/runs", json={"sample": "invoice_1015.csv"})).json()
    await settle(run["id"])
    resp = await client.post(f"/api/runs/{run['id']}/decision", json={"approved": True})
    assert resp.status_code == 409


async def test_unknown_run_and_sample(client):
    assert (await client.get("/api/runs/nope")).status_code == 404
    assert (await client.get("/api/runs/nope/events")).status_code == 404
    for name in ["missing.json", "../pyproject.toml", "conftest.py"]:
        resp = await client.post("/api/runs", json={"sample": name})
        assert resp.status_code == 404


async def test_free_text_offline_fails_with_a_reason(client):
    run = (await client.post("/api/runs", json={"sample": "invoice_1001.txt"})).json()
    await settle(run["id"])
    events = manager().get(run["id"]).events
    assert manager().get(run["id"]).status == "failed"
    assert "needs an LLM" in events[-1]["message"]
    # "error" is EventSource's reserved name for connection errors.
    assert events[-1]["type"] == "run_error"


async def test_upload(client):
    content = (INVOICES / "invoice_1016.json").read_bytes()
    resp = await client.post(
        "/api/runs/upload", files={"file": ("my invoice.json", content)}
    )
    assert resp.status_code == 202
    run = resp.json()
    assert run["source"] == "my invoice.json"
    await settle(run["id"])
    assert manager().get(run["id"]).status == "rejected"

    bad = await client.post("/api/runs/upload", files={"file": ("x.exe", b"MZ")})
    assert bad.status_code == 415


async def test_a_scan_without_ocr_fails_with_the_install_hint(
    client, monkeypatch, caplog
):
    from invoice_agents import graph
    from invoice_agents.ingestion.ocr import OcrUnavailableError

    def no_ocr(path):
        raise OcrUnavailableError("needs OCR: uv sync --extra ocr")

    monkeypatch.setitem(graph.OCR_BACKENDS, "docling", no_ocr)
    resp = await client.post("/api/runs/upload", files={"file": ("scan.png", b"png")})
    await settle(resp.json()["id"])
    run = manager().get(resp.json()["id"])
    assert run.status == "failed"
    [error] = [e for e in run.events if e["type"] == "run_error"]
    assert error["message"] == "needs OCR: uv sync --extra ocr"
    # An expected condition, not a crash: no stack trace in the log.
    assert not [r for r in caplog.records if r.exc_info]


async def test_batch_runs_in_order(client):
    runs = (await client.post("/api/runs/batch")).json()
    assert len(runs) == 20
    for run in runs:
        await settle(run["id"])

    status = {r["source"]: manager().get(r["id"]).status for r in runs}
    assert status["invoice_1004.json"] == "paid"
    # The revision arrives after the original was paid: paying it as well
    # would pay twice, so it is rejected with the reason.
    revised = next(r for r in runs if r["source"] == "invoice_1004_revised.json")
    assert status["invoice_1004_revised.json"] == "rejected"
    summary = manager().get(revised["id"]).summary
    assert summary is not None
    assert "ALREADY_PAID" in {f["code"] for f in summary["findings"]}
    assert status["invoice_1007.csv"] == "rejected"
    assert status["invoice_1001.txt"] == "failed"
    listed = (await client.get("/api/runs")).json()
    assert len(listed) == 20


async def test_global_stream_carries_every_run_in_order(client):
    a = (await client.post("/api/runs", json={"sample": "invoice_1015.csv"})).json()
    b = (await client.post("/api/runs", json={"sample": "invoice_1014.xml"})).json()
    await settle(a["id"])
    await settle(b["id"])

    seen = []
    async for event in manager().all_events():
        seen.append(event)
        if len(seen) == len(manager().log):
            break
    assert [e["seq"] for e in seen] == list(range(len(seen)))
    assert {e["run_id"] for e in seen} == {a["id"], b["id"]}
    # Each event carries its own time, so the console can time each step.
    assert all(e["at"] >= e["created_at"] for e in seen)
    paused = next(e for e in seen if e["type"] == "awaiting_vp")
    assert paused["status"] == "awaiting_vp"
    assert paused["summary"]["invoice"]["invoice_number"] == "INV-1014"
    done = next(e for e in seen if e["type"] == "done")
    assert done["status"] == "paid"
    # Progress steps must not overwrite the run's status.
    assert all(e["status"] == "running" for e in seen if e["type"] == "progress")
    assert {e["step"] for e in seen if e["type"] == "progress"} >= {"start"}


async def test_open_decisions_say_so(client):
    assert (await client.get("/api/health")).json()["vp_auth"] is False


@pytest.mark.parametrize(
    "vp_tokens", ['{"Dana Whitfield": "s3cret", "Lee Park": "other"}']
)
async def test_only_a_vp_token_can_decide(client):
    assert (await client.get("/api/health")).json()["vp_auth"] is True
    run = (await client.post("/api/runs", json={"sample": "invoice_1014.xml"})).json()
    await settle(run["id"])
    url = f"/api/runs/{run['id']}/decision"
    # The client may claim to be anyone; the token decides who it is.
    body = {"approved": False, "note": "Wrong rate", "by": "The CEO"}

    missing = await client.post(url, json=body)
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    for bad in ("Bearer nope", "Basic s3cret", "s3cret", "Bearer "):
        assert (
            await client.post(url, json=body, headers={"Authorization": bad})
        ).status_code == 401
    # Nothing was decided by the failed attempts.
    assert manager().get(run["id"]).status == "awaiting_vp"

    ok = await client.post(url, json=body, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 202
    await settle(run["id"])
    detail = (await client.get(f"/api/runs/{run['id']}")).json()
    assert detail["status"] == "rejected"
    messages = [t["message"] for t in detail["summary"]["trace"]]
    assert "VP rejected (Dana Whitfield). Note: Wrong rate" in messages
    assert detail["summary"]["reason"].startswith(
        "Rejected by the VP (Dana Whitfield)."
    )
    decision = next(e for e in detail["events"] if e["type"] == "decision")
    assert decision["by"] == "Dana Whitfield"


async def test_a_tool_call_reaches_the_stream_once_before_the_draft(
    inventory, tmp_path
):
    """Tool calls are sent as they finish, not with the draft, and only once."""
    from langgraph.checkpoint.memory import InMemorySaver

    from invoice_agents.graph import Deps, build_graph, checkpoint_serde
    from invoice_agents.llm import Agents
    from tests.test_graph import (
        INV_1001,
        RecordingCritic,
        StubExtractor,
        ToolUsingApprover,
    )

    agents = Agents(StubExtractor(INV_1001), ToolUsingApprover(), RecordingCritic())
    settings = Settings(_env_file=None, llm_provider="offline")  # pyright: ignore[reportCallIssue]
    deps = Deps(agents=agents, inventory=inventory, settings=settings)
    graph = build_graph(deps, checkpointer=InMemorySaver(serde=checkpoint_serde()))
    runs = RunManager(graph)
    run = runs.start(INVOICES / "invoice_1001.txt", "invoice_1001.txt")
    await run.settled.wait()

    lines = [e["message"] for e in run.events if e["type"] == "trace"]
    tool = [m for m in lines if m.startswith("check_inventory")]
    assert len(tool) == 1
    assert lines.index(tool[0]) < next(
        i for i, m in enumerate(lines) if m.startswith("Draft 1")
    )


async def test_a_run_waiting_for_a_vp_survives_a_restart(client, tmp_path):
    # Before: after a restart GET /api/runs was empty and the decision 404'd.
    run = (await client.post("/api/runs", json={"sample": "invoice_1014.xml"})).json()
    await settle(run["id"])
    paid = (await client.post("/api/runs", json={"sample": "invoice_1015.csv"})).json()
    await settle(paid["id"])
    assert manager().get(run["id"]).status == "awaiting_vp"

    async with lifespan(app):  # a new process on the same files
        listed = (await client.get("/api/runs")).json()
        assert [r["id"] for r in listed] == [run["id"]]
        restored = manager().get(run["id"])
        assert restored.status == "awaiting_vp"
        assert restored.source == "invoice_1014.xml"
        kinds = [e["type"] for e in restored.events]
        assert kinds[0] == "started" and kinds[-1] == "awaiting_vp"
        nodes = [e["node"] for e in restored.events if e["type"] == "trace"]
        assert nodes[:2] == ["ingest", "validate"] and "review" in nodes

        resp = await client.post(
            f"/api/runs/{run['id']}/decision", json={"approved": True}
        )
        assert resp.status_code == 202
        await settle(run["id"])
        assert manager().get(run["id"]).status == "paid"


async def test_reset_forgets_runs_and_received_invoices(client):
    run = (await client.post("/api/runs", json={"sample": "invoice_1015.csv"})).json()
    await settle(run["id"])
    assert (await client.post("/api/reset")).status_code == 204
    assert (await client.get("/api/runs")).json() == []
    # Every open console hears it on the event stream.
    assert manager().log[-1]["type"] == "reset"

    # The same invoice again is not a duplicate any more, and the bank has
    # forgotten its payment key too, so it really pays.
    again = (await client.post("/api/runs", json={"sample": "invoice_1015.csv"})).json()
    await settle(again["id"])
    assert manager().get(again["id"]).status == "paid"
    summary = manager().get(again["id"]).summary
    assert summary is not None and summary["payment"]["duplicate"] is False


async def test_reset_waits_for_running_work(client):
    manager().runs["busy"] = Run(id="busy", source="x", path=Path("x"), created_at="")
    assert (await client.post("/api/reset")).status_code == 409


async def test_inventory_comes_from_the_database(client, tmp_path):
    with sqlite3.connect(tmp_path / "inventory.db") as conn:
        conn.execute("UPDATE inventory SET stock = 3 WHERE item = 'WidgetA'")
    rows = {r["item"]: r for r in (await client.get("/api/inventory")).json()}
    assert rows["WidgetA"]["stock"] == 3
    assert rows["GadgetX"]["unit_price"] == 750


async def test_a_rejection_needs_a_reason(client):
    run = (await client.post("/api/runs", json={"sample": "invoice_1014.xml"})).json()
    await settle(run["id"])
    resp = await client.post(
        f"/api/runs/{run['id']}/decision", json={"approved": False, "note": "  "}
    )
    assert resp.status_code == 422
    assert "needs a reason" in resp.json()["detail"]
    assert manager().get(run["id"]).status == "awaiting_vp"
