"""The HTTP API, run in-process with the offline agents."""

import json
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from invoice_agents.api.app import app, lifespan
from invoice_agents.api.runs import RunManager
from invoice_agents.config import get_settings
from tests.conftest import INVOICES


@pytest.fixture
async def client(tmp_path, monkeypatch) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "inventory.db"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
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


async def test_batch_runs_in_order(client):
    runs = (await client.post("/api/runs/batch")).json()
    assert len(runs) == 20
    for run in runs:
        await settle(run["id"])

    status = {r["source"]: manager().get(r["id"]).status for r in runs}
    assert status["invoice_1004.json"] == "paid"
    # The revision arrives after the original, so it is flagged for the VP.
    assert status["invoice_1004_revised.json"] == "awaiting_vp"
    assert status["invoice_1007.csv"] == "rejected"
    assert status["invoice_1001.txt"] == "failed"
    listed = (await client.get("/api/runs")).json()
    assert len(listed) == 20
