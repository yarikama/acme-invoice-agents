"""Talking to the API from the CLI, and starting it (with the console) on demand.

When the API is up, the CLI runs invoices through it rather than in-process,
so every run shows in the web console too.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]


async def reachable(url: str, client: httpx.AsyncClient | None = None) -> bool:
    try:
        if client is not None:
            return (await client.get(url, timeout=1.0)).status_code < 500
        async with httpx.AsyncClient() as c:
            return (await c.get(url, timeout=1.0)).status_code < 500
    except httpx.HTTPError:
        return False


def seconds(iso: str | None) -> float | None:
    """An ISO 8601 time as epoch seconds; None if missing or unreadable."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


class Api:
    """The few endpoints the CLI needs."""

    def __init__(self, base: str, client: httpx.AsyncClient) -> None:
        self.base = base.rstrip("/")
        self.client = client

    async def start(self, path: Path) -> str:
        r = await self.client.post(
            f"{self.base}/api/runs/upload",
            files={"file": (path.name, await asyncio.to_thread(path.read_bytes))},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]

    async def events(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """The run's events, replayed then live, until the run settles."""
        async with self.client.stream(
            "GET", f"{self.base}/api/runs/{run_id}/events", timeout=None
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue  # ids, event names and keepalive comments
                event = json.loads(line[6:])
                if "at" in event:
                    event["at"] = seconds(event["at"])
                yield event
                if event["type"] in ("done", "run_error"):
                    return

    async def decide(self, run_id: str, approved: bool, note: str) -> None:
        r = await self.client.post(
            f"{self.base}/api/runs/{run_id}/decision",
            json={"approved": approved, "note": note},
            timeout=30,
        )
        r.raise_for_status()

    async def result(self, run_id: str) -> dict[str, Any]:
        r = await self.client.get(f"{self.base}/api/runs/{run_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        return {
            "run_id": run_id,
            **(data.get("summary") or {}),
            "status": data["status"],
            "reason": (data.get("summary") or {}).get("reason") or data.get("error"),
        }


def _spawn(cmd: list[str], cwd: Path, log: Path) -> subprocess.Popen[bytes]:
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        cmd, cwd=cwd, stdout=log.open("wb"), stderr=subprocess.STDOUT
    )


def _port(url: str) -> str:
    return url.rsplit(":", 1)[-1].split("/")[0]


async def serve(
    api_url: str, web_url: str, log_dir: Path
) -> list[subprocess.Popen[bytes]]:
    """Start the API and the console unless they already run; wait until both answer."""
    started: list[subprocess.Popen[bytes]] = []
    if not await reachable(f"{api_url}/api/health"):
        cmd = [sys.executable, "-m", "uvicorn", "invoice_agents.api.app:app"]
        cmd += ["--port", _port(api_url)]
        started.append(await asyncio.to_thread(_spawn, cmd, REPO, log_dir / "api.log"))
    if not await reachable(web_url):
        npm = shutil.which("npm")
        if npm is None or not (REPO / "web" / "node_modules").is_dir():
            stop(started)
            raise RuntimeError(
                "The console needs Node and its packages: run `make install` first."
            )
        cmd = [npm, "run", "dev", "--", "--port", _port(web_url), "--strictPort"]
        started.append(
            await asyncio.to_thread(_spawn, cmd, REPO / "web", log_dir / "web.log")
        )
    for _ in range(120):
        if await reachable(f"{api_url}/api/health") and await reachable(web_url):
            return started
        if any(p.poll() is not None for p in started):
            break
        await asyncio.sleep(0.25)
    stop(started)
    raise RuntimeError(f"The API or the console did not start; see {log_dir}.")


def stop(processes: list[subprocess.Popen[bytes]]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
