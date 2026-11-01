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
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from invoice_agents import memory
from invoice_agents.config import get_settings

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

    def _vp_auth(self) -> dict[str, str] | None:
        # A VP's token, when the API requires one (VP_TOKENS on the server),
        # from the environment or .env like every other setting.
        token = get_settings().vp_token
        return (
            {"Authorization": f"Bearer {token.get_secret_value()}"} if token else None
        )

    async def decide(self, run_id: str, approved: bool, note: str) -> None:
        r = await self.client.post(
            f"{self.base}/api/runs/{run_id}/decision",
            json={"approved": approved, "note": note},
            headers=self._vp_auth(),
            timeout=30,
        )
        r.raise_for_status()

    async def reset(self) -> str | None:
        """Forget every run and received invoice. The reason if refused."""
        r = await self.client.post(
            f"{self.base}/api/reset", headers=self._vp_auth(), timeout=30
        )
        if r.status_code in (401, 403, 409):
            return str(r.json().get("detail", r.text))
        r.raise_for_status()
        return None

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
        cmd,
        cwd=cwd,
        stdout=log.open("wb"),
        stderr=subprocess.STDOUT,
        # Not the terminal: Vite reads keyboard shortcuts from stdin, and two
        # readers on one terminal split the keys (and the bytes of a CJK
        # character) between them, so typing in the console went missing.
        stdin=subprocess.DEVNULL,
        # Own session: Ctrl-C in the console stops a run, not the servers;
        # `stop` shuts them down when the console exits.
        start_new_session=True,
    )


def _port(url: str) -> str:
    return url.rsplit(":", 1)[-1].split("/")[0]


async def serve(
    api_url: str, web_url: str, log_dir: Path
) -> list[subprocess.Popen[bytes]]:
    """Start the API and the console unless they already run; wait until both answer."""
    started: list[subprocess.Popen[bytes]] = []
    # Redis first: the API decides at startup whether the assistant's
    # conversation is shared with the terminal.
    redis_server = await asyncio.to_thread(_start_redis, log_dir)
    if redis_server is not None:
        started.append(redis_server)
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


def _start_redis(log_dir: Path) -> subprocess.Popen[bytes] | None:
    """Start a local Redis for the shared chat memory, if one is configured on
    this machine, isn't running, and redis-server is installed. In memory
    only: the conversation lasts as long as the servers do."""
    url = get_settings().redis_url
    parsed = urlparse(url)
    server = shutil.which("redis-server")
    local = parsed.hostname in ("localhost", "127.0.0.1")
    if not url or not local or server is None or memory.redis_up(url):
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [server, "--port", str(parsed.port or 6379), "--save", ""]
    cmd += ["--appendonly", "no", "--dir", str(log_dir.resolve())]
    process = _spawn(cmd, REPO, log_dir / "redis.log")
    for _ in range(40):
        if memory.redis_up(url):
            return process
        time.sleep(0.05)
    stop([process])
    return None


def stop(processes: list[subprocess.Popen[bytes]]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
