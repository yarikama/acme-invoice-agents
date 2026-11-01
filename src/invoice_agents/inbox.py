"""A watched folder standing in for the AP mailbox.

The brief's invoices arrive by email. Here they arrive in a folder: whatever
lands in it is processed and then filed under processed/, the way a mailbox
rule files a handled message, so each file is processed once.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

DONE = "processed"
# A file changed this recently may still be being written (a copy or a
# download in progress), so it waits for the next look.
SETTLE_SECONDS = 1.0


class Inbox:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.done = folder / DONE
        self.done.mkdir(parents=True, exist_ok=True)

    def arrived(self, now: float | None = None) -> list[Path]:
        """Files ready to process, oldest first."""
        now = time.time() if now is None else now
        files = [
            p
            for p in self.folder.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and now - p.stat().st_mtime >= SETTLE_SECONDS
        ]
        return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))

    def file_away(self, path: Path) -> Path:
        """Move a processed file under processed/, never over another one."""
        target = self.done / path.name
        if target.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            target = self.done / f"{stamp}-{path.name}"
        return path.rename(target)


async def watch(
    inbox: Inbox,
    handle: Callable[[list[Path]], Awaitable[None]],
    interval: float = 2.0,
    stop: asyncio.Event | None = None,
) -> None:
    """Look every `interval` seconds; hand over what arrived, then file it.

    Runs until `stop` is set (or forever). A file is filed even when its run
    failed: the result is in the ledger, and a retry would fail the same way.
    One interrupted mid-batch (Ctrl-C) stays, to be processed next time.
    """
    stop = stop or asyncio.Event()
    while not stop.is_set():
        batch = inbox.arrived()
        if batch:
            await handle(batch)
            for path in batch:
                if path.exists():
                    inbox.file_away(path)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)
