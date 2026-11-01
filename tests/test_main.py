"""`python main.py` works from a shell where the virtualenv isn't active."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if k != "ACME_INVOICE_AGENTS_REEXEC"}
    return subprocess.run(
        [*args, "main.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=clean | env,
        timeout=120,
    )


def test_without_the_packages_it_reruns_in_the_virtualenv() -> None:
    # -S hides site-packages, as a system Python without the deps would.
    done = run(sys.executable, "-S")
    assert done.returncode == 0, done.stderr
    assert "Process invoices end to end." in done.stdout


def test_it_says_what_to_do_when_there_is_no_virtualenv_to_use() -> None:
    done = run(sys.executable, "-S", ACME_INVOICE_AGENTS_REEXEC="1")
    assert done.returncode == 1
    assert "run `make install`" in done.stderr
