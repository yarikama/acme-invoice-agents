"""python main.py --invoice_path=data/invoices/invoice_1001.txt"""

import os
import sys
from pathlib import Path

# The brief runs `python main.py`, often from a shell where the project's
# virtualenv isn't active. Rather than fail on the import, run again with the
# virtualenv's Python that `make install` (uv sync) created.
VENV = Path(__file__).resolve().parent / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
AGAIN = "ACME_INVOICE_AGENTS_REEXEC"

try:
    from invoice_agents.cli import main
except ModuleNotFoundError:
    if PYTHON.exists() and not os.environ.get(AGAIN):
        os.environ[AGAIN] = "1"
        os.execv(PYTHON, [str(PYTHON), *sys.argv])
    sys.exit("Dependencies are missing: run `make install` (or `uv sync`) first.")

if __name__ == "__main__":
    sys.exit(main())
