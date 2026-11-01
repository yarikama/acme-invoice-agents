"""Reading text off scanned invoices and photos, for the extraction agent.

Optional: `uv sync --extra ocr` installs Docling (MIT, runs on CPU, no
network after its models download once) and langchain-docling, its LangChain
document loader. The backend is one function, so a
heavier engine such as MinerU, stronger on complex layouts and best with a
GPU, can be added beside it without touching the callers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any


class OcrUnavailableError(RuntimeError):
    """The file needs OCR, and no OCR backend is installed."""


Ocr = Callable[[Path], str]


@cache
def _docling_loader() -> tuple[Any, Any]:
    """LangChain's Docling integration, and one converter for every file."""
    try:
        # Optional extra; absent from a plain `uv sync`.
        from docling.document_converter import (  # pyright: ignore[reportMissingImports]
            DocumentConverter,
        )
        from langchain_docling import (  # pyright: ignore[reportMissingImports]
            DoclingLoader,
        )
    except ImportError as exc:
        raise OcrUnavailableError(
            "This invoice is a scanned image, so reading it needs OCR."
            " Install it with `uv sync --extra ocr`."
        ) from exc
    # Loading the layout and OCR models takes seconds; do it once and hand the
    # same converter to each loader.
    return DoclingLoader, DocumentConverter()


# One conversion at a time: the models are large, and runs share them.
_lock = threading.Lock()


def docling_text(path: Path) -> str:
    """The document's text as Markdown, read by Docling's layout and OCR models.

    Goes through langchain-docling's DoclingLoader, exporting the whole file as
    one Markdown document: an invoice is short, so there is nothing to chunk.
    """
    loader_cls, converter = _docling_loader()
    loader = loader_cls(str(path), converter=converter, export_type="markdown")
    with _lock:
        docs = loader.load()
    return "\n\n".join(d.page_content for d in docs).strip()


BACKENDS: dict[str, Ocr] = {"docling": docling_text}
