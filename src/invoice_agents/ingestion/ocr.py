"""Reading text off scanned invoices and photos, for the extraction agent.

Two backends, chosen with `OCR_BACKEND`. Both read the characters with
Tesseract, which must be on the machine (`brew install tesseract`,
`apt-get install tesseract-ocr`; the Docker image has it):

- `pymupdf` (default): PyMuPDF renders each page and Tesseract reads it whole,
  one line at a time, so a table row stays one line. Under a second a page,
  nothing to download.
- `docling` (optional: `uv sync --extra docling`, about 1 GB with PyTorch):
  Docling's layout model finds the page's blocks and tables first and returns
  Markdown, with tables as tables. Better on multi-column layouts; on the
  sample scans (slightly skewed tables) its table model drops cells that the
  whole-page read keeps. Seconds a page on CPU; the layout model downloads
  once from Hugging Face. Goes through langchain-docling.

A third parser, `grok`, has Grok read the page image itself (see
agents.LLMVisionReader): no Tesseract, best on photos and poor scans, but it
needs an API key and costs a model call.

A backend is one function from a path to text, so another engine (MinerU, a
cloud OCR API) can be added beside these without touching the callers.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any


class OcrUnavailableError(RuntimeError):
    """The file needs OCR, and the OCR engine can't run here."""


Ocr = Callable[[Path], str]

NO_TESSERACT = (
    "This invoice is a scanned image, so reading it needs OCR, which uses"
    " Tesseract. Install it with `brew install tesseract` (macOS) or"
    " `apt-get install tesseract-ocr` (Debian, Ubuntu)."
)


NO_DOCLING = (
    "The docling parser is optional and not installed: run"
    " `uv sync --extra docling` (about 1 GB, with PyTorch), or pick the"
    " pymupdf or grok parser."
)


def docling_installed() -> bool:
    return importlib.util.find_spec("docling") is not None


def available() -> list[str]:
    """The parsers this install can run; grok also needs a model, checked
    when a run uses it."""
    return [p for p in PARSERS if p != "docling" or docling_installed()]


def _need_tesseract() -> None:
    if shutil.which("tesseract") is None:
        raise OcrUnavailableError(NO_TESSERACT)


# Tesseract's page segmentation: 4 reads the page as one column of lines of
# varying size, which keeps an invoice's table rows whole (item, qty, price on
# one line). The default (3) splits the table into columns and loses the rows.
PSM = 4


# One conversion at a time: the models are large, and runs share them.
_lock = threading.Lock()


# Text already read, by (backend, path, size, mtime): an extraction retry
# reads the same file again, and OCR takes seconds a page.
_read: dict[tuple[str, str, int, int], str] = {}


def cached(backend: str, path: Path, read: Callable[[], str]) -> str:
    """The file's text from this backend, read once per file version."""
    stat = path.stat()
    key = (backend, str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _lock:
        if key not in _read:
            _read[key] = read().strip()
        return _read[key]


@cache
def _docling_loader() -> tuple[Any, Any]:
    """LangChain's Docling loader, and one converter for every file, reading
    characters with Tesseract."""
    _need_tesseract()
    if not docling_installed():
        raise OcrUnavailableError(NO_DOCLING)
    # The optional extra; absent from a plain `uv sync`.
    from docling.datamodel.base_models import (  # pyright: ignore[reportMissingImports]
        InputFormat,
    )
    from docling.datamodel.pipeline_options import (  # pyright: ignore[reportMissingImports]
        OcrMode,
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import (  # pyright: ignore[reportMissingImports]
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )
    from langchain_docling import (  # pyright: ignore[reportMissingImports]
        DoclingLoader,
    )

    # Read the whole page at once: Tesseract on each table cell alone loses
    # short cells such as item names.
    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=TesseractCliOcrOptions(
            mode=OcrMode.FULL_PAGE, lang=["eng"], psm=PSM
        ),
    )
    # Loading the layout model takes seconds; do it once and hand the same
    # converter to each loader.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        }
    )
    return DoclingLoader, converter


def docling_text(path: Path) -> str:
    """The document's text as Markdown, laid out by Docling.

    The whole file is one Markdown document: an invoice is short, so there is
    nothing to chunk.
    """

    def read() -> str:
        loader_cls, converter = _docling_loader()
        loader = loader_cls(str(path), converter=converter, export_type="markdown")
        return "\n\n".join(d.page_content for d in loader.load())

    return cached("docling", path, read)


def page_images(path: Path, dpi: int = 300) -> list[bytes]:
    """Each page of a PDF, or a photo as one page, rendered to PNG."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return [page.get_pixmap(dpi=dpi).tobytes("png") for page in doc]


def pymupdf_text(path: Path) -> str:
    """The document's text, page by page: PyMuPDF renders each page (a photo
    is one page) at 300 dpi and Tesseract reads it."""
    _need_tesseract()

    def read() -> str:
        pages = []
        for png in page_images(path):
            out = subprocess.run(
                ["tesseract", "stdin", "stdout", "--psm", str(PSM), "-l", "eng"],
                input=png,
                capture_output=True,
                check=True,
                timeout=120,
            )
            pages.append(out.stdout.decode())
        return "\n\n".join(pages)

    return cached("pymupdf", path, read)


BACKENDS: dict[str, Ocr] = {"pymupdf": pymupdf_text, "docling": docling_text}

# Every parser a run can ask for; "grok" is built from the run's agents.
PARSERS = ("pymupdf", "docling", "grok")
