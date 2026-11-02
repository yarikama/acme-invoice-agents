"""Reading text off scanned invoices and photos, for the extraction agent.

Two backends, chosen with `OCR_BACKEND`. Both read the characters with
Tesseract, which must be on the machine (`brew install tesseract`,
`apt-get install tesseract-ocr`; the Docker image has it):

- `pymupdf` (default): PyMuPDF renders each page and Tesseract reads it whole,
  one line at a time, so a table row stays one line. Under a second a page,
  nothing to download.
- `docling`: Docling's layout model finds the page's blocks and tables first
  and returns Markdown, with tables as tables. Better on multi-column layouts;
  on the sample scans (slightly skewed tables) its table model drops cells
  that the whole-page read keeps. Seconds a page on CPU; the layout model
  downloads once from Hugging Face. Goes through langchain-docling.

A backend is one function from a path to text, so another engine (MinerU, a
cloud OCR API) can be added beside these without touching the callers.
"""

from __future__ import annotations

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


def _cached(backend: str, path: Path, read: Callable[[], str]) -> str:
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
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrMode,
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )
    from langchain_docling import DoclingLoader

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

    return _cached("docling", path, read)


def pymupdf_text(path: Path) -> str:
    """The document's text, page by page: PyMuPDF renders each page (a photo
    is one page) at 300 dpi and Tesseract reads it."""
    _need_tesseract()
    import pymupdf

    def read() -> str:
        pages = []
        with pymupdf.open(path) as doc:
            for page in doc:
                png = page.get_pixmap(dpi=300).tobytes("png")
                out = subprocess.run(
                    ["tesseract", "stdin", "stdout", "--psm", str(PSM), "-l", "eng"],
                    input=png,
                    capture_output=True,
                    check=True,
                    timeout=120,
                )
                pages.append(out.stdout.decode())
        return "\n\n".join(pages)

    return _cached("pymupdf", path, read)


BACKENDS: dict[str, Ocr] = {"docling": docling_text, "pymupdf": pymupdf_text}
