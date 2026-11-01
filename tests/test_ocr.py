"""Scanned PDFs and photos go to OCR; text PDFs don't. No OCR model runs here."""

import asyncio
import builtins
from pathlib import Path
from typing import Any

import pytest

from invoice_agents.ingestion import load_document
from invoice_agents.ingestion import ocr as ocr_module
from invoice_agents.ingestion.ocr import OcrUnavailableError
from invoice_agents.models import SourceFormat

SCANNED = Path(__file__).parent.parent / "data" / "scanned"
INVOICES = Path(__file__).parent.parent / "data" / "invoices"


def fake_ocr(path: Path) -> str:
    return f"OCR text of {path.name}"


def test_a_scanned_pdf_goes_to_ocr() -> None:
    doc = load_document(SCANNED / "invoice_1011_scan.pdf", fake_ocr)
    assert doc.ocr and doc.text == "OCR text of invoice_1011_scan.pdf"
    assert doc.invoice is None  # the extraction agent still reads it


def test_an_image_goes_to_ocr() -> None:
    doc = load_document(SCANNED / "invoice_1011_scan.png", fake_ocr)
    assert doc.ocr and doc.format is SourceFormat.PNG


def test_a_text_pdf_skips_ocr() -> None:
    def boom(path: Path) -> str:
        raise AssertionError("OCR ran on a PDF with text")

    doc = load_document(INVOICES / "invoice_1011.pdf", boom)
    assert not doc.ocr and "INV-1011" in doc.text


def test_without_docling_the_error_says_how_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def no_docling(name: str, *args: Any, **kwargs: Any) -> Any:
        if "docling" in name:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    ocr_module._docling_loader.cache_clear()
    monkeypatch.setattr(builtins, "__import__", no_docling)
    try:
        with pytest.raises(OcrUnavailableError, match="uv sync --extra ocr"):
            load_document(SCANNED / "invoice_1011_scan.png")
    finally:
        ocr_module._docling_loader.cache_clear()


def test_docling_text_goes_through_the_langchain_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, Any]] = []
    converter = object()

    class Page:
        def __init__(self, text: str) -> None:
            self.page_content = text

    class Loader:
        def __init__(self, path: str, **kwargs: Any) -> None:
            made.append({"path": path, **kwargs})

        def load(self) -> list[Page]:
            return [Page("  # INVOICE"), Page("INV-1011  ")]

    monkeypatch.setattr(ocr_module, "_docling_loader", lambda: (Loader, converter))
    assert ocr_module.docling_text(Path("a.png")) == "# INVOICE\n\nINV-1011"
    assert made == [
        {"path": "a.png", "converter": converter, "export_type": "markdown"}
    ]


async def test_the_graph_reads_and_checks_off_the_event_loop(
    inventory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading (with OCR) and validating run in worker threads, not on the loop."""
    from invoice_agents import graph
    from tests.test_graph import config, make_graph

    seen: list[str] = []
    real = asyncio.to_thread

    async def spy(fn: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(fn.__name__)
        return await real(fn, *args, **kwargs)

    monkeypatch.setattr(graph.asyncio, "to_thread", spy)
    monkeypatch.setitem(graph.OCR_BACKENDS, "docling", fake_ocr)
    state = await make_graph(inventory).ainvoke(
        {"path": str(SCANNED / "invoice_1011_scan.png")}, config()
    )
    assert {"load_document", "validate"} <= set(seen)
    assert "Read the scan with OCR (docling)" in state["trace"][0]["message"]
