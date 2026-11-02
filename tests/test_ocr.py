"""Scanned PDFs and photos go to OCR; text PDFs don't. No OCR model runs here."""

import asyncio
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


@pytest.mark.parametrize("backend", ["pymupdf", "docling"])
def test_without_tesseract_the_error_says_how_to_install(
    backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ocr_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ocr_module, "_read", {})
    ocr_module._docling_loader.cache_clear()
    scan = tmp_path / "scan.png"
    scan.write_bytes((SCANNED / "invoice_1011_scan.png").read_bytes())
    try:
        with pytest.raises(OcrUnavailableError, match="brew install tesseract"):
            load_document(scan, ocr_module.BACKENDS[backend])
    finally:
        ocr_module._docling_loader.cache_clear()


def test_pymupdf_renders_each_page_and_tesseract_reads_it_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)
        assert kwargs["input"].startswith(b"\x89PNG")

        class Done:
            stdout = b"WidgetA 6 $250.00 $1,500.00\n"

        return Done()

    monkeypatch.setattr(ocr_module.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(ocr_module.subprocess, "run", fake_run)
    monkeypatch.setattr(ocr_module, "_read", {})
    text = ocr_module.pymupdf_text(SCANNED / "invoice_1011_scan.pdf")
    assert text == "WidgetA 6 $250.00 $1,500.00"
    assert calls == [["tesseract", "stdin", "stdout", "--psm", "4", "-l", "eng"]]
    # Read once: a retry gets the same text without running OCR again.
    ocr_module.pymupdf_text(SCANNED / "invoice_1011_scan.pdf")
    assert len(calls) == 1


def test_docling_text_goes_through_the_langchain_loader_once_per_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    monkeypatch.setattr(ocr_module, "_read", {})
    scan = tmp_path / "a.png"
    scan.write_bytes(b"one")
    assert ocr_module.docling_text(scan) == "# INVOICE\n\nINV-1011"
    assert made == [
        {"path": str(scan), "converter": converter, "export_type": "markdown"}
    ]
    # An extraction retry reads the same file again: no second OCR pass.
    ocr_module.docling_text(scan)
    assert len(made) == 1
    # A changed file is read again.
    scan.write_bytes(b"two, longer")
    ocr_module.docling_text(scan)
    assert len(made) == 2


async def test_the_graph_reads_files_off_the_event_loop(
    inventory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading (with OCR) runs in a worker thread. Validation stays on the loop,
    in one step with recording the invoice, so overlapping runs see each other."""
    from invoice_agents import graph
    from tests.test_graph import config, make_graph

    seen: list[str] = []
    real = asyncio.to_thread

    async def spy(fn: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(fn.__name__)
        return await real(fn, *args, **kwargs)

    monkeypatch.setattr(graph.asyncio, "to_thread", spy)
    monkeypatch.setitem(graph.OCR_BACKENDS, "pymupdf", fake_ocr)
    state = await make_graph(inventory).ainvoke(
        {"path": str(SCANNED / "invoice_1011_scan.png")}, config()
    )
    assert "load_document" in seen and "validate" not in seen
    assert "Read the scan with OCR (pymupdf)" in state["trace"][0]["message"]
