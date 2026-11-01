"""Make the scanned test invoices: INV-1011 as an image with no text layer.

Run with the OCR extra installed (it brings pypdfium2 and Pillow):
    uv run --extra ocr python data/make_scans.py
"""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import ImageFilter

HERE = Path(__file__).parent

page = pdfium.PdfDocument(HERE / "invoices" / "invoice_1011.pdf")[0]
img = page.render(scale=2).to_pil().convert("L")
# A light blur and a slight tilt, like a desk scanner.
img = img.filter(ImageFilter.GaussianBlur(0.6)).rotate(0.8, expand=True, fillcolor=255)
out = HERE / "scanned"
out.mkdir(exist_ok=True)
img.save(out / "invoice_1011_scan.png")
img.convert("RGB").save(out / "invoice_1011_scan.pdf", "PDF", resolution=144)
