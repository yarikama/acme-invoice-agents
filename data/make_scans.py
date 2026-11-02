"""Make the scanned test invoices: INV-1011 as an image with no text layer.

Run:
    uv run python data/make_scans.py
"""

from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter

HERE = Path(__file__).parent

with pymupdf.open(HERE / "invoices" / "invoice_1011.pdf") as doc:
    pix = doc[0].get_pixmap(dpi=144)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
# A light blur and a slight tilt, like a desk scanner.
img = img.filter(ImageFilter.GaussianBlur(0.6)).rotate(0.8, expand=True, fillcolor=255)
out = HERE / "scanned"
out.mkdir(exist_ok=True)
img.save(out / "invoice_1011_scan.png")
img.convert("RGB").save(out / "invoice_1011_scan.pdf", "PDF", resolution=144)
