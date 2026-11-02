"""Benchmark the scan parsers on scans of varying quality.

Each sample PDF is rendered, then degraded the way real scans are: clean,
skewed, low resolution, a noisy fax, a phone photo. Every parser reads every
image, and its text is scored against the PDF's own text layer:

- key tokens: the tokens with a digit in them (amounts, quantities, dates,
  invoice numbers), read exactly. A wrong one can pay the wrong amount.
- all tokens: every word, read exactly.

Scoring is on the parser's text alone, before any agent, so it measures the
parser and nothing else. Reading time and, for Grok, model cost are measured
too.

    uv run python -m invoice_agents.parser_eval [--parsers pymupdf docling grok]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pdfplumber
import pymupdf
from PIL import Image, ImageDraw, ImageFilter
from rich.console import Console
from rich.table import Table

from invoice_agents.config import get_settings
from invoice_agents.ingestion import ocr
from invoice_agents.usage import collecting, cost

console = Console()

SOURCES = ("invoice_1011.pdf", "invoice_1012.pdf", "invoice_1013.pdf")


# --- The scans --------------------------------------------------------------


def _render(pdf: Path, dpi: int) -> Image.Image:
    with pymupdf.open(pdf) as doc:
        pix = doc[0].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _clean(pdf: Path, rng: random.Random) -> Image.Image:
    return _render(pdf, 300).convert("L")


def _skewed(pdf: Path, rng: random.Random) -> Image.Image:
    img = _render(pdf, 300).convert("L").filter(ImageFilter.GaussianBlur(0.8))
    return img.rotate(
        3.0, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC
    )


def _low_res(pdf: Path, rng: random.Random) -> Image.Image:
    return _render(pdf, 100).convert("L")


def _fax(pdf: Path, rng: random.Random) -> Image.Image:
    """Speckle, a little blur, then 1-bit: what a fax machine prints."""
    img = (
        _render(pdf, 200)
        .convert("L")
        .rotate(1.2, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC)
    )
    px = img.load()
    assert px is not None
    w, h = img.size
    for _ in range(w * h // 60):
        px[rng.randrange(w), rng.randrange(h)] = rng.choice((0, 255))
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    return img.point([255 if v > 150 else 0 for v in range(256)]).convert("L")


def _photo(pdf: Path, rng: random.Random) -> Image.Image:
    """A phone photo of the page: tilted, a shadow across it, warm light,
    soft focus and heavy JPEG compression."""
    img = _render(pdf, 200).convert("RGB")
    w, h = img.size
    img = img.transform(
        (w, h),
        Image.Transform.AFFINE,
        (1.0, 0.06, -0.03 * w, -0.025, 1.0, 0.02 * h),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(205, 200, 190),
    )
    shade = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(shade)
    for x in range(w):
        draw.line([(x, 0), (x, h)], fill=int(90 * x / w))
    img = Image.composite(Image.new("RGB", (w, h), (60, 55, 50)), img, shade)
    img = Image.blend(img, Image.new("RGB", (w, h), (255, 236, 200)), 0.12)
    return img.filter(ImageFilter.GaussianBlur(1.1))


CONDITIONS: dict[str, Callable[[Path, random.Random], Image.Image]] = {
    "clean": _clean,
    "skewed": _skewed,
    "low_res": _low_res,
    "fax": _fax,
    "photo": _photo,
}


def make_scans(sources: list[Path], folder: Path) -> list[tuple[Path, Path, str]]:
    """(scan, source PDF, condition) for every source and condition. The same
    seed every time, so every run scores the same images."""
    folder.mkdir(parents=True, exist_ok=True)
    scans = []
    for pdf in sources:
        for name, degrade in CONDITIONS.items():
            img = degrade(pdf, random.Random(f"{pdf.name}:{name}"))
            path = folder / f"{pdf.stem}_{name}.jpg"
            # The photo is saved as a phone would; the rest as scanners do.
            img.save(path, quality=40 if name == "photo" else 92)
            scans.append((path, pdf, name))
    return scans


# --- Scoring -----------------------------------------------------------------

_EDGE = re.compile(r"^[^\w$]+|[^\w%]+$")


def tokens(text: str) -> list[str]:
    """Words as printed, without the table rules and punctuation around them."""
    out = []
    for raw in text.replace("|", " ").split():
        token = _EDGE.sub("", raw)
        if token:
            out.append(token.lower())
    return out


def recall(truth: list[str], read: list[str]) -> float:
    """How much of the truth was read exactly, counting repeats."""
    if not truth:
        return 1.0
    have = Counter(read)
    found = sum(min(n, have[t]) for t, n in Counter(truth).items())
    return found / len(truth)


def _digits(token: str) -> str:
    """A numeric token with letter O as zero: the brief's sample PDFs print
    "2O26" and "$3,500.O0" on purpose, and a parser that reads a zero there
    reads what the number means."""
    return token.replace("o", "0") if any(c.isdigit() for c in token) else token


def score(truth_text: str, read_text: str) -> tuple[float, float]:
    """(key tokens, all tokens) recall."""
    truth = [_digits(t) for t in tokens(truth_text)]
    read = [_digits(t) for t in tokens(read_text)]
    key = [t for t in truth if any(c.isdigit() for c in t)]
    return recall(key, read), recall(truth, read)


def truth_of(pdf: Path) -> str:
    with pdfplumber.open(pdf) as doc:
        return "\n".join(page.extract_text() or "" for page in doc.pages)


# --- Running -----------------------------------------------------------------


@dataclass
class Result:
    parser: str
    source: str
    condition: str
    key: float | None
    words: float | None
    seconds: float
    cost_usd: float | None
    error: str | None = None


def grok_reader() -> ocr.Ocr:
    from invoice_agents.agents import LLMVisionReader
    from invoice_agents.llm import chat_model

    s = get_settings()
    if s.llm_provider not in ("xai", "openrouter"):
        raise SystemExit("The grok parser needs XAI_API_KEY or OPENROUTER_API_KEY.")
    llm = chat_model(s, s.vision_model, s.extract_reasoning)
    return LLMVisionReader(llm, s.vision_model).read


def read_one(
    parser: str, read: ocr.Ocr, scan: Path, pdf: Path, condition: str, pricing: dict
) -> Result:
    start = time.perf_counter()
    with collecting() as used:
        try:
            text = read(scan)
            error = None
        except Exception as exc:  # a parser that fails on an image scores 0
            text, error = "", f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    key, words = score(truth_of(pdf), text)
    return Result(
        parser,
        pdf.name,
        condition,
        round(key, 3),
        round(words, 3),
        round(seconds, 2),
        cost(dict(used), pricing) if used else None,
        error,
    )


def run(parsers: list[str], out: Path | None) -> list[Result]:
    s = get_settings()
    sources = [s.samples_dir / name for name in SOURCES]
    readers: dict[str, ocr.Ocr] = {
        name: ocr.BACKENDS[name] for name in parsers if name != "grok"
    }
    pricing: dict = {}
    if "grok" in parsers:
        from invoice_agents.llm import load_pricing

        readers["grok"] = grok_reader()
        pricing = load_pricing(s)
    results: list[Result] = []
    with TemporaryDirectory() as tmp:
        scans = make_scans(sources, Path(tmp))
        for name, read in readers.items():
            console.print(f"  {name}: reading {len(scans)} scans…")
            # Tesseract and Docling share the CPU (and Docling a lock), so
            # they read one at a time; Grok's calls can overlap.
            workers = 5 if name == "grok" else 1
            with ThreadPoolExecutor(workers) as pool:
                results += pool.map(
                    lambda job, n=name, r=read: read_one(
                        n, r, job[0], job[1], job[2], pricing
                    ),
                    scans,
                )
    report(results)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = out / f"parsers-{stamp}.json"
        path.write_text(json.dumps([asdict(r) for r in results], indent=1) + "\n")
        console.print(f"Saved {path}")
    return results


def summary(results: list[Result]) -> dict[str, dict[str, float | None]]:
    """Per parser: mean key and word recall, mean seconds, mean cost."""
    out: dict[str, dict[str, float | None]] = {}
    for name in dict.fromkeys(r.parser for r in results):
        mine = [r for r in results if r.parser == name]
        costs = [r.cost_usd for r in mine if r.cost_usd is not None]
        out[name] = {
            "key": sum(r.key or 0 for r in mine) / len(mine),
            "words": sum(r.words or 0 for r in mine) / len(mine),
            "seconds": sum(r.seconds for r in mine) / len(mine),
            "cost_usd": sum(costs) / len(costs) if costs else None,
        }
    return out


def report(results: list[Result]) -> None:
    table = Table(title="Key tokens read exactly (amounts, quantities, dates, numbers)")
    table.add_column("Parser")
    for condition in CONDITIONS:
        table.add_column(condition, justify="right")
    table.add_column("all words", justify="right")
    table.add_column("s / scan", justify="right")
    table.add_column("$ / scan", justify="right")
    for name, totals in summary(results).items():
        cells = []
        for condition in CONDITIONS:
            mine = [r for r in results if r.parser == name and r.condition == condition]
            cells.append(f"{sum(r.key or 0 for r in mine) / len(mine):.0%}")
        price = totals["cost_usd"]
        table.add_row(
            name,
            *cells,
            f"{totals['words'] or 0:.0%}",
            f"{totals['seconds'] or 0:.1f}",
            f"{price:.4f}" if price is not None else "0",
        )
    console.print(table)
    for r in results:
        if r.error:
            console.print(
                f"[red]ERROR[/] {r.parser} {r.source} {r.condition}: {r.error}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--parsers",
        nargs="+",
        choices=ocr.PARSERS,
        default=ocr.available(),
        help="Default: every parser installed (docling is an optional extra)",
    )
    parser.add_argument("--out", type=Path, default=Path("eval-results"))
    args = parser.parse_args(argv)
    if missing := [p for p in args.parsers if p not in ocr.available()]:
        raise SystemExit(f"{', '.join(missing)}: {ocr.NO_DOCLING}")
    run(args.parsers, args.out)


if __name__ == "__main__":
    main()
