"""The parser benchmark's scans and scoring; no parser runs here."""

from pathlib import Path

from PIL import Image

from invoice_agents import parser_eval
from tests.conftest import INVOICES


def test_key_tokens_are_numbers_read_exactly() -> None:
    truth = "INV NO: INV 1012\nDATE: 26-Jan-2O26\nWidget A  12  $250  $3,000.00"
    # Letter O for zero is the same number; a dropped digit or a $ read as s is not.
    key, words = parser_eval.score(
        truth, "inv no: inv 1012 date: 26-jan-2026 widget a 12 $250 $3,000.00"
    )
    assert key == 1.0 and words == 1.0
    key, _ = parser_eval.score(truth, "INV 101 26-Jan-2026 Widget A 12 s250 $3,000.00")
    assert key == 3 / 5  # 1012 and $250 missed


def test_table_rules_and_punctuation_are_not_words() -> None:
    assert parser_eval.tokens("| WidgetA | 6 | $250.00, |") == [
        "widgeta",
        "6",
        "$250.00",
    ]
    assert parser_eval.recall(["6", "6"], ["6"]) == 0.5  # repeats count


def test_every_source_gets_every_condition_the_same_way(tmp_path: Path) -> None:
    pdf = INVOICES / "invoice_1013.pdf"
    first = parser_eval.make_scans([pdf], tmp_path / "a")
    assert [c for _, _, c in first] == list(parser_eval.CONDITIONS)
    second = parser_eval.make_scans([pdf], tmp_path / "b")
    for (a, _, _), (b, _, _) in zip(first, second, strict=True):
        assert (
            a.read_bytes() == b.read_bytes()
        )  # seeded: every run scores the same images
    assert Image.open(first[-1][0]).mode == "RGB"  # the photo is in colour
