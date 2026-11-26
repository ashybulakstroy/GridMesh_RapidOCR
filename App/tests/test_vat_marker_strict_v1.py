from App.ocr.text_quality_gate import vat_marker as quality_vat_marker
from App.ocr.vat_marker_strict_v1 import vat_marker


def test_tax_labels_are_not_product_substrings():
    for text in ("Костюм Виндстопер осений", "Краска с НДС", "НДС 16% краска"):
        assert not vat_marker(text)
        assert not quality_vat_marker(text)
    for text in ("НДС", "НДС 16 %", "VAT 16%", "НДС 16 9%"):
        assert vat_marker(text)
        assert quality_vat_marker(text)


def test_below_grid_recovery_still_requires_a_short_rate():
    assert vat_marker("НДС 16 %", require_rate=True)
    assert not vat_marker("НДС", require_rate=True)
    assert not vat_marker("НДС 16 9%", require_rate=True)
