"""Torgsoft VAT label recognition: entire label, never a substring of a product."""
import re

# Preserve the previously supported Cyrillic / mixed-script spelling and VAT.
_MARKER = r'(?:[нh][дd][сc]|vat)'
# OCR can insert spaces inside the rate (for example, "НДС 16 9%").
_LABEL = re.compile(_MARKER + r'(?:\s*\d(?:[\d\s]*\d)?\s*%)?', re.I)
# Below-grid rescue must still see an explicit percentage, as before.
_LABEL_WITH_RATE = re.compile(_MARKER + r'\s*\d{1,2}\s*%', re.I)


def vat_marker(text, *, require_rate=False):
    """Recognize only a complete tax label, optionally requiring a printed rate."""
    pattern = _LABEL_WITH_RATE if require_rate else _LABEL
    return pattern.fullmatch(str(text or '').strip()) is not None
