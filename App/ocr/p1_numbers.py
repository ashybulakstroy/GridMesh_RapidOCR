"""P1: Strict parsing of photographed receipt numbers (never silently delete OCR errors)."""
from decimal import Decimal, InvalidOperation
import re

_CURRENCY = re.compile(r"\s*(?:₸|[ТT]|тг\.?|тенге)\s*$", re.IGNORECASE)
_SPACES = re.compile(r"[ \u00a0\u202f\u2009]")
_ASCII_GROUPED = re.compile(r"^\d{1,3}(?: \d{3})+(?:[.,]\d{1,3})?$")


def parse_number(value):
    """Return Decimal or None for corrupt/ambiguous OCR; no digit guessing.

    Accepted: 1250, 1250.00, 1 250,00, 1.250,00, 1,250.00,
              1.250.000 and 12,5. Ambiguous 1.234 is rejected, not guessed.
    """
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r'^₸\s*', '', s)
    s = _CURRENCY.sub('', s).strip()
    s = _SPACES.sub(' ', s)
    if not s or not re.fullmatch(r'[0-9., ]+', s):
        return None
    if ' ' in s:
        if not _ASCII_GROUPED.fullmatch(s):
            return None
        s = s.replace(' ', '')
    if ',' in s and '.' in s:
        dec = ',' if s.rfind(',') > s.rfind('.') else '.'
        group = '.' if dec == ',' else ','
        before, after = s.rsplit(dec, 1)
        if not (1 <= len(after) <= 3 and after.isdigit()):
            return None
        if not re.fullmatch(r'\d{1,3}(?:' + re.escape(group) + r'\d{3})+', before):
            return None
        s = before.replace(group, '') + '.' + after
    elif ',' in s or '.' in s:
        sep = ',' if ',' in s else '.'
        parts = s.split(sep)
        if any(not x.isdigit() for x in parts):
            return None
        if len(parts) > 2:
            if 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
                s = ''.join(parts)
            else:
                return None
        elif len(parts) == 2:
            a, b = parts
            if len(b) == 3 and 1 <= len(a) <= 3:
                return None  # 1.234 could mean 1234 or 1.234
            if not 1 <= len(b) <= 3:
                return None
            s = a + '.' + b
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def arithmetic_equal(left, right, tolerance=Decimal('0.01')):
    return left is not None and right is not None and abs(left - right) <= tolerance
