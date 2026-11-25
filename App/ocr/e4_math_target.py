"""E4 math-guided OCR *target selection*, without synthesizing OCR numbers.

An inferred number is used only to prioritize a Tesseract read. It is never
inserted into CandidateBook and cannot be applied without photographed OCR.
"""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
from decimal import Decimal
from .p1_numbers import parse_number


@_deep_timed("totals", "App/ocr/e4_math_target.py:_digits")
def _digits(value):
    return ''.join(c for c in str(value) if '0' <= c <= '9')


@_deep_timed("totals", "App/ocr/e4_math_target.py:_distance")
def _distance(a, b):
    """Levenshtein distance for two short numeric strings."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@_deep_timed("totals", "App/ocr/e4_math_target.py:select_math_targets")
def select_math_targets(rows, printed_total, book=None, *, max_cells=2):
    """Select probable OCR failure locations, NOT corrected values.

    Requires all numbers to parse, a printed total and a real sum mismatch.
    Ignores tax cells as targets in this fast pass; normal E4 covers those.
    A required replacement must differ by <= 1 (short) or <= 2 (long)
    digit edits from existing print recognition. No number is invented as OCR.
    """
    total = parse_number(printed_total)
    if total is None or total <= 0 or not rows or max_cells <= 0:
        return []
    numeric = []
    for row in rows:
        q = parse_number(row.get('cells', {}).get('quantity', {}).get('text'))
        p = parse_number(row.get('cells', {}).get('price', {}).get('text'))
        if q is None or p is None or q <= 0 or p <= 0:
            return []
        numeric.append((row, q, p))
    current = sum((q * p for _, q, p in numeric), Decimal(0))
    gap = total - current
    # Sub-tenge rounding and tiny VAT differences are not a reason to alter products.
    # Small discrepancies remain the responsibility of normal E4 recovery.
    if abs(gap) <= max(Decimal('5'), total * Decimal('0.0005')):
        return []
    proposed = []
    for row, q, p in numeric:
        if row.get('kind') != 'product':
            continue
        for field, current_value, counterpart in [('price', p, q), ('quantity', q, p)]:
            # Solver is only a targeting heuristic, never a candidate generator.
            required = current_value + gap / counterpart
            if not (Decimal(0) < required < Decimal('1000000000')):
                continue
            if required != required.quantize(Decimal('0.01')):
                continue
            current_text = str(row['cells'][field].get('text') or '')
            observed_digits = _digits(current_text)
            required_digits = _digits(format(required.normalize(), 'f'))
            if not observed_digits or not required_digits or observed_digits == required_digits:
                continue
            edit = _distance(observed_digits, required_digits)
            max_edits = 1 if max(len(observed_digits), len(required_digits)) <= 3 else 2
            if edit > max_edits:
                continue
            # Prefer suspicious formatted prices but never trust score to edit data.
            confidence = row['cells'][field].get('confidence')
            conf = float(confidence) if confidence is not None else 0.0
            priority = (edit, 0 if field == 'price' else 1,
                        0 if ('.' in current_text or ',' in current_text) else 1,
                        conf, row['row_id'])
            proposed.append((priority, {'row_id':row['row_id'], 'field':field,
                                       'edit_distance':edit,
                                       'reason':'receipt_sum_mismatch_and_nearby_print_digits'}))
    proposed.sort(key=lambda item:item[0])
    # Do not assert uniqueness of the target. Only OCR + full matrix may do so.
    return [item for _, item in proposed[:max_cells]]
