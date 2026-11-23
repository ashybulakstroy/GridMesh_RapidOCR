"""Direct evidence of printed VAT from Torgsoft tax row; never infer from rate.

The receipt has columns such as NDS 16% | 23200 | 1, where 23200 is the
photographed tax amount, not a percentage-derived calculation.
"""
from __future__ import annotations

from decimal import Decimal
from .p1_numbers import parse_number

NUMERIC_FIELDS = ('quantity', 'price', 'printed_total')


def _cell_value(cell):
    if not isinstance(cell, dict):
        return None
    # Never promote a constructed/inferred value into independent printed OCR.
    if cell.get('derived') and not cell.get('user_verified'):
        return None
    return parse_number(cell.get('text'))


def _one_tax_row(rows):
    taxes = [row for row in rows if row.get('kind') == 'tax']
    return taxes[0] if len(taxes) == 1 else None


def vat_source(rows):
    """Choose a *photographed* cell, not an arithmetic estimate of VAT.

    Numeric printed_total wins only when it actually contains a valid number.
    Otherwise the non-unit cell is VAT if the other printed column says 1.
    """
    tax = _one_tax_row(rows)
    if tax is None:
        return None, 'no_single_tax_row'
    cells, rid = tax.get('cells') or {}, tax['row_id']
    if _cell_value(cells.get('printed_total')) is not None:
        return (rid, 'printed_total'), 'printed_tax_row_total'
    q = _cell_value(cells.get('quantity'))
    p = _cell_value(cells.get('price'))
    if q == Decimal(1) and p is not None:
        return (rid, 'price'), 'single_unit_tax_price'
    if p == Decimal(1) and q is not None:
        return (rid, 'quantity'), 'single_unit_tax_quantity'
    return None, 'tax_amount_not_directly_observed'


def _observations(book, rid, field):
    bucket = book.cells.get(book.key(rid, field)) or {}
    if bucket.get('user_locked'):
        locked = bucket.get('user_value')
        return [x for x in bucket.get('observations', [])
                if x.get('number') is not None and x['number'] == locked]
    return [x for x in bucket.get('observations', []) if x.get('number') is not None]


def _candidate_source(row, book):
    """Fallback if primary cells are unreadable but real OCR alternatives exist."""
    rid = row['row_id']
    printed = _observations(book, rid, 'printed_total')
    if printed:
        return (rid, 'printed_total'), 'candidate_printed_tax_row_total'
    q = _observations(book, rid, 'quantity')
    p = _observations(book, rid, 'price')
    if any(parse_number(x['number']) == Decimal(1) for x in q) and p:
        return (rid, 'price'), 'candidate_single_unit_tax_price'
    if any(parse_number(x['number']) == Decimal(1) for x in p) and q:
        return (rid, 'quantity'), 'candidate_single_unit_tax_quantity'
    return None, 'tax_amount_not_directly_observed'


def seed_printed_vat(rows, book):
    """Add only values actually read in a tax-row cell by OCR or a user.

    Supports candidate observations from E4 before auto-correction, as well as
    a directly read amount, and works when printed_total exists but is blank.
    """
    direct, reason = vat_source(rows)
    row = _one_tax_row(rows)
    if row is None:
        return {'source': None, 'status': reason, 'text': None, 'number': None}
    source = direct
    if source is None:
        source, reason = _candidate_source(row, book)
    if source is None:
        return {'source': None, 'status': reason, 'text': None, 'number': None}
    rid, field = source
    cell = row['cells'].get(field, {})
    direct_number = _cell_value(cell) if direct == source else None
    # Source priority: current observed cell first, then real OCR candidates.
    chosen = None
    candidates = _observations(book, rid, field)
    if direct_number is not None:
        if cell.get('auto_corrected'):
            # The visible E4 value came from a candidate, NOT from the original
            # cell recognizer. Never mislabel it as primary model evidence.
            matched = [x for x in candidates if parse_number(x['number']) == direct_number]
            if matched:
                chosen = str(matched[0]['text'])
        else:
            chosen = str(cell['text'])
            book.add_vat(chosen, cell.get('confidence'), 'vat_row_primary_' + field,
                         user=bool(cell.get('user_verified')),
                         model=cell.get('model_id', 'v5_cyrillic_mobile'),
                         row_id=rid, field=field)
    for item in candidates:
        book.add_vat(item['text'], item.get('confidence'),
                     'vat_row_' + str(item.get('path', 'ocr_candidate')),
                     user=item.get('authority') == 'user_confirmed',
                     model=item.get('model', 'v5_cyrillic_mobile'),
                     row_id=rid, field=field)
    if chosen is None and candidates:
        ranked = sorted(candidates, key=lambda x: -float(x.get('confidence') or 0))
        chosen = str(ranked[0]['text'])
    observed = parse_number(chosen)
    return {'source': {'row_id': rid, 'field': field}, 'status': reason,
            'text': chosen if observed is not None else None,
            'number': str(observed) if observed is not None else None}


def observed_vat_text(book, preferred=None):
    """Select an actual numeric observation; never return an invalid first entry.

    A uniquely solved E4 amount is preferred only if it is already an OCR VAT
    witness. If ambiguous, fall back to an observed candidate for review.
    """
    valid = [x for x in book.vat if x.get('number') is not None]
    if not valid:
        return None
    users = [x for x in valid if x.get('authority') == 'user_confirmed']
    if users:
        return str(users[-1]['text'])
    target = parse_number(preferred)
    if target is not None:
        matching = [x for x in valid if parse_number(x['number']) == target]
        if matching:
            return str(max(matching, key=lambda x: float(x.get('confidence') or 0))['text'])
    return str(max(valid, key=lambda x: float(x.get('confidence') or 0))['text'])
