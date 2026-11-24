"""Conservative arithmetic gates and targeting for E4; no invented OCR evidence."""
from __future__ import annotations
from decimal import Decimal
from .p1_numbers import parse_number


def primary_consistency(rows, printed_total, footer_attempts, *, min_cell_conf=.75,
                        min_footer_conf=.86):
    """True only when existing optical readings and Decimal arithmetic agree.

    A matching sum does not prove optical ground truth. Model uncertainty and
    evidence provenance still have to be checked by the matrix solver.
    """
    notes = []
    total = parse_number(printed_total)
    if total is None or total <= 0:
        notes.append('footer_missing_or_invalid')
    computed = Decimal(0)
    for row in rows:
        cells = row.get('cells', {})
        q, p = cells.get('quantity'), cells.get('price')
        if not q or not p:
            notes.append(f"row_{row.get('row_id')}_missing_field")
            continue
        a, b = parse_number(q.get('text')), parse_number(p.get('text'))
        if a is None or b is None or a < 0 or b < 0:
            notes.append(f"row_{row.get('row_id')}_invalid")
            continue
        for field, cell in (('quantity', q), ('price', p)):
            try:
                conf = float(cell.get('confidence'))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < min_cell_conf:
                notes.append(f"row_{row.get('row_id')}_{field}_low_confidence")
        if 'printed_total' in cells:
            line_total = parse_number(cells['printed_total'].get('text'))
            if line_total is None or abs(line_total - a * b) > Decimal('.01'):
                notes.append(f"row_{row.get('row_id')}_line_total_mismatch")
        computed += a * b
    if not rows:
        notes.append('no_rows')
    if total is not None and abs(computed-total) > Decimal('.01'):
        notes.append('total_mismatch')
    # Accept an actual focused/primary footer OCR observation, not merely a
    # computed number that happens to equal the requested total.
    footer_trusted = False
    for attempt in footer_attempts or []:
        observed = parse_number(attempt.get('total'))
        if observed is None or total is None or abs(observed-total) > Decimal('.01'):
            continue
        conf = attempt.get('confidence')
        if conf is None:
            conf = min([line.get('confidence') or 0.0 for line in (attempt.get('lines') or [])] or [0.0])
        if float(conf or 0) >= min_footer_conf:
            footer_trusted = True
            break
    if not footer_trusted:
        notes.append('printed_footer_not_confirmed')
    return {'eligible': not notes, 'footer_trusted': footer_trusted,
            'computed_total': str(computed), 'printed_total': str(total) if total is not None else None,
            'reasons': notes, 'scope': 'primary_ocr_and_arithmetic_not_ground_truth'}


def prioritize_suspects(rows, crops, *, limit=4):
    """Put malformed/low-confidence and suspect VAT fields first.

    These are PRIORITIES for OCR, not guessed values. The normal E4 can still
    inspect remaining fields when an early unique solution is not established.
    """
    by_id = {r['row_id']: r for r in rows}
    order = {'quantity': 0, 'price': 1, 'printed_total': 2}
    def rank(key):
        rid, field = key
        row = by_id.get(rid, {})
        cell = row.get('cells', {}).get(field, {})
        parsed = parse_number(cell.get('text'))
        try:
            conf = float(cell.get('confidence'))
        except (TypeError, ValueError):
            conf = 0.0
        tax_penalty = -1 if row.get('kind') == 'tax' and field == 'quantity' else 0
        conflict = row.get('printed_line_match') is False
        return (0 if parsed is None else 1, 0 if conflict else 1,
                tax_penalty, conf, rid, order.get(field, 9))
    keys = [k for k in crops if k[0] in by_id and k[1] in order]
    return sorted(keys, key=rank)[:limit]
