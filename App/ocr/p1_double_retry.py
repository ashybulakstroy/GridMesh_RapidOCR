"""Retry suspect numeric cells with an expanded preparation budget."""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
from decimal import Decimal

from .p1_batch import read_batches
from .p1_numbers import arithmetic_equal, parse_number
from .p1_recovery import _recalculate


@_deep_timed("totals", "App/ocr/p1_double_retry.py:retry_suspect_cells")
def retry_suspect_cells(rows, cells, initial_stats, expected_total, vat_rate, *, batch_size=8):
    skipped = {(item['row_id'], item['column_id']) for item in
               initial_stats.get('numeric_roi_skipped_cells', []) +
               initial_stats.get('numeric_roi_currency_skipped_cells', [])}
    by_key = {(cell.row_id, cell.column_id): cell for cell in cells
              if cell.column_id in ('quantity', 'price', 'printed_total')}
    options = []
    for row in rows:
        row_incomplete = any(parse_number(row['cells'][field]['text']) is None
                             for field in ('quantity', 'price'))
        for field in ('quantity', 'price', 'printed_total'):
            key = (row['row_id'], field)
            if field not in row['cells'] or key not in by_key:
                continue
            cell = row['cells'][field]
            invalid = parse_number(cell['text']) is None
            line_mismatch = row.get('printed_line_match') is False
            confidence = float(cell.get('confidence') or 0)
            if invalid or line_mismatch or row_incomplete or key in skipped or confidence < .80:
                priority = 0 if invalid else 1 if line_mismatch or row_incomplete else 2 if key in skipped else 3
                options.append((priority, confidence, key, row))
    if not options:
        for row in rows:
            for field in ('quantity', 'price'):
                key = (row['row_id'], field)
                if key in by_key:
                    options.append((4, float(row['cells'][field].get('confidence') or 0), key, row))
        options = sorted(options, key=lambda item: (item[0], item[1], item[2]))[:4]
    else:
        options.sort(key=lambda item: (item[0], item[1], item[2]))
    if not options:
        return {'attempted':0, 'details':[], 'changed':[], 'candidates':[], 'batch':None}

    results, stats = read_batches([by_key[key] for _, _, key, _ in options],
                                  batch_size=batch_size, budget_multiplier=2)
    details, changed, candidates = [], [], []
    for _, _, key, row in options:
        field = key[1]
        original, alternate = row['cells'][field], results[key]
        old, new = parse_number(original['text']), parse_number(alternate['text'])
        detail = {'row_id':row['row_id'], 'column_id':field,
                  'reason':'double_budget_after_check', 'original_text':original['text'],
                  'original_confidence':original.get('confidence'),
                  'alternative_text':alternate['text'],
                  'alternative_confidence':alternate.get('confidence'),
                  'alternative_valid':new is not None, 'different':new is not None and new != old,
                  'accepted':False, 'strategy':alternate.get('strategy'),
                  'numeric_roi':alternate.get('numeric_roi')}
        original.setdefault('ocr_attempts', []).append(detail)
        details.append(detail)
        if new is None or new == old or float(alternate.get('confidence') or 0) < .90:
            continue
        if row.get('printed_line_match') is False and 'printed_total' in row['cells']:
            printed = alternate if field == 'printed_total' else row['cells']['printed_total']
            if float(printed.get('confidence') or 0) >= .75:
                row['cells'][field] = alternate
                _recalculate(row)
                if row['printed_line_match'] is True:
                    detail.update(accepted=True, confirmation='independent_printed_row_total')
                    alternate['ocr_attempts'] = original['ocr_attempts']
                    changed.append((row['row_id'], field, detail['confirmation']))
                    continue
                row['cells'][field] = original
                _recalculate(row)
        candidates.append((row, field, original, alternate, detail))

    # A four-column receipt has no printed line total. Require a unique
    # candidate supported by both the printed receipt total and VAT check.
    confirmed = []
    if expected_total is not None and any(row['kind'] == 'tax' for row in rows):
        for row, field, original, alternate, detail in candidates:
            row['cells'][field] = alternate
            for item in rows:
                _recalculate(item)
            if (all(item['line_total'] is not None for item in rows) and
                    all(item.get('printed_line_match') is not False for item in rows)):
                total = sum((Decimal(item['line_total']) for item in rows), Decimal(0))
                products = sum((Decimal(item['line_total']) for item in rows if item['kind'] == 'product'), Decimal(0))
                tax = sum((Decimal(item['line_total']) for item in rows if item['kind'] == 'tax'), Decimal(0))
                if arithmetic_equal(total, expected_total) and abs(tax-products*vat_rate/100) <= Decimal('1'):
                    confirmed.append((row, field, original, alternate, detail))
            row['cells'][field] = original
        for item in rows:
            _recalculate(item)
    if len(confirmed) == 1:
        row, field, original, alternate, detail = confirmed[0]
        row['cells'][field] = alternate
        detail.update(accepted=True, confirmation='unique_total_and_vat')
        alternate['ocr_attempts'] = original['ocr_attempts']
        _recalculate(row)
        changed.append((row['row_id'], field, detail['confirmation']))
    remaining = [{'row_id':row['row_id'], 'field':field, 'text':alternate['text'],
                  'reason':'double_budget_after_check'}
                 for row, field, original, alternate, detail in candidates if not detail['accepted']]
    return {'attempted':len(details), 'details':details, 'changed':changed,
            'candidates':remaining, 'batch':stats}
