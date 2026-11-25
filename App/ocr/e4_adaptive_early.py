"""E4 Adaptive Recovery / Early Preview, adapted for the current application.

First-math-match is explicitly provisional: additional OCR observations can
reveal an ambiguous matrix. Normal E4 continues AFTER the preview is published.
No external models or dependencies are required.
"""
from __future__ import annotations
from App.ocr.performance_deep.live_capture import timed as _deep_timed

from copy import deepcopy
from decimal import Decimal
from time import perf_counter
from .experiment_matrix import solve_matrix
from .experiment_vat import seed_printed_vat, vat_source
from .p1_numbers import parse_number

NUMERIC_FIELDS = ('quantity', 'price', 'printed_total')


@_deep_timed("totals", "App/ocr/e4_adaptive_early.py:_observed_numbers")
def _observed_numbers(book, key):
    item = book.cells.get(book.key(*key), {})
    values = set()
    for obs in item.get('observations', []):
        if obs.get('number') is not None:
            values.add(Decimal(str(obs['number'])))
    return values


@_deep_timed("totals", "App/ocr/e4_adaptive_early.py:rank_candidates")
def rank_candidates(rows, crops):
    """Invalid/missing > row mismatch > low OCR confidence; stable order."""
    index = {int(row['row_id']): row for row in rows}
    field_index = {'quantity': 0, 'price': 1, 'printed_total': 2}

    def priority(key):
        rid, field = key
        row = index.get(int(rid), {})
        cell = row.get('cells', {}).get(field, {})
        number = parse_number(cell.get('text'))
        invalid = number is None
        mismatch = row.get('printed_line_match') is False
        raw_conf = cell.get('confidence')
        try:
            confidence = max(0.0, min(1.0, float(raw_conf))) if raw_conf is not None else 0.0
        except (ValueError, TypeError):
            confidence = 0.0
        return (0 if invalid else 1, 0 if mismatch else 1, confidence, int(rid), field_index[field])

    return sorted((key for key in crops if key[1] in NUMERIC_FIELDS and int(key[0]) in index), key=priority)


@_deep_timed("totals", "App/ocr/e4_adaptive_early.py:eligible_first_result")
def eligible_first_result(matrix, rows, book):
    """Only unique, complete, nontruncated arithmetic; VAT must be observed.

    This DOES NOT prove optical correctness and MUST be labelled provisional.
    """
    if (matrix.get('status') != 'unique_math_match' or matrix.get('search_truncated')
            or matrix.get('candidate_pool_truncated') or matrix.get('review_required')
            or matrix.get('critical_ambiguity') or len(matrix.get('solutions', [])) != 1):
        return False
    if not rows or not book.footer_values():
        return False
    if any(r.get('kind') == 'tax' for r in rows):
        if not matrix.get('printed_vat_observed') or not book.vat_values():
            return False
    solution = matrix['solutions'][0]
    assignments = {int(x['row_id']): x for x in solution.get('assignments', [])}
    if (len(solution.get('assignments', [])) != len(rows) or
            set(assignments) != {int(row['row_id']) for row in rows}):
        return False
    for row in rows:
        cell_data = row.get('cells', {})
        assn = assignments.get(int(row['row_id']))
        if not assn:
            return False
        for field in NUMERIC_FIELDS:
            if field in cell_data and (field not in assn or parse_number(assn[field]) is None):
                return False
        if any(field not in assn for field in ('quantity', 'price')):
            return False
    return True


@_deep_timed("totals", "App/ocr/e4_adaptive_early.py:make_early_snapshot")
def make_early_snapshot(rows, book, matrix, total, vat_rate, elapsed_ms, scans, job_id):
    """Freeze selected observed OCR digits; original rows stay untouched."""
    solution = matrix['solutions'][0]
    assignments = {int(a['row_id']): a for a in solution.get('assignments', [])}
    frozen = deepcopy(rows)
    for row in frozen:
        chosen = assignments[int(row['row_id'])]
        for field in NUMERIC_FIELDS:
            if field in chosen and field in row['cells']:
                row['cells'][field]['text'] = chosen[field]
        q = parse_number(row['cells']['quantity']['text'])
        p = parse_number(row['cells']['price']['text'])
        row['line_total'] = str(q * p) if q is not None and p is not None else None
        if 'printed_total' in row['cells']:
            pt = parse_number(row['cells']['printed_total']['text'])
            row['printed_line_total'] = str(pt) if pt is not None else None
            row['printed_line_match'] = (pt == q*p) if pt is not None and q is not None and p is not None else None
    return {
        'schema': 'e4-adaptive-first-result-v1',
        'job_id': job_id,
        'stage': 'first_math_match',
        'provisional': True,
        'read_only': True,
        'notice': 'Математически согласованный первый OCR-результат; не окончательная проверка изображения.',
        'elapsed_ms': round(float(elapsed_ms), 2),
        'v5_scans_before_first': int(scans),
        'rows': frozen,
        'total': str(solution['footer']),
        'initial_printed_total': total,
        'computed_total': str(solution['computed_total']),
        'printed_vat': solution.get('vat_printed'),
        'vat_rate_percent': str(vat_rate),
        'matrix_status_at_first': matrix['status'],
        'original_ocr_conflicts': len(matrix.get('recommended_changes', [])),
        'candidate_observations_at_first': sum(len(x.get('observations', [])) for x in book.cells.values()),
    }


@_deep_timed("totals", "App/ocr/e4_adaptive_early.py:adaptive_collect")
def adaptive_collect(rows, book, crops, vat_rate, collect_one, on_first_ready,
                      *, max_nodes=2500, priority_keys=(), max_probes=None):
    """Try weakest numeric cells first; solve E4 on each NEW numeric candidate.

    Does not perform the full sweep after the early signal: the calling pipeline
    runs its standard E4 collection/verification in the SAME background job.
    """
    seed_printed_vat(rows, book)
    vat_key, _ = vat_source(rows)
    attempted = set()
    observations, errors = [], []
    ordered = rank_candidates(rows, crops)
    # MULTILINE_VAT_FAST_E4_V1: examine mathematical suspects first, not all fields.
    priority = [tuple(k) for k in priority_keys if tuple(k) in ordered]
    ordered = list(dict.fromkeys(priority + ordered))
    all_numeric_count = len(ordered)
    if max_probes is not None:
        ordered = ordered[:max(0, int(max_probes))]
    checks = 0
    first_reported = False
    last_status = 'not_run'
    start = perf_counter()

    def check():
        nonlocal checks, first_reported, last_status
        checks += 1
        matrix = solve_matrix(rows, book, vat_rate, max_nodes=max_nodes)
        last_status = matrix['status']
        if not first_reported and eligible_first_result(matrix, rows, book):
            try:
                on_first_ready(matrix, book, len(attempted))
            except Exception as exc:
                errors.append({'row_id': None, 'field': 'first_result',
                               'error': f'Publish failed: {type(exc).__name__}: {exc}'})
                return False
            first_reported = True
            return True
        return False

    if not check():
        for key in ordered:
            attempted.add(key)
            try:
                old_numbers = _observed_numbers(book, key)
                # Use the CURRENT complete one-cell collector. In particular,
                # retain price_digits_candidate, which did not exist in V1.
                collected = collect_one({key: crops[key]})
                observations.extend(collected['observations'])
                errors.extend(collected['errors'])
                for record in collected['observations']:
                    text, score = str(record.get('text', '')), record.get('confidence')
                    path = record.get('path', 'v5_numeric_candidate')
                    model = record.get('model_id', 'v5_cyrillic_mobile')
                    book.add(key[0], key[1], text, score, path, model=model)
                    if vat_key == key:
                        book.add_vat(text, score, 'vat_row_'+path,
                                     row_id=key[0], field=key[1], model=model)
                # A newly read 1-unit tax column may unlock a real VAT witness.
                # Rebuild the VAT evidence from actual OCR candidates before E4.
                seed_printed_vat(rows, book)
                if _observed_numbers(book, key) - old_numbers:
                    if check():
                        break
            except Exception as exc:
                errors.append({'row_id': key[0], 'field': key[1],
                               'error': f'{type(exc).__name__}: {exc}'})

    return {
        'collected': {'observations': observations, 'errors': errors, 'attempted': attempted},
        'early_found': first_reported,
        'probes_before_first': len(attempted) if first_reported else None,
        'probes_used': len(attempted),
        'matrix_checks': checks,
        'last_matrix_status': last_status,
        'elapsed_ms': round((perf_counter() - start) * 1000, 3),
        'total_numeric_cells': all_numeric_count,
        'targeted_max_probes': max_probes,
    }


@_deep_timed("totals", "App/ocr/e4_adaptive_early.py:compare_early_final")
def compare_early_final(early, final):
    """Separate changed displayed digits from newly found OCR candidates."""
    old_rows = {int(x['row_id']): x for x in early.get('rows', [])}
    new_rows = {int(x['row_id']): x for x in final.get('rows', [])}
    changes, formatting = [], []
    def add_change(rid, field, old, new):
        if old == new:
            return
        before, after = parse_number(old), parse_number(new)
        item = {'row_id': rid, 'field': field, 'before': old, 'after': new}
        (formatting if before is not None and after is not None and before == after else changes).append(item)
    for rid in sorted(old_rows.keys() & new_rows.keys()):
        before = old_rows[rid].get('cells', {})
        after = new_rows[rid].get('cells', {})
        for field in NUMERIC_FIELDS:
            if field not in before and field not in after:
                continue
            old, new = before.get(field, {}).get('text'), after.get(field, {}).get('text')
            add_change(rid, field, old, new)
    for rid in sorted(old_rows.keys() ^ new_rows.keys()):
        changes.append({'row_id': rid, 'field': 'row',
                        'before': 'present' if rid in old_rows else None,
                        'after': 'present' if rid in new_rows else None})
    for field in ('total', 'printed_vat'):
        old, new = early.get(field), final.get(field)
        add_change(None, field, old, new)
    book = final.get('numeric_candidate_book', {})
    total_observations = sum(len(x.get('observations', [])) for x in book.get('cells', {}).values())
    final_matrix = final.get('numeric_matrix', {})
    status = final_matrix.get('status', 'unknown')
    return {
        'schema': 'e4-adaptive-comparison-v1',
        'first_result_found': True,
        'first_elapsed_ms': early.get('elapsed_ms'),
        'full_verification_ms': round(float(final.get('timings', {}).get('total_s', 0)) * 1000, 2),
        'final_matrix_status': status,
        'final_review_required': bool(final_matrix.get('review_required', True)),
        'changed_fields': changes,
        'changed_count': len(changes),
        'formatting_changes': formatting,
        'observations_after_first': max(0, total_observations - int(early.get('candidate_observations_at_first', 0))),
        'note': ('Различия являются результатами OCR/матрицы; только изображение или подтверждение пользователя '
                 'определяет истинность цифр.'),
    }
