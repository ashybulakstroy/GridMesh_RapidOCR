"""Bounded, auditable numeric candidate matrix for Torgsoft receipts.

A mathematical match is *evidence*, not optical ground truth.  The only
explicit ground truth in this experiment is user-confirmed numeric input.
No values are synthesised from arithmetic; only actual OCR observations or
user edits can participate in a candidate assignment.
"""
from __future__ import annotations
from App.ocr.performance_deep.live_capture import timed as _deep_timed

from decimal import Decimal
from collections import defaultdict

from .p1_numbers import parse_number, arithmetic_equal

NUMERIC_FIELDS = ('quantity', 'price', 'printed_total')
MODEL = 'PP-OCRv5 Cyrillic Mobile / RapidOCR 3.x'


@_deep_timed("totals", "App/ocr/experiment_matrix.py:_decimal_string")
def _decimal_string(value):
    parsed = parse_number(value)
    return format(parsed.normalize(), 'f') if parsed is not None else None


@_deep_timed("totals", "App/ocr/experiment_matrix.py:_confidence")
def _confidence(value):
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


@_deep_timed("totals", "App/ocr/experiment_matrix.py:model_identity")
def model_identity(model):
    aliases={MODEL:'v5_cyrillic_mobile','v5':'v5_cyrillic_mobile',
             'PP-OCRv5 Mobile Cyrillic':'v5_cyrillic_mobile',
             'PP-OCRv6 Small official':'v6_small_official',
             'PP-OCRv6 Tiny official':'v6_tiny_official'}
    return aliases.get(model,model)


@_deep_timed("totals", "App/ocr/experiment_matrix.py:observation")
def observation(text, confidence, path, *, model=MODEL, user=False, note=None):
    """An observation has immutable provenance and may be invalid as a number."""
    result = {'text': str(text if text is not None else ''),
              'number': _decimal_string(text),
              'confidence': None if user else _confidence(confidence),
              'model': 'user' if user else model_identity(model), 'path': path,
              'authority': 'user_confirmed' if user else 'ocr_candidate'}
    if note:
        result['note'] = note
    return result


class CandidateBook:
    """Per-cell arrays with provenance, no implicit winner from higher confidence."""

    def __init__(self):
        self.cells = {}
        self.footer = []
        self.vat = []  # independently observed *printed* VAT, not computed tax

    @staticmethod
    def key(row_id, field):
        return f'{int(row_id)}:{field}'

    def add(self, row_id, field, text, confidence, path, *, user=False, model=MODEL):
        key = self.key(row_id, field)
        if key not in self.cells:
            self.cells[key] = {'row_id': int(row_id), 'field': field,
                               'observations': [], 'user_locked': False}
        item = self.cells[key]
        record = observation(text, confidence, path, model=model, user=user)
        if not any(all(existing.get(k) == record.get(k) for k in ('text', 'path', 'model', 'authority'))
                   for existing in item['observations']):
            item['observations'].append(record)
        if user:
            item['user_locked'] = True
            item['user_value'] = record['number']
        return record

    def add_footer(self, text, confidence, path, *, user=False, model=MODEL):
        record = observation(text, confidence, path, model=model, user=user)
        if not any(all(existing.get(k) == record.get(k) for k in ('text', 'path', 'model', 'authority'))
                   for existing in self.footer):
            self.footer.append(record)
        return record

    def add_vat(self, text, confidence, path, *, user=False, model=MODEL,
                row_id=None, field=None):
        record = observation(text, confidence, path, model=model, user=user)
        record['row_id'] = row_id
        record['field'] = field or 'printed_vat'
        if not any(all(item.get(k)==record.get(k) for k in
                       ('text','path','model','authority','row_id','field'))
                   for item in self.vat):
            self.vat.append(record)
        return record

    def vat_values(self, *, limit=6):
        by_number=defaultdict(list)
        for item in self.vat:
            if item['number'] is not None:
                by_number[item['number']].append(item)
        locks=[o for o in self.vat if o['authority']=='user_confirmed']
        if locks:
            last=locks[-1]['number']
            by_number={last:by_number.get(last,[])} if last is not None else {}
        result=[]
        for value,items in by_number.items():
            score=max(o['confidence'] if o['confidence'] is not None else
                      (1.0 if o['authority']=='user_confirmed' else 0.0)
                      for o in items)
            result.append({'number':value,'score':score,'observations':items,
                           'user_locked':any(o['authority']=='user_confirmed' for o in items)})
        return sorted(result,key=lambda x:(-x['score'],Decimal(x['number'])))[:limit]

    @classmethod
    def from_receipt(cls, rows, initial, recovery, total, footer_attempts=()):
        book = cls()
        for row in rows:
            for field in NUMERIC_FIELDS:
                if field not in row.get('cells', {}):
                    continue
                key = cls.key(row['row_id'], field)
                original = initial.get(key)
                if original is not None:
                    book.add(row['row_id'], field, original['text'], original.get('confidence'), 'gridmesh_initial', model=original.get('model_id','v5_cyrillic_mobile'))
                cell = row['cells'][field]
                book.add(row['row_id'], field, cell['text'], cell.get('confidence'),
                         cell.get('strategy') or 'gridmesh_final', model=cell.get('model_id','v5_cyrillic_mobile'))
                # Recoveries may have already replaced the original cell; audit both.
                for item in cell.get('ocr_attempts', []):
                    book.add(row['row_id'], field, item.get('alternative_text'),
                             item.get('alternative_confidence'), item.get('strategy') or 'selective_retry')
        for stage, block in (recovery or {}).items():
            for detail in block.get('details', []):
                rid, field = detail.get('row_id'), detail.get('column_id')
                if rid is None or field not in NUMERIC_FIELDS:
                    continue
                book.add(rid, field, detail.get('original_text'),
                         detail.get('original_confidence'), f'{stage}_before')
                book.add(rid, field, detail.get('alternative_text'),
                         detail.get('alternative_confidence'),
                         detail.get('strategy') or f'{stage}_alternative')
        book.add_footer(total, None, 'pipeline_selected_footer')
        for attempt in footer_attempts or ():
            value = attempt.get('total')
            if value is not None:
                score = attempt.get('confidence')
                if score is None and attempt.get('lines'):
                    score = min((line.get('confidence') or 0) for line in attempt['lines'])
                book.add_footer(value, score, attempt.get('strategy') or f"footer_scale_{attempt.get('scale',1)}")
        return book

    def possible_values(self, row_id, field, *, limit=4):
        # PARTIAL_DIGIT_MATRIX_TRIAL_V1: only the isolated copy sets this attribute.
        limit = max(limit, min(16, getattr(self, 'partial_trial_limit', 4)))
        bucket = self.cells.get(self.key(row_id, field))
        if bucket is None:
            return []
        values = defaultdict(list)
        for rec in bucket['observations']:
            if rec['number'] is not None:
                values[rec['number']].append(rec)
        if bucket.get('user_locked'):
            locked = bucket.get('user_value')
            values = {locked: values.get(locked, [])} if locked is not None else {}
        ranked = []
        for number, items in values.items():
            score = max((x['confidence'] if x['confidence'] is not None else (1.0 if x['authority']=='user_confirmed' else 0.0))
                        for x in items)
            ranked.append({'number': number, 'score': score,
                           'observations': items, 'user_locked': bool(bucket.get('user_locked'))})
        ranked.sort(key=lambda item: (-item['score'], Decimal(item['number'])))
        return ranked[:limit]

    def footer_values(self, *, limit=6):
        by_value = defaultdict(list)
        for obs in self.footer:
            if obs['number'] is not None:
                by_value[obs['number']].append(obs)
        locked = [o for o in self.footer if o['authority'] == 'user_confirmed']
        if locked:
            choice = locked[-1]['number']
            by_value = {choice: by_value.get(choice, [])} if choice is not None else {}
        ranked = []
        for number, items in by_value.items():
            score = max(o['confidence'] if o['confidence'] is not None else (1.0 if o['authority']=='user_confirmed' else 0.0) for o in items)
            ranked.append({'number': number, 'score': score,
                           'observations': items,
                           'user_locked': any(o['authority'] == 'user_confirmed' for o in items)})
        ranked.sort(key=lambda x: (-x['score'], Decimal(x['number'])))
        return ranked[:limit]

    def export(self):
        return {'cells': self.cells, 'footer': self.footer, 'vat': self.vat}


@_deep_timed("totals", "App/ocr/experiment_matrix.py:_build_row_options")
def _build_row_options(row, book, cap):
    rid = row['row_id']
    qs = book.possible_values(rid, 'quantity')
    ps = book.possible_values(rid, 'price')
    if not qs or not ps:
        return [], False
    printed = ('printed_total' in row['cells'])
    totals = book.possible_values(rid, 'printed_total') if printed else [None]
    if printed and not totals:
        return [], False
    result = []
    for q in qs:
        for p in ps:
            amount = Decimal(q['number']) * Decimal(p['number'])
            for t in totals:
                if t is not None and not arithmetic_equal(amount, Decimal(t['number'])):
                    continue
                choice = {'quantity': q['number'], 'price': p['number']}
                if t is not None:
                    choice['printed_total'] = t['number']
                score = q['score'] + p['score'] + (t['score'] if t else 0)
                result.append({'row_id':rid, 'kind':row['kind'], 'amount':amount,
                               'choice':choice,'score':score})
    result.sort(key=lambda item: -item['score'])
    return result[:cap], len(result) > cap


@_deep_timed("totals", "App/ocr/experiment_matrix.py:solve_matrix")
def solve_matrix(rows, book, vat_rate, *, max_nodes=100000, per_row_cap=64):
    """Find real combinations with bounded DFS and Decimal exact arithmetic.

    Never call a search 'unique' if any bound or candidate truncation hit.
    """
    footer = book.footer_values()
    printed_vat = book.vat_values()
    has_tax = any(row.get('kind') == 'tax' for row in rows)
    # An unexamined 5th cell value or 7th footer variant must not be hidden
    # behind a false uniqueness verdict.
    footer_all = {o['number'] for o in book.footer if o['number'] is not None}
    vat_all = {o['number'] for o in book.vat if o['number'] is not None}
    vat_overflow = len(vat_all)>6 and not any(o['authority']=='user_confirmed' for o in book.vat)
    cell_overflow = any(
        len({o['number'] for o in bucket.get('observations', []) if o['number'] is not None}) > max(4, min(16, getattr(book, 'partial_trial_limit', 4)))
        for bucket in book.cells.values() if not bucket.get('user_locked'))
    choices, shortened = [], (len(footer_all)>6 or cell_overflow or vat_overflow)
    for row in rows:
        row_items, cut = _build_row_options(row, book, per_row_cap)
        choices.append(row_items)
        shortened |= cut
    result = {'status':'no_solution', 'review_required':True,
              'footer_variants':[f['number'] for f in footer],
              'footer_ambiguous':len(footer)>1,
              'vat_variants':[v['number'] for v in printed_vat],
              'vat_ambiguous':len(printed_vat)>1,
              'printed_vat_observed':bool(printed_vat),
              'vat_rate_percent':str(vat_rate),
              'vat_rule':'printed_receipt_total_equals_products_plus_printed_tax;rate_check_is_diagnostic_only',
              'vat_rate_diagnostic': {'status':'not_evaluated'},
              'risk_level':'review', 'critical_ambiguity':False,
              'ambiguity_dimensions':[], 'nodes_visited':0,
              'search_truncated':shortened,
              'candidate_pool_truncated':(len(footer_all)>6 or cell_overflow or vat_overflow),
              'solutions':[], 'recommended_changes':[], 'model_assessments':[],
              'note':'Математика подтверждает согласованность, а не оптическую истину. Ставка и схема НДС заданы текущим проектом.'}
    if not rows or not footer or any(not c for c in choices):
        result['status'] = 'insufficient_candidates'
        return result
    # Bounds allow pruning of impossible receipt totals without enumerating all paths.
    mins = [Decimal(0)] * (len(choices)+1)
    maxs = [Decimal(0)] * (len(choices)+1)
    for i in range(len(choices)-1, -1, -1):
        mins[i] = mins[i+1] + min(x['amount'] for x in choices[i])
        maxs[i] = maxs[i+1] + max(x['amount'] for x in choices[i])
    rate = Decimal(str(vat_rate))
    seen = set()
    solutions = []
    limit_reached = False
    for target in footer:
        expected = Decimal(target['number'])
        if limit_reached:
            break
        def walk(index, total, products, tax, picks):
            nonlocal limit_reached
            if limit_reached:
                return
            result['nodes_visited'] += 1
            if result['nodes_visited'] > max_nodes:
                limit_reached = True
                return
            if total + mins[index] > expected + Decimal('.01') or total + maxs[index] < expected - Decimal('.01'):
                return
            if index == len(choices):
                if not arithmetic_equal(total, expected):
                    return
                # Operator-entered printed VAT can be arithmetically wrong.
                # Do NOT reject a genuine OCR reading merely because the tax
                # diverges from the configured nominal tax rate.
                vat_matches = ([v for v in printed_vat if arithmetic_equal(tax,Decimal(v['number']))]
                               if has_tax and printed_vat else [None])
                if not vat_matches:
                    return
                for vat_evidence in vat_matches:
                    vat_label = vat_evidence['number'] if vat_evidence is not None else None
                    signature=(target['number'],vat_label,
                               tuple(tuple(sorted(p['choice'].items())) for p in picks))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    expected_vat=(products*rate/Decimal(100)) if has_tax else None
                    tax_delta=(tax-expected_vat) if has_tax else None
                    score_sum=target['score']+sum(p['score'] for p in picks)
                    score_count=1+sum(len(p['choice']) for p in picks)
                    if vat_evidence is not None:
                        score_sum+=vat_evidence['score'];score_count+=1
                    solutions.append({'confidence_score':score_sum/score_count,'footer':target['number'], 'computed_total':str(total),
                                      'product_total':str(products), 'vat_total':str(tax) if has_tax else None,
                                      'vat_rate_expected':str(expected_vat) if has_tax else None,
                                      'vat_rate_delta':str(tax_delta) if has_tax else None,
                                      'vat_rate_consistent':(abs(tax_delta)<=Decimal('1')) if has_tax else None,
                                      'vat_printed':vat_label,
                                      'vat_crosscheck':('direct_ocr_match' if vat_evidence is not None else
                                                        'calculated_only' if has_tax else 'not_applicable'),
                                      'assignments':[{'row_id':p['row_id'], **p['choice']} for p in picks]})

                return
            for option in choices[index]:
                amount=option['amount']
                walk(index+1, total+amount,
                     products+(amount if option['kind'] != 'tax' else Decimal(0)),
                     tax+(amount if option['kind'] == 'tax' else Decimal(0)),
                     picks+[option])
                if limit_reached:
                    return
        walk(0,Decimal(0),Decimal(0),Decimal(0),[])
    result['search_truncated'] |= limit_reached
    solutions.sort(key=lambda solution: -solution['confidence_score'])
    result['solutions'] = solutions
    if not has_tax:
        result['vat_rate_diagnostic'] = {'status':'not_applicable', 'rate_percent':str(rate)}
    elif not solutions:
        result['vat_rate_diagnostic'] = {'status':'not_evaluated_no_consistent_ocr_solution',
                                         'rate_percent':str(rate)}
    else:
        flags={solution['vat_rate_consistent'] for solution in solutions}
        result['vat_rate_diagnostic'] = {
            'status':'consistent' if flags=={True} else 'operator_calculation_anomaly' if flags=={False} else 'ambiguous',
            'rate_percent':str(rate),
            'expected_values':sorted({sol['vat_rate_expected'] for sol in solutions},key=Decimal),
            'printed_values':sorted({sol['vat_total'] for sol in solutions},key=Decimal),
            'deltas':sorted({sol['vat_rate_delta'] for sol in solutions},key=Decimal),
            'threshold':'1',
            'classification':'financial_diagnostic_not_ocr_failure'}
    if result['search_truncated']:
        result['status']='bounded_search'
        return result
    if not solutions:
        return result
    if len(solutions)>1:
        result['status']='ambiguous'
        result['risk_level']='critical'
        result['critical_ambiguity']=True
        differing=[]
        for name in ('footer','vat_printed'):
            if len({str(solution.get(name)) for solution in solutions})>1:
                differing.append(name)
        for row in rows:
            rid=row['row_id']
            for field in NUMERIC_FIELDS:
                values={next((a.get(field) for a in solution['assignments']
                              if a['row_id']==rid),None) for solution in solutions}
                if len(values)>1:differing.append(f'{rid}:{field}')
        result['ambiguity_dimensions']=differing
        result['note']='Выбрана комбинация с наибольшей средней уверенностью OCR. Проверьте цифры: несколько комбинаций проходят арифметическую проверку.'
    else:
        result['status'] = 'unique_math_match'
    # Conflicting observations are candidates; a unique complete solution resolves them.
    result['review_required'] = len(solutions)>1
    selected=solutions[0]
    result['risk_level']='review' if result['review_required'] else 'normal'
    for assignment in selected['assignments']:
        row=next(r for r in rows if r['row_id']==assignment['row_id'])
        for field in NUMERIC_FIELDS:
            if field not in assignment or field not in row['cells']:
                continue
            chosen=assignment[field]
            old=_decimal_string(row['cells'][field].get('text'))
            if chosen!=old:
                result['recommended_changes'].append({'row_id':assignment['row_id'],
                    'field':field,'from':row['cells'][field].get('text'), 'to':chosen,
                    'status':'proposed_math_supported_not_ground_truth'})
            bucket=book.cells.get(book.key(assignment['row_id'],field),{})
            for obs in bucket.get('observations',[]):
                result['model_assessments'].append({'row_id':assignment['row_id'],
                    'field':field,'model':obs['model'],'path':obs['path'],
                    'observed':obs['number'], 'expected_by_matrix':chosen,
                    'assessment':('user_confirmed' if obs['authority']=='user_confirmed' else
                        'math_supported' if obs['number']==chosen else 'math_conflicting')})
    for obs in book.footer:
        result['model_assessments'].append({'row_id':None,'field':'receipt_total',
            'model':obs['model'],'path':obs['path'],'observed':obs['number'],
            'expected_by_matrix':selected['footer'],
            'assessment':('user_confirmed' if obs['authority']=='user_confirmed' else
                         'math_supported' if obs['number']==selected['footer'] else 'math_conflicting')})
    for obs in book.vat:
        result['model_assessments'].append({'row_id':obs.get('row_id'), 'field':'printed_vat',
            'model':obs['model'],'path':obs['path'],'observed':obs['number'],
            'expected_by_matrix':selected['vat_total'],
            'assessment':('user_confirmed' if obs['authority']=='user_confirmed' else
                         'math_supported' if obs['number']==selected['vat_total'] else 'math_conflicting')})
    return result
