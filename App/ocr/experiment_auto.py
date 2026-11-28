"""Apply the highest confidence complete arithmetic solution; flag ambiguity."""
from __future__ import annotations
from decimal import Decimal
from .p1_numbers import parse_number, arithmetic_equal
from .experiment_matrix import model_identity


def _support(observations, value, minimum_confidence):
    providers={}
    for obs in observations:
        if obs.get('authority')=='user_confirmed':
            # Manually confirmed values are handled separately; not model votes.
            continue
        if (obs.get('number')==value and obs.get('confidence') is not None
                and float(obs['confidence'])>=minimum_confidence):
            provider=model_identity(obs.get('model'))
            if provider:
                providers[provider]=max(providers.get(provider,0),float(obs['confidence']))
    return providers


def _repair_invalid_printed_rows(rows, book):
    """Promote an observed value only with independent, trusted row arithmetic.

    A different row may prevent the complete receipt matrix from resolving.
    No candidate is calculated from the row total; only existing OCR is used.
    Valid readings and user locks are never replaced by this limited repair.
    """
    from .p1_recovery import _recalculate
    changes=[]
    for row in rows:
        cells=row['cells']
        printed=cells.get('printed_total')
        if printed is None or parse_number(printed.get('text')) is None:
            continue
        invalid=[field for field in ('quantity','price')
                 if field in cells and parse_number(cells[field].get('text')) is None]
        if len(invalid)!=1:
            continue
        field=invalid[0];cell=cells[field]
        other=cells.get('price' if field=='quantity' else 'quantity',{})
        multiplier=parse_number(other.get('text'));total=parse_number(printed['text'])
        trusted=lambda c: bool(c.get('user_verified') or float(c.get('confidence') or 0)>=.90)
        bucket=book.cells.get(book.key(row['row_id'],field),{})
        if (cell.get('user_verified') or bucket.get('user_locked') or
                multiplier is None or multiplier<=0 or total<=0 or
                not trusted(other) or not trusted(printed)):
            continue
        matching={}
        for obs in bucket.get('observations',[]):
            value=parse_number(obs.get('text'))
            provider=model_identity(obs.get('model'))
            if (value is None or value<=0 or float(obs.get('confidence') or 0)<.90 or
                    not provider or provider==model_identity(cell.get('model_id')) or
                    obs.get('path') in ('gridmesh_initial','primary_rec_cell') or
                    not arithmetic_equal(multiplier*value,total)):
                continue
            matching.setdefault(value,[]).append(obs)
        if len(matching)!=1:
            continue
        value,observations=next(iter(matching.items()))
        before=cell['text'];after=str(value)
        evidence=[{key:obs.get(key) for key in ('model','path','text','confidence')}
                  for obs in observations]
        chosen=max(observations,key=lambda obs:obs['confidence'])
        models={}
        for obs in observations:
            provider=model_identity(obs['model'])
            models[provider]=max(models.get(provider,0),obs['confidence'])
        cell.update(text=after,confidence=max(obs['confidence'] for obs in observations),
                    model_id=model_identity(chosen['model']),auto_model_votes=models,
                    strategy='e4_printed_row_observed_candidate',auto_corrected=True,
                    auto_corrected_from=before,auto_ocr_evidence=evidence)
        _recalculate(row)
        changes.append({'row_id':row['row_id'],'field':field,'before':before,'after':after,
                        'confirmation':'independent_ocr_and_printed_row_total',
                        'printed_row_total':printed['text'],'ocr_evidence':evidence,
                        'models':models})
    return changes


def auto_apply_consensus(rows, book, matrix, current_total, *, enabled=True):
    """Model provenance is audit data, never a quorum requirement."""
    report={'enabled':bool(enabled),'status':'not_needed','changes':[],
            'reasons':[],'printed_total':current_total,'review_required':bool(matrix.get('review_required'))}
    if not enabled:
        report['status']='disabled';return report
    if matrix.get('status') != 'unique_math_match' or matrix.get('search_truncated'):
        report['changes']=_repair_invalid_printed_rows(rows,book)
        if report['changes']:
            report.update(status='partial_row_repair',review_required=True)
            report['reasons'].append('receipt_matrix_unresolved_row_evidence_only')
            return report
        report['status']='unresolved_preserved';report['reasons'].append('matrix_not_uniquely_confirmed')
        return report
    if len(matrix.get('solutions') or []) != 1:
        report['status']='unresolved_preserved';report['reasons'].append('solution_count_not_one')
        return report
    selection=matrix['solutions'][0]
    proposed=matrix.get('recommended_changes') or []
    total_chosen=selection['footer']
    total_changed=parse_number(current_total)!=parse_number(total_chosen)
    if not proposed and not total_changed:
        return report
    checks=[]
    for move in proposed:
        row=next((r for r in rows if r['row_id']==move['row_id']),None)
        if row is None or move['field'] not in row.get('cells',{}):
            report['reasons'].append('missing_cell');continue
        cell=row['cells'][move['field']]
        if cell.get('user_verified'):
            report['reasons'].append('user_locked');continue
        observed=book.cells.get(book.key(move['row_id'],move['field']),{}).get('observations',[])
        support=_support(observed,move['to'],0)
        evidence=[{'model':obs.get('model'), 'path':obs.get('path'),
                   'text':obs.get('text'), 'confidence':obs.get('confidence')}
                  for obs in observed if obs.get('number')==move['to']]
        if not evidence:
            report['reasons'].append('no_matching_ocr_evidence')
        checks.append((move,row,cell,support,evidence))
    footer_support=_support(book.footer,total_chosen,0)
    if report['reasons'] or len(checks)!=len(proposed):
        report['status']='not_enough_independent_evidence';return report
    for move,row,cell,support,evidence in checks:
        before=cell['text']
        cell['text']=move['to']
        cell['confidence']=max(support.values(), default=None)
        cell['strategy']='e4_unique_matrix'
        cell['auto_corrected']=True
        cell['auto_corrected_from']=before
        cell['auto_model_votes']=support
        cell['auto_ocr_evidence']=evidence
        report['changes'].append({'row_id':move['row_id'],'field':move['field'],
                                  'before':before,'after':move['to'],'models':support,
                                  'ocr_evidence':evidence})
    if total_changed:
        report['changes'].append({'row_id':None,'field':'receipt_total',
                                  'before':current_total,'after':total_chosen,
                                  'models':footer_support})
    report['printed_total']=total_chosen
    for row in rows:
        cells=row['cells']
        q=parse_number(cells['quantity']['text']);p=parse_number(cells['price']['text'])
        printed=parse_number(cells['printed_total']['text']) if 'printed_total' in cells else None
        row['line_total']=str(q*p) if q is not None and p is not None else None
        row['printed_line_total']=str(printed) if printed is not None else None
        row['printed_line_match']=(arithmetic_equal(q*p,printed) if
                                  q is not None and p is not None and printed is not None else None)
    report['status']='applied'
    return report
