"""Server-authoritative manual numeric corrections and OCR model feedback.

Input from the browser is *not* trusted as a precomputed math status: the
server redoes all calculations with Decimal and pins verified user values.
"""
from __future__ import annotations
import copy
from decimal import Decimal

from .experiment_matrix import CandidateBook, solve_matrix, NUMERIC_FIELDS
from .p1_numbers import parse_number, arithmetic_equal


def apply_user_receipt(original_receipt, submitted):
    """Accept edits only for existing rows, produce an auditable new receipt.

    A nonnumeric user edit remains visible and flagged as invalid rather than
    turning an impossible figure into fake ground truth.
    """
    if not isinstance(submitted,dict) or not isinstance(submitted.get('rows'),list):
        raise ValueError('Ожидаются строки чека')
    original_rows=original_receipt.get('rows',[])
    incoming_rows=submitted['rows']
    if len(original_rows)!=len(incoming_rows):
        raise ValueError('Запрещено менять число строк при подтверждении OCR')
    result=copy.deepcopy({k:v for k,v in original_receipt.items() if k not in ('original','rectified')})
    book=CandidateBook()
    old_book=result.get('numeric_candidate_book') or {}
    book.cells=copy.deepcopy(old_book.get('cells',{}))
    book.footer=copy.deepcopy(old_book.get('footer',[]))
    book.vat=copy.deepcopy(old_book.get('vat',[]))
    if not book.cells:
        for row in original_rows:
            for field in NUMERIC_FIELDS:
                if field in row.get('cells', {}):
                    item=row['cells'][field]
                    book.add(row['row_id'],field,item.get('text'),item.get('confidence'),'saved_primary')
    if not book.footer:
        book.add_footer(original_receipt.get('total'),None,'saved_footer')
    manual_truth=[]
    for idx,(before,after) in enumerate(zip(original_rows,incoming_rows)):
        if before.get('row_id')!=after.get('row_id'):
            raise ValueError(f'Порядок/номер строки изменён: {idx+1}')
        row=result['rows'][idx]
        kind=after.get('kind',row.get('kind'))
        if kind not in ('product','tax'):
            raise ValueError('Некорректный тип строки')
        row['kind']=kind
        for field in ('name','quantity','price','printed_total'):
            if field not in row['cells']:
                continue
            submitted_cell=after.get('cells',{}).get(field)
            if not isinstance(submitted_cell,dict):
                continue
            old=str(row['cells'][field].get('text',''))
            new=str(submitted_cell.get('text',old))
            if len(new)>512:
                raise ValueError('Значение ячейки слишком длинное')
            changed=new!=old
            user_flag=bool(submitted_cell.get('user_verified'))
            if field=='name':
                row['cells'][field]['text']=new
                continue
            if field not in NUMERIC_FIELDS:
                continue
            if not (changed or user_flag):
                continue
            row['cells'][field]['text']=new
            valid=parse_number(new) is not None
            row['cells'][field]['user_verified']=valid
            row['cells'][field]['manual_source']='web_user'
            audit={'row_id':row['row_id'],'field':field,'before':old,'after':new,
                   'valid':valid,'authority':'user_confirmed' if valid else 'user_edit_invalid'}
            manual_truth.append(audit)
            if valid:
                book.add(row['row_id'],field,new,None,'web_user',user=True)
                # An edited tax amount must supersede the OCR witness when
                # it is the same printed numeric cell, not a derived product.
                original_source=(original_receipt.get('vat_evidence') or {}).get('source') or {}
                if (original_source.get('row_id')==row['row_id'] and
                    original_source.get('field')==field):
                    book.add_vat(new,None,'web_user_vat_row',user=True,
                                 row_id=row['row_id'],field=field)
    old_footer=str(original_receipt.get('total') or '')
    new_footer=str(submitted.get('total',old_footer))
    if len(new_footer)>256:
        raise ValueError('Печатный итог слишком длинный')
    result['total']=new_footer
    if new_footer!=old_footer or submitted.get('user_total_verified'):
        valid=parse_number(new_footer) is not None
        manual_truth.append({'row_id':None,'field':'receipt_total','before':old_footer,
                             'after':new_footer,'valid':valid,
                             'authority':'user_confirmed' if valid else 'user_edit_invalid'})
        result['user_total_verified']=valid
        if valid:
            book.add_footer(new_footer,None,'web_user',user=True)
    # An explicit user-entered printed VAT amount is authoritative evidence,
    # but may still contradict the other fields; never hide that contradiction.
    old_vat=str(original_receipt.get('printed_vat') or '')
    new_vat=str(submitted.get('printed_vat',old_vat) or '')
    if len(new_vat)>256:
        raise ValueError('Напечатанный НДС слишком длинный')
    result['printed_vat']=new_vat
    if new_vat!=old_vat or submitted.get('user_vat_verified'):
        valid=parse_number(new_vat) is not None
        result['user_vat_verified']=valid
        manual_truth.append({'row_id':None,'field':'printed_vat','before':old_vat,
                             'after':new_vat,'valid':valid,
                             'authority':'user_confirmed' if valid else 'user_edit_invalid'})
        if valid:
            book.add_vat(new_vat,None,'web_user_printed_vat',user=True)
    calc=Decimal(0)
    incomplete=False
    product=Decimal(0)
    vat=Decimal(0)
    row_mismatch=False
    for row in result['rows']:
        q=parse_number(row['cells'].get('quantity',{}).get('text'))
        p=parse_number(row['cells'].get('price',{}).get('text'))
        amount=q*p if q is not None and p is not None else None
        row['line_total']=str(amount) if amount is not None else None
        printed=parse_number(row['cells'].get('printed_total',{}).get('text')) if 'printed_total' in row['cells'] else None
        row['printed_line_total']=str(printed) if printed is not None else None
        row['printed_line_match']=(arithmetic_equal(amount,printed) if printed is not None and amount is not None else None)
        row_mismatch |= row['printed_line_match'] is False
        if amount is None:
            incomplete=True
        else:
            calc+=amount
            if row['kind']=='tax':vat+=amount
            else:product+=amount
    vat_rate=Decimal(str(result.get('vat',{}).get('rate_percent','16')))
    has_tax=any(r['kind']=='tax' for r in result['rows'])
    vat_ok=None if not has_tax or incomplete else abs(vat-product*vat_rate/100)<=Decimal('1')
    expected=parse_number(new_footer)
    math_match=not incomplete and arithmetic_equal(calc,expected)
    result['computed_total']=str(calc)
    result['math_match']=math_match
    vat_printed=parse_number(new_vat)
    vat_printed_match=(arithmetic_equal(vat_printed,vat) if has_tax and vat_printed is not None else None)
    result['vat']={'status':'not_applicable' if vat_ok is None else 'match' if vat_ok else 'mismatch',
                   'rate_percent':str(vat_rate),'amount':str(vat) if has_tax else None,
                   'printed_amount':new_vat or None,'printed_match':vat_printed_match,
                   'rate_diagnostic_only':True}
    # A deliberately or accidentally wrong VAT entered by the operator is a
    # financial anomaly; matching its printed amount is still valid OCR.
    result['financial_warnings'] = (
        ['Напечатанная сумма НДС расходится с расчётом по ставке; возможна ошибка оператора.']
        if vat_ok is False else [])
    result['numeric_candidate_book']=book.export()
    matrix=solve_matrix(result['rows'],book,vat_rate)
    result['numeric_matrix']=matrix
    result['vat_rate_diagnostic']=matrix.get('vat_rate_diagnostic')
    for assessment in matrix.get('model_assessments',[]):
        rid=assessment['row_id']; field=assessment['field']
        verified = next((x for x in manual_truth if x['row_id']==rid and x['field']==field and x['valid']),None)
        if verified and assessment['model']!='user':
            actual=parse_number(verified['after'])
            assessment['assessment']='user_confirmed_match' if (actual is not None and
                assessment['observed']==str(actual)) else 'user_confirmed_conflict'
    result['numeric_user_truth']=manual_truth
    # Manual feedback is independent of whether matrix search found a solution.
    feedback=[]
    for truth in manual_truth:
        if not truth['valid']:
            continue
        target=parse_number(truth['after'])
        observations=(book.footer if truth['field']=='receipt_total' else
             book.vat if truth['field']=='printed_vat' else
             book.cells.get(book.key(truth['row_id'],truth['field']),{}).get('observations',[]))
        for obs in observations:
            if obs.get('authority')=='user_confirmed':
                continue
            feedback.append({'row_id':truth['row_id'],'field':truth['field'],
                             'model':obs.get('model'),'path':obs.get('path'),
                             'observed':obs.get('number'),'user_truth':str(target),
                             'assessment':('user_confirmed_match' if obs.get('number')==str(target)
                                           else 'user_confirmed_conflict')})
    result['numeric_model_feedback']=feedback
    result['ocr_validation_status']=('consistent' if math_match and not row_mismatch and
        (not has_tax or vat_printed_match is True) and not matrix['review_required'] else 'review')
    # Ignore outdated VAT-rate warnings from earlier OCR phases, but do not
    # suppress independent geometry, missing-data or OCR warnings.
    nonfinancial_warnings=[w for w in result.get('geometry',{}).get('warnings',[])
        if w != 'Сумма НДС не совпадает с расчётом по установленной ставке.']
    result['status']='ok' if (math_match and not row_mismatch and
                               (not has_tax or vat_printed_match is True) and
                               not any(not e['valid'] for e in manual_truth) and
                               not matrix['review_required'] and
                               not nonfinancial_warnings) else 'review'
    return result
