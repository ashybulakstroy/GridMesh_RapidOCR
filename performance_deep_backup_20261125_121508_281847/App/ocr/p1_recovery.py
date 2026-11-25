"""Selective numeric OCR recovery with an audit trail.

Never fabricate missing digits from arithmetic. A valid retry can replace an
invalid reading without a printed row total; other replacements require an
independently printed row total to support them.
"""
from decimal import Decimal
import cv2
from .engine import read_cell
from .p1_batch import read_batches
from .p1_numbers import parse_number, arithmetic_equal
from .numeric_roi import prepare_numeric_input
from ..config import setting


def _retry(cell, image):
    roi = None
    if cell['column_id'] == 'price' and setting('OCR_PRICE_CURRENCY_CROP','1') == '1':
        image,roi = prepare_numeric_input(image,cell['column_id'])
    enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(enlarged, 12, 12, 12, 12,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    candidate = read_cell(padded, cell['row_id'], cell['column_id'], (0, 0), force_single=True)
    candidate['strategy'] = 'p1_selective_2x'
    if roi is not None:
        candidate['numeric_roi'] = roi
    candidate['source_bbox'] = cell.get('source_bbox')
    candidate['bbox'] = cell['bbox']
    candidate['blocks'] = [{'text':candidate['text'], 'confidence':candidate['confidence'],
                            'bbox':cell['bbox'], 'strategy':'p1_selective_2x'}]
    return candidate


def _numeric(row, field):
    return parse_number(row['cells'][field]['text'])


def _recalculate(row):
    q, p = _numeric(row, 'quantity'), _numeric(row, 'price')
    printed = _numeric(row, 'printed_total') if 'printed_total' in row['cells'] else None
    row['line_total'] = str(q*p) if q is not None and p is not None else None
    row['printed_line_total'] = str(printed) if printed is not None else None
    row['printed_line_match'] = (arithmetic_equal(q*p, printed) if
                                  printed is not None and q is not None and p is not None else None)


def _consider(row, field, image, reason):
    """Save both readings and accept a valid retry according to available evidence."""
    original = row['cells'][field]
    alternative = _retry(original, image)
    old = parse_number(original['text'])
    new = parse_number(alternative['text'])
    tried = dict(reason=reason, original_text=original['text'],
                 original_confidence=original.get('confidence'),
                 alternative_text=alternative['text'],
                 alternative_confidence=alternative['confidence'],
                 alternative_valid=new is not None, different=(new is not None and new != old),
                 accepted=False)
    attempts = original.setdefault('ocr_attempts', [])
    attempts.append(tried)
    if new is None or new == old or alternative['confidence'] < .70:
        return False, tried
    if 'printed_total' not in row['cells']:
        if old is None and reason == 'invalid_number':
            row['cells'][field] = alternative
            _recalculate(row)
            tried.update(accepted=True, confirmation='valid_retry_without_printed_row_total')
            alternative['ocr_attempts'] = attempts
            return True, tried
        return False, tried
    # A low-confidence printed figure is not sufficient verification.
    printed_evidence = alternative if field == 'printed_total' else row['cells']['printed_total']
    if float(printed_evidence.get('confidence') or 0) < .75:
        return False, tried
    # A value already passing row arithmetic must not be replaced.
    if row.get('printed_line_match') is True:
        return False, tried
    row['cells'][field] = alternative
    _recalculate(row)
    confirmed = row['printed_line_match'] is True
    if confirmed:
        tried['accepted'] = True
        tried['confirmation'] = 'independent_printed_row_total'
        alternative['ocr_attempts'] = attempts
        return True, tried
    row['cells'][field] = original
    _recalculate(row)
    return False, tried


def recover_invalid_and_rows(rows, crop_map, *, max_retries=8):
    """Risk-prioritized retries, capped per receipt, with all alternatives logged."""
    for row in rows:
        _recalculate(row)
    options = []
    for row in rows:
        for field in ('quantity', 'price', 'printed_total'):
            if field not in row['cells'] or (row['row_id'], field) not in crop_map:
                continue
            cell = row['cells'][field]
            value = _numeric(row,field)
            confidence = float(cell.get('confidence') or 0)
            if value is None:
                priority, reason = 0, 'invalid_number'
            elif row.get('printed_line_match') is False:
                priority, reason = 1, 'printed_row_mismatch'
            else:
                continue
            options.append((priority, confidence, row['row_id'], field, row, reason))
    options.sort(key=lambda item:(item[0], item[1], item[2]))
    attempts, changed, candidates, details = 0, [], [], []
    for _,_,_,field,row,reason in options:
        if attempts >= max_retries:
            break
        # Previous recovery may have fixed an arithmetic mismatch.
        if reason == 'printed_row_mismatch' and row.get('printed_line_match') is True:
            continue
        attempts += 1
        accepted, info = _consider(row, field, crop_map[(row['row_id'],field)],reason)
        details.append({'row_id':row['row_id'], 'column_id':field, **info})
        if accepted:
            changed.append((row['row_id'], field, 'printed_row_match' if
                            info['confirmation'] == 'independent_printed_row_total' else
                            info['confirmation']))
        elif info['different'] and info['alternative_confidence'] >= .70:
            candidates.append({'row_id':row['row_id'], 'field':field,
                               'text':info['alternative_text'], 'reason':reason})
    return {'attempted':attempts, 'changed':changed, 'candidates':candidates, 'details':details}


def recover_total_mismatch(rows, crop_map, expected_total, *, max_retries=8):
    """Seek OCR alternatives on footer mismatch; never use footer alone to replace."""
    if expected_total is None:
        return {'attempted':0,'changed':[],'candidates':[]}
    for row in rows:
        _recalculate(row)
    if any(row['line_total'] is None for row in rows):
        return {'attempted':0,'changed':[],'candidates':[]}
    current = sum((Decimal(row['line_total']) for row in rows), Decimal(0))
    if arithmetic_equal(current, expected_total):
        return {'attempted':0,'changed':[],'candidates':[]}
    options = []
    for row in rows:
        for field in ('quantity','price'):
            if (row['row_id'],field) not in crop_map:
                continue
            # Avoid retrying the same region twice; preserve the initial evidence.
            cell = row['cells'][field]
            if cell.get('ocr_attempts'):
                continue
            score = float(cell.get('confidence') or 0)
            options.append((score,row['row_id'],field,row))
    options.sort(key=lambda item:(item[0],item[1]))
    attempts, changed, candidates, details = 0, [], [], []
    for _,_,field,row in options[:max_retries]:
        attempts += 1
        accepted, info = _consider(row,field,crop_map[(row['row_id'],field)],'footer_mismatch')
        details.append({'row_id':row['row_id'], 'column_id':field, **info})
        if accepted:
            changed.append((row['row_id'],field,'printed_row_match'))
        elif info['different'] and info['alternative_confidence'] >= .70:
            candidates.append({'row_id':row['row_id'],'field':field,
                               'text':info['alternative_text'], 'reason':'footer_mismatch'})
    return {'attempted':attempts,'changed':changed,'candidates':candidates,'details':details}
