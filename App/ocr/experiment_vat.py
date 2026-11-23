"""Independent OCR observations of the *printed* VAT amount from original pixels.

A VAT amount is a photographed figure, NOT one computed from percentage. Only a
printed line total, or a price with an observed quantity=1 on the VAT row,
can be used as a standalone printed-VAT OCR witness. If neither exists the
scanner returns 'unavailable' rather than inventing an observed VAT amount.
"""
from __future__ import annotations

from decimal import Decimal
from time import perf_counter
import cv2

from .numeric_roi import crop_numeric_ink
from .p1_numbers import parse_number
from .experiment_rec_trace import rec_scope
from .vat_witness import vat_source, seed_printed_vat, observed_vat_text


def _prep(image, mode):
    if mode == 'raw':
        return image
    if mode == '2x':
        return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    if mode=='otsu':
        _,out=cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return out
    if mode=='clahe':
        return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    raise ValueError(mode)


def scan_printed_vat(rows, cache, book, vat_rate, *, read_line=None,
                     model_pool=None, max_probes=6, always_check=True):
    """Same evidentiary standard as footer: multiple OCR readings, no auto-fix.

    Independent model probes are prioritized after the first raw read so a
    tight probe budget does not get consumed by preprocessing of the same model.
    """
    if read_line is None:
        from .engine import read_line
    seed=seed_printed_vat(rows,book)
    out={'source':seed['source'], 'source_status':seed['status'],
         'observations':[], 'probes':0, 'errors':[], 'budget_exhausted':False}
    if seed['source'] is None:
        return out
    key=(seed['source']['row_id'],seed['source']['field'])
    image=cache.crops.get(key)
    if image is None or not image.size:
        out['source_status']='original_vat_roi_missing'
        return out
    row=next(x for x in rows if x['row_id']==key[0])
    initial=parse_number(row['cells'][key[1]].get('text'))
    # Match the independently photographed VAT to the printed tax row.
    # The nominal rate is *not* an OCR reliability criterion: an operator may
    # print a VAT amount inconsistent with 16% while OCR reads it correctly.
    current_tax=Decimal(0)
    product_total=Decimal(0)
    complete=True
    for item in rows:
        amount=item.get('line_total')
        if amount is None:
            complete=False
            continue
        if item.get('kind')=='tax':current_tax+=Decimal(str(amount))
        else:product_total+=Decimal(str(amount))
    baseline_ok=(complete and initial is not None and
                 abs(current_tax-initial)<=Decimal('.01'))
    if baseline_ok and not always_check:
        out['source_status']='primary_consistent_not_independently_rescanned'
        return out
    try:
        prepared,metadata=crop_numeric_ink(image)
        out['numeric_roi']=metadata
    except Exception as exc:
        prepared=image
        out['errors'].append(f'crop_numeric_ink: {type(exc).__name__}: {exc}')
    if not prepared.size:
        out['errors'].append('empty_vat_crop')
        return out

    def add_variant(label, mode):
        start=perf_counter()
        try:
            with rec_scope([key],'vat_original_variants'):
                response=read_line(_prep(prepared,mode)) or {}
            val=parse_number(response.get('text'))
            record={'strategy':label, 'text':response.get('text'),
                    'vat':str(val) if val is not None else None,
                    'confidence':response.get('confidence'),
                    'model':'PP-OCRv5 Cyrillic Mobile / RapidOCR 3.x',
                    'wall_ms':round((perf_counter()-start)*1000,3)}
            if val is not None:
                book.add_vat(record['text'], record['confidence'], label,
                             model=record['model'],row_id=key[0],field=key[1])
        except Exception as exc:
            record={'strategy':label,'vat':None,
                    'error':f'{type(exc).__name__}: {exc}',
                    'wall_ms':round((perf_counter()-start)*1000,3)}
            out['errors'].append(record['error'])
        out['observations'].append(record)
        out['probes']+=1

    if out['probes']<max_probes:
        add_variant('vat_original_raw_rec','raw')
    # Different recognizer, not merely a different image transformation.
    if model_pool is not None:
        for model_id in model_pool.models_for('vat'):
            if out['probes']>=max_probes:break
            with rec_scope([key], 'vat_independent_model'):
                response=model_pool.recognize(model_id,prepared,'printed_vat_original_roi')
            val=parse_number(response.get('text'))
            record={'strategy':'vat_original_model_'+model_id,
                    'text':response.get('text'),
                    'vat':str(val) if val is not None else None,
                    'confidence':response.get('confidence'),
                    'model_id':model_id,'model':response.get('model',model_id),
                    'wall_ms':response.get('wall_ms'),
                    **({'error':response['error']} if 'error' in response else {})}
            out['observations'].append(record)
            out['probes']+=1
            if 'error' in record:out['errors'].append(record['error'])
            if val is not None:
                book.add_vat(record['text'],record['confidence'],record['strategy'],
                             model=model_id,row_id=key[0],field=key[1])
    values={r['number'] for r in book.vat if r['number'] is not None}
    conflict=len(values)>1
    # Once consistent, avoid repeated recognition by the SAME model.
    if not baseline_ok or conflict:
        for mode in ('2x','otsu','clahe'):
            if out['probes']>=max_probes:break
            add_variant('vat_original_'+mode+'_rec',mode)
    out['budget_exhausted']=out['probes']>=max_probes and (not baseline_ok or conflict)
    out['source_status']='rescanned'
    out['vat_variants']=sorted({r['number'] for r in book.vat if r['number'] is not None},
                               key=Decimal)
    return out
