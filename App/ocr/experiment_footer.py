"""Footer-only OCR cross-check. Original pixels; variant array, not a guessed total.

Uses the RapidOCR engines already available in this project. Additional
preprocessing variants are NOT independent neural networks.
"""
from __future__ import annotations
import re
from decimal import Decimal
from time import perf_counter
import cv2
from .p1_numbers import parse_number
from .experiment_rec_trace import rec_scope

_PATTERN = re.compile(r'^[\d\s.,]+\s*(?:[ТT₸]|тг\.?)?\s*$',re.I)


def _check_text(text):
    if text is None:
        return None
    txt=str(text).strip()
    if not _PATTERN.fullmatch(txt):
        return None
    num=parse_number(txt)
    return str(num) if num is not None and num>0 else None


def scan_footer(image, y_bottom, current_total, current_sum, *, read_line=None,
                read_full=None, always_check=True, max_probes=6, model_pool=None):
    """At least one original-pixel cross-check; deeper only on uncertainty.

    Never return a replacement total; return only independently OCR-observed
    candidate values to be considered in the bounded matrix solver.
    """
    if read_line is None or read_full is None:
        from .engine import read_line as engine_line, read_full as engine_full
        read_line=read_line or engine_line
        read_full=read_full or engine_full
    h,w=image.shape[:2]
    current=parse_number(current_total)
    mismatch=(current is None or abs(current-Decimal(str(current_sum)))>Decimal('.01'))
    if not mismatch and not always_check:
        return []
    result=[]
    # First REC-only probes retain source detail. Crops deliberately overlap
    # and differ in position/scale so that table lines do not dominate all.
    presets=(('footer_source_rec',.48,-10,.06,1.0,'raw'),
             ('footer_source_2x',.43,-18,.085,2.0,'raw'),
             ('footer_source_otsu',.43,-23,.085,1.0,'otsu'),
             ('footer_source_clahe',.48,-14,.07,1.0,'clahe'))
    for strategy, left_fraction, offset, vertical, scale, method in presets:
        if len(result)>=max_probes:
            break
        if not mismatch and len(result)>=2 and all(x.get('total')==str(current) for x in result):
            break
        x0=int(w*left_fraction)
        y0=max(0,y_bottom+offset)
        y1=min(h,y0+max(14,int(w*vertical)))
        crop=image[y0:y1,x0:]
        if crop.size==0:
            continue
        start=perf_counter()
        try:
            if method!='raw':
                gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
                if method=='otsu':
                    _,crop=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                else:
                    crop=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
            if scale != 1:
                crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            with rec_scope(['receipt_total'], "footer_source_variants"):
                response=read_line(crop)
            line=response or {}
            result.append({'strategy':strategy,'total':_check_text(line.get('text')),
                'source_roi_bbox':[x0,y0,w,y1], 'preprocessing':method,'scale':scale,
                'text':line.get('text'),'confidence':line.get('confidence'),
                'model':'PP-OCRv5 Cyrillic Mobile / RapidOCR 3.x',
                'wall_ms':round((perf_counter()-start)*1000,3)})
        except Exception as exc:
            result.append({'strategy':strategy,'total':None,
                'error':f'{type(exc).__name__}: {exc}',
                'wall_ms':round((perf_counter()-start)*1000,3)})
    variants={x['total'] for x in result if x.get('total') is not None}
    if mismatch or (variants and (current is None or variants!={str(current)})):
        if len(result)<max_probes:
            x0=int(w*.42);y0=max(0,y_bottom-27)
            y1=min(h,max(y0+15,y_bottom+int(w*.08)))
            start=perf_counter()
            try:
                with rec_scope(['receipt_total'], "footer_source_detector"):
                    lines=read_full(image[y0:y1,x0:],footer_mode=True)
                # The crop overlaps the last table row and VAT.  Retain only
                # the printed footer baseline closest to the table bottom;
                # never treat every number in the crop as a receipt total.
                observed=[]
                for line in lines:
                    if _check_text(line.get('text')) is None or not line.get('bbox'):
                        continue
                    box=line['bbox']
                    local_y=sum(float(p[1]) for p in box)/len(box)
                    local_x=min(float(p[0]) for p in box)
                    source_y=y0+local_y
                    if y_bottom-12 <= source_y <= y1:
                        observed.append((source_y,local_x,line))
                if observed:
                    anchor=min(observed,key=lambda item:abs(item[0]-y_bottom))
                    baseline=sorted((x for x in observed if abs(x[0]-anchor[0])<=12),
                                    key=lambda item:item[1])
                    combined=''.join(x[2]['text'].strip() for x in baseline)
                    value=_check_text(combined)
                    if value is not None:
                        result.append({'strategy':'footer_source_detector_baseline',
                            'total':value,'text':combined,
                            'confidence':min(float(x[2].get('confidence') or 0) for x in baseline),
                            'model':'PP-OCRv5 Cyrillic Mobile / RapidOCR 3.x',
                            'wall_ms':round((perf_counter()-start)*1000,3)})
            except Exception as exc:
                result.append({'strategy':'footer_source_detector', 'total':None,
                    'error':f'{type(exc).__name__}: {exc}',
                    'wall_ms':round((perf_counter()-start)*1000,3)})
    # E2: alternative RapidOCR REC models, not Tesseract. Use original pixels.
    # All observations remain candidates; no reading can silently replace total.
    if model_pool is not None:
        x0=int(w*.45); y0=max(0,y_bottom-12)
        y1=min(h,max(y0+12,y_bottom+int(w*.065)))
        source=image[y0:y1,x0:]
        if source.size:
            for model_id in model_pool.models_for('footer'):
                if len(result)>=max_probes:
                    break
                with rec_scope(['receipt_total'],'footer_independent_model'):
                    probe=model_pool.recognize(model_id,source,'footer_source_model')
                result.append({'strategy':'footer_source_model_'+model_id,
                    'total':_check_text(probe.get('text')),
                    'text':probe.get('text'), 'confidence':probe.get('confidence'),
                    'model':probe.get('model',model_id), 'model_id':model_id,
                    'wall_ms':probe.get('wall_ms'),
                    **({'error':probe['error']} if 'error' in probe else {})})
    return result
