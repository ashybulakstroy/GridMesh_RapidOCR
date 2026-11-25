"""RapidOCR 3.x adapter. Models are initialized once and shared by all cells."""
import threading
from contextvars import ContextVar
from contextlib import contextmanager
from time import perf_counter
from .experiment_rec_trace import record_rec
import numpy as np
import cv2
import re
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, OCRVersion, ModelType
from ..config import positive_int, setting

_LANE = ContextVar('ocr_engine_lane', default='table')
_NAME_MODEL = ContextVar('ocr_name_model', default='v5_cyrillic_mobile')
_LAST_REC_MODEL = ContextVar('ocr_last_rec_model', default=None)
_models = {}
_locks = {lane: threading.RLock() for lane in
          ('table', 'footer', 'table_names', 'table_numeric')}


class _LaneLock:
    def __enter__(self):
        return _locks[_LANE.get()].__enter__()
    def __exit__(self, *args):
        return _locks[_LANE.get()].__exit__(*args)


_lock = _LaneLock()


@contextmanager
def inference_lane(lane):
    """Separate cached sessions and locks; never share mutable RapidOCR workers."""
    token = _LANE.set(lane)
    try:
        yield
    finally:
        _LANE.reset(token)


def onnx_threads():
    """Divide the existing table thread budget between simultaneous columns."""
    configured = positive_int('OCR_ONNX_THREADS', 2)
    lane = _LANE.get()
    if lane == 'table_names':
        table_budget = positive_int('OCR_TABLE_ONNX_THREADS', configured)
        return max(1, (table_budget + 1) // 2)
    if lane == 'table_numeric':
        table_budget = positive_int('OCR_TABLE_ONNX_THREADS', configured)
        return max(1, table_budget // 2)
    return configured


def engine():
    key = (_LANE.get(), 'primary')
    if key not in _models:
        with _lock:
            if key not in _models:
                _models[key] = RapidOCR(params={
                    'Rec.lang_type': LangRec.CYRILLIC,
                    'Rec.ocr_version': OCRVersion.PPOCRV5,
                    'Rec.model_type': ModelType.MOBILE,
                    'Rec.rec_batch_num': positive_int('OCR_REC_BATCH_NUM', 8),
                    'Det.ocr_version': OCRVersion.PPOCRV5,
                    'Det.model_type': ModelType.MOBILE,
                    'Global.use_cls': False,
                    'Global.text_score': 0.25,
                    'EngineConfig.onnxruntime.intra_op_num_threads': onnx_threads(),
                    'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                })
    return _models[key]


def footer_engine():
    """Detector profile for short, wide receipt footer regions."""
    key = (_LANE.get(), 'detector')
    if key not in _models:
        with _lock:
            if key not in _models:
                _models[key] = RapidOCR(params={
                    'Rec.lang_type': LangRec.CYRILLIC,
                    'Rec.ocr_version': OCRVersion.PPOCRV5,
                    'Rec.model_type': ModelType.MOBILE,
                    'Rec.rec_batch_num': positive_int('OCR_REC_BATCH_NUM', 8),
                    'Det.ocr_version': OCRVersion.PPOCRV5,
                    'Det.model_type': ModelType.MOBILE,
                    'Det.limit_type': 'max',
                    'Det.limit_side_len': 736,
                    'Global.use_cls': False,
                    'Global.text_score': 0.25,
                    'EngineConfig.onnxruntime.intra_op_num_threads': onnx_threads(),
                    'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                })
    return _models[key]


def _run_cyrillic_rec(image):
    start=perf_counter()
    try:
        model_id=_NAME_MODEL.get()
        if model_id in ('v6_tiny_cyrillic_optional','v6_tiny_official'):
            from .experiment_models import ExperimentModelPool, load_catalog
            key=(_LANE.get(),model_id)
            if key not in _models:
                pool=ExperimentModelPool()
                _models[key]=pool._get_session(model_id,load_catalog()[model_id])
            session=_models[key]
            result=session(image,use_det=False,use_cls=False)
        else:
            result=engine()(image,use_det=False,use_cls=False)
        if model_id != 'v5_cyrillic_mobile' and not result.txts:
            raise RuntimeError('Модель вернула пустой результат')
        _LAST_REC_MODEL.set(model_id)
        record_rec(model=model_id,ms=(perf_counter()-start)*1000,image=image,result=result)
        return result
    except Exception as exc:
        record_rec(model=_NAME_MODEL.get(),ms=(perf_counter()-start)*1000,
                   success=False,note=str(exc),image=image)
        if _NAME_MODEL.get() != 'v5_cyrillic_mobile':
            from .debug_performance import report_issue
            report_issue(source='RapidOCR',operation='primary_rec',kind='model_exception',error=exc,
                         context={'model_id':_NAME_MODEL.get()},fallback='v5_cyrillic_mobile')
            start=perf_counter()
            result=engine()(image,use_det=False,use_cls=False)
            _LAST_REC_MODEL.set('v5_cyrillic_mobile')
            record_rec(model='v5_cyrillic_mobile',ms=(perf_counter()-start)*1000,image=image,result=result)
            return result
        raise

def _run_english_rec(image):
    start=perf_counter()
    try:
        result=english_engine()(image,use_det=False,use_cls=False)
        record_rec(model='v5_english_mobile',ms=(perf_counter()-start)*1000,image=image,result=result)
        return result
    except Exception as exc:
        record_rec(model='v5_english_mobile',ms=(perf_counter()-start)*1000,
                   success=False,note=str(exc),image=image)
        raise


def read_full(image, *, footer_mode=False, cell_mode=False):
    if image is None or not image.size or min(image.shape[:2]) < 4: return []
    scale = max(1., 160/min(image.shape[:2])) if cell_mode else 1.
    if scale > 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    with _lock:
        clock=perf_counter()
        try:
            result = (footer_engine() if footer_mode or cell_mode else engine())(image, use_det=True, use_cls=False)
            record_rec(mode='DET+REC',model='v5_cyrillic_mobile',ms=(perf_counter()-clock)*1000,
                       note='Detector may dispatch multiple internal REC calls; exact count unavailable',image=image,result=result)
        except Exception as exc:
            record_rec(mode='DET+REC',model='v5_cyrillic_mobile',ms=(perf_counter()-clock)*1000,
                       success=False,note=str(exc),image=image)
            raise
    if result.boxes is None or result.txts is None: return []
    lines = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        text, score = str(text), float(score)
        # A separate English recognizer is used only for Latin-only text
        # segments; it can never overwrite a segment containing Cyrillic.
        if (setting('OCR_NAME_ENGLISH_FALLBACK', '0') == '1' and
                len(re.findall(r'[A-Za-z]', text)) >= 2 and
                not re.search(r'[А-Яа-яЁё]', text)):
            x0, y0 = np.floor(np.min(box, axis=0)).astype(int)
            x1, y1 = np.ceil(np.max(box, axis=0)).astype(int)
            crop = image[max(0,y0):min(image.shape[0],y1+1), max(0,x0):min(image.shape[1],x1+1)]
            if crop.size:
                with _lock:
                    alternate = _run_english_rec(crop)
                if alternate.txts and alternate.scores[0] > score + .03:
                    candidate = str(alternate.txts[0])
                    if re.search(r'[A-Za-z]', candidate):
                        text, score = candidate, float(alternate.scores[0])
        lines.append({'text': text, 'confidence': score, 'bbox': np.asarray(box).round(2).tolist()})
    if scale > 1:
        for line in lines:
            line['bbox'] = [[x/scale,y/scale] for x,y in line['bbox']]
    return lines


def english_engine():
    key = (_LANE.get(), 'english')
    if key not in _models:
        with _lock:
            if key not in _models:
                _models[key] = RapidOCR(params={
                    'Rec.lang_type': LangRec.EN, 'Rec.ocr_version': OCRVersion.PPOCRV5,
                    'Rec.model_type': ModelType.MOBILE, 'Global.use_det': False,
                    'Global.use_cls': False,
                    'EngineConfig.onnxruntime.intra_op_num_threads': onnx_threads(),
                    'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                })
    return _models[key]


def read_line(image):
    """Recognize one observed line without running the text detector."""
    if image is None or not image.size: return None
    with _lock:
        result = _run_cyrillic_rec(image)
    if not result.txts: return None
    return {'text':str(result.txts[0]), 'confidence':float(result.scores[0])}


def _has_separate_numeric_lines(image):
    """Use image evidence to decide whether a numeric cell needs detection."""
    if image.shape[0] < 35 or image.shape[1] < 6:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[:2,:] = 0; binary[-2:,:] = 0; binary[:,:2] = 0; binary[:,-2:] = 0
    density = np.count_nonzero(binary, axis=1)
    mask = density >= max(2, image.shape[1]*.025)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    bands = [(a,b) for a,b in zip(starts,ends) if b-a >= 5]
    return any(b[0]-a[1] >= 4 for a,b in zip(bands,bands[1:]))


def _read_observed_name_lines(image, row_id, column_id, offset):
    """Recognize physically separated text bands; fall back when uncertain."""
    if image.shape[0] < 30 or image.shape[1] < 40:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[:2,:]=0; binary[-2:,:]=0; binary[:,:2]=0; binary[:,-2:]=0
    mask = np.count_nonzero(binary,axis=1) >= max(2, image.shape[1]*.015)
    edges = np.diff(np.r_[False,mask,False].astype(np.int8))
    bands = []
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if bands and start - bands[-1][1] <= 3:
            bands[-1] = (bands[-1][0], end)
        else:
            bands.append((start,end))
    bands = [(int(a),int(b)) for a,b in bands if b-a >= 6]
    if not 2 <= len(bands) <= 8:
        return None
    blocks = []
    for start,end in bands:
        top, bottom = max(0,start-2), min(image.shape[0],end+2)
        crop = image[top:bottom,:]
        with _lock:
            recognized = _run_cyrillic_rec(crop)
        if not recognized.txts or recognized.scores[0] < .60 or not re.search(r'\w',recognized.txts[0]):
            return None
        text, score = str(recognized.txts[0]), float(recognized.scores[0])
        if (setting('OCR_NAME_ENGLISH_FALLBACK', '0') == '1' and
                len(re.findall(r'[A-Za-z]',text)) >= 2 and
                not re.search(r'[А-Яа-яЁё]',text)):
            with _lock:
                alternative = _run_english_rec(crop)
            if alternative.txts and alternative.scores[0] > score+.03 and re.search(r'[A-Za-z]',alternative.txts[0]):
                text, score = str(alternative.txts[0]), float(alternative.scores[0])
        blocks.append(dict(text=text, confidence=score,
            bbox=[offset[0], offset[1]+top, offset[0]+image.shape[1], offset[1]+bottom],
            cell_id=f'{row_id}-{column_id}', row_id=row_id, column_id=column_id))
    return {'text':' '.join(x['text'] for x in blocks), 'confidence':min(x['confidence'] for x in blocks),
            'bbox':[offset[0],offset[1],offset[0]+image.shape[1],offset[1]+image.shape[0]],
            'cell_id':f'{row_id}-{column_id}', 'row_id':row_id, 'column_id':column_id,
            'strategy':'observed_line_segments', 'blocks':blocks}


def read_cell(image, row_id, column_id, offset, *, force_single=False):
    # Experiment: one Cyrillic model for all three primary table fields.
    # Set OCR_UNIFIED_TINY_ENABLE=0 to compare with the previous primary routing.
    model = ('v6_tiny_cyrillic_optional'
             if column_id == 'name' or (
                 column_id in ('quantity', 'price') and
                 setting('OCR_UNIFIED_TINY_ENABLE', '1') == '1')
             else 'v6_tiny_official'
             if column_id in ('quantity', 'price')
             else 'v5_cyrillic_mobile')
    token = _NAME_MODEL.set(model)
    last_token = _LAST_REC_MODEL.set(None)
    try:
        fast_mode = setting('OCR_TEXT_FAST_MODE','E').upper()
        if column_id == 'name' and not force_single and fast_mode != 'A' and (fast_mode == 'G' or row_id != _TEXT_LAST_ROW.get()):
            from .text_selective import read_name
            return read_name(image, row_id, column_id, offset, primary=_run_cyrillic_rec,
                             secondary=_text_selective_v5, last_model=_LAST_REC_MODEL.get,
                             lock=_lock, full=read_full)
        result = _read_cell(image, row_id, column_id, offset, force_single=force_single)
        result['model_id'] = _LAST_REC_MODEL.get() or 'v5_cyrillic_mobile'
        return result
    finally:
        _LAST_REC_MODEL.reset(last_token)
        _NAME_MODEL.reset(token)


def _read_cell(image, row_id, column_id, offset, *, force_single=False):
    # Numeric cells and short single-line names benefit from recognition of the
    # whole physical cell: a detector can otherwise miss small digits entirely.
    if column_id == 'name' and not force_single and _has_separate_numeric_lines(image):
        segmented = _read_observed_name_lines(image, row_id, column_id, offset)
        if segmented is not None:
            return segmented
    # MULTILINE_VAT_FAST_E4_V1: split observed numeric rows BEFORE whole-cell OCR.
    # Never infer a digit from arithmetic; every joined part is actual REC text.
    if (column_id in ('quantity', 'price', 'printed_total') and
            setting('OCR_NUMERIC_MULTILINE_FAST', '1') == '1' and
            not force_single and _has_separate_numeric_lines(image)):
        from .numeric_multiline_fast import read_numeric_multiline
        with _lock:
            segmented_digits = read_numeric_multiline(image, _run_cyrillic_rec)
        if segmented_digits is not None:
            bbox = [offset[0],offset[1],offset[0]+image.shape[1],offset[1]+image.shape[0]]
            blocks = [dict(text=part['text'], confidence=part['confidence'],
                           bbox=[offset[0], offset[1]+part['top'],
                                 offset[0]+image.shape[1], offset[1]+part['bottom']],
                           cell_id=f'{row_id}-{column_id}', row_id=row_id,
                           column_id=column_id)
                      for part in segmented_digits['lines']]
            return {'text':segmented_digits['text'],
                    'confidence':segmented_digits['confidence'], 'bbox':bbox,
                    'cell_id':f'{row_id}-{column_id}', 'row_id':row_id,
                    'column_id':column_id, 'blocks':blocks,
                    'strategy':'numeric_multiline_rec_only',
                    'numeric_multiline':segmented_digits}
    if force_single or column_id != 'name' or image.shape[0] < 45 or not _has_separate_numeric_lines(image):
        with _lock:
            direct = _run_cyrillic_rec(image)
        if direct.txts:
            text, score = str(direct.txts[0]), float(direct.scores[0])
            if column_id in {'quantity', 'price'} and _has_separate_numeric_lines(image):
                parts = read_full(image)
                parts.sort(key=lambda item: min(p[1] for p in item['bbox']))
                if (len(parts) >= 2 and all(re.fullmatch(r'[\d,.\sTТ₸]+', p['text']) for p in parts)
                        and all(min(p[1] for p in b['bbox']) > min(p[1] for p in a['bbox'])+5 for a,b in zip(parts,parts[1:]))):
                    text = ''.join(p['text'].strip() for p in parts)
                    score = min(p['confidence'] for p in parts)
            bbox = [offset[0], offset[1], offset[0]+image.shape[1], offset[1]+image.shape[0]]
            block = dict(text=text, confidence=score, bbox=bbox, cell_id=f'{row_id}-{column_id}', row_id=row_id, column_id=column_id)
            return {**block, 'blocks': [block.copy()]}
    # Cell crops are often tiny; extra white border makes edge-adjacent text
    # detectable without drawing or inventing any OCR characters.
    factor = 2
    enlarged = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(enlarged, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255,255,255))
    lines = read_full(padded, cell_mode=True) if column_id == 'name' else read_full(padded)
    if column_id == 'name' and not lines:
        lines = read_full(padded)
    for line in lines:
        line['bbox'] = [[(x-12)/factor, (y-12)/factor] for x,y in line['bbox']]
    # Baseline grouping keeps words on the same line in reading order.
    lines.sort(key=lambda line: sum(p[1] for p in line['bbox'])/4)
    groups = []
    for line in lines:
        cy = sum(p[1] for p in line['bbox'])/4
        height = max(p[1] for p in line['bbox'])-min(p[1] for p in line['bbox'])
        if groups and abs(cy-groups[-1][0]) <= max(5, height*.6):
            groups[-1][1].append(line)
        else:
            groups.append((cy, [line]))
    lines = [line for _, group in groups for line in sorted(group, key=lambda x: min(p[0] for p in x['bbox']))]
    for line in lines:
        line['bbox'] = [[x+offset[0], y+offset[1]] for x, y in line['bbox']]
        line.update(cell_id=f'{row_id}-{column_id}', row_id=row_id, column_id=column_id)
    return {'text': ' '.join(line['text'] for line in lines),
            'confidence': min((line['confidence'] for line in lines), default=0.0),
            'bbox': [offset[0], offset[1], offset[0]+image.shape[1], offset[1]+image.shape[0]],
            'cell_id': f'{row_id}-{column_id}', 'row_id': row_id,
            'column_id': column_id, 'blocks': lines}

_TEXT_LAST_ROW = ContextVar('text_fast_last_row', default=None)

def _text_selective_v5(image):
    start=perf_counter()
    try:
        result=engine()(image,use_det=False,use_cls=False)
        _LAST_REC_MODEL.set('v5_cyrillic_mobile')
        record_rec(model='v5_cyrillic_mobile',ms=(perf_counter()-start)*1000,image=image,result=result)
        return result
    except Exception as exc:
        record_rec(model='v5_cyrillic_mobile',ms=(perf_counter()-start)*1000,success=False,note=str(exc),image=image)
        raise
