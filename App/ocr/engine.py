"""RapidOCR 3.x adapter. Models are initialized once and shared by all cells."""
import threading
import numpy as np
import cv2
import re
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, OCRVersion, ModelType
from ..config import positive_int

_engine = None
_english_engine = None
_lock = threading.RLock()


def engine():
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = RapidOCR(params={
                    'Rec.lang_type': LangRec.CYRILLIC,
                    'Rec.ocr_version': OCRVersion.PPOCRV5,
                    'Rec.model_type': ModelType.MOBILE,
                    'Det.ocr_version': OCRVersion.PPOCRV5,
                    'Det.model_type': ModelType.MOBILE,
                    'Global.use_cls': False,
                    'Global.text_score': 0.25,
                    'EngineConfig.onnxruntime.intra_op_num_threads': positive_int('OCR_ONNX_THREADS', 2),
                    'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                })
    return _engine


def read_full(image):
    if image is None or not image.size or min(image.shape[:2]) < 4: return []
    with _lock:
        result = engine()(image, use_det=True, use_cls=False)
    if result.boxes is None or result.txts is None: return []
    lines = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        text, score = str(text), float(score)
        # A separate English recognizer is used only for Latin-only text
        # segments; it can never overwrite a segment containing Cyrillic.
        if len(re.findall(r'[A-Za-z]', text)) >= 2 and not re.search(r'[А-Яа-яЁё]', text):
            x0, y0 = np.floor(np.min(box, axis=0)).astype(int)
            x1, y1 = np.ceil(np.max(box, axis=0)).astype(int)
            crop = image[max(0,y0):min(image.shape[0],y1+1), max(0,x0):min(image.shape[1],x1+1)]
            if crop.size:
                with _lock:
                    alternate = english_engine()(crop, use_det=False, use_cls=False)
                if alternate.txts and alternate.scores[0] > score + .03:
                    candidate = str(alternate.txts[0])
                    if re.search(r'[A-Za-z]', candidate):
                        text, score = candidate, float(alternate.scores[0])
        lines.append({'text': text, 'confidence': score, 'bbox': np.asarray(box).round(2).tolist()})
    return lines


def english_engine():
    global _english_engine
    if _english_engine is None:
        with _lock:
            if _english_engine is None:
                _english_engine = RapidOCR(params={
                    'Rec.lang_type': LangRec.EN, 'Rec.ocr_version': OCRVersion.PPOCRV5,
                    'Rec.model_type': ModelType.MOBILE, 'Global.use_det': False,
                    'Global.use_cls': False,
                    'EngineConfig.onnxruntime.intra_op_num_threads': positive_int('OCR_ONNX_THREADS', 2),
                    'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                })
    return _english_engine


def read_line(image):
    """Recognize one observed line without running the text detector."""
    if image is None or not image.size: return None
    with _lock:
        result = engine()(image, use_det=False, use_cls=False)
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
    if not 2 <= len(bands) <= 4:
        return None
    blocks = []
    for start,end in bands:
        top, bottom = max(0,start-2), min(image.shape[0],end+2)
        crop = image[top:bottom,:]
        with _lock:
            recognized = engine()(crop, use_det=False, use_cls=False)
        if not recognized.txts or recognized.scores[0] < .60 or not re.search(r'\w',recognized.txts[0]):
            return None
        text, score = str(recognized.txts[0]), float(recognized.scores[0])
        if len(re.findall(r'[A-Za-z]',text)) >= 2 and not re.search(r'[А-Яа-яЁё]',text):
            with _lock:
                alternative = english_engine()(crop, use_det=False, use_cls=False)
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
    # Numeric cells and short single-line names benefit from recognition of the
    # whole physical cell: a detector can otherwise miss small digits entirely.
    if column_id == 'name' and not force_single and _has_separate_numeric_lines(image):
        segmented = _read_observed_name_lines(image, row_id, column_id, offset)
        if segmented is not None:
            return segmented
    if force_single or column_id != 'name' or image.shape[0] < 45 or not _has_separate_numeric_lines(image):
        with _lock:
            direct = engine()(image, use_det=False, use_cls=False)
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
