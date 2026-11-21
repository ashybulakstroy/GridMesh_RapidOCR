"""P1: Real RapidOCR 3.10.0 recognizer batching for physical receipt cells.

Use one existing cyrillic model; numeric/text are separate *logical* queues.
Do not create ONNX sessions per cell or spawn competing Python threads.
"""
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np

from .engine import engine, read_cell, _has_separate_numeric_lines, _lock
from .numeric_roi import crop_numeric_ink, split_price_currency
import re
from ..config import setting


@dataclass
class BatchCell:
    image: np.ndarray
    row_id: int
    column_id: str
    offset: tuple
    source_bbox: list


def _convert(cell, text, score):
    bbox = [cell.offset[0], cell.offset[1],
            cell.offset[0] + cell.image.shape[1],
            cell.offset[1] + cell.image.shape[0]]
    info = dict(text=str(text), confidence=float(score), bbox=bbox,
                cell_id=f'{cell.row_id}-{cell.column_id}',
                row_id=cell.row_id, column_id=cell.column_id,
                source_bbox=cell.source_bbox, strategy='p1_rec_batch')
    return {**info, 'blocks':[info.copy()]}


def read_batches(cells, *, batch_size=8, budget_multiplier=1):
    """Read one-line cells as batches, fallback to legacy for multiline cells.

    Returns (dict[(row_id, column_id)] -> legacy cell result, stats).
    Any ONNX batch incompatibility is disclosed and falls back to legacy OCR.
    """
    if batch_size not in (1, 2, 4, 6, 8, 12, 16):
        raise ValueError('OCR_REC_BATCH_NUM: допустимы 1, 2, 4, 6, 8, 12, 16')
    if budget_multiplier not in (1, 2):
        raise ValueError('Допустимый множитель бюджета: 1 или 2')
    pending = {'numeric': [], 'text': []}
    out = {}
    stats = {'requested':len(cells), 'batch_size':batch_size, 'batch_calls':0,
             'multiline_fallback':0, 'batch_fallback':0, 'errors': []}
    # Cooperative bound: stop starting optional ROI work after 16 ms per job.
    # An individual OpenCV call is synchronous and cannot be preempted.
    crop_limit_s = .012 * budget_multiplier
    roi_limit_s = .016 * budget_multiplier
    roi_seconds = 0.0
    roi_metadata = {}
    originals = {(c.row_id, c.column_id): c for c in cells}
    working = list(cells)
    roi_skipped = 0
    skipped_cells = []
    roi_enabled = setting('OCR_NUMERIC_ROI', '1') == '1'
    if roi_enabled:
        # The last row often carries VAT, where a misread unit price
        # multiplies the error by the entire tax amount.
        last_price = next((i for i in range(len(cells)-1, -1, -1)
                           if cells[i].column_id == 'price'), None)
        indices = list(range(len(cells)))
        if last_price is not None:
            indices.remove(last_price)
            indices.insert(0, last_price)
        for i in indices:
            cell = cells[i]
            if cell.column_id not in ('quantity','price','printed_total'):
                continue
            if roi_seconds >= crop_limit_s:
                roi_skipped += 1
                skipped_cells.append({'row_id':cell.row_id, 'column_id':cell.column_id})
            else:
                started = perf_counter()
                cropped, info = crop_numeric_ink(cell.image)
                roi_seconds += perf_counter() - started
                roi_metadata[(cell.row_id,cell.column_id)] = info
                working[i] = replace(cell, image=cropped)
    crop_seconds = roi_seconds
    stats.update(numeric_roi_ms=round(roi_seconds*1000,3),
                 numeric_roi_enabled=roi_enabled,
                 numeric_roi_crop_ms=round(crop_seconds*1000,3),
                 numeric_roi_budget_ms=round(roi_limit_s*1000),
                 numeric_roi_crop_budget_ms=round(crop_limit_s*1000),
                 numeric_roi_skipped=roi_skipped,
                 numeric_roi_skipped_cells=skipped_cells,
                 numeric_roi_crop_exceeded=crop_seconds > crop_limit_s,
                 numeric_roi_changed=sum(info.get('changed',False) for info in roi_metadata.values()))
    for cell in working:
        key = (cell.row_id, cell.column_id)
        if _has_separate_numeric_lines(cell.image):
            item = read_cell(cell.image, cell.row_id, cell.column_id, cell.offset)
            item['source_bbox'] = cell.source_bbox
            item['strategy'] = item.get('strategy','legacy_multiline')
            out[key] = item
            stats['multiline_fallback'] += 1
        else:
            group = 'numeric' if cell.column_id in ('quantity','price','printed_total') else 'text'
            pending[group].append(cell)
    for group in ('numeric','text'):
        bucket = pending[group]
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start:start + batch_size]
            stats['batch_calls'] += 1
            try:
                # RapidOCR 3.10.0: RapidOCR.recognize_txt(List[np.ndarray])
                # calls TextRecognizer and uses its configured rec_batch_num.
                with _lock:
                    response = engine().recognize_txt([c.image for c in chunk])
                if response.txts is None or len(response.txts) != len(chunk):
                    raise RuntimeError('Batch returned incorrect number of results')
                for c,txt,score in zip(chunk,response.txts,response.scores):
                    out[(c.row_id,c.column_id)] = _convert(c, txt, score)
            except Exception as exc:
                stats['batch_fallback'] += len(chunk)
                stats['errors'].append(f'{group} batch: {type(exc).__name__}: {exc}')
                for c in chunk:
                    item = read_cell(c.image, c.row_id, c.column_id, c.offset)
                    item['source_bbox'] = c.source_bbox
                    item['strategy'] = 'legacy_batch_fallback'
                    out[(c.row_id,c.column_id)] = item
    # Inspect only suspicious all-numeric prices ending in 7. Correct currency
    # text needs no extra work; image evidence is required before changing a digit.
    currency_skipped = []
    for cell in working:
        key = (cell.row_id, cell.column_id)
        raw = out[key]['text'].strip()
        if cell.column_id != 'price' or not re.fullmatch(r'[0-9]+(?:[,.][0-9]+)?7', raw):
            continue
        if roi_seconds >= roi_limit_s:
            currency_skipped.append({'row_id':cell.row_id, 'column_id':cell.column_id})
        else:
            started = perf_counter()
            _, currency = split_price_currency(cell.image)
            roi_seconds += perf_counter() - started
            if currency is not None:
                roi_metadata.setdefault(key, {})['currency'] = currency
    stats['numeric_roi_ms'] = round(roi_seconds*1000,3)
    stats['numeric_roi_budget_exceeded'] = roi_seconds > roi_limit_s
    stats['numeric_roi_currency_skipped_cells'] = currency_skipped
    for key, info in roi_metadata.items():
        result = out[key]
        result['numeric_roi'] = info
        if info.get('currency'):
            result['currency_symbol'] = info['currency']
            result['ocr_text'] = result['text']
            raw = result['text'].strip()
            if re.fullmatch(r'[0-9]+(?:[,.][0-9]+)?', raw):
                digits = sum(char.isdigit() for char in raw)
                observed_digits = info['currency']['numeric_glyph_count']
                if digits == observed_digits+1 and raw.endswith('7'):
                    result['text'] = raw[:-1] + ' ₸'
                    result['currency_symbol']['ocr_confusion'] = '7'
                elif digits == observed_digits:
                    result['text'] = raw + ' ₸'
        # ROI is internal to OCR: displayed/source coordinates still describe
        # the original physical cell, not the smaller input image.
        cell = originals[key]
        bbox = [cell.offset[0],cell.offset[1],cell.offset[0]+cell.image.shape[1],
                cell.offset[1]+cell.image.shape[0]]
        result['bbox'] = bbox
        for block in result.get('blocks', []):
            block['bbox'] = bbox.copy()
    return out, stats
