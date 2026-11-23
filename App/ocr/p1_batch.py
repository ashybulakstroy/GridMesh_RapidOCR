"""P1: Real RapidOCR 3.10.0 recognizer batching for physical receipt cells.

Use one existing cyrillic model; numeric/text are separate *logical* queues.
Model sessions are reused; the footer lane owns separate sessions.
"""
from dataclasses import dataclass, replace
from time import perf_counter
from time import perf_counter as rec_counter
import numpy as np

from .engine import engine, read_cell, _has_separate_numeric_lines, _lock
from .experiment_rec_trace import rec_scope, record_rec
from .debug_performance import get_debug, report_issue
from .numeric_roi import crop_numeric_ink, split_price_currency, exclude_price_currency
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


def read_batches(cells, *, batch_size=8, budget_multiplier=1, mode="batch", unlimited_roi=False):
    """Read one-line cells as batches, fallback to legacy for multiline cells.

    Returns (dict[(row_id, column_id)] -> legacy cell result, stats).
    Any ONNX batch incompatibility is disclosed and falls back to legacy OCR.
    """
    if batch_size not in (1, 2, 4, 6, 8, 12, 16):
        raise ValueError('OCR_REC_BATCH_NUM: допустимы 1, 2, 4, 6, 8, 12, 16')
    if budget_multiplier not in (1, 2):
        raise ValueError('Допустимый множитель бюджета: 1 или 2')
    if mode not in ('batch','cell'):raise ValueError('Unknown OCR mode: '+mode)
    _debug=get_debug()
    pending = {'numeric': [], 'text': []}
    out = {}
    stats = {'mode':mode, 'cell_calls':0, 'cell_fallback':0, 'requested':len(cells), 'batch_size':batch_size, 'batch_calls':0,
             'multiline_fallback':0, 'batch_fallback':0, 'errors': [], 'numeric_roi_unlimited':unlimited_roi}
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
    currency_crop_enabled = setting('OCR_PRICE_CURRENCY_CROP', '1') == '1'
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
            if not unlimited_roi and roi_seconds >= crop_limit_s:
                roi_skipped += 1
                skipped_cells.append({'row_id':cell.row_id, 'column_id':cell.column_id})
            else:
                started = perf_counter()
                cropped, info = crop_numeric_ink(cell.image)
                if cell.column_id == 'price' and currency_crop_enabled:
                    cropped, info = exclude_price_currency(cropped, info)
                roi_seconds += perf_counter() - started
                roi_metadata[(cell.row_id,cell.column_id)] = info
                working[i] = replace(cell, image=cropped)
    crop_seconds = roi_seconds
    stats.update(numeric_roi_ms=round(roi_seconds*1000,3),
                 numeric_roi_enabled=roi_enabled,
                 numeric_currency_crop_enabled=currency_crop_enabled,
                 numeric_currency_removed=sum(bool(info.get('currency_removed_before_ocr')) for info in roi_metadata.values()),
                 numeric_roi_crop_ms=round(crop_seconds*1000,3),
                 numeric_roi_budget_ms=round(roi_limit_s*1000),
                 numeric_roi_crop_budget_ms=round(crop_limit_s*1000),
                 numeric_roi_skipped=roi_skipped,
                 numeric_roi_skipped_cells=skipped_cells,
                 numeric_roi_crop_exceeded=(not unlimited_roi and crop_seconds > crop_limit_s),
                 numeric_roi_changed=sum(info.get('changed',False) for info in roi_metadata.values()))
    for cell in working:
        key = (cell.row_id, cell.column_id)
        if _debug is not None:
            _debug.capture_numeric_input(cell)
        if mode == 'cell':
            _cell_t0=rec_counter() if _debug is not None else None
            try:
                with rec_scope([key], 'primary_rec_cell'):
                    item = read_cell(cell.image, cell.row_id, cell.column_id, cell.offset)
                item['source_bbox'] = cell.source_bbox
                item['strategy'] = 'primary_rec_cell'
                out[key] = item
                stats['cell_calls'] += 1
                if _debug is not None:
                    _debug.add_cell(cell,(rec_counter()-_cell_t0)*1000,
                                    strategy=item.get('strategy','primary_rec_cell'))
                if not str(item.get('text','')).strip():
                    report_issue(source='RapidOCR',operation='read_cell',
                                 kind='empty_text',severity='warning',
                                 context={'cell_id':f'{cell.row_id}:{cell.column_id}',
                                          'mode':'cell'})
            except Exception as exc:
                report_issue(source='RapidOCR',operation='read_cell',
                             kind='cell_exception',error=exc,
                             context={'cell_id':f'{cell.row_id}:{cell.column_id}'},
                             fallback='batch')
                if _debug is not None:
                    _debug.add_cell(cell,(rec_counter()-_cell_t0)*1000,
                                    status='error',error=f'{type(exc).__name__}: {exc}')
                stats['cell_fallback'] += 1
                stats['errors'].append(f'{key} cell: {type(exc).__name__}: {exc}')
                # Only failed cells use the existing batch route.
                fallback, _ = read_batches([originals[key]], batch_size=batch_size,
                                           budget_multiplier=budget_multiplier)
                out.update(fallback)
            continue
        if _has_separate_numeric_lines(cell.image):
            _multi_t0=rec_counter() if _debug is not None else None
            with rec_scope([(cell.row_id,cell.column_id)],'primary_multiline_cell'):
                item = read_cell(cell.image, cell.row_id, cell.column_id, cell.offset)
            item['source_bbox'] = cell.source_bbox
            item['strategy'] = item.get('strategy','legacy_multiline')
            out[key] = item
            stats['multiline_fallback'] += 1
            if _debug is not None:
                _debug.add_fallback(cell,(rec_counter()-_multi_t0)*1000,
                                    reason='multiline',strategy=item.get('strategy','legacy_multiline'))
            if not str(item.get('text','')).strip():
                report_issue(source='RapidOCR',operation='read_cell',kind='empty_text',
                             severity='warning',context={'cell_id':f'{cell.row_id}:{cell.column_id}',
                                                         'mode':'multiline'})
        else:
            group = 'numeric' if cell.column_id in ('quantity','price','printed_total') else 'text'
            pending[group].append(cell)
    for group in ('numeric','text'):
        bucket = pending[group]
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start:start + batch_size]
            stats['batch_calls'] += 1
            _batch_t0=rec_counter() if _debug is not None else None
            try:
                # RapidOCR 3.10.0: RapidOCR.recognize_txt(List[np.ndarray])
                # calls TextRecognizer and uses its configured rec_batch_num.
                started_rec=rec_counter()
                try:
                    with _lock:
                        response = engine().recognize_txt([c.image for c in chunk])
                    record_rec(keys=[f'{c.row_id}:{c.column_id}' for c in chunk],
                               stage='primary_rec_batch',model='v5_cyrillic_mobile',
                               ms=(rec_counter()-started_rec)*1000)
                except Exception as error:
                    record_rec(keys=[f'{c.row_id}:{c.column_id}' for c in chunk],
                               stage='primary_rec_batch',model='v5_cyrillic_mobile',
                               ms=(rec_counter()-started_rec)*1000,
                               success=False,note=str(error))
                    raise
                if response.txts is None or len(response.txts) != len(chunk):
                    raise RuntimeError('Batch returned incorrect number of results')
                for c,txt,score in zip(chunk,response.txts,response.scores):
                    out[(c.row_id,c.column_id)] = _convert(c, txt, score)
                    if not str(txt).strip():
                        report_issue(source='RapidOCR',operation='recognize_txt',
                                     kind='empty_text',severity='warning',
                                     context={'cell_id':f'{c.row_id}:{c.column_id}',
                                              'group':group})
                if _debug is not None:
                    _debug.add_batch(group,chunk,(rec_counter()-_batch_t0)*1000,status='ok')
            except Exception as exc:
                report_issue(source='RapidOCR',operation='recognize_txt',
                             kind='batch_exception_or_invalid_response',error=exc,
                             context={'group':group, 'size':len(chunk),
                                      'cell_ids':[f'{c.row_id}:{c.column_id}' for c in chunk]},
                             fallback='read_cell')
                if _debug is not None:
                    _debug.add_batch(group,chunk,(rec_counter()-_batch_t0)*1000,
                                     status='error',error=f'{type(exc).__name__}: {exc}')
                stats['batch_fallback'] += len(chunk)
                stats['errors'].append(f'{group} batch: {type(exc).__name__}: {exc}')
                for c in chunk:
                    _fallback_t0=rec_counter() if _debug is not None else None
                    with rec_scope([(c.row_id,c.column_id)],'primary_batch_exception_fallback'):
                        item = read_cell(c.image, c.row_id, c.column_id, c.offset)
                    if _debug is not None:
                        _debug.add_fallback(c,(rec_counter()-_fallback_t0)*1000,
                                            reason='batch_error',strategy='read_cell')
                    item['source_bbox'] = c.source_bbox
                    item['strategy'] = 'legacy_batch_fallback'
                    out[(c.row_id,c.column_id)] = item
    # Inspect only suspicious all-numeric prices ending in 7. Correct currency
    # text needs no extra work; image evidence is required before changing a digit.
    currency_skipped = []
    for cell in working:
        key = (cell.row_id, cell.column_id)
        raw = out[key]['text'].strip()
        if roi_metadata.get(key,{}).get('currency_removed_before_ocr'):
            continue  # A real final 7 must never be reclassified after cropping.
        if cell.column_id != 'price' or not re.fullmatch(r'[0-9]+(?:[,.][0-9]+)?7', raw):
            continue
        if not unlimited_roi and roi_seconds >= roi_limit_s:
            currency_skipped.append({'row_id':cell.row_id, 'column_id':cell.column_id})
        else:
            started = perf_counter()
            _, currency = split_price_currency(cell.image)
            roi_seconds += perf_counter() - started
            if currency is not None:
                roi_metadata.setdefault(key, {})['currency'] = currency
    stats['numeric_roi_ms'] = round(roi_seconds*1000,3)
    stats['numeric_roi_budget_exceeded'] = not unlimited_roi and roi_seconds > roi_limit_s
    stats['numeric_roi_currency_skipped_cells'] = currency_skipped
    for key, info in roi_metadata.items():
        result = out[key]
        result['numeric_roi'] = info
        if info.get('currency'):
            result['currency_symbol'] = info['currency']
            result['ocr_text'] = result['text']
            raw = result['text'].strip()
            if not info.get('currency_removed_before_ocr') and re.fullmatch(r'[0-9]+(?:[,.][0-9]+)?', raw):
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


def read_cells(cells, *, batch_size=8):
    """Prepare all ROI without a time cap, then recognize individually by column."""
    order = {'name':0, 'quantity':1, 'price':2, 'printed_total':3}
    ordered = sorted(cells, key=lambda c:(order.get(c.column_id,4), c.row_id))
    results, stats = read_batches(ordered, batch_size=batch_size, mode='cell', unlimited_roi=True)
    stats['roi_budget_scope'] = 'unlimited'
    return results, stats
