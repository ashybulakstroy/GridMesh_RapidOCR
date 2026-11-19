"""Receipt processing: detected physical grid defines every OCR cell."""
import base64
import re
import time
import uuid
import json
from decimal import Decimal, InvalidOperation

import cv2
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO

from .geometry.service import build_grid, cell_crop, cell_polygon
from .ocr.engine import read_cell, read_full, read_line
from .ocr.p1_batch import BatchCell, read_batches
from .ocr.p1_numbers import parse_number
from .ocr.p1_recovery import recover_invalid_and_rows, recover_total_mismatch
from .config import positive_int, setting, ROOT
from .ocr.ocr_performance import OCRPerformance


def number(text):
    # P1: reject ambiguous/malformed digits instead of silently deleting OCR characters.
    return parse_number(text)



def png(image):
    ok, data = cv2.imencode('.png', image)
    if not ok: raise ValueError('Не удалось сохранить изображение')
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def is_footer_label(text):
    """Identify an observed total label; it is evidence, not a grid line."""
    text = str(text or '').strip().casefold()
    if re.fullmatch(r'total\s*[:.]?', text):
        return True
    text = text.translate(str.maketrans({'қ':'к', 'ғ':'г', 'і':'и',
                                         'k':'к', 'o':'о', 'p':'р', 'c':'с',
                                         'a':'а', 'e':'е', 'b':'в', 'h':'н'}))
    text = re.sub(r'[^а-яёa-z]', '', text)
    return text in {'корытынды', 'жиыны', 'барлыгы', 'барлыгысы',
                    'итог', 'итого', 'всего'}


def recover_tax_below_grid(image, rect, columns, row_id):
    """Read a visible VAT row below the detected grid without inventing a line."""
    if rect.method != 'original_cells' or not {'name', 'quantity', 'price'} <= columns.keys():
        return None
    grid = rect.info['source_grid']
    if len(grid['horizontal']) < 3:
        return None
    height, width = image.shape[:2]
    last = grid['horizontal'][-1]
    gaps = np.diff([rule.position for rule in grid['horizontal']])
    band_height = max(12, min(80, round(float(np.median(gaps[-4:])) * .75)))

    def observed_band(field, *, start=2, end_col=None):
        col = columns[field]
        right_col = col+1 if end_col is None else end_col
        left = int(grid['vertical'][col].eval((last.position-height/2)/height)) + 2
        right = int(grid['vertical'][right_col].eval((last.position-height/2)/height)) - 2
        left, right = max(0, left), min(width, right)
        if right-left < 4:
            return None
        x = np.arange(left, right)
        baseline = np.asarray(last.eval((x-width/2)/width))
        mx = np.broadcast_to(x, (band_height, len(x))).astype(np.float32)
        my = (baseline[None,:] + np.arange(start, start+band_height)[:,None]).astype(np.float32)
        if my.min() < 0 or my.max() >= height:
            return None
        crop = cv2.remap(image, mx, my, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        corners = [[round(float(mx[y,x]),2), round(float(my[y,x]),2)]
                   for y,x in ((0,0),(0,-1),(-1,-1),(-1,0))]
        return crop, corners

    name_band = observed_band('name')
    if name_band is None:
        return None
    name = read_line(name_band[0])
    if not name or name['confidence'] < .65 or not re.search(r'[нh][дd][сc]\s*\d{1,2}\s*%', name['text'], re.I):
        return None
    # The total label can cross the number/name column boundary. Read both
    # together below the VAT band and require a known, observed footer word.
    footer_band = observed_band('number', start=band_height,
                                end_col=columns['name']+1)
    if footer_band is None:
        return None
    marker = read_line(footer_band[0])
    if not marker or marker['confidence'] < .65 or not is_footer_label(marker['text']):
        return None
    cells = {}
    for field in ('name', 'quantity', 'price'):
        band = name_band if field == 'name' else observed_band(field)
        if band is None:
            return None
        result = name if field == 'name' else read_line(band[0])
        if not result or result['confidence'] < .65:
            return None
        if field != 'name' and number(result['text']) is None:
            return None
        box = band[1]
        cells[field] = {'text':result['text'], 'confidence':result['confidence'],
                        'bbox':[min(p[0] for p in box),min(p[1] for p in box),
                                max(p[0] for p in box),max(p[1] for p in box)],
                        'source_bbox':box, 'cell_id':f'{row_id}-{field}',
                        'row_id':row_id, 'column_id':field, 'blocks':[],
                        'strategy':'visible_tax_below_grid'}
    cells['number'] = {'text':'', 'confidence':None, 'derived':True, 'blocks':[],
                       'source_bbox':[], 'row_id':row_id, 'column_id':'number'}
    q,p = number(cells['quantity']['text']),number(cells['price']['text'])
    bottom = int(max(point[1] for cell in cells.values() for point in cell['source_bbox']) + 1)
    return {'row_id':row_id, 'kind':'tax', 'cells':cells, 'line_total':str(q*p),
            'printed_line_total':None, 'printed_line_match':None}, bottom, marker


def process(data, publish=None, job_id=None, *, rectify=False):
    if job_id is None:
        job_id = uuid.uuid4().hex
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('Некорректный идентификатор задания')
    output_dir = (ROOT / setting('OUTPUT_DIR', 'data/output')).resolve()
    if not output_dir.is_relative_to(ROOT): raise ValueError('OUTPUT_DIR должен быть внутри проекта')
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    profile = OCRPerformance(publish)
    try:
        profile.phase('Сохранение исходного изображения')
        (job_dir/'input.bin').write_bytes(data)
        result = _process(data, profile, job_dir, rectify)
        profile.phase('Сохранение результата')
        for key in ('original', 'rectified'):
            (job_dir/f'{key}.png').write_bytes(base64.b64decode(result[key].split(',', 1)[1]))
    except Exception as exc:
        failure_profile = profile.finish()
        (job_dir/'performance_profile.json').write_text(json.dumps(failure_profile, ensure_ascii=False, indent=2), encoding='utf-8')
        (job_dir/'error.json').write_text(json.dumps({'error':str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    performance = profile.finish()
    result['performance'] = performance
    result['job_id'] = job_id
    result['job_directory'] = str(job_dir.relative_to(ROOT)).replace('\\', '/')
    result['performance_file'] = f"{result['job_directory']}/performance_profile.json"
    result['timings']['total_s'] = round(performance['wall_ms']/1000, 3)
    (job_dir/'performance_profile.json').write_text(json.dumps(performance, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt = {key:value for key,value in result.items() if key not in ('original', 'rectified')}
    (job_dir/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def _process(data, profile, job_dir, rectify):
    started = time.perf_counter()
    profile.phase('Подготовка и ориентация изображения')
    with Image.open(BytesIO(data)) as photo:
        if photo.width * photo.height > 25_000_000: raise ValueError('Слишком большое изображение')
        rgb = ImageOps.exif_transpose(photo).convert('RGB')
    image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    cv2.setNumThreads(positive_int('OCR_OPENCV_THREADS', 2))
    profile.phase('Поиск физической сетки')
    # Portrait photos are the supported automatic orientation. Landscape is
    # tested in both directions using the same physical-grid detector.
    candidates = [image]
    if image.shape[1] > image.shape[0]:
        candidates = [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    failures = []
    options = []
    for oriented in candidates:
        try:
            rect, overlay, geometry = build_grid(oriented, rectify=rectify, phase=profile.phase)
            if len(candidates) == 1:
                options.append((1.0, oriented, rect, overlay, geometry))
                break
            x0,x1 = int(rect.xs[0]),int(rect.xs[-1])
            y0,y1 = int(rect.ys[0]),int(rect.ys[1])
            header = read_full(rect.image[y0:y1, x0:x1])
            words = ' '.join(line['text'].casefold() for line in header)
            known = sum(token in words for token in ('атау', 'наимен', 'саны', 'кол', 'баға', 'цена'))
            confidence = sum(line['confidence'] for line in header)/max(1,len(header))
            options.append((known + confidence*.1, oriented, rect, overlay, geometry))
        except Exception as exc:
            failures.append(str(exc))
    if not options:
        raise ValueError('Не удалось построить сетку: ' + '; '.join(failures))
    score, image, rect, overlay, geometry = max(options, key=lambda item: item[0])
    if len(candidates)>1 and score < 1:
        geometry['warnings'].append('Ориентация не подтверждена по заголовку; проверьте фото.')
    (job_dir/'rectified.png').write_bytes(cv2.imencode('.png', overlay)[1].tobytes())
    profile.publish({'preview_ready': True, 'rectification': geometry['rectification']})
    geo_s = time.perf_counter() - started
    ocr_start = time.perf_counter()
    profile.phase('Подготовка OCR и чтение ячеек')
    rows = []
    numeric_issues = []
    xs = [int(round(x)) for x in rect.xs]; ys = [int(round(y)) for y in rect.ys]
    if len(xs) == 5:
        columns = {'number':0, 'name':1, 'quantity':2, 'price':3}
    else:
        profile.phase('Определяем столбцы по заголовку')
        columns = {}
        header_texts = []
        for k, (left,right) in enumerate(zip(xs,xs[1:])):
            header_crop, header_offset, _ = cell_crop(rect, 0, k)
            head = read_cell(header_crop, 0, 'header', header_offset)['text']
            header_texts.append(head)
            clean = head.casefold().translate(str.maketrans({'r':'г','b':'ы','ғ':'г','ң':'н','қ':'к'}))
            if re.search(r'атау|наимен|назван', clean): role = 'name'
            elif re.search(r'саны|кол.?во|колич', clean): role = 'quantity'
            elif re.search(r'бага|цена|стоим', clean): role = 'price'
            elif re.search(r'сома|сумм', clean): role = 'printed_total'
            elif re.fullmatch(r'\s*(?:№|n[oеe]?|no\.?|#)\s*', clean): role = 'number'
            else: continue
            if role in columns:
                raise ValueError('Заголовки таблицы прочитаны неоднозначно; проверьте фотографию.')
            columns[role] = k
        missing = {'number','name','quantity','price'} - columns.keys()
        if missing:
            raise ValueError('Не удалось уверенно определить столбцы по заголовку: ' + ', '.join(header_texts))
    geometry['columns'] = columns
    # First band is the table header. All subsequent bands are physical rows.
    # P1: extract each cell once, then submit single-line recognition in batches.
    # Physical row numbers are DERIVED from the ordered grid, not claimed as OCR.
    profile.phase('OCR P1: подготовка пакетов ячеек')
    cell_requests, crop_map = [], {}
    body_row_count = len(ys)-2
    field_order = ('name','quantity','price') + (('printed_total',) if 'printed_total' in columns else ())
    for idx in range(1, body_row_count+1):
        for field in field_order:
            crop, offset, source_bbox = cell_crop(rect, idx, columns[field])
            cell_requests.append(BatchCell(crop, idx, field, offset, source_bbox))
            if field in ('quantity','price','printed_total'):
                crop_map[(idx,field)] = crop
    profile.phase('OCR P1: пакетный RapidOCR')
    readouts, batch_stats = read_batches(cell_requests, batch_size=positive_int('OCR_REC_BATCH_NUM',8))
    geometry['ocr_batch'] = batch_stats
    if batch_stats['batch_fallback']:
        geometry['warnings'].append('Пакетный OCR был частично заменён поячейковым: ' + '; '.join(batch_stats['errors'][:2]))
    profile.phase('OCR P1: сборка товарных строк')
    for idx in range(1, body_row_count+1):
        k = columns['number']
        cells = {'number': {
            'text':str(idx), 'confidence':None, 'derived':True,
            'strategy':'physical_row_order', 'cell_id':f'{idx}-number',
            'row_id':idx, 'column_id':'number',
            'bbox':[xs[k],ys[idx],xs[k+1],ys[idx+1]],
            'source_bbox':[], 'blocks':[]}}
        cells.update({field:readouts[(idx,field)] for field in field_order})
        if idx == body_row_count and not re.search(r'[нh][дd][сc]|vat',cells['name']['text'],re.I):
            # VAT name may be split across several observed printed fragments.
            old = cells['name']
            candidate = next(c for c in cell_requests if c.row_id == idx and c.column_id == 'name')
            alternate = read_cell(candidate.image, idx, 'name', candidate.offset, force_single=True)
            if re.search(r'[нh][дd][сc]|vat',alternate['text'],re.I):
                alternate['source_bbox'] = candidate.source_bbox
                cells['name'] = alternate
        name_text = cells['name']['text'].strip()
        if name_text and cells['name']['confidence'] < .5:
            geometry['warnings'].append(f'Строка {idx}: низкая уверенность в наименовании, проверьте текст.')
        if not name_text and not cells['quantity']['text'] and not cells['price']['text']:
            geometry['warnings'].append(f'Строка {idx} не распознана: проверьте сетку.')
        vat = bool(re.search(r'[нh][дd][сc]|vat',name_text,re.I))
        if vat: cells['number']['text'] = ''  # Tax line is not a product ordinal.
        q,p = number(cells['quantity']['text']),number(cells['price']['text'])
        printed = number(cells['printed_total']['text']) if 'printed_total' in cells else None
        rows.append({'row_id':idx,'kind':'tax' if vat else 'product','cells':cells,
                     'line_total':str(q*p) if q is not None and p is not None else None,
                     'printed_line_total':str(printed) if printed is not None else None,
                     'printed_line_match':(abs(q*p-printed) <= Decimal('0.01')) if
                         q is not None and p is not None and printed is not None else None})
    profile.phase('OCR P1: выборочное восстановление чисел')
    retry_budget = positive_int('OCR_P1_MAX_RETRIES', 4)
    initial_recovery = recover_invalid_and_rows(rows,crop_map,max_retries=retry_budget)
    geometry['ocr_recovery'] = {'initial':initial_recovery}
    recovered_tax_bottom = None
    if not any(row['kind'] == 'tax' for row in rows):
        profile.phase('Проверка строки НДС под сеткой')
        recovered_tax = recover_tax_below_grid(image, rect, columns, len(rows)+1)
        if recovered_tax:
            row, recovered_tax_bottom, footer_marker = recovered_tax
            rows.append(row)
            geometry['footer_marker'] = footer_marker
            geometry['warnings'].append('Строка НДС прочитана под границей сетки; проверьте её расположение и данные.')
            # Show the independently read band below the existing mesh in a
            # different color; do not draw it as a detected physical grid line.
            left, top = rect.info['source_offset']
            right = min(image.shape[1], left + overlay.shape[1])
            bottom = min(image.shape[0], max(top + overlay.shape[0], recovered_tax_bottom + 16))
            expanded = image[top:bottom, left:right].copy()
            expanded[:overlay.shape[0], :overlay.shape[1]] = overlay
            for field in ('name', 'quantity', 'price'):
                points = np.rint(np.asarray(row['cells'][field]['source_bbox']) - [left, top]).astype(np.int32)
                cv2.polylines(expanded, [points], True, (0, 160, 255), 2, cv2.LINE_AA)
            overlay = expanded
    # The printed total is below the table, outside the mesh; use the inverse
    # map to locate that region in the original, then OCR it independently.
    last_y = min(rect.map_y.shape[0]-1, max(0, ys[-1]))
    if rect.method == 'original_cells':
        y_bottom = int(max(np.max(cell_polygon(rect, len(ys)-2, k)[34:51, 1]) for k in range(len(xs)-1)) + rect.info['source_offset'][1])
    else:
        y_bottom = int(np.max(rect.map_y[last_y, max(0,xs[0]):min(rect.map_y.shape[1],xs[-1])]))
    if recovered_tax_bottom is not None:
        y_bottom = max(y_bottom, recovered_tax_bottom)
    footer_top = max(0,y_bottom-20)
    footer = image[footer_top:min(image.shape[0], y_bottom+max(55,int(image.shape[1]*.09))), :]
    profile.phase('Распознавание напечатанного итога')
    footer_left = int(image.shape[1]*.45)
    total_crop = footer[:,footer_left:]
    def read_total(scale, *, fast=False):
        sample = cv2.resize(total_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else total_crop
        if not sample.size:
            lines = []
        elif fast:
            lines = read_full(sample, footer_mode=True)
        else:
            lines = read_full(sample)
        for line in lines:
            line['bbox'] = [[x/scale+footer_left, y/scale+footer_top] for x,y in line['bbox']]
        # Numeric blocks at the right of the same baseline form the printed
        # total (e.g. OCR separates "5" and "730,00").
        numeric = [x for x in lines if re.fullmatch(r'[\d\s,.]+\s*[TТ₸]?', x['text'].strip()) and
                   x['confidence'] >= .5 and
                   min(p[0] for p in x['bbox']) >= image.shape[1]*.45 and
                   number(x['text']) is not None and number(x['text']) > 0]
        numeric.sort(key=lambda x: min(p[0] for p in x['bbox']))
        if numeric:
            # The footer crop can contain the last product/VAT row. Select
            # the numeric baseline closest to the table's lower boundary.
            anchor = min(numeric, key=lambda x: abs(sum(p[1] for p in x['bbox'])/len(x['bbox']) - y_bottom))
            top = min(p[1] for p in anchor['bbox'])
            same_line = [x for x in numeric if abs(min(p[1] for p in x['bbox'])-top)<12]
            candidate = number(''.join(x['text'] for x in same_line))
            if candidate is not None: return str(candidate), lines, min(x['confidence'] for x in same_line)
        return None, lines, 0.0
    current_sum = sum((Decimal(r['line_total']) for r in rows if r['line_total'] is not None), Decimal(0))
    fast_x = int(image.shape[1]*.50)
    fast_top, fast_bottom = max(0,y_bottom+2), min(image.shape[0], y_bottom+int(image.shape[1]*.05))
    direct = read_line(image[fast_top:fast_bottom,fast_x:])
    footer_attempts = []
    total, footer_lines = None, []
    if direct:
        direct['bbox'] = [[fast_x,fast_top], [image.shape[1],fast_top],
                          [image.shape[1],fast_bottom], [fast_x,fast_bottom]]
        candidate = number(direct['text']) if re.fullmatch(r'[\d\s,.]+\s*[TТ₸]?', direct['text'].strip()) else None
        footer_attempts.append({'strategy':'single_line', 'total':str(candidate) if candidate is not None else None, 'lines':[direct]})
        if (candidate is not None and (direct['confidence'] >= .93 or
                (direct['confidence'] >= .8 and all(r['line_total'] is not None for r in rows)
                 and abs(current_sum-candidate) <= Decimal('0.01')))):
            total, footer_lines = str(candidate), [direct]
    # The printed baseline can slope above the lowest table intersection.
    # Try small overlapping footer crops before invoking the costly detector.
    # Agreement never changes product values or hides a sum mismatch.
    if total is None:
        candidates = []
        for frac, offset, height in ((.45, -20, .06), (.50, -25, .08)):
            left = int(image.shape[1]*frac)
            top = max(0, y_bottom+offset)
            bottom = min(image.shape[0], top+int(image.shape[1]*height))
            line = read_line(image[top:bottom, left:])
            if not line:
                continue
            line['bbox'] = [[left,top], [image.shape[1],top], [image.shape[1],bottom], [left,bottom]]
            candidate = number(line['text']) if re.fullmatch(r'[\d\s,.]+\s*[TТ₸]?', line['text'].strip()) else None
            footer_attempts.append({'strategy':'compact_line', 'total':str(candidate) if candidate is not None else None, 'lines':[line]})
            if candidate is None or candidate <= 0:
                continue
            if line['confidence'] >= .93 or (line['confidence'] >= .75 and all(r['line_total'] is not None for r in rows) and abs(current_sum-candidate) <= Decimal('0.01')):
                total, footer_lines = str(candidate), [line]
                break
            if line['confidence'] >= .80:
                if any(value == candidate for value in candidates):
                    total, footer_lines = str(candidate), [line]
                    break
                candidates.append(candidate)
    if total is None:
        try:
            total, footer_lines, total_confidence = read_total(1, fast=True)
            footer_attempts.append({'strategy':'fast_detector', 'scale':1, 'total':total,
                                    'confidence':total_confidence, 'lines':footer_lines})
        except Exception as exc:
            total, footer_lines, total_confidence = None, [], 0.0
            footer_attempts.append({'strategy':'fast_detector', 'scale':1,
                                    'error':f'{type(exc).__name__}: {exc}'})
        if total is None or total_confidence < .85:
            total, footer_lines, total_confidence = read_total(1)
            footer_attempts.append({'strategy':'standard_detector', 'scale':1, 'total':total,
                                    'confidence':total_confidence, 'lines':footer_lines})
        # A reliable printed total is evidence in its own right. A mismatch can
        # come from a product cell, so don't keep rerunning the footer for it.
        if total is None or (total_confidence < .85 and abs(current_sum - number(total)) > Decimal('0.01')):
            second_total, second_lines, second_confidence = read_total(2)
            footer_attempts.append({'scale':2, 'total':second_total, 'confidence':second_confidence, 'lines':second_lines})
            if second_total is not None and (total is None or second_confidence > total_confidence):
                total, footer_lines = second_total, second_lines
    # Only investigate footer mismatch AFTER normal OCR; never invent numbers.
    profile.phase('OCR P1: проверка напечатанного итога')
    footer_recovery = recover_total_mismatch(rows,crop_map,number(total),
        max_retries=max(0,retry_budget-initial_recovery['attempted']))
    geometry['ocr_recovery']['footer'] = footer_recovery
    if initial_recovery['candidates'] or footer_recovery['candidates']:
        geometry['warnings'].append('Есть альтернативные варианты OCR чисел, требующие проверки.')
    invalid_printed = [r['row_id'] for r in rows if r.get('printed_line_match') is False]
    if invalid_printed:
        geometry['warnings'].append('Несовпадение напечатанных сумм в строках: ' + ', '.join(map(str,invalid_printed)))
    for row in rows:
        for col in ('quantity','price'):
            value = row['cells'][col]['text']
            if number(value) is None:
                issue = {'row_id':row['row_id'], 'column_id':col, 'text':str(value),
                         'reason':'Число не распознано в безопасном формате; исходный текст сохранён.'}
                row['cells'][col]['numeric_issue'] = issue['reason']
                numeric_issues.append(issue)
                label = 'количество' if col == 'quantity' else 'цену'
                shown = str(value) if str(value).strip() else 'пустое значение'
                geometry['warnings'].append(f'Строка {row["row_id"]}: проверьте {label} «{shown}» — число не рассчитано.')
    ocr_s = time.perf_counter() - ocr_start
    profile.phase('Сборка и проверка электронного чека')
    computed = sum((Decimal(r['line_total']) for r in rows if r['line_total'] is not None), Decimal(0))
    incomplete = any(r['line_total'] is None for r in rows)
    match = not incomplete and number(total) is not None and abs(computed-number(total)) <= Decimal('0.01')
    taxes = [r for r in rows if r['kind'] == 'tax']
    product_sum = sum((Decimal(r['line_total']) for r in rows if r['kind'] == 'product' and r['line_total'] is not None), Decimal(0))
    vat_sum = sum((Decimal(r['line_total']) for r in taxes if r['line_total'] is not None), Decimal(0))
    vat_rate = Decimal(setting('OCR_VAT_RATE_PERCENT', '16'))
    vat_ok = None if not taxes or incomplete else abs(vat_sum-product_sum*vat_rate/100) <= Decimal('1')
    if incomplete: geometry['warnings'].append('Есть ячейки с нераспознанным количеством или ценой.')
    if total is None: geometry['warnings'].append('Печатный итог не найден; проверьте вручную.')
    elif not match: geometry['warnings'].append('Сумма строк не совпадает с печатным итогом.')
    if vat_ok is False: geometry['warnings'].append('Сумма НДС не совпадает с расчётом по установленной ставке.')
    map_path = job_dir/'rectification_map.npz'
    np.savez_compressed(map_path, map_x=rect.map_x, map_y=rect.map_y)
    geometry['inverse_map'] = str(map_path.relative_to(ROOT)).replace('\\', '/')
    return {'status': 'review' if geometry['warnings'] else 'ok', 'geometry': geometry,
            'original': png(image), 'rectified': png(overlay), 'rows': rows,
            'numeric_issues': numeric_issues,
            'total': total, 'computed_total': str(computed), 'math_match': match,
            'vat': {'status': 'not_applicable' if vat_ok is None else 'match' if vat_ok else 'mismatch',
                    'rate_percent': str(vat_rate), 'amount': str(vat_sum) if taxes else None},
            'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts, 'timings': {'geometry_s': round(geo_s, 3),
            'ocr_s': round(ocr_s, 3), 'total_s': round(time.perf_counter()-started, 3)}}
