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
from .config import positive_int, setting, ROOT
from .ocr.ocr_performance import OCRPerformance


def number(text):
    text = re.sub(r'[^\d,.]', '', str(text or '')).replace(',', '.')
    try:
        return Decimal(text) if text and text.count('.') <= 1 else None
    except InvalidOperation:
        return None


def png(image):
    ok, data = cv2.imencode('.png', image)
    if not ok: raise ValueError('Не удалось сохранить изображение')
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


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
    for idx, (top, bottom) in enumerate(zip(ys[1:-1], ys[2:]), 1):
        cells = {}
        for col in ('number', 'name', 'quantity', 'price'):
            k = columns[col]
            left, right = xs[k], xs[k+1]
            label = {'number':'номер', 'name':'наименование', 'quantity':'количество', 'price':'цена'}[col]
            profile.phase(f'OCR: строка {idx}, {label}')
            crop, offset, source_bbox = cell_crop(rect, idx, k)
            cells[col] = read_cell(crop, idx, col, offset)
            if col == 'name' and bottom == ys[-1] and not re.search(r'[нh][дd][сc]|vat', cells[col]['text'], re.I):
                alternate = read_cell(crop, idx, col, offset, force_single=True)
                if re.search(r'[нh][дd][сc]|vat', alternate['text'], re.I):
                    cells[col] = alternate
            cells[col]['source_bbox'] = source_bbox
        name = cells['name']['text'].strip()
        if name and cells['name']['confidence'] < .5:
            geometry['warnings'].append(f'Строка {idx}: низкая уверенность в наименовании, проверьте текст.')
        if not name and not cells['quantity']['text'] and not cells['price']['text']:
            geometry['warnings'].append(f'Строка {idx} не распознана: проверьте сетку.')
        vat = bool(re.search(r'[нh][дd][сc]|vat', name, re.I))
        q, p = number(cells['quantity']['text']), number(cells['price']['text'])
        rows.append({'row_id': idx, 'kind': 'tax' if vat else 'product', 'cells': cells,
                     'line_total': str(q*p) if q is not None and p is not None else None})
    # The printed total is below the table, outside the mesh; use the inverse
    # map to locate that region in the original, then OCR it independently.
    last_y = min(rect.map_y.shape[0]-1, max(0, ys[-1]))
    if rect.method == 'original_cells':
        y_bottom = int(max(np.max(cell_polygon(rect, len(ys)-2, k)[34:51, 1]) for k in range(len(xs)-1)) + rect.info['source_offset'][1])
    else:
        y_bottom = int(np.max(rect.map_y[last_y, max(0,xs[0]):min(rect.map_y.shape[1],xs[-1])]))
    footer_top = max(0,y_bottom-20)
    footer = image[footer_top:min(image.shape[0], y_bottom+max(55,int(image.shape[1]*.09))), :]
    profile.phase('Распознавание напечатанного итога')
    footer_left = int(image.shape[1]*.45)
    total_crop = footer[:,footer_left:]
    def read_total(scale):
        sample = cv2.resize(total_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else total_crop
        lines = read_full(sample) if sample.size else []
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
    current_sum = sum((number(r['line_total']) or Decimal(0) for r in rows), Decimal(0))
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
        total, footer_lines, total_confidence = read_total(1)
        footer_attempts.append({'scale':1, 'total':total, 'confidence':total_confidence, 'lines':footer_lines})
        # A reliable printed total is evidence in its own right. A mismatch can
        # come from a product cell, so don't keep rerunning the footer for it.
        if total is None or (total_confidence < .85 and abs(current_sum - number(total)) > Decimal('0.01')):
            second_total, second_lines, second_confidence = read_total(2)
            footer_attempts.append({'scale':2, 'total':second_total, 'confidence':second_confidence, 'lines':second_lines})
            if second_total is not None and (total is None or second_confidence > total_confidence):
                total, footer_lines = second_total, second_lines
    ocr_s = time.perf_counter() - ocr_start
    profile.phase('Сборка и проверка электронного чека')
    computed = sum((number(r['line_total']) or Decimal(0) for r in rows), Decimal(0))
    incomplete = any(r['line_total'] is None for r in rows)
    match = not incomplete and number(total) is not None and abs(computed-number(total)) <= Decimal('0.01')
    taxes = [r for r in rows if r['kind'] == 'tax']
    product_sum = sum((number(r['line_total']) or Decimal(0) for r in rows if r['kind'] == 'product'), Decimal(0))
    vat_sum = sum((number(r['line_total']) or Decimal(0) for r in taxes), Decimal(0))
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
            'total': total, 'computed_total': str(computed), 'math_match': match,
            'vat': {'status': 'not_applicable' if vat_ok is None else 'match' if vat_ok else 'mismatch',
                    'rate_percent': str(vat_rate), 'amount': str(vat_sum) if taxes else None},
            'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts, 'timings': {'geometry_s': round(geo_s, 3),
            'ocr_s': round(ocr_s, 3), 'total_s': round(time.perf_counter()-started, 3)}}
