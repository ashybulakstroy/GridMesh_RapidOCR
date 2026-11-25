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
from .ocr.p1_batch import BatchCell, read_batches, read_cells
from .ocr.cell_mode import read_parallel
from .ocr.e4_adaptive_early import adaptive_collect, make_early_snapshot, compare_early_final
from .ocr.e4_fast_gate import primary_consistency, prioritize_suspects
from .ocr.p1_numbers import parse_number
from .ocr.numeric_roi import crop_numeric_ink, prepare_numeric_input
from .ocr.price_candidate import prepare_price_candidate
from .ocr.p1_recovery import recover_invalid_and_rows, recover_total_mismatch
from .ocr.p1_double_retry import retry_suspect_cells
from .ocr.e5_tesseract import recover_with_tesseract
from .ocr.e4_math_target import select_math_targets
from .ocr.experiment_matrix import CandidateBook, solve_matrix
from .ocr.experiment_original import OriginalNumericCache
from .ocr.experiment_footer import scan_footer
from .ocr.footer_line_guided_v1 import guide_from_grid, attempt_focused_footer
from .ocr.experiment_vat import scan_printed_vat, seed_printed_vat, observed_vat_text
from .ocr.experiment_models import ExperimentModelPool
from .ocr.experiment_rec_trace import RecTracker, activate, deactivate, rec_scope, current_trace, current_inputs
from .ocr.experiment_auto import auto_apply_consensus
from .config import positive_int, setting, ROOT
from .ocr.ocr_performance import OCRPerformance
from .ocr.debug_performance import (activate_telemetry, deactivate_telemetry,
    bind_profile, get_debug, report_issue)


def number(text):
    # P1: reject ambiguous/malformed digits instead of silently deleting OCR characters.
    return parse_number(text)


def has_multiple_consistent_footer_values(matrix):
    """Only flag distinct totals that belong to complete matrix solutions."""
    values = {number(solution.get('footer')) for solution in matrix.get('solutions', [])}
    values.discard(None)
    return len(values) > 1


def log_budget_events(log, stats, stage):
    if not stats:
        return
    skipped = stats.get('numeric_roi_skipped_cells', [])
    currency_skipped = stats.get('numeric_roi_currency_skipped_cells', [])
    if stats.get('numeric_roi_crop_exceeded') or skipped:
        log.append({'type':'budget', 'stage':stage, 'kind':'crop',
                    'limit_ms':stats['numeric_roi_crop_budget_ms'],
                    'elapsed_ms':stats['numeric_roi_crop_ms'],
                    'exceeded':stats['numeric_roi_crop_exceeded'], 'skipped_cells':skipped,
                    'message':f"{stage}: подготовка {stats['numeric_roi_crop_ms']:.3f} мс "
                              f"при лимите {stats['numeric_roi_crop_budget_ms']} мс, "
                              f"пропущено ячеек: {len(skipped)}."})
    if stats.get('numeric_roi_budget_exceeded') or currency_skipped:
        log.append({'type':'budget', 'stage':stage, 'kind':'total',
                    'limit_ms':stats['numeric_roi_budget_ms'],
                    'elapsed_ms':stats['numeric_roi_ms'],
                    'exceeded':stats['numeric_roi_budget_exceeded'],
                    'skipped_cells':currency_skipped,
                    'message':f"{stage}: общий бюджет {stats['numeric_roi_ms']:.3f} мс "
                              f"при лимите {stats['numeric_roi_budget_ms']} мс, "
                              f"пропущено проверок валюты: {len(currency_skipped)}."})


def log_retry_details(log, recovery, stage):
    labels = {'quantity':'количество', 'price':'цена', 'printed_total':'сумма строки'}
    for attempt in recovery.get('details', []):
        field = labels.get(attempt['column_id'], attempt['column_id'])
        outcome = 'принято' if attempt['accepted'] else 'оставлено исходное значение'
        log.append({'type':'retry', 'stage':stage, **attempt,
                    'message':f"{stage}: строка {attempt['row_id']}, {field}: "
                              f"«{attempt['original_text']}» → «{attempt['alternative_text']}»; {outcome}."})



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
    numeric_crops = {}
    for field in ('name', 'quantity', 'price'):
        band = name_band if field == 'name' else observed_band(field)
        if band is None:
            return None
        price_roi = None
        prepared = band[0]
        if field == 'price':
            prepared, price_roi = prepare_numeric_input(prepared, field)
        result = name if field == 'name' else read_line(prepared)
        if not result or result['confidence'] < .65:
            return None
        if field != 'name' and number(result['text']) is None:
            return None
        if field != 'name':
            numeric_crops[field] = band[0]
        box = band[1]
        cells[field] = {'text':result['text'], 'confidence':result['confidence'],
                        'bbox':[min(p[0] for p in box),min(p[1] for p in box),
                                max(p[0] for p in box),max(p[1] for p in box)],
                        'source_bbox':box, 'cell_id':f'{row_id}-{field}',
                        'row_id':row_id, 'column_id':field, 'blocks':[],
                        'strategy':'visible_tax_below_grid'}
        if price_roi is not None:
            cells[field]['numeric_roi'] = price_roi
    cells['number'] = {'text':'', 'confidence':None, 'derived':True, 'blocks':[],
                       'source_bbox':[], 'row_id':row_id, 'column_id':'number'}
    q,p = number(cells['quantity']['text']),number(cells['price']['text'])
    bottom = int(max(point[1] for cell in cells.values() for point in cell['source_bbox']) + 1)
    return {'row_id':row_id, 'kind':'tax', 'cells':cells, 'line_total':str(q*p),
            'printed_line_total':None, 'printed_line_match':None}, bottom, marker, numeric_crops


def process(data, publish=None, job_id=None, *, rectify=False):
    tracker=RecTracker()
    token=activate(tracker)
    telemetry_token=activate_telemetry(job_id)
    from .ocr.performance_deep.live_capture import activate as deep_activate, deactivate as deep_deactivate
    deep_token=deep_activate(setting('LOG_LEVEL','INFO').strip().upper()=='DEBUG')
    try:
        return _process_job(data,publish,job_id,rectify=rectify,rec_tracker=tracker)
    finally:
        deep_deactivate(deep_token)
        deactivate_telemetry(telemetry_token)
        deactivate(token)


def _process_job(data, publish=None, job_id=None, *, rectify=False, rec_tracker):
    if job_id is None:
        job_id = uuid.uuid4().hex
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('Некорректный идентификатор задания')
    output_dir = (ROOT / setting('OUTPUT_DIR', 'data/output')).resolve()
    if not output_dir.is_relative_to(ROOT): raise ValueError('OUTPUT_DIR должен быть внутри проекта')
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    profile = OCRPerformance(publish)
    bind_profile(profile, job_id)
    profile.processing_log = []
    try:
        profile.phase('Сохранение исходного изображения')
        (job_dir/'input.bin').write_bytes(data)
        result = _process(data, profile, job_dir, rectify)
        profile.phase('Сохранение результата')
        for key in ('original', 'rectified'):
            (job_dir/f'{key}.png').write_bytes(base64.b64decode(result[key].split(',', 1)[1]))
    except Exception as exc:
        if (job_dir/'e4_early_result.json').is_file():
            profile.publish({'e4_background_verification':'failed'})
        report_issue(source='pipeline',operation=profile.name or 'unknown',
                     kind='uncaught_exception',error=exc,fallback='none')
        failure_profile = profile.finish()
        debug_perf=get_debug()
        if debug_perf is not None:
            failure_profile['debug']=debug_perf.finish(rec_tracker.export())
            failure_profile['debug'].setdefault('configuration', {}).setdefault('log_level', 'DEBUG')
        profile.processing_log.append({'type':'error', 'stage':'Обработка',
                                       'message':f'Обработка завершилась ошибкой: {type(exc).__name__}: {exc}'})
        failure_profile['budget_events'] = [event for event in profile.processing_log if event['type'] == 'budget']
        (job_dir/'performance_profile.json').write_text(json.dumps(failure_profile, ensure_ascii=False, indent=2), encoding='utf-8')
        (job_dir/'processing_log.json').write_text(json.dumps(profile.processing_log, ensure_ascii=False, indent=2), encoding='utf-8')
        (job_dir/'error.json').write_text(json.dumps({'error':str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        if setting('LOG_LEVEL','INFO').strip().upper()=='DEBUG' and failure_profile['wall_ms']>2000:
            try:
                from .ocr.performance_deep.runtime_export import export as export_performance
                export_performance(job_dir,rec_tracker)
            except Exception as diagnostic_error:
                import logging
                logging.getLogger(__name__).warning('problem_performance export failed: %s', diagnostic_error)
        raise
    result['rec_trace']=rec_tracker.export()
    for item in result.get('processing_log',[]):
        if item.get('type') == 'error':
            report_issue(source='processing_log',operation=str(item.get('stage','pipeline')),
                         kind='caught_pipeline_error',context={'message':str(item.get('message',''))[:250]})
    performance = profile.finish()
    debug_perf=get_debug()
    if debug_perf is not None:
        performance['debug']=debug_perf.finish(result['rec_trace'])
    result['performance'] = performance
    result['job_id'] = job_id
    result['job_directory'] = str(job_dir.relative_to(ROOT)).replace('\\', '/')
    result['performance_file'] = f"{result['job_directory']}/performance_profile.json"
    result['processing_log_file'] = f"{result['job_directory']}/processing_log.json"
    result['timings']['total_s'] = round(performance['wall_ms']/1000, 3)
    if result.get('e4_adaptive', {}).get('enabled'):
        performance['e4_adaptive'] = result['e4_adaptive']
        early_path = job_dir/'e4_early_result.json'
        if early_path.is_file():
            comparison = compare_early_final(json.loads(early_path.read_text(encoding='utf-8')), result)
            comparison['snapshot_created_ms'] = comparison['first_elapsed_ms']
            comparison['first_elapsed_ms'] = result['e4_adaptive']['first_elapsed_ms']
            result['e4_early_comparison'] = comparison
            (job_dir/'e4_early_comparison.json').write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
            profile.publish({'e4_background_verification':'done',
                             'e4_verification_change_count':comparison['changed_count']})
    performance['budget_events'] = [event for event in result['processing_log'] if event['type'] == 'budget']
    (job_dir/'performance_profile.json').write_text(json.dumps(performance, ensure_ascii=False, indent=2), encoding='utf-8')
    (job_dir/'processing_log.json').write_text(json.dumps(result['processing_log'], ensure_ascii=False, indent=2), encoding='utf-8')
    # E1 metadata is stored beside receipt.json; raw ROIs remain in RAM only.
    result['numeric_experiment_file'] = f"{result['job_directory']}/numeric_experiment.json"
    experiment_payload = {key:result.get(key) for key in
                          ('numeric_candidate_book','numeric_matrix','original_roi_experiment',
                           'footer_variant_attempts','vat_evidence','printed_vat',
                           'numeric_auto_corrections','rec_trace')}
    experiment_payload['model_ensemble'] = (result.get('geometry',{})
                                            .get('numeric_experiment',{})
                                            .get('model_ensemble'))
    (job_dir/'numeric_experiment.json').write_text(
        json.dumps(experiment_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt = {key:value for key,value in result.items() if key not in ('original', 'rectified')}
    (job_dir/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    if setting('LOG_LEVEL','INFO').strip().upper()=='DEBUG' and performance['wall_ms']>2000:
        try:
            from .ocr.performance_deep.runtime_export import export as export_performance
            diagnostic_path=export_performance(job_dir,rec_tracker)
            if diagnostic_path is not None:
                result['problem_performance']={'report':diagnostic_path.relative_to(ROOT).as_posix()}
                receipt['problem_performance']=result['problem_performance']
                (job_dir/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning('problem_performance export failed: %s',exc)
    return result


# ADAPTIVE_DETECTOR_ROUTING_V1
def _process(data, profile, job_dir, rectify):
    from .geometry import detector_routing_v1 as _route
    _mode=setting('OCR_GRID_ROUTE_V1','auto')
    # Never bypass V2 unless the existing anomaly guard is enabled.
    if setting('OCR_GRID_ANOMALY_V3','1')!='1': _mode='legacy'
    _state,_token=_route.start(_mode, int(setting('OCR_GRID_ROUTE_MIN_H','10')))
    _retry=None
    try:
        try:
            result=_process_route_body(data,profile,job_dir,rectify)
        except (ValueError,cv2.error) as exc:
            if not _route.can_retry(): raise
            _route.record('legacy_retry',reason=str(exc))
            _retry=str(exc)
            # Only a pre-product-OCR error may enter this branch.
            # Keep prior diagnostics under a distinct name, never publish partial rows.
            for _filename in ('grid_anomaly.json','grid_anomaly_initial.png','rectified.png'):
                _p=job_dir/_filename
                if _p.exists(): _p.replace(job_dir/('routing_trial_'+_filename))
            _state.mode='legacy'; _state.candidate_used=False
            result=_process_route_body(data,profile,job_dir,rectify)
        result.setdefault('geometry',{})['detector_routing_v1']={'requested_mode':_mode,'final_mode':_state.mode,'events':_state.events,'legacy_retry':_retry}
        return result
    finally:
        try:
            (job_dir/'detector_routing.json').write_text(__import__('json').dumps({'requested_mode':_mode,'final_mode':_state.mode,'events':_state.events,'legacy_retry':_retry},ensure_ascii=False,indent=2),encoding='utf-8')
        except OSError:
            pass  # diagnostic persistence must not mask original exception
        finally:
            _route.end(_token)


def _process_route_body(data, profile, job_dir, rectify):
    started = time.perf_counter()
    e4_enabled = setting('OCR_E4_ENABLE','1') == '1'
    adaptive_enabled = e4_enabled and setting('OCR_E4_ADAPTIVE','1') == '1'
    adaptive_info = {'enabled':adaptive_enabled, 'first_ready':False,
                     'first_elapsed_ms':None, 'probes_before_first':None, 'early_solver_checks':0}
    if adaptive_enabled:
        profile.publish({'e4_adaptive_enabled':True})
    processing_log = profile.processing_log
    profile.phase('Подготовка и ориентация изображения')
    with Image.open(BytesIO(data)) as photo:
        if photo.width * photo.height > 25_000_000: raise ValueError('Слишком большое изображение')
        rgb = ImageOps.exif_transpose(photo).convert('RGB')
    image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    cv2.setNumThreads(positive_int('OCR_OPENCV_THREADS', 2))
    debug_perf=get_debug()
    if debug_perf is not None:
        debug_perf.capture_runtime(image.shape)
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
            # PAPER_FIRST_FULLGRID_V16: rescue only on error; verify the actual
            # printed header for each candidate before promoting its grid.
            def _v12_validate_header(_candidate):
                if len(_candidate.xs) != 5:
                    return False
                _texts=[]
                for _col in (1,2,3):
                    _crop,_off,_=cell_crop(_candidate,0,_col)
                    _texts.append(read_cell(_crop,0,'header',_off)['text'].casefold())
                _header=' '.join(_texts).translate(str.maketrans({'ғ':'г','ң':'н','қ':'к'}))
                return (bool(re.search(r'тауар|атау|наимен|назван',_header)) and
                        bool(re.search(r'саны|кол.?во|колич|бага|бағ|цена',_header)))
            try:
                rect, overlay, geometry = build_grid(oriented, rectify=rectify, phase=profile.phase)
            except (ValueError, cv2.error):
                from .geometry.service import build_grid_paper_rescue
                _rescue = build_grid_paper_rescue(oriented, rectify=rectify,
                                                  phase=profile.phase, header_validator=_v12_validate_header)
                if _rescue is None:
                    raise
                rect, overlay, geometry = _rescue
            if len(candidates) == 1:
                options.append((1.0, oriented, rect, overlay, geometry))
                break
            x0,x1 = int(rect.xs[0]),int(rect.xs[-1])
            y0,y1 = int(rect.ys[0]),int(rect.ys[1])
            # Grid cells already locate header text: try recognition without
            # a full text detector before constructing the opposite orientation.
            header=[]
            for column in range(1,len(rect.xs)-1):
                crop,_,_=cell_crop(rect,0,column)
                line=read_line(crop)
                if line:header.append(line)
            words=' '.join(line['text'].casefold() for line in header)
            roles=sum(any(token in words for token in group) for group in
                      (('атау','наимен','назван'),('саны','колич','кол-во'),('баға','бага','цена')))
            confidence=sum(line['confidence'] for line in header)/max(1,len(header))
            if roles>=2 and confidence>=.80:
                geometry['orientation']={'method':'observed_header_cells','roles':roles,'confidence':confidence}
                options.append((roles+confidence*.1,oriented,rect,overlay,geometry))
                break
            header = read_full(rect.image[y0:y1, x0:x1])
            words = ' '.join(line['text'].casefold() for line in header)
            known = sum(token in words for token in ('атау', 'наимен', 'саны', 'кол', 'баға', 'цена'))
            confidence = sum(line['confidence'] for line in header)/max(1,len(header))
            options.append((known + confidence*.1, oriented, rect, overlay, geometry))
        except Exception as exc:
            failures.append(str(exc))
    if not options:
        from .geometry.service import build_grid_source_rescue
        _candidate_image=candidates[0]
        _candidate=build_grid_source_rescue(_candidate_image,rectify=rectify,
                              phase=profile.phase,header_validator=_v12_validate_header)
        if _candidate is None:
            raise ValueError('Не удалось построить сетку: ' + '; '.join(failures))
        _rect,_overlay,_geometry=_candidate
        options.append((1.0,_candidate_image,_rect,_overlay,_geometry))
    score, image, rect, overlay, geometry = max(options, key=lambda item: item[0])
    # PAPER_FIRST_FULLGRID_V16: reject an anomalously tall first band before
    # numerical OCR starts. V4 runs at its normal speed on ordinary receipts.
    if setting('OCR_SOURCE_GRID_V16','1') == '1' and len(rect.ys)>=5:
        _bands=np.diff(np.asarray(rect.ys,dtype=float))
        _typical=float(np.median(_bands[1:])) if len(_bands)>2 else 0.
        _giant=bool(_typical>0 and _bands[0]>max(90.,3.0*_typical))
        if _giant:
            from .geometry.service import build_grid_source_rescue
            _fixed=build_grid_source_rescue(image,rectify=rectify,
                        phase=profile.phase,header_validator=_v12_validate_header)
            if _fixed is None:
                raise ValueError('V16: первая полоса сетки слишком высокая; '
                    'печатная шапка не подтверждена, исходная сетка отклонена до OCR')
            rect,overlay,geometry=_fixed
            processing_log.append({'type':'operation',
                'stage':'V16: восстановление полной сетки',
                'message':'Заменена неполная сетка на источник-подтверждённую физическую сетку.'})
    if geometry.get('fold_recovered_horizontal'):
        count = geometry['fold_recovered_horizontal']
        processing_log.append({'type':'operation', 'stage':'Построение сетки',
                               'message':f'Восстановлены линии сетки у перегиба: {count}.'})
    if len(candidates)>1 and score < 1:
        geometry['warnings'].append('Ориентация не подтверждена по заголовку; проверьте фото.')
    geo_s = time.perf_counter() - started
    ocr_start = time.perf_counter()
    profile.phase('Подготовка OCR и чтение ячеек')
    rows = []
    numeric_issues = []
    xs = [int(round(x)) for x in rect.xs]; ys = [int(round(y)) for y in rect.ys]
    footer_focus_guide = guide_from_grid(rect, image.shape)
    # PAPER_FIRST_FULLGRID_V16: VAT/НДС is optional. Header must be the
    # photographed header, not the tenth product row of a partial grid.
    def _v12_read_columns(grid, rescued=False):
        if len(grid.xs)==5:
            # Only rescued grids pay for a header OCR proof; preserve V4 speed.
            if rescued:
                observed=[]
                for idx in (1,2,3):
                    crop,off,_=cell_crop(grid,0,idx)
                    observed.append(read_cell(crop,0,'header',off)['text'].casefold())
                header=' '.join(observed).translate(str.maketrans({'ғ':'г','ң':'н','қ':'к'}))
                names=bool(re.search(r'тауар|атау|наимен|назван',header))
                numeric=bool(re.search(r'саны|кол.?во|колич|бага|бағ|цена',header))
                if not (names and numeric):
                    raise ValueError('V12: верхняя строка не подтверждена как заголовок: '+header)
            return {'number':0,'name':1,'quantity':2,'price':3}
        profile.phase('Определяем столбцы по заголовку')
        seen={}; texts=[]
        for k in range(len(grid.xs)-1):
            crop,offset,_=cell_crop(grid,0,k)
            head=read_cell(crop,0,'header',offset)['text']
            texts.append(head)
            clean=head.casefold().translate(str.maketrans({'r':'г','b':'ы','ғ':'г','ң':'н','қ':'к'}))
            if re.search(r'атау|наимен|назван',clean):role='name'
            elif re.search(r'саны|кол.?во|колич',clean):role='quantity'
            elif re.search(r'бага|цена|стоим',clean):role='price'
            elif re.search(r'сома|сумм',clean):role='printed_total'
            elif re.fullmatch(r'\s*(?:№|n[oеe]?|no\.?|#)\s*',clean):role='number'
            else:continue
            if role in seen:
                raise ValueError('Заголовки определены неоднозначно: '+', '.join(texts))
            seen[role]=k
        if {'number','name','quantity','price'}-seen.keys():
            raise ValueError('Не определены заголовки: '+', '.join(texts))
        return seen
    try:
        columns=_v12_read_columns(rect, rescued=bool(geometry.get('paper_first_v12')))
    except ValueError as _v12_error:
        from .geometry.service import build_grid_paper_rescue
        _v12_rescue=build_grid_paper_rescue(image,rectify=rectify,
                                         phase=profile.phase,header_validator=_v12_validate_header)
        if _v12_rescue is None:
            from .geometry.service import build_grid_source_rescue
            _v12_rescue=build_grid_source_rescue(image,rectify=rectify,
                             phase=profile.phase,header_validator=_v12_validate_header)
            if _v12_rescue is None:
                raise _v12_error
        _v12_rect,_v12_overlay,_v12_geometry=_v12_rescue
        # Never promote a graph whose actual OCR header is unreadable.
        columns=_v12_read_columns(_v12_rect,rescued=True)
        rect,overlay,geometry=_v12_rect,_v12_overlay,_v12_geometry
        xs=[int(round(x)) for x in rect.xs]
        ys=[int(round(y)) for y in rect.ys]
        footer_focus_guide=guide_from_grid(rect,image.shape)
        processing_log.append({'type':'operation','stage':'V12 PaperFirst recovery',
            'message':'Частичная таблица отклонена; использована полная сетка по бумаге.'})
    # GRID_ANOMALY_V3: independent source coverage and printed ordinals.
    # Run on every selected route (including SMART/full-image fallback),
    # before publishing preview or starting expensive product OCR.
    if setting('OCR_GRID_ANOMALY_V3','1') == '1':
        import json as _grid_json
        from .geometry.grid_anomaly_v3 import audit_grid, build_direct
        from .geometry.service import build_grid_paper_rescue, build_grid_source_rescue
        profile.phase('Контроль полноты сетки и первого номера')
        _grid_gate = audit_grid(image,rect,geometry,crop=cell_crop,read=read_line,
                                number_col=columns.get('number'))
        _grid_trace = {'initial':_grid_gate,'attempts':[],'status':_grid_gate['status']}
        _grid_gate_file = job_dir/'grid_anomaly.json'
        def _grid_save():
            _grid_gate_file.write_text(_grid_json.dumps(_grid_trace,ensure_ascii=False,indent=2),encoding='utf-8')
        if _grid_gate['status'] != 'pass':
            (job_dir/'grid_anomaly_initial.png').write_bytes(cv2.imencode('.png',overlay)[1].tobytes())
        # Only confirmed geometric anomalies trigger repair.
        if _grid_gate['status']=='suspect':
            _grid_fixed = False
            def _grid_direct_builder(*args, **kwargs):
                if 'invalid_mesh_geometry' in _grid_gate['reasons']:
                    raise ValueError('direct_repair_requires_valid_lower_grid')
                _direct = build_direct(image,rect,geometry,_grid_gate['source_extension'],rectify=rectify)
                if not _v12_validate_header(_direct[0]):
                    raise ValueError('direct_header_not_confirmed')
                return _direct
            for _grid_route,_grid_builder in (('direct_anchor_v3',_grid_direct_builder),
                                               ('source_v16',build_grid_source_rescue),
                                               ('paper_v12',build_grid_paper_rescue)):
                _grid_attempt = {'route':_grid_route}
                _grid_trace['attempts'].append(_grid_attempt)
                try:
                    _grid_candidate = _grid_builder(image,rectify=rectify,phase=profile.phase,
                                                     header_validator=_v12_validate_header)
                except (ValueError,cv2.error,TypeError,IndexError) as _ge:
                    _grid_attempt.update(status='rejected',error=str(_ge))
                    continue
                if _grid_candidate is None:
                    _grid_attempt['status']='no_candidate'
                    _grid_save()
                    continue
                _gr,_go,_gg = _grid_candidate
                try:
                    _gc = _v12_read_columns(_gr,rescued=False)  # rescue builder already verified header
                    _ga = audit_grid(image,_gr,_gg,crop=cell_crop,read=read_line,
                                     number_col=_gc.get('number'),force_number=True)
                except (ValueError,cv2.error) as _ge:
                    _grid_attempt.update(status='rejected',error=str(_ge))
                    _grid_save()
                    continue
                _grid_attempt['audit']=_ga
                if _gg.get('direct_anchor_v3'):
                    _grid_attempt['direct_recovery'] = _gg['direct_anchor_v3']
                if _ga['status']!='pass':
                    _grid_attempt['status']='rejected'
                    _grid_save()
                    continue
                rect,overlay,geometry,columns = _gr,_go,_gg,_gc
                _grid_gate = _ga
                _grid_attempt['status']='accepted'
                _grid_trace['status']='recovered'
                _grid_fixed=True
                processing_log.append({'type':'operation','stage':'Grid Anomaly V3',
                     'message':'Неполная или повреждённая сетка заменена после проверки исходных линий, шапки и первого номера.'})
                break
            if not _grid_fixed:
                _grid_trace['status']='rejected'
                _grid_save()
                raise ValueError('Grid Anomaly V3: сетка не подтверждена; '+
                                 ', '.join(_grid_gate['reasons'] or ['геометрический дефект'])+
                                 '. Подробности: grid_anomaly.json')
        _grid_trace['final']=_grid_gate
        geometry['grid_anomaly_v3']=_grid_trace
        _grid_save()
        xs=[int(round(x)) for x in rect.xs]
        ys=[int(round(y)) for y in rect.ys]
        footer_focus_guide=guide_from_grid(rect,image.shape)
    # GRID_SPEED_COMBINED_V1: premesh candidate never bypasses header or safety.
    if geometry.get('combo_premesh'):
        if geometry.get('grid_anomaly_v3',{}).get('final',{}).get('status')!='pass':
            raise ValueError('combo premesh: repaired grid not confirmed')
        # Existing adaptive guard below checks header unless V4 already did.
        if geometry.get('grid_anomaly_v3',{}).get('status')=='recovered' and not _v12_validate_header(rect):
            raise ValueError('combo premesh: repaired header not confirmed')
    # ADAPTIVE_DETECTOR_ROUTING_V1: the existing gate must have run.
    from .geometry import detector_routing_v1 as _route
    if _route.used():
        _final_trace=geometry.get('grid_anomaly_v3',{})
        if _final_trace.get('final',{}).get('status')!='pass':
            raise ValueError('routing: anomaly gate did not confirm candidate')
        # Recovered candidates have already passed header validation.
        if _final_trace.get('status')!='recovered' and not _v12_validate_header(rect):
            raise ValueError('routing: printed header not confirmed')
        _route.record('candidate_confirmed',status=_final_trace.get('status'))
    _route.finish_geometry()
    (job_dir/'rectified.png').write_bytes(cv2.imencode('.png',overlay)[1].tobytes())
    profile.publish({'preview_ready':True,'rectification':geometry['rectification']})
    geo_s = time.perf_counter() - started
    ocr_start = time.perf_counter()
    geometry['columns'] = columns
    # First band is the table header. All subsequent bands are physical rows.
    # P1: extract each cell once, then submit single-line recognition in batches.
    # Physical row numbers are DERIVED from the ordered grid, not claimed as OCR.
    ocr_mode = setting('OCR_TABLE_MODE', 'cell')
    if ocr_mode not in ('cell','batch'):raise ValueError('OCR_TABLE_MODE: cell или batch')
    profile.phase('Подготовка ячеек таблицы')
    cell_requests, crop_map = [], {}
    # E1: cache original source pixels; the usual remapped crop remains primary.
    original_numeric_cache = OriginalNumericCache(
        image, rect, max_bytes=positive_int('OCR_ORIGINAL_CACHE_MB',32)*1024*1024)
    body_row_count = len(ys)-2
    field_order = ('name','quantity','price') + (('printed_total',) if 'printed_total' in columns else ())
    for idx in range(1, body_row_count+1):
        for field in field_order:
            _crop_t0=time.perf_counter() if debug_perf is not None else None
            crop, offset, source_bbox = cell_crop(rect, idx, columns[field])
            if debug_perf is not None:
                debug_perf.add_crop(field,(time.perf_counter()-_crop_t0)*1000,crop.shape)
            cell_requests.append(BatchCell(crop, idx, field, offset, source_bbox))
            if field in ('quantity','price','printed_total'):
                crop_map[(idx,field)] = crop
    from .ocr.engine import _TEXT_LAST_ROW
    text_last_token = _TEXT_LAST_ROW.set(body_row_count)
    try:
        parallel_footer = None
        if ocr_mode == 'cell':
            profile.phase('Поячеечный OCR: итог, названия и цифры параллельно'
                          if setting('OCR_TABLE_PARALLEL', '1') == '1'
                          else 'Поячеечный OCR: итог и таблица параллельно')
            def read_footer_and_candidates():
                footer_start = time.perf_counter()
                footer_result = read_printed_footer(image, table_source_bottom(rect,xs,ys), footer_guide=footer_focus_guide)
                footer_result['footer_read_ms'] = round((time.perf_counter()-footer_start)*1000,3)
                if e4_enabled and not adaptive_enabled and setting('OCR_E4_FAST_GATE','1') != '1':
                    candidate_start = time.perf_counter()
                    footer_result['numeric_candidates'] = collect_v5_candidates(crop_map)
                    footer_result['candidate_read_ms'] = round((time.perf_counter()-candidate_start)*1000,3)
                return footer_result
            readouts, batch_stats, parallel_footer = read_parallel(
                cell_requests, read_footer_and_candidates,
                read_cells, batch_size=positive_int('OCR_REC_BATCH_NUM',8))
        else:
            profile.phase('OCR P1: пакетный RapidOCR')
            table_started = time.perf_counter()
            readouts, batch_stats = read_batches(cell_requests, batch_size=positive_int('OCR_REC_BATCH_NUM',8))
            batch_stats['table_wall_ms'] = round((time.perf_counter()-table_started)*1000,3)
    finally:
        _TEXT_LAST_ROW.reset(text_last_token)
    geometry['ocr_mode'] = ocr_mode
    geometry['ocr_batch'] = batch_stats
    if ocr_mode == 'batch' and debug_perf is not None:
        debug_perf.add_lane('table',batch_stats.get('table_wall_ms'),status='batch')
    log_budget_events(processing_log, batch_stats, 'Первичная подготовка')
    if batch_stats.get('cell_fallback'):
        geometry['warnings'].append('Поячеечный OCR использовал резервный пакетный режим для части ячеек.')
    if batch_stats['batch_fallback']:
        geometry['warnings'].append('Пакетный OCR был частично заменён поячейковым: ' + '; '.join(batch_stats['errors'][:2]))
        processing_log.append({'type':'operation', 'stage':'Пакетный OCR',
                               'message':f"Пакетный OCR заменён поячейковым для {batch_stats['batch_fallback']} ячеек."})
    from .ocr import text_quality_gate as tqg
    text_review_v3 = tqg.enabled()
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
        _grid_final = geometry.get('grid_anomaly_v3',{}).get('final',{})
        _grid_observed = _grid_final.get('number_check',{}).get('observations',{}).get(idx)
        if _grid_observed is not None:
            cells['number']['printed_observation'] = _grid_observed
        if text_review_v3:
            if idx == body_row_count:
                decision = tqg.plan(cells, rows, number, setting('OCR_VAT_RATE_PERCENT','16'))
                cells['name']['vat_gate'] = decision
                if decision == 'verify':
                    candidate = next(c for c in cell_requests if c.row_id == idx and c.column_id == 'name')
                    try:
                        with rec_scope([(idx,'name')], 'vat_name_gate_retry'):
                            alternate = tqg.verify(candidate, cells['name'], read_cell, read_full)
                    except Exception as exc:
                        alternate = None
                        processing_log.append({'type':'error','stage':'VAT gate','message':str(exc)})
                    if alternate and tqg.vat_marker(alternate['text']):
                        alternate['source_bbox'] = candidate.source_bbox
                        alternate['vat_gate'] = 'verified_marker'
                        cells['name'] = alternate
                    else:
                        cells['name']['vat_name_review'] = True
                elif decision == 'uncertain_no_retry':
                    cells['name']['vat_name_review'] = True
        else:
            if idx == body_row_count and not re.search(r'[нh][дd][сc]|vat',cells['name']['text'],re.I):
                # VAT name may be split across several observed printed fragments.
                old = cells['name']
                candidate = next(c for c in cell_requests if c.row_id == idx and c.column_id == 'name')
                with rec_scope([(idx,'name')], 'vat_name_retry'):
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
    # Snapshot pre-retry observations, including OCR scores for provenance.
    primary_numeric_observations = {
        CandidateBook.key(row['row_id'], field): {
            'text': row['cells'][field]['text'],
            'confidence': row['cells'][field].get('confidence'),
            'model_id':row['cells'][field].get('model_id','v5_cyrillic_mobile')}
        for row in rows for field in ('quantity','price','printed_total')
        if field in row['cells']
    }
    profile.phase('OCR P1: выборочное восстановление чисел')
    retry_budget = positive_int('OCR_P1_MAX_RETRIES', 4)
    initial_recovery = ({'attempted':0,'changed':[],'candidates':[],
                         'details':[],'delegated':'e4_independent_original_roi'}
                        if e4_enabled and setting('OCR_E4_ROUTE_RETRIES','1') == '1' else
                        recover_invalid_and_rows(rows,crop_map,max_retries=retry_budget))
    geometry['ocr_recovery'] = {'initial':initial_recovery}
    log_retry_details(processing_log, initial_recovery, 'Первичная проверка чисел')
    recovered_tax_bottom = None
    if not any(row['kind'] == 'tax' for row in rows):
        profile.phase('Проверка строки НДС под сеткой')
        recovered_tax = recover_tax_below_grid(image, rect, columns, len(rows)+1)
        if recovered_tax:
            row, recovered_tax_bottom, footer_marker, tax_crops = recovered_tax
            rows.append(row)
            for field, crop in tax_crops.items():
                cell = row['cells'][field]
                cell_requests.append(BatchCell(crop, row['row_id'], field,
                                               tuple(cell['bbox'][:2]), cell['source_bbox']))
                crop_map[(row['row_id'],field)] = crop
                # VAT outside the mesh is already an original-pixel extraction.
            processing_log.append({'type':'operation', 'stage':'Проверка НДС',
                                   'message':'Строка НДС найдена под границей сетки.'})
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
    y_bottom = table_source_bottom(rect, xs, ys)
    if recovered_tax_bottom is not None:
        y_bottom = max(y_bottom, recovered_tax_bottom)
    if parallel_footer is None or parallel_footer['y_bottom'] != y_bottom:
        footer_result = read_printed_footer(image, y_bottom, rows, footer_guide=footer_focus_guide)
    else:
        footer_result = parallel_footer
    total, footer_lines, footer_attempts = (footer_result['total'],
        footer_result['lines'], footer_result['attempts'])
    # Only investigate footer mismatch AFTER normal OCR; never invent numbers.
    profile.phase('OCR P1: проверка напечатанного итога')
    expected_total = number(total)
    missing_total = expected_total is None
    incomplete_before = any(row['line_total'] is None for row in rows)
    before_sum = sum((Decimal(row['line_total']) for row in rows if row['line_total'] is not None), Decimal(0))
    receipt_mismatch = expected_total is not None and (incomplete_before or abs(before_sum-expected_total) > Decimal('0.01'))
    row_mismatch = any(row.get('printed_line_match') is False for row in rows)
    taxes_before = [row for row in rows if row['kind'] == 'tax']
    vat_rate = Decimal(setting('OCR_VAT_RATE_PERCENT', '16'))
    products_before = sum((Decimal(row['line_total']) for row in rows
                           if row['kind'] == 'product' and row['line_total'] is not None), Decimal(0))
    tax_before = sum((Decimal(row['line_total']) for row in taxes_before if row['line_total'] is not None), Decimal(0))
    vat_mismatch = bool(taxes_before and not incomplete_before and
                        abs(tax_before-products_before*vat_rate/100) > Decimal('1'))
    budget_limited = bool(batch_stats.get('numeric_roi_crop_exceeded') or
                          batch_stats.get('numeric_roi_budget_exceeded') or
                          batch_stats.get('numeric_roi_skipped_cells') or
                          batch_stats.get('numeric_roi_currency_skipped_cells'))
    low_confidence = any(float(row['cells'][field].get('confidence') or 0) < .80
                         for row in rows for field in ('quantity', 'price'))
    retry_required = incomplete_before or receipt_mismatch or row_mismatch or missing_total
    expanded_recovery = {'attempted':0, 'details':[], 'changed':[], 'candidates':[], 'batch':None}
    if retry_required and (not e4_enabled or setting('OCR_E4_ROUTE_RETRIES','1') != '1'):
        processing_log.append({'type':'check', 'stage':'Контроль сумм',
                               'printed_total':str(expected_total) if expected_total is not None else None,
                               'computed_total':str(before_sum), 'incomplete':incomplete_before,
                               'receipt_mismatch':receipt_mismatch, 'row_mismatch':row_mismatch,
                               'vat_mismatch':vat_mismatch, 'missing_total':missing_total,
                               'budget_limited':budget_limited, 'low_confidence':low_confidence,
                               'message':f'Контроль сумм: напечатано {total if total is not None else "не распознано"}, рассчитано {before_sum}; '
                                         f'неполных строк: {sum(row["line_total"] is None for row in rows)}; '
                                         f'НДС {"не совпадает" if vat_mismatch else "проверен или недоступен"}; '
                                         f'итоги строк {"не совпадают" if row_mismatch else "без выявленных расхождений"}; '
                                         f'{"есть числа с низкой уверенностью после исчерпания бюджета" if budget_limited and low_confidence else "проверка чисел завершена"}.'})
        profile.phase('OCR P1: повтор проблемных ячеек с удвоенным бюджетом')
        try:
            expanded_recovery = retry_suspect_cells(
                rows, cell_requests, batch_stats, expected_total, vat_rate,
                batch_size=positive_int('OCR_REC_BATCH_NUM',8))
            processing_log.append({'type':'operation', 'stage':'Повтор с удвоенным бюджетом',
                                   'attempted':expanded_recovery['attempted'],
                                   'message':f"Повтор с удвоенным бюджетом: проверено ячеек "
                                             f"{expanded_recovery['attempted']}; бюджет 24 + 8 мс."})
            if expanded_recovery['batch'] and not expanded_recovery['batch']['numeric_roi_enabled']:
                processing_log.append({'type':'warning', 'stage':'Повтор с удвоенным бюджетом',
                                       'message':'Подготовка числовых ячеек отключена настройкой OCR_NUMERIC_ROI.'})
            log_budget_events(processing_log, expanded_recovery['batch'], 'Повтор с удвоенным бюджетом')
            log_retry_details(processing_log, expanded_recovery, 'Повтор с удвоенным бюджетом')
        except Exception as exc:
            processing_log.append({'type':'error', 'stage':'Повтор с удвоенным бюджетом',
                                   'message':f'Повтор не выполнен: {type(exc).__name__}: {exc}'})
        profile.phase('OCR P1: дополнительная проверка итога')
    geometry['ocr_recovery']['double_budget'] = expanded_recovery
    footer_recovery = ({'attempted':0,'changed':[],'candidates':[],
                        'details':[],'delegated':'e4_independent_original_roi'}
                       if e4_enabled and setting('OCR_E4_ROUTE_RETRIES','1') == '1' else
                       recover_total_mismatch(rows,crop_map,number(total),
                           max_retries=max(0,retry_budget-initial_recovery['attempted'])))
    geometry['ocr_recovery']['footer'] = footer_recovery
    log_retry_details(processing_log, footer_recovery, 'Дополнительная проверка итога')
    profile.phase('E5: проверка проблемных цифр Tesseract')
    try:
        e5_result = recover_with_tesseract(rows, crop_map, geometry['ocr_recovery'])
    except Exception as exc:
        e5_result = {'status':'error', 'details':[], 'candidates':[],
                     'error':f'{type(exc).__name__}: {exc}'}
    geometry['e5_tesseract'] = e5_result
    if e5_result['status'] in ('error', 'unavailable'):
        geometry['warnings'].append('Резервный Tesseract для проблемных цифр недоступен; проверьте результат.')
    for detail in e5_result['details']:
        observations = ', '.join(a['text'] or 'нет результата' for a in detail['attempts'])
        processing_log.append({'type':'retry', 'stage':'E5 Tesseract KAZ', **detail,
            'message':f"Строка {detail['row_id']}, {detail['column_id']}: Tesseract прочитал {observations}; "
                      f"{'принято' if detail['accepted'] else 'сохранено для проверки'}."})
    if e5_result['status'] in ('error', 'unavailable'):
        processing_log.append({'type':'warning', 'stage':'E5 Tesseract KAZ',
                               'message':f"Резервное чтение недоступно: {e5_result.get('error')}"})
    # E1: finish the normal OCR pipeline first.  Do not mutate its numbers;
    # independent source-pixel observations form alternatives for a bounded
    # candidate matrix, not automatic replacements.
    if e4_enabled:profile.phase('Сверка матрицы чисел')
    e5_book = CandidateBook()
    for candidate in e5_result['candidates']:
        e5_book.add(candidate['row_id'], candidate['field'], candidate['text'], None,
                    'e5_'+candidate['variant'], model='tesseract_kaz')
    numeric_book_export = e5_book.export()
    numeric_matrix = {'status':'not_run' if e4_enabled else 'disabled', 'review_required':e4_enabled}
    original_roi_experiment = {}
    numeric_auto_corrections = {'status':'not_run','changes':[]}
    footer_variant_attempts = []
    vat_evidence = {'source': None, 'source_status': 'not_run', 'observations': []}
    printed_vat_observed = None
    if e4_enabled:
        try:
            audit_sum = sum((Decimal(r['line_total']) for r in rows if r['line_total'] is not None), Decimal(0))
            book = CandidateBook.from_receipt(rows, primary_numeric_observations,
                                             geometry['ocr_recovery'], total, footer_attempts)
            # MULTILINE_VAT_FAST_E4_V1: cheap proof of internal consistency first.
            # Arithmetic is never used to CREATE OCR candidates.
            gate = primary_consistency(rows, total, footer_attempts,
                min_cell_conf=float(setting('OCR_E4_GATE_MIN_CELL_CONF','0.75')),
                min_footer_conf=float(setting('OCR_E4_GATE_MIN_FOOTER_CONF','0.86')))
            fast_matrix = None
            skip_secondary = False
            if gate['eligible'] and setting('OCR_E4_FAST_GATE','1') == '1':
                seed_printed_vat(rows, book)
                fast_matrix = solve_matrix(rows, book, vat_rate,
                    max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
                skip_secondary = (fast_matrix['status']=='unique_math_match' and
                                  not fast_matrix.get('review_required') and
                                  not fast_matrix.get('search_truncated') and
                                  not fast_matrix.get('recommended_changes'))
            geometry['e4_fast_gate'] = {
                **gate, 'enabled':setting('OCR_E4_FAST_GATE','1') == '1',
                'skipped_independent_v5':skip_secondary,
                'primary_matrix_status':fast_matrix.get('status') if fast_matrix else None}
            if skip_secondary:
                processing_log.append({'type':'operation','stage':'E4 быстрый контроль',
                    'message':'Основные OCR-цифры, наблюдаемый итог и матрица согласованы; '
                              'дополнительный полный V5 не запускался.'})
                if adaptive_enabled:
                    # Keep the existing early-preview contract when fast checks
                    # resolve the matrix before adaptive V5 sampling.
                    elapsed = (time.perf_counter()-profile.start_wall)*1000
                    snapshot = make_early_snapshot(rows, book, fast_matrix, total,
                                                  vat_rate, elapsed, 0, job_dir.name)
                    temp = job_dir/'e4_early_result.json.tmp'
                    temp.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
                    temp.replace(job_dir/'e4_early_result.json')
                    adaptive_info.update(first_ready=True, first_elapsed_ms=round(elapsed,2),
                                         probes_before_first=0, probes_used=0,
                                         early_solver_checks=1,
                                         first_matrix_status='unique_math_match',
                                         adaptive_collect_ms=0)
                    profile.publish({'e4_early_ready':True,'e4_first_result':snapshot,
                                     'e4_background_verification':'running'})
            collected = None
            early_scan = None
            if adaptive_enabled and not skip_secondary:
                profile.phase('E4: поиск первого предварительного результата')
                def publish_first_matrix(matrix, candidate_book, scans):
                    # This is a read-only, immutable snapshot, separate from receipt.json.
                    elapsed = (time.perf_counter()-profile.start_wall)*1000
                    snapshot = make_early_snapshot(rows, candidate_book, matrix, total,
                                                  vat_rate, elapsed, scans, job_dir.name)
                    temp = job_dir/'e4_early_result.json.tmp'
                    temp.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
                    temp.replace(job_dir/'e4_early_result.json')
                    adaptive_info.update(first_ready=True, first_elapsed_ms=round(
                        (time.perf_counter()-profile.start_wall)*1000,2), probes_before_first=scans)
                    profile.publish({'e4_early_ready':True, 'e4_first_result':snapshot,
                                     'e4_background_verification':'running'})
                    processing_log.append({'type':'operation','stage':'E4 предварительный результат',
                        'message':f'Первый согласованный снимок сохранён через {adaptive_info["first_elapsed_ms"]:.0f} мс; полная проверка продолжается.'})
                try:
                    suspicious = prioritize_suspects(rows, crop_map,
                        limit=positive_int('OCR_E4_TARGETED_MAX_PROBES',4))
                    early_scan = adaptive_collect(rows, book, crop_map, vat_rate,
                        collect_v5_candidates, publish_first_matrix,
                        max_nodes=positive_int('OCR_E4_ADAPTIVE_MAX_NODES',2500),
                        priority_keys=suspicious,
                        max_probes=(positive_int('OCR_E4_TARGETED_MAX_PROBES',4)
                                    if setting('OCR_E4_TARGETED_FIRST','1')=='1'
                                    else None))
                    collected = early_scan['collected']
                    adaptive_info.update(probes_used=early_scan['probes_used'],
                        early_solver_checks=early_scan['matrix_checks'],
                        first_matrix_status=early_scan['last_matrix_status'],
                        adaptive_collect_ms=early_scan['elapsed_ms'])
                except Exception as exc:
                    adaptive_info['error'] = f'{type(exc).__name__}: {exc}'
                    report_issue(source='E4_adaptive',operation='early_collect',
                                 kind='early_exception',error=exc,fallback='full_e4')
                # Early ranking/VAT evidence must not reorder or restrict the
                # normal final audit. Rebuild it with the baseline provenance.
                book = CandidateBook.from_receipt(rows, primary_numeric_observations,
                                                 geometry['ocr_recovery'], total, footer_attempts)
                profile.phase('E4: полная проверка чисел')
                # Build the FINAL book in the baseline insertion order. Early
                # ranking must not change confidence ties or candidate caps.
                book = CandidateBook.from_receipt(rows, primary_numeric_observations,
                                                 geometry['ocr_recovery'], total, footer_attempts)
                profile.phase('E4: полная проверка после предварительного результата')
            else:
                collected = ({'observations':[], 'errors':[], 'attempted':set()}
                             if skip_secondary else (parallel_footer or {}).get('numeric_candidates'))
            if collected is None:
                collected = collect_v5_candidates(crop_map)
            # A full independent sweep is a fallback, not the normal path.
            # Stop after E4 found an eligible unique first result; remaining
            # cells are not silently labelled as having been OCR-checked.
            targeted_resolved = bool(early_scan and early_scan.get('early_found') and
                                     setting('OCR_E4_TARGETED_FIRST','1')=='1')
            geometry['e4_fast_gate']['targeted_resolved'] = targeted_resolved
            missing = ({} if skip_secondary or targeted_resolved else
                       {key:crop for key,crop in crop_map.items()
                        if key not in collected['attempted']})
            if missing:
                extra = collect_v5_candidates(missing)
                collected['observations'].extend(extra['observations'])
                collected['errors'].extend(extra['errors'])
                collected['attempted'].update(extra['attempted'])
            if adaptive_enabled:
                # Some attempts may be intentionally omitted on an eligible
                # unique partial solution; numbers reflect ACTUAL calls.
                # Never rescan price/currency variants already observed.
                order = {key:i for i,key in enumerate(crop_map)}
                collected['observations'].sort(key=lambda o:order[(o['row_id'],o['field'])])
                adaptive_info['full_v5_cells'] = len(collected['attempted'])
            candidate_calls = len(collected['attempted'])
            candidate_failures = collected['errors']
            for candidate in collected['observations']:
                book.add(candidate['row_id'],candidate['field'],candidate['text'],
                         candidate['confidence'],candidate.get('path','v5_numeric_candidate'),
                         model=candidate.get('model_id','v5_cyrillic_mobile'))
            if candidate_failures:
                geometry['warnings'].append('Не удалось собрать все резервные OCR-варианты чисел; проверьте результат.')
            for candidate in e5_result['candidates']:
                book.add(candidate['row_id'], candidate['field'], candidate['text'], None,
                         'e5_'+candidate['variant'], model='tesseract_kaz')
            seed = seed_printed_vat(rows, book)
            vat_evidence = {'source':seed['source'], 'source_status':seed['status'],
                            'observations':[], 'probes':0, 'errors':[]}
            printed_vat_observed = (observed_vat_text(book, preferred=seed.get('text'))
                                    or printed_vat_observed)
            numeric_matrix = solve_matrix(rows, book, vat_rate,
                max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
            initial_matrix_status = numeric_matrix['status']
            # E4_MATH_TARGET_EXPERIMENT_V1: bounded, read-only location prediction.
            # Inferred numbers NEVER enter CandidateBook as observations.
            fast_e5 = {'status':'not_needed', 'details':[], 'candidates':[]}
            fast_targets = []
            if (setting('OCR_E4_FAST_MATH_E5','1') == '1' and
                    numeric_matrix['status'] not in ('unique_math_match', 'ambiguous') and
                    not numeric_matrix.get('search_truncated')):
                fast_targets = select_math_targets(
                    rows, total, book,
                    max_cells=positive_int('OCR_E4_FAST_MATH_E5_MAX_CELLS',2))
                if fast_targets:
                    profile.phase('E4 FAST: адресная проверка Tesseract KAZ')
                    _fast_e5_started = time.perf_counter()
                    try:
                        fast_e5 = recover_with_tesseract(
                            rows, crop_map, {}, accept_invalid=False,
                            force_keys=[(x['row_id'],x['field']) for x in fast_targets],
                            max_cells=len(fast_targets))
                    except Exception as exc:
                        fast_e5 = {'status':'error', 'details':[], 'candidates':[],
                                   'error':f'{type(exc).__name__}: {exc}'}
                    fast_e5['wall_ms'] = round((time.perf_counter()-_fast_e5_started)*1000,3)
                    for candidate in fast_e5.get('candidates', []):
                        book.add(candidate['row_id'],candidate['field'],candidate['text'],None,
                                 'e5_math_target_'+candidate['variant'],model='tesseract_kaz')
                    if fast_e5.get('candidates'):
                        numeric_matrix = solve_matrix(
                            rows, book, vat_rate,
                            max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
                    processing_log.append({'type':'operation',
                        'stage':'E4 FAST адресная E5',
                        'message':'Адресная E5: '
                            +str([(x['row_id'],x['field']) for x in fast_targets])
                            +'; результат '+str(numeric_matrix['status']),
                        'targets':fast_targets, 'e5_status':fast_e5.get('status'),
                        'elapsed_ms':fast_e5.get('wall_ms'),
                        'matrix_before':initial_matrix_status,
                        'matrix_after':numeric_matrix['status']})
            geometry['e4_math_target'] = {
                'enabled':setting('OCR_E4_FAST_MATH_E5','1') == '1',
                'targets':fast_targets,'e5_status':fast_e5.get('status'),
                'elapsed_ms':fast_e5.get('wall_ms',0),
                'matrix_before':initial_matrix_status,
                'matrix_after':numeric_matrix['status']}
            geometry['e5_tesseract']['early_math_target'] = fast_e5
            original_roi_experiment = {'cache_cells':0,'cache_bytes':0,'rec_calls':0,
                                       'skipped':'matrix_resolved'}
            model_pool = ExperimentModelPool()
            if numeric_matrix['status'] not in ('unique_math_match','ambiguous') or numeric_matrix.get('search_truncated'):
                profile.phase('Резервная проверка оригинальных пикселей')
                for row in rows:
                    for field in ('quantity','price','printed_total'):
                        key=(row['row_id'],field)
                        if key not in crop_map:continue
                        if row['row_id'] <= body_row_count:
                            original_numeric_cache.store(row['row_id'],field,columns[field])
                        else:
                            crop=crop_map[key]
                            if original_numeric_cache.bytes+crop.nbytes<=original_numeric_cache.max_bytes:
                                original_numeric_cache.crops[key]=crop.copy()
                                original_numeric_cache.bytes+=crop.nbytes
                footer_variant_attempts = scan_footer(
                    image, y_bottom, total, audit_sum,
                    read_line=read_line, read_full=read_full,
                    # A confident source OCR footer equal to the primary
                    # mathematical total needs no further duplicate OCR.
                    always_check=(setting('OCR_FOOTER_VERIFY_ALWAYS', '1') == '1'
                                  and not (setting('OCR_E4_SKIP_VERIFIED_FOOTER','1') == '1'
                                           and gate['footer_trusted']
                                           and 'total_mismatch' not in gate['reasons'])),
                    max_probes=positive_int('OCR_FOOTER_MAX_PROBES',8),
                    model_pool=model_pool)
                for alternative in footer_variant_attempts:
                    if alternative.get('total') is not None:
                        book.add_footer(alternative['total'], alternative.get('confidence'),
                                        alternative['strategy'],
                                        model=alternative.get('model_id') or alternative.get('model','PP-OCRv5 Cyrillic Mobile / RapidOCR 3.x'))
                # E3: photograph-based OCR of VAT, with its own independent observations.
                # Never infer 'printed VAT' solely by applying the tax percentage.
                vat_evidence = scan_printed_vat(
                    rows, original_numeric_cache, book, vat_rate, read_line=read_line,
                    model_pool=model_pool,
                    always_check=setting('OCR_VAT_VERIFY_ALWAYS','1') == '1',
                    max_probes=positive_int('OCR_VAT_MAX_PROBES',6))
                printed_vat_observed = (observed_vat_text(book, preferred=printed_vat_observed)
                                        or printed_vat_observed)
                original_roi_experiment = original_numeric_cache.run_suspects(
                    rows, book, printed_total=total,
                    max_calls=positive_int('OCR_ORIGINAL_MAX_RECOGNITIONS',8),
                    read_line=read_line, model_pool=model_pool)
                numeric_matrix = solve_matrix(rows, book, vat_rate,
                    max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
                if ((numeric_matrix['status'] != 'unique_math_match' or
                     numeric_matrix.get('review_required') or numeric_matrix.get('recommended_changes')) and
                        setting('OCR_E4_SECOND_PASS','1') == '1'):
                    target_keys={(v['row_id'],v['field']) for v in numeric_matrix.get('recommended_changes',[])}
                    second_pass = original_numeric_cache.escalate_suspects(
                        book, original_roi_experiment, model_pool,
                        max_calls=positive_int('OCR_E4_SECOND_PASS_MAX',3),
                        only_keys=target_keys if numeric_matrix['status']=='unique_math_match' and target_keys else None)
                    original_roi_experiment['second_pass'] = second_pass
                    if second_pass['recognitions']:
                        numeric_matrix = solve_matrix(rows, book, vat_rate,
                            max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
            # E4 may only now have observed letters/symbols in a numeric cell.
            # Give those cells the remaining E5 budget before the final solve.
            e5_seen = {(d['row_id'], d['column_id']) for d in e5_result['details']}
            e5_seen.update((d['row_id'],d['column_id']) for d in fast_e5.get('details',[]))
            try:
                late_e5 = recover_with_tesseract(
                    rows, crop_map, {}, book=book, exclude_keys=e5_seen,
                    max_cells=max(0, 3-len(e5_seen)), accept_invalid=False)
            except Exception as exc:
                late_e5 = {'status':'error', 'details':[], 'candidates':[],
                           'error':f'{type(exc).__name__}: {exc}'}
            geometry['e5_tesseract']['after_e4'] = late_e5
            if late_e5['status'] == 'completed':
                e5_result['status'] = 'completed'
            if late_e5['status'] == 'budget_exhausted':
                geometry['warnings'].append('Не все числовые ячейки с символами проверены Tesseract: исчерпан лимит.')
                processing_log.append({'type':'warning', 'stage':'E5 Tesseract KAZ после E4',
                    'message':'Не все числовые ячейки с символами проверены: исчерпан лимит Tesseract.'})
            for detail in late_e5['details']:
                observations = ', '.join(a['text'] or 'нет результата' for a in detail['attempts'])
                processing_log.append({'type':'retry', 'stage':'E5 Tesseract KAZ после E4', **detail,
                    'message':f"Строка {detail['row_id']}, {detail['column_id']}: Tesseract прочитал {observations}; варианты переданы в матрицу."})
            if late_e5['status'] in ('error', 'unavailable'):
                geometry['warnings'].append('Tesseract недоступен для дополнительной проверки цифр после E4.')
                processing_log.append({'type':'warning', 'stage':'E5 Tesseract KAZ после E4',
                    'message':f"Резервное чтение недоступно: {late_e5.get('error')}"})
            for candidate in late_e5['candidates']:
                book.add(candidate['row_id'], candidate['field'], candidate['text'], None,
                         'e5_'+candidate['variant'], model='tesseract_kaz')
            if late_e5['candidates']:
                numeric_matrix = solve_matrix(rows, book, vat_rate,
                    max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
            # Display only a VAT figure that was actually observed by OCR.
            if numeric_matrix.get('status') == 'unique_math_match' and numeric_matrix.get('solutions'):
                printed_vat_observed = (observed_vat_text(
                    book, preferred=numeric_matrix['solutions'][0].get('vat_printed'))
                    or printed_vat_observed)
            numeric_auto_corrections = auto_apply_consensus(
                rows,book,numeric_matrix,total,
                enabled=setting('OCR_E4_AUTO_APPLY','1') == '1')
            if numeric_auto_corrections['status']=='applied':
                total=numeric_auto_corrections['printed_total']
                # Corrections may turn an unreadable VAT row into a valid
                # photographed value (e.g. quantity=23200, price=1).
                post_e4_vat_seed = seed_printed_vat(rows, book)
                if post_e4_vat_seed['source'] is not None:
                    vat_evidence['source'] = post_e4_vat_seed['source']
                    vat_evidence['post_e4_source_status'] = post_e4_vat_seed['status']
                printed_vat_observed = (observed_vat_text(
                    book, preferred=numeric_matrix['solutions'][0].get('vat_printed'))
                    or printed_vat_observed)
                numeric_matrix = solve_matrix(rows, book, vat_rate,
                    max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
                processing_log.append({'type':'operation','stage':'E4 автоматическая сверка',
                    'message':'Выбрана единственная арифметически согласованная комбинация: '
                              + str(len(numeric_auto_corrections['changes'])) + ' исправлений.'})
            if skip_secondary:
                numeric_matrix['candidate_scope'] = 'primary_only_fast_math'
                numeric_matrix['candidate_scope_note'] = (
                    'Сумма согласуется с исходным OCR, но дополнительные V5-кандидаты '
                    'для всех ячеек не собирались; это не оптическая истина.')
            elif targeted_resolved:
                numeric_matrix['candidate_scope'] = 'targeted_v5_early_only'
                numeric_matrix['candidate_scope_note'] = (
                    'Были прочитаны только адресные V5-кандидаты. При новом конфликте '
                    'математики или OCR применяется штатное восстановление.')
            elif (geometry['e4_math_target']['matrix_after'] == 'unique_math_match' and
                    original_roi_experiment.get('skipped') == 'matrix_resolved'):
                numeric_matrix['candidate_scope'] = 'early_targeted_without_full_source_scan'
                numeric_matrix['candidate_scope_note'] = (
                    'Единственное решение относится к собранным вариантам; '
                    'полный поиск альтернатив по исходным пикселям не выполнялся.')
            numeric_matrix['candidate_collection'] = {'v5_calls':candidate_calls, 'errors':candidate_failures}
            numeric_matrix['initial_status'] = initial_matrix_status
            numeric_matrix['fallback_used'] = original_roi_experiment.get('skipped') != 'matrix_resolved'
            numeric_matrix['auto_applied'] = numeric_auto_corrections
            numeric_matrix['user_action_required'] = numeric_matrix['status']=='ambiguous'
            numeric_matrix['automatic_resolution'] = (
                'consistent' if numeric_matrix['status']=='unique_math_match' and not numeric_matrix['review_required']
                else 'selected_by_confidence_review_required' if numeric_matrix['status']=='ambiguous'
                else 'unresolved_preserved_without_forced_edit')
            numeric_book_export = book.export()
            geometry['numeric_experiment'] = {
                'status':numeric_matrix['status'],
                'original_cache_cells':original_roi_experiment['cache_cells'],
                'original_cache_bytes':original_roi_experiment['cache_bytes'],
                'source_rec_calls':original_roi_experiment['rec_calls'],
                'footer_probe_calls':len(footer_variant_attempts),
                'vat_probe_calls':vat_evidence['probes'],
                'vat_source_status':vat_evidence['source_status'],
                'vat_variants':numeric_matrix['vat_variants'],
                'critical_ambiguity':numeric_matrix['critical_ambiguity'],
                'matrix_nodes':numeric_matrix['nodes_visited'],
                'matrix_bounded':numeric_matrix['search_truncated'],
                'model_ensemble':model_pool.stats()}
            if has_multiple_consistent_footer_values(numeric_matrix):
                geometry['warnings'].append('OCR печатного итога дал разные варианты; альтернативы сохранены для анализа.')
            if numeric_matrix['vat_ambiguous']:
                geometry['warnings'].append('OCR напечатанного НДС дал разные значения; альтернативы сохранены для анализа.')
            if numeric_matrix.get('vat_rate_diagnostic',{}).get('status')=='operator_calculation_anomaly':
                processing_log.append({'type':'financial_warning','stage':'Контроль НДС',
                    'message':'Напечатанный НДС расходится с расчётом по ставке, но это не ошибка OCR само по себе.'})
            if numeric_matrix['critical_ambiguity']:
                geometry['warnings'].append('Найдены разные математически согласованные комбинации цифр. Выбрана комбинация с наибольшей уверенностью OCR. Пожалуйста, проверьте цифры.')
            if any(row.get('kind')=='tax' for row in rows) and not numeric_matrix['printed_vat_observed']:
                geometry['warnings'].append('Напечатанный НДС не удалось надёжно прочитать из числовых ячеек строки НДС; сумма, рассчитанная по ставке, не подставлялась.')
            if numeric_matrix['recommended_changes']:
                geometry['warnings'].append('Матрица сохранила альтернативные цифры; исходные OCR-значения оставлены без автоматической подмены.')
            processing_log.append({'type':'operation', 'stage':'Эксперимент E1',
                'message':f"Матрица: исходных ROI {original_roi_experiment['cache_cells']}, "
                          f"повторов {original_roi_experiment['rec_calls']}, "
                          f"вариантов итога {len(numeric_matrix['footer_variants'])}, "
                          f"вариантов НДС {len(numeric_matrix['vat_variants'])}, "
                          f"матрица {numeric_matrix['status']}."})
        except Exception as exc:
            numeric_matrix={'status':'experiment_error','review_required':True,
                            'error':f'{type(exc).__name__}: {exc}'}
            geometry['numeric_experiment']={'status':'experiment_error'}
            geometry['warnings'].append('Дополнительная проверка альтернатив OCR завершилась ошибкой; основной результат сохранён.')
            processing_log.append({'type':'error','stage':'Эксперимент E1',
                                   'message':f'Эксперимент E1: {type(exc).__name__}: {exc}'})
    if initial_recovery['candidates'] or expanded_recovery['candidates'] or footer_recovery['candidates']:
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
                geometry['warnings'].append(f'Строка {row["row_id"]}: {label} «{shown}» не удалось надёжно распознать; число не рассчитано.')
    ocr_s = time.perf_counter() - ocr_start
    profile.phase('Сборка и проверка электронного чека')
    computed = sum((Decimal(r['line_total']) for r in rows if r['line_total'] is not None), Decimal(0))
    incomplete = any(r['line_total'] is None for r in rows)
    match = not incomplete and number(total) is not None and abs(computed-number(total)) <= Decimal('0.01')
    taxes = [r for r in rows if r['kind'] == 'tax']
    product_sum = sum((Decimal(r['line_total']) for r in rows if r['kind'] == 'product' and r['line_total'] is not None), Decimal(0))
    vat_sum = sum((Decimal(r['line_total']) for r in taxes if r['line_total'] is not None), Decimal(0))
    vat_ok = None if not taxes or incomplete else abs(vat_sum-product_sum*vat_rate/100) <= Decimal('1')
    if retry_required:
        processing_log.append({'type':'check', 'stage':'Итоговая проверка',
                               'computed_total':str(computed), 'math_match':match,
                               'vat_match':vat_ok,
                               'message':f"Итоговая проверка: сумма {computed}, "
                                         f"{'совпадает' if match else 'требует проверки'}; "
                                         f"НДС {'совпадает' if vat_ok is True else 'требует проверки'}."})
    if incomplete: geometry['warnings'].append('Есть ячейки с нераспознанным количеством или ценой.')
    if total is None: geometry['warnings'].append('Печатный итог не получен однозначно; доступные OCR-варианты сохранены.')
    elif not match: geometry['warnings'].append('Сумма строк не совпадает с печатным итогом.')
    # Financial validation is orthogonal to whether printed digits were read.
    if not e4_enabled and vat_ok is False:
        geometry['warnings'].append('Сумма НДС не совпадает с расчётом по установленной ставке.')
    operator_vat_warning=('Напечатанный НДС расходится с установленной ставкой; возможна ошибка оператора.'
                           if vat_ok is False else None)
    map_path = job_dir/'rectification_map.npz'
    np.savez_compressed(map_path, map_x=rect.map_x, map_y=rect.map_y)
    geometry['inverse_map'] = str(map_path.relative_to(ROOT)).replace('\\', '/')
    problem_text = None
    try:
        from .ocr.text_debug import save_text_debug
        from .ocr.experiment_rec_trace import current_text_inputs
        text_inputs, text_dropped = current_text_inputs()
        problem_text = save_text_debug(job_dir, rows, cell_requests, image, rect, columns, current_trace(), text_inputs, text_dropped)
        if problem_text:
            processing_log.append({'type':'debug','stage':'Проблемный текст','message':f"Текст: {problem_text['suspects']} подозрительных ячеек; отчёт problem_text/report.html"})
    except Exception as exc:
        processing_log.append({'type':'error','stage':'Проблемный текст','message':f'{type(exc).__name__}: {exc}'})
    problem_digits = None
    if debug_perf is not None:
        from .ocr.problem_digits import save_problem_digits
        try:
            problem_digits = save_problem_digits(job_dir, rows, crop_map,
                numeric_book_export, numeric_auto_corrections, original_numeric_cache.crops,
                debug_perf.primary_numeric_inputs,
                footer_context={'image':image,'total':total,'lines':footer_lines,
                                'attempts':footer_attempts+footer_variant_attempts},
                rec_trace=current_trace(), attempt_inputs=current_inputs(),
                vat_context=vat_evidence, math_match=match)
            if problem_digits['count']:
                processing_log.extend(problem_digits.pop('log_entries',[]))
                processing_log.append({'type':'debug', 'stage':'Проблемные цифры',
                    'message':f"Фото проблемных цифровых ячеек: {problem_digits['count']}. Отчёт: {job_dir.relative_to(ROOT).as_posix()}/problem_digits/report.html"})
            else:
                problem_digits = None
        except Exception as exc:
            processing_log.append({'type':'error','stage':'Проблемные цифры',
                'message':f'Не удалось сохранить фото цифровых ячеек: {type(exc).__name__}: {exc}'})
    return {'status': 'review' if geometry['warnings'] else 'ok', 'geometry': geometry,
            'problem_digits': problem_digits,
            'problem_text': problem_text,
            'processing_log': processing_log,
            'original': png(image), 'rectified': png(overlay), 'rows': rows,
            'numeric_issues': numeric_issues,
             'numeric_candidate_book': numeric_book_export,
             'ocr_user_action_required': False,
             'numeric_auto_corrections': numeric_auto_corrections,
             'e4_adaptive': adaptive_info,
             'numeric_matrix': numeric_matrix,
             'original_roi_experiment': original_roi_experiment,
             'footer_variant_attempts': footer_variant_attempts,
             'vat_evidence': vat_evidence,
             'printed_vat': printed_vat_observed,
             'vat_rate_diagnostic': numeric_matrix.get('vat_rate_diagnostic'),
             'financial_warnings': [operator_vat_warning] if operator_vat_warning else [],
             'ocr_validation_status':('consistent' if match and not numeric_matrix.get('review_required',True)
                                      else 'review'),
            'total': total, 'computed_total': str(computed), 'math_match': match,
            'vat': {'status': 'not_applicable' if vat_ok is None else 'match' if vat_ok else 'mismatch',
                    'rate_percent': str(vat_rate), 'amount': str(vat_sum) if taxes else None},
            'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts,
            'footer_focus': footer_result.get('footer_focus'), 'timings': {'geometry_s': round(geo_s, 3),
            'ocr_s': round(ocr_s, 3), 'total_s': round(time.perf_counter()-started, 3)}}


def read_printed_footer(image, y_bottom, rows=(), *, footer_guide=None):
    """Read photographed footer independently; optional legacy sum evidence."""
    def read_footer_line(source_crop):
        # Only change OCR pixels when a separated currency glyph is confirmed.
        # Uncertain crops retain the original image, including all digits.
        prepared, roi = prepare_numeric_input(source_crop, 'price')
        removed = roi.get('currency_removed_before_ocr', False)
        result = read_line(prepared if removed else source_crop)
        if result is not None:
            result['currency_crop'] = {
                'removed_before_ocr': bool(removed),
                'status': roi.get('currency_crop_status', 'not_confirmed'),
                'ocr_roi_bbox': roi.get('ocr_roi_bbox') if removed else None,
            }
        return result
    # FOCUSED_FOOTER_EXPERIMENT_V1: optical evidence strictly between source rules.
    focused_attempt=None
    focus_info={'status':'disabled'}
    if setting('OCR_FOOTER_FOCUS_ENABLE','1') == '1':
        def focused_reader(crop):
            with rec_scope(['receipt_total'], 'printed_footer_focused'):
                return read_footer_line(crop)
        min_conf=float(setting('OCR_FOOTER_FOCUS_MIN_CONF','0.86'))
        # A recovered VAT row can extend the table after this guide was built.
        # Do not read that newly recovered row as though it were the footer.
        current_guide=footer_guide
        if (current_guide and abs(float(current_guide['grid_bottom_y'])-y_bottom) >
                max(12,float(current_guide['last_row_height'])*.6)):
            current_guide=None
        focused=attempt_focused_footer(image,current_guide,focused_reader,number,
                                       min_confidence=min_conf)
        focus_info={**focused['diagnostics'],
                    'accepted':focused['accepted'],
                    'result':focused['total']}
        focused_attempt=focused['attempt']
        if not focused['accepted'] and focused_attempt and rows and all(r['line_total'] is not None for r in rows):
            observed = number(focused_attempt.get('unaccepted_numeric_candidate'))
            calculated = sum((Decimal(r['line_total']) for r in rows), Decimal(0))
            if observed is not None and abs(calculated-observed) <= Decimal('0.01'):
                x0,y0,x1,y1 = focused_attempt['source_roi_bbox']
                line = {'text':focused_attempt['text'], 'confidence':focused_attempt['confidence'],
                        'bbox':[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]}
                focused_attempt.update(total=str(observed), lines=[line], status='accepted_by_arithmetic')
                focus_info.update(accepted=True, result=str(observed))
                return {'total':str(observed), 'lines':[line], 'attempts':[focused_attempt],
                        'y_bottom':y_bottom, 'footer_focus':focus_info}
        if focused['accepted']:
            return {'total':focused['total'],'lines':focused['lines'],
                    'attempts':[focused_attempt], 'y_bottom':y_bottom,
                    'footer_focus':focus_info}
    footer_top = max(0,y_bottom-20)
    # All footer recognitions are attributable to one printed total.
    def footer_read_line(source_crop):
        with rec_scope(['receipt_total'],'printed_footer_primary'):
            return read_footer_line(source_crop)
    def footer_read_full(source_crop, **kwargs):
        with rec_scope(['receipt_total'],'printed_footer_detector'):
            return read_full(source_crop, **kwargs)
    footer = image[footer_top:min(image.shape[0], y_bottom+max(55,int(image.shape[1]*.09))), :]
    footer_left = int(image.shape[1]*.45)
    total_crop = footer[:,footer_left:]
    def read_total(scale, *, fast=False):
        sample = cv2.resize(total_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else total_crop
        if not sample.size:
            lines = []
        elif fast:
            lines = footer_read_full(sample, footer_mode=True)
        else:
            lines = footer_read_full(sample)
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
    direct = footer_read_line(image[fast_top:fast_bottom,fast_x:])
    footer_attempts = [focused_attempt] if focused_attempt is not None else []
    total, footer_lines = None, []
    if direct:
        direct['bbox'] = [[fast_x,fast_top], [image.shape[1],fast_top],
                          [image.shape[1],fast_bottom], [fast_x,fast_bottom]]
        candidate = number(direct['text']) if re.fullmatch(r'[\d\s,.]+\s*[TТ₸]?', direct['text'].strip()) else None
        footer_attempts.append({'strategy':'single_line', 'total':str(candidate) if candidate is not None else None, 'lines':[direct]})
        if (candidate is not None and (direct['confidence'] >= .93 or
                (bool(rows) and all(r['line_total'] is not None for r in rows)
                 and abs(current_sum-candidate) <= Decimal('0.01')))):
            total, footer_lines = str(candidate), [direct]
        # An independent recognizer can confirm a readable but sub-threshold
        # amount before trying multiple crops and detector-based recognition.
        if total is None and candidate is not None and candidate > 0:
            source_crop = image[fast_top:fast_bottom,fast_x:]
            prepared, roi = prepare_numeric_input(source_crop, 'price')
            if not roi.get('currency_removed_before_ocr', False):
                prepared = source_crop
            with rec_scope(['receipt_total'], 'printed_footer_v6_confirmation'):
                check = ExperimentModelPool().recognize(
                    'v6_tiny_cyrillic_optional', prepared, 'printed_footer_v6_confirmation')
            check_text = str(check.get('text') or '').strip()
            check_value = number(check_text) if re.fullmatch(r'[\d\s,.]+\s*[TТт₸]?', check_text) else None
            check_confidence = check.get('confidence')
            matched = (check_value is not None and check_value > 0 and
                       abs(check_value-candidate) <= Decimal('0.01'))
            footer_attempts.append({'strategy':'v6_tiny_cyrillic_confirmation',
                'model_id':'v6_tiny_cyrillic_optional', 'text':check_text,
                'confidence':check_confidence, 'total':str(check_value) if check_value is not None else None,
                'matched':matched, 'wall_ms':check.get('wall_ms'),
                'lines':[{**check, 'bbox':direct['bbox']}],
                **({'error':check['error']} if check.get('error') else {})})
            if matched:
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
            line = footer_read_line(image[top:bottom, left:])
            if not line:
                continue
            line['bbox'] = [[left,top], [image.shape[1],top], [image.shape[1],bottom], [left,bottom]]
            candidate = number(line['text']) if re.fullmatch(r'[\d\s,.]+\s*[TТ₸]?', line['text'].strip()) else None
            footer_attempts.append({'strategy':'compact_line', 'total':str(candidate) if candidate is not None else None, 'lines':[line]})
            if candidate is None or candidate <= 0:
                continue
            if line['confidence'] >= .93 or (bool(rows) and all(r['line_total'] is not None for r in rows) and abs(current_sum-candidate) <= Decimal('0.01')):
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
        if total is None or (total_confidence < .85 and not (bool(rows) and all(r['line_total'] is not None for r in rows) and abs(current_sum-number(total)) <= Decimal('0.01'))):
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
    return {'total':total, 'lines':footer_lines, 'attempts':footer_attempts,
            'y_bottom':y_bottom, 'footer_focus':focus_info}


def table_source_bottom(rect, xs, ys):
    last_y = min(rect.map_y.shape[0]-1, max(0, ys[-1]))
    if rect.method == 'original_cells':
        return int(max(np.max(cell_polygon(rect, len(ys)-2, k)[34:51, 1])
                       for k in range(len(xs)-1)) + rect.info['source_offset'][1])
    return int(np.max(rect.map_y[last_y, max(0,xs[0]):min(rect.map_y.shape[1],xs[-1])]))


def collect_v5_candidates(crops):
    """Independent numeric observations; called only after footer recognition."""
    out={'observations':[], 'errors':[], 'attempted':set()}
    for (row_id,field),crop in crops.items():
        if field not in ('quantity','price','printed_total'):continue
        out['attempted'].add((row_id,field))
        try:
            with rec_scope([(row_id,field)], 'v5_numeric_candidate'):
                prepared,roi = prepare_numeric_input(crop,field)
                alternative=read_line(prepared)
            if alternative:
                out['observations'].append({'row_id':row_id,'field':field,
                                            'text':alternative['text'],
                                            'confidence':alternative['confidence'], 'numeric_roi':roi})
                if field == 'price' and not roi.get('currency_removed_before_ocr'):
                    price_crop=prepare_price_candidate(crop,alternative['text'])
                    if price_crop is not None:
                        with rec_scope([(row_id,field)], 'price_digits_candidate'):
                            item=read_line(price_crop)
                        if item and item.get('text'):
                            out['observations'].append({'row_id':row_id,'field':field,
                                'text':item['text'],'confidence':item.get('confidence'),
                                'model_id':'v5_cyrillic_mobile',
                                'path':'price_digits_candidate'})
        except Exception as exc:
            out['errors'].append({'row_id':row_id,'field':field,
                                  'error':f'{type(exc).__name__}: {exc}'})
    return out
