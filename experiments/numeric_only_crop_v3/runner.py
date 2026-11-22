"""Read-only CnOCR number-densenet_lite_136-fc comparison for GridMesh output.

Pinned to upstream GridMesh_RapidOCR a19598a3899e2da174ba94348885aa5e3474c493.
Does not modify App/, .env, receipt.json or any existing OCR results.
"""
from __future__ import annotations
import argparse
import html
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import FIELDS, prepare_integer_cell

MODEL = 'number-densenet_lite_136-fc'
EXPECTED_HEAD = 'a19598a3899e2da174ba94348885aa5e3474c493'
FIELDS_ORDERED = ('quantity', 'price', 'printed_total')


def configuration(name, default):
    """Mirror App/config.py priority: process environment, then root .env."""
    if name in os.environ:
        return os.environ[name]
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            key, sep, val = line.partition('=')
            if sep and key.strip() == name:
                return val.strip().strip('"\'')
    return default


def output_dir():
    p = (ROOT / configuration('OUTPUT_DIR', 'data/output')).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError('OUTPUT_DIR must remain within GridMesh project')
    return p


def detect_git_head():
    try:
        proc = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=4, check=True)
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def normalize_numeric(value):
    text = str(value or '').strip()
    text = re.sub(r'[₸тТTt\s]', '', text)
    return text if re.fullmatch(r'[0-9]+', text) else None


def percentile_summary(values):
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    return {'count': len(vals), 'mean_ms': round(statistics.mean(vals), 3),
            'median_ms': round(statistics.median(vals), 3),
            'p95_ms': round(vals[max(0, int(np.ceil(.95 * len(vals))) - 1)], 3),
            'sum_ms': round(sum(vals), 3)}


def find_job(job=None, latest=False):
    base = output_dir()
    if job:
        p = Path(job)
        if not p.is_dir():
            p = base / job
        if not p.is_dir():
            raise FileNotFoundError('Job directory not found: ' + str(job))
        if not (p/'receipt.json').is_file():
            raise FileNotFoundError('Missing receipt.json in ' + str(p))
        return p.resolve()
    if not latest:
        raise ValueError('Use --job DIR_OR_JOB_ID or --latest')
    if not base.is_dir():
        raise FileNotFoundError('No GridMesh output directory: ' + str(base))
    candidates = [p for p in base.iterdir() if p.is_dir() and
                  all((p/name).is_file() for name in
                      ('receipt.json', 'original.png', 'rectification_map.npz'))]
    if not candidates:
        raise FileNotFoundError('No completed job with source images and saved map')
    return max(candidates, key=lambda p: (p/'receipt.json').stat().st_mtime)


def load_bgr(path):
    # cv2.imread can fail with Cyrillic filesystem paths on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    result = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return result


def extract_cell(image, map_x, map_y, bbox):
    if map_x.shape != map_y.shape or map_x.ndim != 2:
        return None, 'invalid_rectification_maps'
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None, 'invalid_bbox'
    try:
        x0, y0, x1, y1 = map(int, map(round, bbox))
    except (ValueError, TypeError):
        return None, 'invalid_bbox_values'
    h, w = map_x.shape
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1-x0 < 5 or y1-y0 < 5:
        return None, 'bbox_outside_map'
    # In the current pipeline these are the exact same grid-map coordinates
    # used by cell_crop() and the chosen oriented original.png.
    mx, my = map_x[y0:y1,x0:x1], map_y[y0:y1,x0:x1]
    finite = np.isfinite(mx) & np.isfinite(my)
    if not finite.all():
        return None, 'nonfinite_mapping'
    crop = cv2.remap(image, mx, my, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(255,255,255))
    return crop, None


def initial_observation(book, cell_id):
    bucket = ((book.get('cells') or {}).get(cell_id) or {})
    for item in bucket.get('observations') or []:
        if item.get('path') == 'gridmesh_initial':
            return {'text': item.get('text'), 'model': item.get('model'),
                    'confidence': item.get('confidence')}
    return None


def parallel_v5_observation(book, cell_id):
    """Independent v5 candidate from parallel footer lane (not a primary fallback).

    Keep the recorded provenance and never treat it as ground truth.
    """
    bucket = ((book.get('cells') or {}).get(cell_id) or {})
    observations = [o for o in bucket.get('observations') or []
                    if o.get('path') == 'v5_numeric_candidate'
                    and o.get('model') == 'v5_cyrillic_mobile']
    if not observations:
        return None
    latest = observations[-1]
    return {'text': latest.get('text'), 'model': latest.get('model'),
            'confidence': latest.get('confidence'), 'path': latest.get('path'),
            'count': len(observations)}


def trace_for_cell(trace, cell_id, stage):
    """Read only explicitly attributed REC calls; detector internal calls opaque."""
    events = [e for e in (trace.get('events') or [])
              if e.get('stage') == stage and cell_id in (e.get('cells') or [])]
    if not events:
        return None
    return {'calls': len(events),
            'models': sorted({str(e.get('model_id')) for e in events}),
            'total_call_wall_ms': round(sum(float(e.get('wall_ms') or 0) for e in events),3),
            'failed_calls': sum(not e.get('success', True) for e in events),
            'note': 'Model invocation accounting, not a separate addition to receipt wall time.'}


def get_debug_map(saved_profile):
    entries = (saved_profile.get('debug') or {}).get('ocr_cells') or []
    return {str(e['cell_id']): e for e in entries
            if e.get('mode') == 'cell' and e.get('cell_id')}


def model_call(ocr, image):
    # CnOCR's single-line API skips the text detector. BGR to RGB.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    stamp = time.perf_counter()
    result = ocr.ocr_for_single_line(rgb)
    ms = (time.perf_counter() - stamp) * 1000
    if not isinstance(result, dict):
        raise TypeError('Unexpected CnOCR result type: ' + type(result).__name__)
    text = str(result.get('text', ''))
    score = result.get('score')
    if isinstance(score, np.generic):
        score = float(score)
    return {'candidate_text': text, 'confidence': score,
            'wall_ms': round(ms, 3),
            'valid_digits': bool(re.fullmatch(r'[0-9]+', text))}


def write_html(report, folder):
    def esc(x): return html.escape(str('' if x is None else x))
    rows = report['cells']
    summary = report['summary']
    css = ('body{font:15px/1.5 system-ui,sans-serif;background:#f7f9fc;color:#253448;margin:20px}'
           'main{max-width:1400px;margin:auto}table{border-collapse:collapse;width:100%;background:#fff}'
           'td,th{padding:8px 10px;border-bottom:1px solid #dfe5ee;text-align:left}'
           '.ok{color:#176b38}.bad{color:#a52c33}.muted{color:#697789}')
    out = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
           '<title>GridMesh CnOCR V3 — цифровые ячейки</title>',
           '<style>'+css+'</style><main>',
           '<h1>GridMesh — CnOCR только для цифр (V3)</h1>',
           '<p><b>Чек:</b> '+esc(report['job_id'])+' · <b>Модель:</b> '+esc(report['model'])+'</p>',
           '<p>Сравнение с первичной PP-OCRv6 Tiny, независимым кандидатом PP-OCRv5, '
           'собираемым в параллельной линии, и конечным результатом. '
           'Резервный кандидат V5 не равен fallback при ошибке. '
           'Никакие значения чека не меняются. Совпадение OCR не доказывает точность.</p>',
           '<p>Вызовы: '+esc(summary['recognized'])+' · совпало с итоговыми значениями: '
           +esc(summary['agreements_final'])+' · отличается: '+esc(summary['disagreements_final'])
           +' · ошибок: '+esc(summary['errors'])+'</p>',
           '<p>Кандидаты V5: '+esc(summary['v5_candidate_available'])
           +' · совпало с CnOCR: '+esc(summary['agreements_parallel_v5'])
           +' · отличается: '+esc(summary['disagreements_parallel_v5'])+'</p>',
           '<p><b>Параллельные линии исходного OCR (мс):</b> таблица '
           +esc(summary['production_parallel_lanes'].get('table_wall_ms'))
           +' · итог '+esc(summary['production_parallel_lanes'].get('footer_wall_ms'))
           +' · кандидаты V5 '+esc(summary['production_parallel_lanes'].get('v5_candidate_wall_ms'))
           +' (время перекрывается, не суммировать)</p>',
           '<p><b>Время CnOCR:</b> '+esc(summary['model_call_wall_ms'])
           +' · <b>Время OpenCV:</b> '+esc(summary['preprocess_wall_ms'])+'</p>',
           '<table><thead><tr><th>Ячейка</th><th>Первичный OCR</th>'
           '<th>Модель первичного OCR</th><th>Кандидат V5</th><th>V5 мс (trace)</th>'
           '<th>Текущее значение</th><th>CnOCR</th>'
           '<th>Препроцесс, мс</th><th>Модель, мс (median)</th><th>Валюта</th>'
           '<th>Состояние</th></tr></thead><tbody>']
    for r in rows:
        val = r.get('candidate_text','')
        status = r.get('status','')
        cls = 'ok' if status == 'agreement_final' else ('bad' if status in
              ('disagreement_final','model_error','model_init_error') else 'muted')
        out.append('<tr class="'+cls+'">'+''.join('<td>'+esc(x)+'</td>' for x in
                   (r.get('cell_id'),(r.get('primary_observation') or {}).get('text'),
                    (r.get('primary_observation') or {}).get('model'),
                    (r.get('parallel_v5_observation') or {}).get('text'),
                    (r.get('parallel_v5_trace') or {}).get('total_call_wall_ms'),
                    r.get('baseline_final_text'),val,
                    r.get('preprocess_wall_ms'),r.get('wall_ms'),r.get('currency_method'),
                    status))+'</tr>')
    out.append('</tbody></table>')
    out.append('<p class="muted">Замер отдельного CnOCR-процесса, не параллельный A/B '
               'тест двух OCR-линий. Время DEBUG основной модели включает Python-обвязку '
               'и другое окружение, поэтому прямое ускорение не утверждается.</p></main></html>')
    (folder/'numeric_only_experiment_v3.html').write_text('\n'.join(out), encoding='utf-8')


def run(job, *, warmups=2, repeats=3, limit=0,
        preprocess_only=False, save_crops=False, ocr_factory=None):
    if repeats < 1 or warmups < 0:
        raise ValueError('repeats >=1, warmups >=0 required')
    receipt = json.loads((job/'receipt.json').read_text(encoding='utf-8'))
    if (receipt.get('geometry') or {}).get('rectification') == 'original_cells':
        raise ValueError('Сохранённая карта original_cells не содержит карту OCR-ячеек; нужно заново построить GridMesh и проверить координаты перед сравнением.')
    image = load_bgr(job/'original.png')
    with np.load(job/'rectification_map.npz') as maps:
        map_x = np.asarray(maps['map_x'], dtype=np.float32)
        map_y = np.asarray(maps['map_y'], dtype=np.float32)
    performance = {}
    profile_path = job/'performance_profile.json'
    if profile_path.is_file():
        try:
            performance = json.loads(profile_path.read_text(encoding='utf-8'))
        except (ValueError, TypeError):
            pass
    debug_cells = get_debug_map(performance)
    book = receipt.get('numeric_candidate_book') or {}
    rec_trace = receipt.get('rec_trace') or {}
    parallel_stats = ((receipt.get('geometry') or {}).get('ocr_batch') or {})
    cells = []
    cropfolder = job/'numeric_only_crops_v3'
    if save_crops:
        cropfolder.mkdir(exist_ok=True)
    for row in receipt.get('rows', []):
        row_id = row.get('row_id')
        for field in FIELDS_ORDERED:
            item = (row.get('cells') or {}).get(field)
            if not isinstance(item, dict):
                continue
            key = f'{row_id}:{field}'
            final = str(item.get('text',''))
            primary = initial_observation(book, key)
            parallel_v5 = parallel_v5_observation(book, key)
            v5_trace = trace_for_cell(rec_trace, key, 'v5_numeric_candidate')
            existing = debug_cells.get(key) or {}
            record = {
                'cell_id': key, 'row_kind': row.get('kind'), 'field': field,
                'baseline_final_text': final,
                'baseline_final_model_id': item.get('model_id'),
                'baseline_final_normalized': normalize_numeric(final),
                'primary_observation': primary,
                'primary_normalized': normalize_numeric(primary['text']) if primary else None,
                'parallel_v5_observation': parallel_v5,
                'parallel_v5_normalized': normalize_numeric(parallel_v5['text']) if parallel_v5 else None,
                'parallel_v5_trace': v5_trace,
                'production_debug_wall_ms': existing.get('wall_ms'),
                'production_debug_model': existing.get('model_id'),
                'status': 'pending',
            }
            cells.append(record)
            t0 = time.perf_counter()
            try:
                raw, problem = extract_cell(image, map_x, map_y, item.get('bbox'))
                if problem:
                    record['status'] = problem
                    continue
                source_ocr = item.get('ocr_text') or ''
                cmeta = item.get('currency_symbol')
                currency_hint = bool(cmeta.get('symbol')) if isinstance(cmeta,dict) else bool(cmeta)
                prepared = prepare_integer_cell(raw, field=field,
                                                baseline_text=final,
                                                baseline_ocr_text=source_ocr,
                                                currency_hint=currency_hint)
                record.update(original_hw=prepared.original_hw,
                              candidate_hw=prepared.candidate_hw,
                              currency_method=prepared.method,
                              currency=prepared.currency,
                              crop_metadata=prepared.crop_metadata)
                if prepared.warning: record['warning'] = prepared.warning
                record['status'] = prepared.status
                if prepared.image is not None:
                    record['_image'] = prepared.image
                if save_crops:
                    base = key.replace(':','_')
                    cv2.imencode('.png',raw)[1].tofile(str(cropfolder/(base+'_before.png')))
                    if prepared.image is not None:
                        cv2.imencode('.png',prepared.image)[1].tofile(str(cropfolder/(base+'_after.png')))
            except Exception as exc:
                record['status'] = 'preprocess_error'
                record['error'] = type(exc).__name__ + ': ' + str(exc)
            finally:
                record['preprocess_wall_ms'] = round((time.perf_counter()-t0)*1000,3)
    active = [r for r in cells if '_image' in r]
    if limit > 0:
        for r in active[limit:]:
            r['status'] = 'over_limit'
            del r['_image']
        active = active[:limit]
    init_ms = 0.0
    init_error = None
    if not preprocess_only and active:
        try:
            t0 = time.perf_counter()
            if ocr_factory is None:
                from cnocr import CnOcr
                ocr = CnOcr(rec_model_name=MODEL, det_model_name='naive_det',
                            rec_model_backend='onnx',context='cpu',
                            rec_root=str(ROOT/'data'/'models'/'cnocr'))
            else:
                ocr = ocr_factory()
            init_ms = round((time.perf_counter()-t0)*1000,3)
            for i in range(warmups):
                model_call(ocr, active[i%len(active)]['_image'])
        except Exception as exc:
            init_error = type(exc).__name__ + ': ' + str(exc)
            for r in active:
                r['status'] = 'model_init_error'
                r['error'] = init_error
                r.pop('_image',None)
    if not preprocess_only and init_error is None:
        for r in active:
            img = r.pop('_image')
            results = []
            try:
                for i in range(repeats):
                    results.append(model_call(ocr,img))
                predictions = [v['candidate_text'] for v in results]
                first = results[0]
                r.update(candidate_text=first['candidate_text'],
                         confidence=first['confidence'],
                         valid_digits=first['valid_digits'],
                         wall_ms=round(statistics.median(v['wall_ms'] for v in results),3),
                         wall_ms_repeats=[v['wall_ms'] for v in results],
                         prediction_consistent=len(set(predictions))==1)
                if not r['prediction_consistent']:
                    r['status'] = 'unstable_prediction'
                elif not first['valid_digits']:
                    r['status'] = 'invalid_prediction'
                elif r['baseline_final_normalized'] is None:
                    r['status'] = 'baseline_final_noninteger'
                elif first['candidate_text'] == r['baseline_final_normalized']:
                    r['status'] = 'agreement_final'
                else:
                    r['status'] = 'disagreement_final'
                if r['primary_normalized'] is not None:
                    r['agreement_primary'] = (first['candidate_text'] == r['primary_normalized'])
                if r['parallel_v5_normalized'] is not None:
                    r['agreement_parallel_v5'] = (first['candidate_text'] == r['parallel_v5_normalized'])
            except Exception as exc:
                r['status'] = 'model_error'
                r['error'] = type(exc).__name__ + ': ' + str(exc)
                r['traceback'] = traceback.format_exc(limit=3)
    if preprocess_only:
        for r in active:
            r.pop('_image',None)
            r['status'] = 'ready_for_model'
    times = [v for r in cells for v in r.get('wall_ms_repeats', [])]
    preprocess_times = [r.get('preprocess_wall_ms',0) for r in cells]
    comparisons = [r for r in cells if r['status'] in ('agreement_final','disagreement_final')]
    per_field = {}
    for f in FIELDS_ORDERED:
        fieldrows = [r for r in cells if r['field'] == f]
        per_field[f] = {
            'total':len(fieldrows), 'recognized':sum('candidate_text' in r for r in fieldrows),
            'agreement_final':sum(r['status']=='agreement_final' for r in fieldrows),
            'disagreement_final':sum(r['status']=='disagreement_final' for r in fieldrows),
            'model_ms':percentile_summary([r['wall_ms'] for r in fieldrows if 'wall_ms' in r]),
        }
    summary = {
        'fields_total':len(cells), 'prepared':len(active),
        'recognized':sum('candidate_text' in r for r in cells),
        'agreements_final':sum(r['status']=='agreement_final' for r in cells),
        'disagreements_final':sum(r['status']=='disagreement_final' for r in cells),
        'agreements_primary':sum(r.get('agreement_primary') is True for r in cells),
        'disagreements_primary':sum(r.get('agreement_primary') is False for r in cells),
        'v5_candidate_available':sum(r.get('parallel_v5_observation') is not None for r in cells),
        'agreements_parallel_v5':sum(r.get('agreement_parallel_v5') is True for r in cells),
        'disagreements_parallel_v5':sum(r.get('agreement_parallel_v5') is False for r in cells),
        'errors':sum(r['status'] in ('model_error','model_init_error','preprocess_error') for r in cells),
        'skipped_or_noninteger':sum(r['status'] not in ('agreement_final','disagreement_final') for r in cells),
        'model_call_wall_ms':percentile_summary(times),
        'preprocess_wall_ms':percentile_summary(preprocess_times),
        'production_debug_wall_ms_on_same_cells':percentile_summary([
            r['production_debug_wall_ms'] for r in cells
            if 'candidate_text' in r and isinstance(r.get('production_debug_wall_ms'),(int,float))]),
        'model_init_ms':init_ms,'model_init_error':init_error,
        'per_field':per_field,
        'production_parallel_lanes':{k:parallel_stats.get(k) for k in
            ('mode','parallel_lanes','table_wall_ms','footer_lane_wall_ms',
             'footer_wall_ms','v5_candidate_wall_ms')},
        'production_v5_per_cell_call_ms_from_rec_trace':percentile_summary([
            r['parallel_v5_trace']['total_call_wall_ms'] for r in cells
            if r.get('parallel_v5_trace')]),
    }
    current_head = detect_git_head()
    report = {
        'version':'numeric-only-v3',
        'job_id':receipt.get('job_id',job.name),
        'model':MODEL,
        'expected_project_commit':EXPECTED_HEAD,
        'detected_local_project_commit':current_head,
        'project_version_warning':bool(current_head and current_head!=EXPECTED_HEAD),
        'source_job':str(job),
        'mode':'preprocess_only' if preprocess_only else 'model_comparison',
        'settings':{'warmups':warmups,'repeats':repeats,'save_crops':save_crops,
                    'output_dir':str(output_dir()),'table_mode':configuration('OCR_TABLE_MODE','cell')},
        'summary':summary,'cells':cells,
        'notes':[
            'Read-only. No modifications to App, receipt, printed VAT or footer total.',
            'Table numeric fields only. Printed receipt.total is excluded.',
            'Initial observations use numeric_candidate_book path=gridmesh_initial when available.',
            'Independent v5 candidate path=v5_numeric_candidate may run concurrently with table OCR.',
            'Parallel footer and v5 times are parts of an overlapping lane; NEVER add to table wall time.',
            'V5 candidate model call times are attributed from rec_trace; some events are not instrumented.',
            'Current final OCR values may differ after selective retries or E4 auto apply.',
            'Baseline agreement is not ground-truth accuracy.',
            'Production DEBUG per-cell time includes different model/config/environment.',
            'CnOCR per-call times exclude model initialization, include Python wrapper.',
            'CnOCR and production are not run concurrently; timing cannot prove speedup.',
        ]
    }
    (job/'numeric_only_experiment_v3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    write_html(report,job)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',help='Existing job directory or job ID')
    parser.add_argument('--latest',action='store_true',help='Most recent completed receipt')
    parser.add_argument('--warmups',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--preprocess-only',action='store_true',help='Test OpenCV without loading CnOCR')
    parser.add_argument('--save-crops',action='store_true',help='Save before/after cropped cell images')
    args = parser.parse_args()
    job = find_job(args.job,args.latest)
    print('Job:',job)
    report = run(job,warmups=args.warmups,repeats=args.repeats,
                 limit=args.limit,preprocess_only=args.preprocess_only,save_crops=args.save_crops)
    summary = report['summary']
    print('Prepared:',summary['prepared'],'Recognized:',summary['recognized'])
    print('Agreements:',summary['agreements_final'],'Disagreements:',summary['disagreements_final'])
    print('Report:',job/'numeric_only_experiment_v3.html')
    if summary['model_init_error']:
        print('MODEL INIT ERROR:',summary['model_init_error'])
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
