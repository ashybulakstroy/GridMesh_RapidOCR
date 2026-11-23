#!/usr/bin/env python3
# coding: utf-8
"""Isolated GridMesh RapidOCR 3.0 geometry benchmark.

Purely in-memory monkeypatch of service detector names in a single process;
no production files or environment settings are changed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
COMMIT = 'c80b36cd9f5156f8394f6c26ae90b6e2655beb4a'
DEFAULT_PHOTOS = (
    'App/tests/fixtures/receipt_short_51040.jpg',
    'App/tests/fixtures/receipt_253309.jpg',
    'App/tests/fixtures/receipt_29464.jpg',
    'App/tests/fixtures/receipt_32596.jpg',
    'App/tests/fixtures/receipt_fold_36981.jpg',
    'App/tests/fixtures/receipt_long_grid.png',
)
VARIANTS = ('baseline', 'physical_only', 'scaled_085', 'scaled_070', 'autogrid_only')


def read_image(path: Path):
    # imdecode handles Unicode paths on Windows.
    source = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(source, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or '.png'
    ok, arr = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f'Could not encode {path}')
    arr.tofile(str(path))


def git_head(root: Path) -> str | None:
    try:
        p = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                           capture_output=True, text=True, check=True, timeout=5)
        return p.stdout.strip()
    except Exception:
        return None


def scale_rule(rule: Any, sx: float, sy: float, shape: tuple[int, ...]):
    """Map original Rule or PolylineRule from downscaled to source coordinates.

    Input coordinates use (coordinate - extent/2)/extent, which is scale-invariant;
    only output coefficients, domain and polyline points need scaling.
    """
    from App.geometry.geometry_lab.detection import Rule
    from App.geometry.geometry_lab.structure_diagnostics import PolylineRule
    if getattr(rule, 'axis', None) not in ('v', 'h'):
        raise ValueError('Unknown rule axis')
    if hasattr(rule, 'points'):
        points = np.asarray(rule.points, float).copy()
        points[:, 0] *= sx
        points[:, 1] *= sy
        return PolylineRule(points, rule.axis, shape)
    if not hasattr(rule, 'coeff'):
        raise ValueError('Unsupported geometry rule type')
    across = sx if rule.axis == 'v' else sy
    along = sy if rule.axis == 'v' else sx
    return Rule(rule.axis, np.asarray(rule.coeff, float) * across,
                float(rule.position) * across, float(rule.support),
                (float(rule.domain[0]) * along, float(rule.domain[1]) * along))


def scale_graph(data: dict, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> dict:
    oh, ow = old_shape
    nh, nw = new_shape
    sx, sy = nw / ow, nh / oh
    return {'vertical': [scale_rule(r, sx, sy, new_shape) for r in data['vertical']],
            'horizontal': [scale_rule(r, sx, sy, new_shape) for r in data['horizontal']],
            'shape': new_shape}


def grid_snapshot(result: Any, image_shape: tuple[int, ...]):
    """Observed grid topology/intersections, not OCR or ground truth."""
    from App.geometry.geometry_lab.algorithms import intersection
    source = result.info.get('source_grid')
    if source is None:
        raise RuntimeError('Expected source_grid in original_cells result')
    vs, hs = source['vertical'], source['horizontal']
    nodes = np.asarray([[intersection(v, h, image_shape) for v in vs] for h in hs], dtype=float)
    if not np.isfinite(nodes).all():
        raise RuntimeError('Non-finite grid intersections')
    return {
        'vertical_lines': len(vs),
        'horizontal_lines': len(hs),
        'body_rows': len(hs)-2,
        'intersections': nodes.round(3).tolist(),
        'last_boundary_y_px': round(float(np.median(nodes[-1, :, 1])), 2),
        'first_boundary_y_px': round(float(np.median(nodes[0, :, 1])), 2),
        'table_width_px': round(float(np.median(nodes[:, -1, 0] - nodes[:, 0, 0])), 1),
    }


def compare_grids(reference: dict | None, candidate: dict | None,
                  *, median_limit: float, max_limit: float) -> dict:
    if reference is None:
        return {'status': 'unverified_no_baseline', 'accepted': False,
                'reason': 'Baseline did not find grid; manual inspection required'}
    if candidate is None:
        return {'status': 'failure', 'accepted': False, 'reason': 'Candidate did not find grid'}
    for f in ('vertical_lines', 'horizontal_lines', 'body_rows'):
        if reference[f] != candidate[f]:
            return {'status': 'different_topology', 'accepted': False,
                    'reason': f'{f}: baseline={reference[f]} candidate={candidate[f]}'}
    a, b = np.asarray(reference['intersections']), np.asarray(candidate['intersections'])
    if a.shape != b.shape:
        return {'status': 'different_topology', 'accepted': False,
                'reason': f'Intersection shapes: {a.shape}, {b.shape}'}
    dist = np.linalg.norm(a-b, axis=-1).ravel()
    med = float(np.median(dist))
    maximum = float(np.max(dist))
    p95 = float(np.percentile(dist, 95))
    passed = med <= median_limit and maximum <= max_limit
    return {'status': 'geometry_match' if passed else 'position_mismatch',
            'accepted': bool(passed), 'median_corner_error_px': round(med, 3),
            'p95_corner_error_px': round(p95, 3),
            'max_corner_error_px': round(maximum, 3),
            'reason': 'Matches the baseline geometry; not independently ground-truthed'
                      if passed else 'Observed line displacements exceed acceptance gate'}


@contextlib.contextmanager
def experimental_route(service: Any, mode: str, timings: dict):
    """Replace only imported aliases in service; always restore them.

    NOT concurrency-safe. Program deliberately runs variants serially.
    """
    original_auto = service.detect_for_lab
    original_physical = service.detect_physical_graph
    from App.geometry.geometry_lab.physical_grid_graph_v3 import evidence, ridge_response

    def auto(image):
        if mode in ('physical_only', 'scaled_085', 'scaled_070'):
            timings['auto_skipped'] = True
            return None, {'status': 'SKIPPED_IN_EXPERIMENT', 'reason': mode}
        start = time.perf_counter()
        try:
            return original_auto(image)
        finally:
            timings['auto_ms'] = round((time.perf_counter()-start)*1000, 3)

    def physical(image):
        start = time.perf_counter()
        try:
            if mode == 'autogrid_only':
                timings['physical_skipped'] = True
                return None, {'status': 'SKIPPED_IN_EXPERIMENT'}
            if mode not in ('scaled_085', 'scaled_070'):
                return original_physical(image)
            scale = .85 if mode == 'scaled_085' else .70
            H, W = image.shape[:2]
            sw = max(20, round(W * scale))
            sh = max(20, round(H * scale))
            small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            graph, report = original_physical(small)
            if graph is None:
                return None, {'status': 'REJECTED', 'reason': 'no_scaled_graph',
                              'details': report.get('selection') or report.get('reason')}
            converted = scale_graph(graph, (sh, sw), (H, W))
            # Strict source-image physical-ink gate; no fabricated lines.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            response, dark = ridge_response(gray)
            checked = [evidence(h, converted['vertical'], (H, W), response, dark)
                       for h in converted['horizontal']]
            accepted = bool(checked) and all(e is not None and e['accepted'] for e in checked)
            timings['original_ink_gate_all_edges'] = accepted
            timings['original_ink_gate_passed'] = sum(e is not None and e['accepted'] for e in checked)
            timings['original_ink_gate_total'] = len(checked)
            if not accepted:
                return None, {'status': 'REJECTED', 'reason': 'failed_original_pixel_ink_gate',
                              'accepted_edges': timings['original_ink_gate_passed'],
                              'total_edges': len(checked)}
            return converted, {'status': 'OBSERVED_NOT_VERIFIED',
                               'selection': f'experiment_scaled_{scale}_source_verified',
                               'edges': checked, 'warnings': ['experiment_scaled_input']}
        finally:
            timings['physical_ms'] = round((time.perf_counter()-start)*1000, 3)

    service.detect_for_lab = auto
    service.detect_physical_graph = physical
    try:
        yield
    finally:
        service.detect_for_lab = original_auto
        service.detect_physical_graph = original_physical


def run_once(service: Any, image: np.ndarray, mode: str, save_overlay: Path | None):
    timings = {}
    start = time.perf_counter()
    cpu_start = time.process_time()
    snapshot = None
    geom_info = {}
    error = None
    overlay = None
    try:
        with experimental_route(service, mode, timings):
            rect, overlay, geometry = service.build_grid(image, rectify=False)
        snapshot = grid_snapshot(rect, image.shape)
        geom_info = {
            'method': geometry.get('method'),
            'physical_detection_selection': geometry.get('physical_detection_selection'),
            'rectification': geometry.get('rectification'),
            'source_valid_fraction': geometry.get('safety', {}).get('source_valid_fraction'),
            'folded_fraction': geometry.get('safety', {}).get('folded_fraction'),
            'warnings': geometry.get('warnings') or [],
        }
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    # Stop the clock before writing diagnostic PNG: disk encoding is not geometry.
    wall_ms = round((time.perf_counter()-start)*1000, 3)
    cpu_ms = round((time.process_time()-cpu_start)*1000, 3)
    overlay_error = None
    if save_overlay is not None and overlay is not None:
        try:
            write_image(save_overlay, overlay)
        except Exception as exc:
            overlay_error = f'{type(exc).__name__}: {exc}'
    return {'status': 'ok' if snapshot is not None else 'error',
            'wall_ms': wall_ms, 'cpu_ms': cpu_ms, 'substeps': timings,
            'grid': snapshot, 'geometry': geom_info, 'error': error,
            'overlay_error': overlay_error}


def collect_paths(project: Path, choices: list[str] | None, all_fixtures: bool,
                  extra_dirs: list[str]) -> list[Path]:
    if choices:
        paths = [Path(p) if Path(p).is_absolute() else project / p for p in choices]
    elif all_fixtures:
        paths = sorted([*project.glob('App/tests/fixtures/receipt_*.jpg'),
                        *project.glob('App/tests/fixtures/receipt_*.png'),
                        *project.glob('App/tests/receipt_*.jpg'),
                        *project.glob('App/tests/receipt_*.png')])
    else:
        paths = [project / p for p in DEFAULT_PHOTOS]
    for dirname in extra_dirs:
        directory = Path(dirname)
        if not directory.is_absolute():
            directory = project / directory
        for ext in ('*.jpg','*.jpeg','*.png'):
            paths += list(directory.glob(ext))
    unique = []
    seen = set()
    for x in paths:
        k = str(x.resolve()).lower()
        if k not in seen:
            seen.add(k)
            unique.append(x.resolve())
    return unique


def file_sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            dig.update(b)
    return dig.hexdigest()


def read_truth(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def add_truth(quality: dict, image_name: str, snapshot: dict | None, truth: dict) -> dict:
    expected = truth.get(image_name, {})
    if not expected:
        return {**quality, 'labelled': False, 'truth_gate': 'unlabelled'}
    expected_rows = expected.get('body_rows')
    if expected_rows is None:
        return {**quality, 'labelled': False, 'truth_gate': 'no_row_label'}
    same = snapshot is not None and snapshot['body_rows'] == expected_rows
    return {**quality, 'labelled': True, 'truth_gate': 'pass' if same else 'fail',
            'truth_expected_body_rows': expected_rows,
            'accepted': bool(quality.get('accepted')) and bool(same)}


def summary_stats(items: list[dict], modes: list[str]) -> list[dict]:
    result = []
    for mode in modes:
        subset = [x for x in items if x['mode'] == mode]
        successful = [x for x in subset if x['status'] == 'ok']
        accepted = [x for x in subset if x.get('quality', {}).get('accepted')]
        times = [x['wall_ms'] for x in successful]
        result.append({
            'mode': mode, 'cases': len(subset), 'found': len(successful),
            'quality_pass': len(accepted),
            'median_wall_ms': round(statistics.median(times), 2) if times else None,
            'mean_wall_ms': round(statistics.mean(times), 2) if times else None,
            'p90_wall_ms': round(float(np.percentile(times,90)), 2) if times else None,
            'mean_cpu_ms': round(statistics.mean(x['cpu_ms'] for x in successful),2) if times else None,
            'median_speedup_x': round(statistics.median(x['speedup_x'] for x in accepted
                                        if x.get('speedup_x') is not None), 2)
                                if any(x.get('speedup_x') is not None for x in accepted) else None,
        })
    return result


def write_csv(path: Path, records: list[dict]):
    names = ['photo', 'orientation', 'mode', 'repetition','status','wall_ms','cpu_ms',
             'auto_ms','physical_ms','physical_skipped','auto_skipped',
             'body_rows','horizontal_lines','vertical_lines', 'bottom_y',
             'quality','quality_accepted','speedup_x','corner_median_px','corner_max_px',
             'truth_gate','error']
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=names, delimiter=';')
        writer.writeheader()
        for a in records:
            g=a.get('grid') or {}; t=a.get('substeps') or {}; q=a.get('quality') or {}
            writer.writerow({'photo':a['photo'], 'orientation':a['orientation'],
                'mode':a['mode'], 'repetition':a['repetition'],
                'status':a['status'], 'wall_ms':a['wall_ms'], 'cpu_ms':a['cpu_ms'],
                'auto_ms':t.get('auto_ms'), 'physical_ms':t.get('physical_ms'),
                'physical_skipped':t.get('physical_skipped'), 'auto_skipped':t.get('auto_skipped'),
                'body_rows':g.get('body_rows'), 'horizontal_lines':g.get('horizontal_lines'),
                'vertical_lines':g.get('vertical_lines'), 'bottom_y':g.get('last_boundary_y_px'),
                'quality':q.get('status'), 'quality_accepted':q.get('accepted'),
                'speedup_x': a.get('speedup_x'),
                'corner_median_px':q.get('median_corner_error_px'),
                'corner_max_px':q.get('max_corner_error_px'),
                'truth_gate':q.get('truth_gate'), 'error':a.get('error')})


def report_html(output: Path, data: dict):
    results = data['results']
    groups: dict[tuple[str,str],list[dict]] = {}
    for item in results:
        groups.setdefault((item['photo'], item['orientation']), []).append(item)
    cards = []
    for (photo,orientation), rows in groups.items():
        entries = []
        for row in rows:
            if row['repetition'] != 1: continue
            grid=row.get('grid') or {}
            quality=row.get('quality') or {}
            uri=row.get('overlay_file')
            path = f'<a href="{html.escape(uri,quote=True)}"><img loading="lazy" src="{html.escape(uri,quote=True)}"></a>' if uri else '<em>Нет изображения сетки</em>'
            entries.append(f'''<div class="variant"><h4>{html.escape(row['mode'])} — {row['wall_ms']:.1f} мс</h4>
                <p>Строк тела: {grid.get('body_rows','—')}; вертикалей: {grid.get('vertical_lines','—')};
                совпадение: {html.escape(str(quality.get('status','—')))};
                gate: {html.escape(str(quality.get('truth_gate','—')))}</p>
                {path}<p class="error">{html.escape(row.get('error') or '')}</p></div>''')
        cards.append(f'<section><h3>{html.escape(photo)} | {html.escape(orientation)}</h3><div class="variants">{"".join(entries)}</div></section>')
    tr = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v if v is not None else "—"))}</td>'
              for v in [r['mode'],r['cases'],r['found'],r['quality_pass'],r['median_wall_ms'],r['mean_wall_ms'],r['p90_wall_ms'],r['median_speedup_x']])+'</tr>'
              for r in data['summary'])
    legend = '''Сравнение геометрии с baseline не является независимой разметкой. Вариант с другой сеткой не считается улучшением автоматически.
    Скорость засчитывайте только для варианта, прошедшего gate качества на ВСЕХ проверяемых фотографиях.
    Вычисления выполнены последовательно, без OCR и без изменения production.'''
    page = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Grid Speed V3</title>
    <style>body{{font:15px/1.48 system-ui,sans-serif;max-width:1440px;margin:30px auto;padding:0 20px;background:#f7f9fc;color:#172234}}
    table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border-bottom:1px solid #d9e2ed;text-align:left}}
    section{{background:#fff;margin-top:25px;padding:18px;border-radius:12px;box-shadow:0 1px 5px #ccc}}
    .variants{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}}
    .variant{{border:1px solid #c5ced9;border-radius:8px;padding:10px}}
    img{{width:100%;max-height:380px;object-fit:contain;background:#ebeff3}}
    .error{{color:#a52a2a;overflow-wrap:anywhere}}
    .note{{padding:14px 0}}</style></head><body>
    <h1>Эксперимент: ускорение физической сетки V3</h1>
    <p>GitHub SHA: <code>{html.escape(data['authored_commit'])}</code><br>
    Локальный SHA: <code>{html.escape(str(data['local_commit']))}</code><br>
    Дата запуска: {html.escape(data['date'])}; фото: {data['photos_count']}</p>
    <p class="note">{html.escape(legend)}</p>
    <p>* Коэффициент ускорения считается только для повторов с принятым сравнением геометрии. 1,0× = без ускорения.</p>
    <table><tr><th>Маршрут</th><th>Попытки</th><th>Сетка найдена</th><th>Геометрия принята</th>
    <th>Медиана, мс</th><th>Среднее, мс</th><th>p90, мс</th><th>Медианный выигрыш ×*</th></tr>{tr}</table>
    {''.join(cards)}<p>Подробные измерения: <a href="results.csv">results.csv</a>, <a href="results.json">results.json</a>.</p>
    </body></html>'''
    (output/'summary.html').write_text(page, encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='GridMesh V3 geometry only experiment')
    parser.add_argument('--project', default=os.environ.get('PROJECT_ROOT',r'C:\Work\Projects\Prj_18_OCR_2'))
    parser.add_argument('--images', nargs='*', help='Image filenames; relative to --project or absolute')
    parser.add_argument('--image-dir', action='append', default=[], help='Extra local image directory')
    parser.add_argument('--all-fixtures',action='store_true',help='All receipt_* photos in App/tests')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=['baseline','physical_only','scaled_085','scaled_070'])
    parser.add_argument('--truth',default=str(ROOT/'truth.json'))
    parser.add_argument('--out',default=None)
    parser.add_argument('--median-tolerance-px',type=float,default=3.0)
    parser.add_argument('--max-tolerance-px',type=float,default=8.0)
    parser.add_argument('--no-copy-photos',action='store_true')
    parser.add_argument('--strict-sha',action='store_true')
    args=parser.parse_args(argv)
    if args.repeats<1 or args.warmups<0:
        parser.error('repeats >=1, warmups >=0')
    if 'baseline' not in args.variants:
        parser.error('baseline must be among --variants, for geometry comparison')
    project=Path(args.project).expanduser().resolve()
    if not (project/'App/geometry/service.py').is_file():
        print(f'ERROR: No project code at {project}. Specify --project.',file=sys.stderr)
        return 2
    local_head=git_head(project)
    if local_head and local_head != COMMIT:
        print(f'WARNING: Different local HEAD: {local_head} (experiment written for {COMMIT})')
        if args.strict_sha: return 3
    # Never import the experiment bundle as App; production repo stays first on sys.path.
    sys.path.insert(0,str(project))
    from App.geometry import service
    from App.config import positive_int
    cv2.setNumThreads(positive_int('OCR_OPENCV_THREADS', 2))
    photo_paths=collect_paths(project,args.images,args.all_fixtures,args.image_dir)
    photos=[];missing=[]
    for path in photo_paths:
        if path.is_file():photos.append(path)
        else:missing.append(str(path))
    if missing:
        print('Missing input photos:', *missing,sep='\n  ',file=sys.stderr)
    if not photos:
        print('ERROR: No real input photos found. See README_RU.md.',file=sys.stderr)
        return 4
    if args.out:
        output=Path(args.out).resolve()
    else:
        output=ROOT/'results'/dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=True)
    (output/'overlays').mkdir(exist_ok=True)
    truth=read_truth(Path(args.truth))
    records=[]
    for pic_index, path in enumerate(photos,1):
        image=read_image(path)
        if image is None:
            print(f'WARNING: unreadable {path}',file=sys.stderr)
            continue
        if not args.no_copy_photos:
            target=output/'photos'/f'{pic_index:02d}_{path.name}'
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path,target)
        name=path.name
        photo_hash = file_sha256(path)
        # Actual pipeline tests CW/CCW on landscape input; run both independently.
        orientations = ([('original',image)] if image.shape[0]>=image.shape[1]
                        else [('clockwise',cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)),
                              ('counterclockwise',cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE))])
        for orient, img in orientations:
            print(f'[{pic_index}/{len(photos)}] {path.name} / {orient}: {img.shape[1]}x{img.shape[0]}')
            # Baseline always first (and includes same geometry operations as production).
            modes=['baseline']+[v for v in args.variants if v!='baseline']
            ref=None
            for mode in modes:
                for _ in range(args.warmups):
                    _=run_once(service,img,mode,None)
                for rep in range(1,args.repeats+1):
                    overlay_name=f'{pic_index:02d}_{orient}_{mode}.png'
                    overlay=output/'overlays'/overlay_name if rep==1 else None
                    measurement=run_once(service,img,mode,overlay)
                    if mode=='baseline' and rep==1:
                        ref=measurement.get('grid')
                    q=compare_grids(ref,measurement.get('grid'),
                                    median_limit=args.median_tolerance_px,
                                    max_limit=args.max_tolerance_px)
                    if mode=='baseline' and ref is not None:
                        q.update(status='reference_baseline',accepted=True)
                    q=add_truth(q,name,measurement.get('grid'),truth)
                    if mode=='baseline' and ref is not None and q['truth_gate']=='unlabelled':
                        q['accepted']=True  # baseline is self-reference, not independently verified
                    row={'photo':name,'photo_path':str(path),'photo_sha256':photo_hash,
                         'orientation':orient,'mode':mode,'repetition':rep,
                         'overlay_file':f'overlays/{overlay_name}' if overlay and overlay.exists() else None,
                         'quality':q,**measurement}
                    records.append(row)
                    print(f'  {mode:17} #{rep} {row["wall_ms"]:8.1f} ms; '
                          f'rows={str((row.get("grid") or {}).get("body_rows","—")):>2} '
                          f'quality={q["status"]} {row.get("error") or ""}')
    if not records:
        print('ERROR: no results',file=sys.stderr)
        return 5
    baseline_medians = {}
    for row in records:
        if row['mode'] == 'baseline' and row['status'] == 'ok':
            key=(row['photo_path'],row['orientation'])
            baseline_medians.setdefault(key,[]).append(row['wall_ms'])
    baseline_medians = {key: statistics.median(vals) for key,vals in baseline_medians.items()}
    for row in records:
        base=baseline_medians.get((row['photo_path'],row['orientation']))
        row['speedup_x']=round(base/row['wall_ms'],3) if base is not None and row['wall_ms']>0 else None
    result={'schema':'grid-speed-v3-exp-1','date':dt.datetime.now().isoformat(timespec='seconds'),
            'authored_commit':COMMIT,'local_commit':local_head,'project_path':str(project),
            'photos_count':len(photos),'photos_missing':missing,'repeats':args.repeats,
            'warmups':args.warmups,'quality_rules':{'median_px':args.median_tolerance_px,
                                                   'max_px':args.max_tolerance_px},
            'summary':summary_stats(records,modes),'results':records,
            'notes':['Production code is never written or modified.',
                     'Quality gate compares baseline, not independently annotated ground truth.',
                     'No OCR is run. Footer/VAT cannot be validated by this experiment.',
                     'Variants are executed sequentially, monkeypatch is not thread-safe.',
                     'Results depend on image size, OpenCV build, operating system and CPU.']}
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(output/'results.csv',records)
    report_html(output,result)
    print(f'\nDONE: {output/"summary.html"}')
    print('NOTE: Inspect missing rows and overlay accuracy before claiming a speed improvement.')
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted by user',file=sys.stderr)
        raise SystemExit(130)
