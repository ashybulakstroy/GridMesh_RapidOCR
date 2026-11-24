"""Benchmark2: reconstruct GridMesh NAME crops from an existing OCR job and benchmark REC.
Does not mutate the main project or Benchmark2's 17-sample dataset.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from benchmark2.preprocess import split_name_lines, join_parts
from benchmark2.adapters import make_adapter

DEFAULT_BASE = Path(os.environ.get('BENCHMARK2_BASE_PROJECT', r'C:\Work\Prj_36_OCR2'))
DEFAULT_JOB = '8ca442c659ab434ca9fd6f7166f01cfa'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def export_names(base, job_id, dataset):
    """Recalculate GridMesh without OCR; verify saved corner coordinates. Reconstructed pixels are not guaranteed bit-identical."""
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('Job ID must be 32 lowercase hexadecimal chars')
    directory = (base/'data'/'output'/job_id).resolve()
    receipt_path = directory/'receipt.json'
    original_path = directory/'original.png'
    if not receipt_path.is_file() or not original_path.is_file():
        raise FileNotFoundError(f'Expected receipt.json + original.png in {directory}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    geometry = receipt.get('geometry') or {}
    columns = geometry.get('columns') or {}
    if 'name' not in columns:
        raise ValueError('receipt.json has no geometry.columns.name')
    rectification = geometry.get('rectification')
    if not rectification:
        raise ValueError('receipt.json has no geometry.rectification; cannot recover exact ROI')
    import cv2
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(base))
    from App.geometry.service import build_grid, cell_crop
    # original.png is the lossless, orientation-corrected source used by the worker.
    raw = np.fromfile(str(original_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read '+str(original_path))
    rectify = rectification != 'original_cells'
    start = time.perf_counter()
    result, _, geo = build_grid(image, rectify=rectify)
    grid_seconds = time.perf_counter() - start
    if result.method != rectification:
        raise ValueError(f'Grid type changed: previously {rectification}, rebuilt {result.method}')
    output = []
    crops_dir = dataset/'cells'
    crops_dir.mkdir(parents=True, exist_ok=True)
    for row in receipt.get('rows', []):
        if row.get('kind') != 'product':
            continue
        row_id = int(row['row_id'])
        if row_id < 1 or row_id >= len(result.ys)-1:
            raise ValueError(f'Row {row_id} outside recalculated GridMesh')
        previous = ((row.get('cells') or {}).get('name') or {}).get('source_bbox')
        if not previous:
            raise ValueError(f'Row {row_id} missing saved name source_bbox')
        crop, offset, source_bbox = cell_crop(result, row_id, int(columns['name']))
        old, new = np.asarray(previous, dtype=float), np.asarray(source_bbox, dtype=float)
        if old.shape != (4, 2) or new.shape != (4, 2):
            raise ValueError(f'Row {row_id}: invalid source bbox shapes {old.shape} {new.shape}')
        drift = float(np.max(np.linalg.norm(old-new, axis=1)))
        # Current local GridMesh implementation might not match the older job's version.
        # Never call an approximate crop 'the original OCR input'.
        if drift > 3.0:
            raise ValueError(f'Row {row_id}: GridMesh corner drift {drift:.2f}px > 3px. '
                             'Recompute under the same project revision/settings or capture actual ROI on next run.')
        if crop.size == 0:
            raise ValueError(f'Row {row_id}: empty crop')
        rel = Path('data')/'job_import'/job_id/'cells'/f'name_{row_id:03}.png'
        target = BENCH/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # PIL supports Cyrillic and non-ASCII Windows file paths reliably.
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(target)
        saved_text = str(((row.get('cells') or {}).get('name') or {}).get('text') or '')
        output.append({'id':f'job_{job_id[:8]}_name_{row_id:03}', 'type':'name',
                       'row_id':row_id, 'source_job_id':job_id,
                       'file':rel.as_posix(), 'width':int(crop.shape[1]),
                       'height':int(crop.shape[0]), 'crop_sha256':sha(target),
                       'saved_ocr_text_NOT_ground_truth':saved_text,
                       'source_bbox_drift_px':round(drift,3), 'source_bbox':source_bbox,
                       'crop_source':'recalculated_gridmesh_corner_verified_against_receipt'})
    if not output:
        raise ValueError('No product name rows found')
    settings={'job_id':job_id, 'source_job':str(directory), 'original_png_sha256':sha(original_path),
              'receipt_json_sha256':sha(receipt_path), 'geometry_time_sec_excluded':round(grid_seconds,3),
              'rectification':rectification,'reconstructed_method':geo.get('method'),
              'count':len(output), 'quality_scoring':'DISABLED_NO_VERIFIED_GT',
              'crop_identity':'RECONSTRUCTED_NOT_BYTE_VERIFIED',
              'model_loading_included_in_rec_timing':False, 'inputs':output}
    write_json(dataset/'inputs.json',settings)
    truth_template={str(x['row_id']):None for x in output}
    template=dataset/'ground_truth_TEMPLATE.json'
    if not template.exists():
        write_json(template, truth_template)
    print(f'[IMPORT] {len(output)} name cells; GridMesh rebuild {grid_seconds:.3f}s (excluded from REC)')
    print('[IMPORT] Dataset:', dataset)
    return settings


def make_report(output_dir, data, records, runs, model_ids, metadata):
    rows=[]
    for model_id in model_ids:
        measured=[r for r in records if r['model_id']==model_id and r['pass']>1]
        ok=[r for r in measured if r['status']=='OK']
        vals=[r['ocr_ms'] for r in ok]
        per_pass=[]
        for p in range(2,runs+1):
            subset=[r for r in ok if r['pass']==p]
            if len(subset)==len(data['inputs']):
                per_pass.append(sum(r['ocr_ms'] for r in subset))
        rows.append({'model_id':model_id,'ok':len(ok),'expected':len(data['inputs'])*(runs-1),
                     'median_cell_ms':round(statistics.median(vals),3) if vals else None,
                     'median_all_names_ms':round(statistics.median(per_pass),3) if per_pass else None,
                     'mean_cell_ms':round(statistics.mean(vals),3) if vals else None,
                     'min_cell_ms':round(min(vals),3) if vals else None,
                     'max_cell_ms':round(max(vals),3) if vals else None})
    result={'status':'FINISHED', 'source_job_id':data['job_id'], 'quality_scoring':'DISABLED_NO_VERIFIED_GT',
            'warmup_passes':1, 'measured_passes':runs-1, 'models':rows,
            'geometry_build_sec_excluded':data['geometry_time_sec_excluded'],
            'model_load_ms':metadata}
    write_json(output_dir/'summary.json',result)
    e=lambda v:html.escape(str(v),quote=True)
    h=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Benchmark2 — job rebench</title>',
       '<style>body{font:15px system-ui;max-width:1100px;margin:28px auto;padding:12px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}img{max-width:220px}</style>',
       '<h1>Benchmark2 — те же вырезки рабочего задания</h1>',
       f'<p>Job: <code>{e(data["job_id"])}</code>; ячеек: {len(data["inputs"])}; '+
       f'GridMesh восстанавливался {data["geometry_time_sec_excluded"]} с — это время исключено.</p>',
       '<p><b>Точность не измерялась, вырезки восстановлены повторным GridMesh и НЕ сверены побайтово с исходными OCR ROI.</b> Ранее распознанные названия — НЕ проверенная разметка. '+
       'Первый проход — прогрев, остальные учитываются в статистике. Один OCR-вызов на один физический сегмент.</p>',
       '<h2>Скорость</h2><table><tr><th>Модель</th><th>Медиана ячейки, мс</th><th>Медиана всех названий, мс</th><th>Готово</th></tr>']
    for a in rows:
        h.append(f'<tr><td>{e(a["model_id"])}</td><td>{e(a["median_cell_ms"])}</td><td>{e(a["median_all_names_ms"])}</td><td>{a["ok"]}/{a["expected"]}</td></tr>')
    h.append('</table><h2>Детализация</h2><table><tr><th>Вырезка GridMesh</th><th>Сохранённый OCR (не эталон)</th><th>Новый OCR</th><th>Время, мс</th></tr>')
    for item in data['inputs']:
        local=[r for r in records if r['cell_id']==item['id'] and r['pass']==2]
        h.append('<tr><td>'+e(item['row_id'])+'<img src="../../../../'+e(item['file'])+'"></td><td>'+e(item['saved_ocr_text_NOT_ground_truth'])+'</td><td>'+e(' | '.join(f'{r["model_id"]}: {r["text"]}' for r in local))+'</td><td>'+e(' | '.join(f'{r["ocr_ms"]:.2f}' for r in local))+'</td></tr>')
    h.append('</table></html>')
    (output_dir/'report.html').write_text(''.join(h),encoding='utf-8')
    return result


def benchmark(data, dataset, models, runs):
    if runs<2:
        raise ValueError('--runs must be >=2 (first is warmup)')
    specs={s['id']:s for s in json.loads((BENCH/'config/models.json').read_text(encoding='utf-8'))}
    if any(x not in specs for x in models):
        raise ValueError('Unknown model id: '+str(set(models)-set(specs)))
    if any(not specs[x].get('cyrillic') for x in models):
        raise ValueError('Only Cyrillic-capable models are allowed for product names')
    from PIL import Image
    from benchmark2.preprocess import precompute
    inputs=data['inputs']
    for x in inputs:
        if sha(BENCH/x['file'])!=x['crop_sha256']:
            raise ValueError('Crop SHA mismatch: '+x['id'])
    # Use exactly the same preprocessing as original Benchmark2.
    processed=precompute(inputs,BENCH)
    output_dir=BENCH/'results'/'job_rebench'/data['job_id']/dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir.mkdir(parents=True, exist_ok=False)
    records=[];initialization=[]
    try:
        for model_id in models:
            spec=specs[model_id]
            print(f'[MODEL] {model_id}',flush=True)
            st=time.perf_counter_ns()
            try:
                adapter=make_adapter(spec,BENCH,Path(data['source_job']).parents[2])
                load_ms=(time.perf_counter_ns()-st)/1e6
                initialization.append({'id':model_id,'load_ms':round(load_ms,3),'status':'OK',
                                       'parameters':getattr(adapter,'effective_parameters',{})})
            except Exception as ex:
                initialization.append({'id':model_id,'load_ms':round((time.perf_counter_ns()-st)/1e6,3),
                                       'status':'ERROR','error':str(ex)})
                print(f'[ERROR MODEL] {model_id}: {ex}',flush=True)
                continue
            try:
                for p in range(1,runs+1):
                    for item in inputs:
                        start=time.perf_counter_ns();parts=[];confs=[]
                        row={'model_id':model_id,'pass':p,'phase':'warmup' if p==1 else 'measured',
                             'cell_id':item['id'],'row_id':item['row_id'],'status':'ERROR',
                             'text':'','confidence':None,'ocr_ms':None,'error':None}
                        try:
                            for im in processed[item['id']]:
                                t,c=adapter.recognize(im)
                                parts.append(t)
                                if c is not None:confs.append(c)
                            row['text']=join_parts(parts)
                            row['confidence']=round(statistics.mean(confs),6) if confs else None
                            row['status']='OK'
                        except Exception as exc:
                            row['error']=f'{type(exc).__name__}: {exc}'
                        row['ocr_ms']=round((time.perf_counter_ns()-start)/1e6,3)
                        records.append(row)
                        print(f' {p}/{runs} row={item["row_id"]}: {row["ocr_ms"]:.2f}ms {row["status"]}',flush=True)
            finally:
                adapter.close()
    finally:
        write_json(output_dir/'metadata.json',{'source_job':data['source_job'],'model_initialization':initialization,
                                               'dataset':str(dataset), 'runs':runs,
                                               'scope':'REC only, GridMesh and loading excluded'})
        with (output_dir/'raw.csv').open('w',encoding='utf-8-sig',newline='') as f:
            columns=['model_id','pass','phase','cell_id','row_id','status','text','confidence','ocr_ms','error']
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(records)
        write_json(output_dir/'raw.json',records)
        result=make_report(output_dir,data,records,runs,models,initialization)
        print('[REPORT]',output_dir/'report.html',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description='Rebench a working OCR job on its own GridMesh name crops')
    ap.add_argument('--job-id',default=DEFAULT_JOB)
    ap.add_argument('--base-project',type=Path,default=DEFAULT_BASE)
    ap.add_argument('--models',default='v6_tiny_cyrillic_optional',help='Comma-separated Cyrillic model ids')
    ap.add_argument('--runs',type=int,default=3,help='1 warmup plus measured passes; default 3')
    ap.add_argument('--dry-run',action='store_true',help='Export and verify crops, no model inference')
    args=ap.parse_args()
    base=args.base_project.resolve()
    if not base.is_dir():raise FileNotFoundError('Main project not found: '+str(base))
    dataset=BENCH/'data'/'job_import'/args.job_id
    data=export_names(base,args.job_id,dataset)
    if args.dry_run:
        print('[DRY RUN] No OCR calls. Review images:',dataset/'cells')
        return
    benchmark(data,dataset,[x.strip() for x in args.models.split(',') if x.strip()],args.runs)

if __name__=='__main__':
    main()
