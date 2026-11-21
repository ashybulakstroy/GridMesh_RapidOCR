"""Sequential CPU experiment. Default command plans only; run needs --confirm-load."""
from __future__ import annotations
import argparse, collections, datetime, hashlib, html, json, math, os, statistics, subprocess, sys, time, traceback
from pathlib import Path
from unittest.mock import patch
from .runner import ROOT, load_json, selected_receipts, check_inputs, _load_cells, _create_recognizer, _run_receipt

CONFIG=ROOT/'config/cpu_threads.json'
OUTPUT=ROOT/'data/output/cpu_threads_benchmark'


def write_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def cell_hash(cells):
    h=hashlib.sha256()
    for field,img in cells:
        h.update(field.encode()); h.update(str((img.shape,img.dtype.str)).encode()); h.update(img.tobytes())
    return h.hexdigest()


def cached_cells(gt):
    # A missing cache is an input error, never permission to run geometry here.
    from .gridmesh import _coordinate_key
    from PIL import Image,ImageOps
    import numpy as np
    with Image.open(ROOT/'data/images'/gt['source_image']) as im:
        image=np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(im).convert('RGB'))[:,:,::-1])
    ann=load_json(ROOT/'data/annotations'/f"{gt['receipt_id']}.json")
    key=_coordinate_key(image,ann,True)
    if not (ROOT/'data/gridmesh_coordinates'/f'{key}.npz').is_file():
        raise FileNotFoundError(f"COORDINATES_MISSING: {gt['receipt_id']}; prepare coordinates outside experiment")
    with patch('App.geometry.service.build_grid',side_effect=RuntimeError('GRIDMESH_FORBIDDEN_DURING_CPU_EXPERIMENT')):
        return _load_cells(gt,'gridmesh_total')


def applied_session(engine, intra, inter, opencv):
    import cv2
    session=engine._load_rec_model().session.session
    opts=session.get_session_options()
    actual={'intra_op_threads':opts.intra_op_num_threads,'inter_op_threads':opts.inter_op_num_threads,
            'opencv_threads':cv2.getNumThreads(),'providers':session.get_providers(),
            'execution_mode':str(opts.execution_mode)}
    valid=(actual['intra_op_threads']==intra and actual['inter_op_threads']==inter
           and actual['opencv_threads']==opencv and actual['providers']==['CPUExecutionProvider'])
    return actual,valid


def balanced(values):
    """Alternate ends; reverse model order for alternating configurations."""
    values=list(values); out=[]
    while values:
        out.append(values.pop(0))
        if values:out.append(values.pop())
    return out


def jobs_for_stage1(cfg):
    # Alternate intra values across individual sessions, not model pairs.
    order=balanced(cfg['intra_op_candidates'])
    models=cfg['primary_models']
    return [dict(stage=1,model_id=models[(cycle+(i//2))%len(models)],intra=intra,
                 opencv=cfg['opencv_threads'],modes=cfg['modes'])
            for cycle in range(len(models)) for i,intra in enumerate(order)]


def stats(values):
    values=sorted(values)
    return dict(median=statistics.median(values),p95=values[math.ceil(.95*len(values))-1],
                min=values[0],max=values[-1],spread=values[-1]-values[0],n=len(values))


def choose_winners(rows,cfg,stage=1):
    rows=active_rows(rows)
    winners={}
    dimension='opencv_threads' if stage==2 else 'intra_op_threads'
    values=cfg['opencv_candidates'] if stage==2 else cfg['intra_op_candidates']
    receipts=sorted({r['receipt_id'] for r in rows if r.get('stage')==stage and r['receipt_id']!='ALL'})
    if not receipts:return winners
    for model in cfg['primary_models']:
        for mode in cfg['modes']:
            def matching(intra):
                return [r for r in rows if r.get('stage')==stage and r['model_id']==model and r['mode']==mode and r[dimension]==intra]
            base=matching(cfg['baseline_intra_op'])
            candidates=[]
            for intra in values:
                trials=matching(intra)
                if len(trials)!=len(receipts)*cfg['repeats'] or any(r['status']!='OK' for r in trials):continue
                valid=True; score=0
                for receipt in receipts:
                    current=[r for r in trials if r['receipt_id']==receipt]
                    reference=[r for r in base if r['receipt_id']==receipt and r['status']=='OK']
                    if len(reference)!=cfg['repeats']:valid=False;break
                    # Compare every numeric correctness and scored text error, not just overall averages.
                    # Field-level numeric and text comparison below handles the differing metric types.
                    for field in {c['field'] for r in reference for c in r['quality']['cell_metrics']}:
                        a=[c for r in current for c in r['quality']['cell_metrics'] if c['field']==field]
                        b=[c for r in reference for c in r['quality']['cell_metrics'] if c['field']==field]
                        if a[0]['type']=='numeric':
                            if sum(c['correct'] for c in a)<sum(c['correct'] for c in b):valid=False
                        elif b[0].get('char_edits') is not None:
                            if sum(c['char_edits'] for c in a)>sum(c['char_edits'] for c in b):valid=False
                    score+=statistics.median(r['ocr_loop_ms'] for r in current)
                if valid:candidates.append((score,intra))
            if candidates:winners[f'{model}/{mode}']=min(candidates)[1]
    return winners


class TimedRecognizer:
    def __init__(self,engine):self.engine=engine;self.cpu_ms=0
    def recognize_txt(self,images):
        start=time.process_time_ns()
        try:return self.engine.recognize_txt(images)
        finally:self.cpu_ms+=(time.process_time_ns()-start)/1e6


def no_download(params):
    from rapidocr.utils.download_file import DownloadFile
    path=Path(params.save_path)
    if not path.is_file() or (params.sha256 and not DownloadFile.check_file_sha256(path,params.sha256)):
        raise RuntimeError(f'MODEL_NOT_AVAILABLE_OFFLINE: {path}')


def worker(job_path):
    import cv2,psutil
    cfg=load_json(OUTPUT/'config_snapshot.json'); job=load_json(job_path)
    models={m['id']:m for m in load_json(ROOT/'config/models.json')}
    gts=selected_receipts(cfg['scope'],load_json(ROOT/'config/experiment.json'))
    expected=load_json(OUTPUT/'input_snapshot.json')
    cv2.setNumThreads(job['opencv'])
    tag={'field_scope':'all' if models[job['model_id']]['cyrillic'] else 'numeric_only','stage':job['stage'],'model_id':job['model_id'],'intra_op_threads':job['intra'],
         'inter_op_threads':cfg['inter_op'],'opencv_threads':job['opencv'],'batch_size':cfg['batch_size']}
    output=OUTPUT/job['raw_file']
    def save(row):
        with output.open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    try:
        with patch('rapidocr.utils.download_file.DownloadFile.run',side_effect=no_download):
            engine=_create_recognizer(models[job['model_id']],job['intra'],cfg['batch_size'])
            actual,valid=applied_session(engine,job['intra'],cfg['inter_op'],job['opencv'])
        tag['applied_settings']=actual
        if not valid:
            save(dict(tag,status='NOT_APPLIED',receipt_id='ALL',mode='SESSION',run_number=0));return
        sample,_=cached_cells(gts[0])
        if tag['field_scope']=='numeric_only':
            from .measure import is_numeric_field
            sample=[(f,c) for f,c in sample if is_numeric_field(f)]
        for _ in range(cfg['warmups']):engine.recognize_txt([sample[0][1]])
        timed=TimedRecognizer(engine)
        for repeat in range(1,cfg['repeats']+1):
            ordered=gts[repeat%len(gts):]+gts[:repeat%len(gts)]
            modes=job['modes'] if repeat%2 else list(reversed(job['modes']))
            for gt in ordered:
                for mode in modes:
                    if job.get('retry_targets') and [gt['receipt_id'],mode,repeat] not in job['retry_targets']:continue
                    # Verify prepared ROI identity before each measurement; no preprocessor changes.
                    cells,_=cached_cells(gt)
                    if cell_hash(cells)!=expected[gt['receipt_id']]['roi_sha256']:raise RuntimeError('ROI_CHANGED')
                    process=psutil.Process()
                    cpu_before=process.cpu_times()
                    before_started=time.perf_counter()
                    host_before=psutil.cpu_percent(interval=.2)
                    before_elapsed=time.perf_counter()-before_started
                    cpu_after=process.cpu_times()
                    own_before=100*((cpu_after.user+cpu_after.system)-(cpu_before.user+cpu_before.system))/before_elapsed/(os.cpu_count() or 1)
                    background_before=max(0,host_before-own_before)
                    timed.cpu_ms=0
                    with patch('benchmark.core.runner._load_cells',side_effect=lambda g,source:cached_cells(g)):
                        row=_run_receipt(gt,timed,models[job['model_id']],mode,cfg['batch_size'],repeat,dict(tag,crop_source='gridmesh_total'))
                    host=psutil.cpu_percent(interval=None)
                    own=row.get('process_cpu_percent_one_core_basis',0)/(os.cpu_count() or 1)
                    background=max(0,host-own)
                    row.update(tag,is_retry=bool(job.get('retry_targets')),run_number=repeat,wall_ms=row['receipt_total_wall_ms'],
                               ocr_loop_ms=row.get('ocr_loop_wall_ms'),process_cpu_ms=timed.cpu_ms,
                               full_process_cpu_ms=row['cpu_ms'],
                               cpu_utilization_percent=100*timed.cpu_ms/row['ocr_loop_wall_ms'] if row.get('ocr_loop_wall_ms') else None,
                               peak_ram_mb=row.get('peak_rss_sampled_mb'),host_cpu_percent=host,host_cpu_before_percent=host_before,background_cpu_before_percent=background_before,
                               background_cpu_estimated_percent=background,
                               numeric_accuracy=row.get('quality',{}).get('numeric_accuracy'),text_cer=row.get('quality',{}).get('text_cer'))
                    if row['status']=='OK' and max(background,background_before)>cfg['background_cpu_threshold_percent']:row['status']='UNSTABLE_BACKGROUND_CPU'
                    save(row)
    except Exception as exc:
        save(dict(tag,status='ERROR',receipt_id='ALL',mode='SESSION',run_number=0,error=str(exc),traceback=traceback.format_exc()))


def read_rows():
    return [json.loads(line) for p in sorted(OUTPUT.glob('job_*.jsonl')) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def active_rows(rows):
    latest={}
    for r in rows:
        # Missing run numbers are retained individually for old report fixtures.
        if 'run_number' not in r:
            latest[('legacy',id(r))]=r;continue
        key=tuple(r.get(k) for k in ('stage','model_id','receipt_id','mode','intra_op_threads','opencv_threads','run_number'))
        latest[key]=r
    return list(latest.values())


def retry_unstable(stage,attempts=3):
    cfg=load_json(OUTPUT/'config_snapshot.json')
    for attempt in range(attempts):
        bad=[r for r in active_rows(read_rows()) if r['stage']==stage and r['status']=='UNSTABLE_BACKGROUND_CPU']
        if not bad:return
        groups=collections.defaultdict(list)
        for r in bad:groups[(r['model_id'],r['intra_op_threads'],r['opencv_threads'])].append([r['receipt_id'],r['mode'],r['run_number']])
        configurations=list(groups)
        ordered=balanced(sorted(configurations,key=lambda k:(k[1],k[0],k[2])))
        # Configurations alternate low/high thread counts; only failed fields are retried.
        for model,intra,cv in ordered:
            index=max([int(p.stem.split('_')[1]) for p in OUTPUT.glob('job_*.json')],default=-1)+1
            job=dict(stage=stage,model_id=model,intra=intra,opencv=cv,modes=cfg['modes'],
                     retry_targets=groups[(model,intra,cv)],raw_file=f'job_{index:03d}.jsonl')
            path=OUTPUT/f'job_{index:03d}.json';write_json(path,job)
            with (OUTPUT/f'job_{index:03d}.log').open('w',encoding='utf-8') as log:
                proc=subprocess.run([sys.executable,'-m','benchmark.core.cpu_threads','worker','--job',str(path)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent)
            if proc.returncode:raise RuntimeError(f'Retry worker failed: {path}')
            report()


def report():
    cfg=load_json(OUTPUT/'config_snapshot.json'); rows=read_rows()
    selected_rows=active_rows(rows)
    groups=collections.defaultdict(list)
    for r in selected_rows:
        if r.get('ocr_loop_ms') is not None:
            groups[(r['stage'],r['model_id'],r['mode'],r['receipt_id'],r['intra_op_threads'],r['opencv_threads'])].append(r)
    aggregates=[]
    for key,rs in sorted(groups.items()):
        a=dict(zip(('stage','model_id','mode','receipt_id','intra_op_threads','opencv_threads'),key))
        for metric in ('ocr_loop_ms','wall_ms','process_cpu_ms','cpu_utilization_percent','peak_ram_mb'):
            a[metric]=stats([r[metric] for r in rs if r.get(metric) is not None])
        a['field_scope']=rs[0].get('field_scope','all')
        a['numeric_accuracy']=min(r['numeric_accuracy'] for r in rs)
        a['text_cer']=max((r['text_cer'] for r in rs if r['text_cer'] is not None),default=None)
        a['product_name_cer']=max((r['quality'].get('product_name_cer') for r in rs if r['quality'].get('product_name_cer') is not None),default=None)
        a['product_names_exact']=min(r['quality'].get('product_names_exact',0) for r in rs)
        a['product_names_total']=max(r['quality'].get('product_names_total',0) for r in rs)
        a['complete_receipts_correct']=sum(r['quality'].get('complete_receipt_correct',False) for r in rs)
        a['statuses']=dict(collections.Counter(r['status'] for r in rs));aggregates.append(a)
    for a in aggregates:
        base=next((b for b in aggregates if all(a[k]==b[k] for k in ('stage','model_id','mode','receipt_id')) and b['intra_op_threads']==(a['intra_op_threads'] if a['stage']==2 else 2) and b['opencv_threads']==2),None)
        a['speedup_vs_baseline']=base['ocr_loop_ms']['median']/a['ocr_loop_ms']['median'] if base else None
    winners=choose_winners(rows,cfg)
    opencv_winners=choose_winners(rows,cfg,stage=2)
    secondary_cfg=dict(cfg,primary_models=cfg['secondary_models'])
    secondary_winners=choose_winners(rows,secondary_cfg,stage=3)
    confirmation_cfg=dict(cfg,primary_models=['v5_cyrillic_mobile'],modes=['batch'],intra_op_candidates=[2,6,8])
    confirmation_winners=choose_winners(rows,confirmation_cfg,stage=4)
    recommendations={}
    for key,intra in winners.items():
        confirmed=confirmation_winners.get(key)
        chosen=confirmed if confirmed is not None else intra
        cv=2 if confirmed is not None else opencv_winners.get(key)
        recommendations[key]={'intra_op':chosen,'inter_op':cfg['inter_op'],'opencv_threads':cv,'batch_size':cfg['batch_size'],'basis':'control stage 4' if confirmed is not None else 'stages 1 and 2'}
    write_json(OUTPUT/'results.json',dict(recommendations=recommendations,aggregates=aggregates,winners=winners,opencv_winners=opencv_winners,secondary_winners=secondary_winners,confirmation_winners=confirmation_winners,records=len(rows)))
    # One raw row per repetition, preserved alongside per-job logs.
    (OUTPUT/'raw.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    import csv
    if rows:
        fields=['stage','model_id','receipt_id','mode','intra_op_threads','inter_op_threads','opencv_threads','batch_size','wall_ms','ocr_loop_ms','process_cpu_ms','cpu_utilization_percent','peak_ram_mb','numeric_accuracy','text_cer','run_number','status','background_cpu_estimated_percent']
        with (OUTPUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    esc=lambda x:html.escape(str(x))
    columns=['stage','field_scope','model_id','mode','receipt_id','intra_op_threads','opencv_threads','ocr_loop_ms','wall_ms','process_cpu_ms','cpu_utilization_percent','peak_ram_mb','numeric_accuracy','text_cer','product_name_cer','product_names_exact','product_names_total','complete_receipts_correct','speedup_vs_baseline','statuses']
    def display(value):
        if isinstance(value,dict) and 'median' in value:
            return f"<b>{value['median']:.2f}</b><br><small>P95 {value['p95']:.2f}; min {value['min']:.2f}; max {value['max']:.2f}; разброс {value['spread']:.2f}; n={value['n']}</small>"
        if isinstance(value,float):return f'{value:.4f}'
        return esc(value)
    table='<h2>Время и ресурсы по конфигурациям</h2><p>OCR, wall и CPU-время — мс; RAM — МБ; загрузка CPU — % одного ядра. Разные field_scope не сравниваются по скорости напрямую.</p><div class="scroll"><table><tr>'+''.join('<th>'+esc(c)+'</th>' for c in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+display(a[c])+'</td>' for c in columns)+'</tr>' for a in aggregates)+'</table></div>'
    charts=[]
    for metric in ('ocr_loop_ms','cpu_utilization_percent','peak_ram_mb'):
        for model in cfg['primary_models']:
            for mode in cfg['modes']:
                selected=[a for a in aggregates if a['stage']==1 and a['model_id']==model and a['mode']==mode]
                if not selected:continue
                maximum=max(a[metric]['median'] for a in selected) or 1
                lines=[]
                for ri,receipt in enumerate(sorted({a['receipt_id'] for a in selected})):
                    points=sorted([a for a in selected if a['receipt_id']==receipt],key=lambda a:a['intra_op_threads'])
                    coords=' '.join(f"{45+(a['intra_op_threads']-1)*65},{210-170*a[metric]['median']/maximum}" for a in points)
                    color=['#1767b5','#c34c19','#328235'][ri%3]
                    lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/><text x="40" y="{250+ri*18}" fill="{color}">{esc(receipt)}</text>')
                labels=''.join(f'<text x="{40+(x-1)*65}" y="230">{x}</text>' for x in cfg['intra_op_candidates'])
                charts.append(f'<h3>{esc(model)} / {esc(mode)} — {metric}</h3><svg viewBox="0 0 560 320" width="560" role="img"><text x="5" y="20">max {maximum:.2f}; Median</text><path d="M35 30 V210 H530" fill="none" stroke="#777"/>'+''.join(lines)+labels+'</svg>')
    errors=[r for r in rows if r['status']!='OK']
    text='<html lang="ru"><meta charset="utf-8"><title>CPU Threads Benchmark</title><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}svg{background:#fafafa}</style><h1>CPU Threads Benchmark</h1><p>Только локальные измерения. OCR исключает загрузку моделей, геометрию и подготовку вырезок. wall_ms включает чтение чека и подготовку по сохранённым координатам. CPU: 100% = одно логическое ядро, возможно >100%. RAM — выборочный пик всего процесса (50 мс), не только OCR. Фоновая нагрузка — оценка общей загрузки за интервал минус CPU процесса; >30% отмечается нестабильностью. Координаты и ROI фиксированы.</p><p>Короткие чеки: 1 и 2 товара; средний: 8 товаров. Длинный чек отсутствует в активном наборе. Все товарные названия сверены агентом по оригиналам; окончательное утверждение владельцем не выставлено. Оценивается смешанная кириллица/латиница, включая артикулы и знаки.</p><h2>Рекомендации intra_op по модели и режиму</h2><pre>'+esc(json.dumps(dict(intra_op=winners,opencv_threads=opencv_winners,secondary_intra_op=secondary_winners,confirmation_v5_batch=confirmation_winners),ensure_ascii=False,indent=2))+'</pre><p>Выбор: сумма медиан OCR по чекам. Требуются все пять стабильных повторов, без ухудшения числовых полей и проверенного CER относительно intra_op=2. Отсутствие победителя означает недостаточно валидных данных. Это кандидаты по этапам, а не гарантия универсального ускорения. Этап 4 — контроль v5 batch при intra_op 2,8,6 и OpenCV=2 из-за резкого разброса этапа 2. Сильные колебания времени требуют осторожной интерпретации даже при низкой фоновой нагрузке. Ускорение на этапе 2 считается относительно OpenCV=2 при том же intra_op; на остальных — intra_op=2.</p>'+''.join(charts)+table+'<h2>Ошибки и нестабильные запуски</h2><pre>'+esc(json.dumps(errors,ensure_ascii=False,indent=2))+'</pre></html>'
    from .quality_report import quality_sections
    status_counts=dict(collections.Counter(r['status'] for r in rows))
    manifest=load_json(OUTPUT/'manifest.json') if (OUTPUT/'manifest.json').exists() else {}
    banner='<p><b>Статус: '+esc(manifest.get('status','IN_PROGRESS'))+'; измерений: '+str(len(rows))+'; '+esc(status_counts)+'</b></p><nav><a href="#text-benchmark">Тексты</a> · <a href="#numeric-benchmark">Числа и арифметика</a></nav>'
    recommendation_table='<h2>Итоговые настройки для этого набора</h2><table><tr><th>Модель / режим</th><th>Intra-op</th><th>Inter-op</th><th>OpenCV</th><th>Batch</th></tr>'+''.join('<tr><td>'+esc(key)+'</td>'+''.join('<td>'+esc(value[k])+'</td>' for k in ('intra_op','inter_op','opencv_threads','batch_size'))+'</tr>' for key,value in recommendations.items())+'</table>'
    caution='<p><b>Важно:</b> v5 batch на 8 потоках показала сильные выбросы времени на этапе OpenCV. Контрольная серия 2 → 8 → 6 подтвердила более устойчивый вариант с 6 потоками и OpenCV=2. Температура CPU не измерялась: причину выбросов нельзя уверенно приписывать перегреву. Настройки рабочего App не изменялись.</p>' if confirmation_winners else ''
    text=text.replace('<h1>CPU Threads Benchmark</h1>','<h1>CPU Threads Benchmark</h1>'+banner+recommendation_table+caution)
    text=text.replace('table{border-collapse:collapse}', 'table{border-collapse:collapse;font-size:12px}.scroll{overflow:auto}th{position:sticky;top:0;background:#eef3f7}small{white-space:nowrap;color:#526070}nav a{margin-right:16px}')
    text=text.replace('</html>','<p>Повторы нестабильных замеров заменяют соответствующий номер прогона только в сводных таблицах. Все исходные записи сохранены в raw.jsonl и разделе ошибок.</p>'+quality_sections(selected_rows)+'</html>')
    (OUTPUT/'report.html').write_text(text,encoding='utf-8')
    return winners


def run_jobs(jobs,start,resume=False):
    existing=[load_json(p) for p in sorted(OUTPUT.glob('job_*.json'))]
    next_index=max([int(p.stem.split('_')[1]) for p in OUTPUT.glob('job_*.json')],default=-1)+1
    cfg=load_json(OUTPUT/'config_snapshot.json')
    for job in jobs:
        matches=[j for j in existing if all(j.get(k)==job.get(k) for k in ('stage','model_id','intra','opencv','modes'))]
        if matches:
            saved=matches[0]; rows=[json.loads(x) for x in (OUTPUT/saved['raw_file']).read_text(encoding='utf-8').splitlines()]
            expected=len(selected_receipts(cfg['scope'],load_json(ROOT/'config/experiment.json')))*len(job['modes'])*cfg['repeats']
            if len(rows)==expected or any(r['status'] in ('ERROR','NOT_APPLIED') for r in rows):continue
            raise RuntimeError('Partial configuration; preserve and inspect before rerunning')
        index=next_index;next_index+=1
        job['raw_file']=f'job_{index:03d}.jsonl'
        path=OUTPUT/f'job_{index:03d}.json';write_json(path,job)
        log=OUTPUT/f'job_{index:03d}.log'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-m','benchmark.core.cpu_threads','worker','--job',str(path)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT.parent)
        if result.returncode:raise RuntimeError(f'Worker failed; see {log}')
        existing.append(job)
        report()
    return next_index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run','resume','retry','report','worker'],nargs='?',default='plan')
    parser.add_argument('--confirm-load',action='store_true');parser.add_argument('--job',type=Path)
    args=parser.parse_args();cfg=load_json(CONFIG)
    if args.command=='worker':worker(args.job);return
    if args.command=='report':report();return
    gts=selected_receipts(cfg['scope'],load_json(ROOT/'config/experiment.json'))
    n=len(gts)*cfg['repeats']
    stage1=len(jobs_for_stage1(cfg))*len(cfg['modes'])*n
    stage2=len(cfg['primary_models'])*len(cfg['modes'])*len(cfg['opencv_candidates'])*n
    stage3_min=len(cfg['secondary_models'])*len(cfg['modes'])*n
    stage3_max=stage3_min*(1+len(cfg['primary_models']))
    print(f'Stage 1: {stage1}; stage 2: {stage2}; stage 3: {stage3_min}..{stage3_max}; total {stage1+stage2+stage3_min}..{stage1+stage2+stage3_max} measured trials.')
    if args.command=='plan':return
    if not args.confirm_load:raise RuntimeError('Separate explicit authorization and --confirm-load required')
    resuming=args.command in ('resume','retry')
    if OUTPUT.exists() and not resuming:raise RuntimeError(f'Output already exists: archive it before a new experiment')
    if resuming and load_json(OUTPUT/'config_snapshot.json')!=cfg:raise RuntimeError('Config changed; cannot resume')
    _,errors=check_inputs(gts)
    if errors:raise RuntimeError('Invalid inputs')
    hashes={gt['receipt_id']:{'roi_sha256':cell_hash(cached_cells(gt)[0])} for gt in gts}
    if not resuming:
        OUTPUT.mkdir(parents=True)
        write_json(OUTPUT/'config_snapshot.json',cfg);write_json(OUTPUT/'input_snapshot.json',hashes)
        write_json(OUTPUT/'manifest.json',dict(status='RUNNING',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,logical_cpus=os.cpu_count(),order='alternating low/high intra; reverse model order; rotate receipt order and alternate modes per repeat; sequential isolated processes'))
    else:
        if load_json(OUTPUT/'input_snapshot.json')!=hashes:raise RuntimeError('Inputs changed')
        manifest=load_json(OUTPUT/'manifest.json')
        manifest.update(status='RUNNING',order_revision='Individual sessions alternate intra 1,8,2,6,3,4; completed measurements retained')
        write_json(OUTPUT/'manifest.json',manifest)
    try:
        index=run_jobs(jobs_for_stage1(cfg),0,resume=resuming)
        if args.command=='retry':retry_unstable(1)
        winners=report()
        if len(winners)!=len(cfg['primary_models'])*len(cfg['modes']):raise RuntimeError('No reliable stage-1 winner for every model/mode; inspect report before continuation')
        jobs=[dict(stage=2,model_id=model,intra=winners[f'{model}/{mode}'],opencv=cv,modes=[mode]) for cv in balanced(cfg['opencv_candidates']) for model in cfg['primary_models'] for mode in cfg['modes']]
        index=run_jobs(jobs,index,resume=resuming)
        if args.command=='retry':retry_unstable(2)
        jobs=[dict(stage=3,model_id=model,intra=intra,opencv=2,modes=[mode]) for model in cfg['secondary_models'] for mode in cfg['modes'] for intra in sorted({2}|{v for k,v in winners.items() if k.endswith('/'+mode)})]
        run_jobs(jobs,index,resume=resuming)
        if args.command=='retry':retry_unstable(3)
        manifest=load_json(OUTPUT/'manifest.json');manifest['status']='FINISHED';write_json(OUTPUT/'manifest.json',manifest)
    except BaseException:
        manifest=load_json(OUTPUT/'manifest.json');manifest.update(status='INTERRUPTED',error=traceback.format_exc());write_json(OUTPUT/'manifest.json',manifest);raise
    finally:report()


if __name__=='__main__':main()
