"""Read-only report generator; standard library only. No OCR calls."""
from __future__ import annotations
import argparse, html, json, os, re, statistics, shutil, time
from pathlib import Path
from collections import defaultdict

GROUPS = ('grid','ocr','totals','other')
def group(name):
    n=name.lower()
    if any(s in n for s in ('сетк','grid','geometry','геометр','mesh')): return 'grid'
    if any(s in n for s in ('поячееч','подготовка ocr','ячее','ocr lane','table_numeric','table_names')): return 'ocr'
    if any(s in n for s in ('итог','footer','e4','e5','чисел','матриц','ндс','проверка электронного')): return 'totals'
    return 'other'
def number(x):
    try: return float(x)
    except (ValueError,TypeError): return 0.0

def read(path, default):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default

def normalized(cid): return re.sub(r'^(\d+)-',r'\1:',str(cid))

def build(profile, receipt, deep=None, job_ms=2000, stage_ms=1000):
    d=profile.get('debug',{}); deep=deep or {}; total=number(profile.get('wall_ms'))
    if str(d.get('configuration',{}).get('log_level','')).upper()!='DEBUG' or total<=job_ms: return None
    stages=[dict(s,group=group(s.get('name',''))) for s in profile.get('samples',[])]
    buckets=defaultdict(list)
    for s in stages: buckets[s['group']].append(s)
    # Existing samples have no intervals. Do not sum possibly nested or parallel measurements.
    sections={g:{'stages':ss,'max_stage_ms':max((number(s.get('wall_ms')) for s in ss),default=0), 'wall_ms':deep.get('section_wall_ms',{}).get(g,round(sum(number(s.get('wall_ms')) for s in ss),3) if profile.get('version')=='ocr-performance-v1' else None)} for g,ss in buckets.items()}
    for g,v in sections.items(): v['detailed']=max(v['max_stage_ms'],number(v['wall_ms']))>stage_ms
    lanes=d.get('ocr_lanes',[])
    events=receipt.get('rec_trace',{}).get('events',[])
    cells={normalized(c['cell_id']):c for c in d.get('ocr_cells',[]) if 'cell_id' in c}
    final={normalized(f"{row.get('row_id')}:{field}"):c for row in receipt.get('rows',[]) for field,c in row.get('cells',{}).items() if isinstance(c,dict)}
    by=defaultdict(list)
    for ev in events:
        for cid in set(map(normalized,ev.get('cells',[]))): by[cid].append(ev)
    med=defaultdict(list)
    for cid,c in cells.items(): med[c.get('field',cid.split(':')[-1])].append(number(c.get('wall_ms')))
    out=[]
    if sections.get('ocr',{}).get('detailed') or sections.get('totals',{}).get('detailed'):
        for cid in sorted(set(cells)|set(by)):
            c=cells.get(cid,{}); evs=by.get(cid,[]); f=final.get(cid,{}); field=c.get('field',cid.split(':')[-1]); is_footer=cid.startswith('receipt_') or cid=='printed_vat' or 'footer' in cid
            if is_footer and not sections.get('totals',{}).get('detailed'): continue
            if not is_footer and not sections.get('ocr',{}).get('detailed'): continue
            w=number(c.get('wall_ms')); m=statistics.median(med[field]) if med[field] else 0
            reasons=[]
            if w>(250 if field=='name' else 150): reasons.append('медленная ячейка')
            if m and w>3*m: reasons.append('более 3× медианы поля')
            if any(number(e.get('wall_ms'))>100 and e.get('mode')=='REC_only' for e in evs): reasons.append('REC_only >100 мс')
            if any(e.get('mode')=='DET+REC' for e in evs): reasons.append('DET+REC')
            if sum(number(e.get('wall_ms')) for e in evs)>(250 if field=='name' else 150): reasons.append('сумма длительностей вызовов выше порога (не wall time)')
            out.append({'cell_id':cid,'field':field,'wall_ms':c.get('wall_ms'),'reasons':reasons,'cell_profile':c,'attempts':evs,'final':{k:f.get(k) for k in ('text','confidence','model_id','strategy','numeric_roi')},'fallback_reason':next((e.get('fallback_reason') for e in evs if e.get('fallback_reason')),None),'selection_reason':f.get('selection_reason') or next((e.get('selection_reason') for e in evs if e.get('selection_reason')),None),'decisions':[x for x in deep.get('ocr_decisions',[]) if x.get('cell_id')==cid]})
    for g,v in sections.items():
        if not v['detailed']: continue
        v['live_spans']=[s for s in deep.get('spans',[]) if s.get('group')==g]
        if g=='grid': v.update(geometry_substeps=d.get('geometry_substeps',[]),grid_anomaly=deep.get('grid_anomaly'),counters=deep.get('grid_counters',{}))
        if g=='ocr': v.update(lanes=lanes,batches=d.get('ocr_batches',[]),fallbacks=d.get('ocr_fallbacks',[]),model_calls=d.get('model_calls_from_rec_trace',{}),crop_summary=d.get('cell_crop_summary',{}))
        if g=='totals': v.update(e4=profile.get('e4_adaptive',{}),footer_attempts=receipt.get('footer_ocr_attempts',[]),footer_variants=receipt.get('footer_variant_attempts',[]),math_match=receipt.get('math_match'),total=receipt.get('total'),computed_total=receipt.get('computed_total'),vat=receipt.get('vat'),numeric_retries=receipt.get('numeric_auto_corrections',[]))
    return {'schema':'problem-performance-v1','job_id':receipt.get('job_id',d.get('job_id')),'error':receipt.get('error'),'wall_ms':total,'thresholds':{'job_ms':job_ms,'section_ms':stage_ms},'sections':sections,'cells':sorted(out,key=lambda x:(not bool(x['reasons']),-number(x['wall_ms']))),'configuration':d.get('configuration',{}),'cpu':{k:profile.get(k) for k in ('process_cpu_ms','cpu_core_equivalents','peak_process_cpu_percent','peak_system_cpu_percent','cpu_history')},'memory':d.get('memory_mb',{}),'capture_warnings':deep.get('warnings',[]),'notes':['Отсутствующие измерения не реконструируются.','Параллельные вызовы и вложенные таймеры нельзя складывать для wall time.','ocr-performance-v1: wall раздела — сумма непересекающихся верхнеуровневых фаз OCRPerformance.phase; вложенные подэтапы отдельно. Для неизвестной схемы используется максимум этапов.']}

def render(report):
    esc=lambda v:html.escape(str(v if v is not None else 'не зафиксировано'))
    def js(v):return esc(json.dumps(v,ensure_ascii=False,indent=2))
    def table(headers,rows):return '<table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    labels={'grid':'Сетка и геометрия','ocr':'Распознавание','totals':'Итоги / НДС / восстановление','other':'Прочие этапы'}
    chunks=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Диагностика производительности</title><style>body{font:15px system-ui;background:#f2f5f8;color:#203040;margin:32px auto;max-width:1150px;padding:0 18px}section{background:white;border:1px solid #d7e0e8;border-radius:10px;margin:18px 0;padding:20px}h1{font-size:29px}h2{font-size:21px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid #d7e0e8;vertical-align:top}th{background:#e8eef4}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;padding:9px 0} .slow{color:#a32c21;font-weight:700}img{max-width:350px;max-height:160px;border:1px solid #ddd}</style><h1>Диагностика производительности OCR</h1>',f'<p>Чек: {esc(report["job_id"])} · <b>{report["wall_ms"]:.2f} мс</b> · DEBUG</p>', '<section><h2>Все этапы</h2>']
    if report.get('error'):
        chunks.append('<p class="slow">Обработка завершилась ошибкой: '+esc(report['error'])+'</p>')
    chunks.append(table(['Раздел','Этап','Wall, мс','CPU, мс'],[(labels[g],s.get('name'),s.get('wall_ms'),s.get('process_cpu_ms')) for g,v in report['sections'].items() for s in v['stages']]))
    chunks.append('</section>')
    for g,v in report['sections'].items():
        if not v['detailed']:continue
        chunks.append(f'<section><h2 class="slow">{labels[g]} — подробная диагностика</h2><p>Максимальный этап: {v["max_stage_ms"]:.2f} мс; wall раздела: {esc(v["wall_ms"])}</p>')
        if g=='grid':
            functions=(v.get('counters') or {}).get('functions') or {}
            selected=[]
            for label in ('trace_seed','graph_verticals'):
                matches=[(name,stats) for name,stats in functions.items() if name.rsplit(':',1)[-1]==label]
                if not matches:
                    selected.append((label,'не зафиксировано','—','—','—','—','—'))
                else:
                    for name,stats in matches:
                        selected.append((name,stats.get('calls'),stats.get('sum_call_ms'),
                                         stats.get('median_ms'),stats.get('p90_ms'),
                                         stats.get('min_ms'),stats.get('max_ms')))
            chunks.append('<h3>Внутренние вызовы поиска сетки</h3><p>Сумма показывает длительность всех вызовов функции; вложенные вызовы могут пересекаться, поэтому её нельзя прибавлять к времени других этапов. «Не зафиксировано» не означает, что вызовов было ноль.</p>')
            chunks.append(table(['Функция','Вызовы','Сумма, мс','Медиана, мс','P90, мс','Мин., мс','Макс., мс'],selected))
            details=(v.get('counters') or {}).get('details') or []
            internal=[d for d in details if d.get('name')=='grid_internal_timing' and isinstance(d.get('breakdown_ms'),dict)]
            chunks.append('<h3>Где тратится время внутри функций</h3>')
            if internal:
                breakdown=defaultdict(list)
                for detail in internal:
                    for name,ms in detail['breakdown_ms'].items():
                        breakdown[(detail.get('function','—'),name)].append(number(ms))
                chunks.append('<p>Суммы по измеренным вызовам; внутренние таймеры собраны в режиме DEBUG и могут добавлять собственные накладные расходы.</p>')
                chunks.append(table(['Функция','Внутренний этап','Измерений','Сумма, мс','Медиана, мс'],
                    [(function,name,len(values),round(sum(values),2),round(statistics.median(values),2))
                     for (function,name),values in breakdown.items()]))
            else:
                chunks.append('<p>Для этого задания внутренняя разбивка не была записана. Повторите обработку в режиме DEBUG после обновления сервера.</p>')
        if g in ('ocr','totals'):
            for c in report['cells']:
                if (g=='totals') != (c['cell_id'].startswith('receipt_') or c['cell_id']=='printed_vat' or 'footer' in c['cell_id']):continue
                chunks.append(f'<details {"open" if c["reasons"] else ""}><summary><b>{esc(c["cell_id"])}</b> · {esc(c["wall_ms"])} мс · {esc(", ".join(c["reasons"]))}</summary>')
                chunks.append(table(['Событие','Модель','Режим','Результат','Уверенность','мс','Причина'],[(e.get('event_id'),e.get('model_id'),e.get('mode'),e.get('texts'),e.get('confidences'),e.get('wall_ms'),e.get('fallback_reason')) for e in c['attempts']]))
                for e in c['attempts']:
                    for path in e.get('report_images',[]):
                        if re.fullmatch(r'images/[a-zA-Z0-9_.-]+\.png',path): chunks.append(f'<p>Вход OCR #{esc(e.get("event_id"))}<br><img src="{esc(path)}"></p>')
                chunks.append(f'<p>Причина fallback: {esc(c["fallback_reason"])}; выбор: {esc(c["selection_reason"])}</p><pre>{js(c["final"])}</pre><pre>{js(c["decisions"])}</pre></details>')
        for key,val in v.items():
            if key in ('stages','max_stage_ms','wall_ms','detailed'):continue
            chunks.append(f'<details><summary>{esc(key)}</summary><pre>{js(val)}</pre></details>')
        chunks.append('</section>')
    if report.get('deep_profile'):
        chunks.append('<p><a href="deep_profile.json">Сырые данные профилирования (JSON)</a></p>')
    chunks.append(f'<section><h2>Конфигурация и ресурсы</h2><pre>{js({k:report[k] for k in ("configuration","cpu","memory","capture_warnings","notes")})}</pre></section></html>')
    return ''.join(chunks)

def generate(job, deep=None, images=None, job_ms=2000, stage_ms=1000, receipt=None):
    job=Path(job); start=time.perf_counter(); p=read(job/'performance_profile.json',{}); r=receipt if receipt is not None else read(job/'receipt.json',{}); has_deep=deep is not None; deep=dict(deep or {})
    if (job/'grid_anomaly.json').exists(): deep.setdefault('grid_anomaly',read(job/'grid_anomaly.json',{}))
    report=build(p,r,deep,job_ms,stage_ms)
    if report is None:return None
    target=job/'problem_performance'; target.mkdir(exist_ok=True)
    if has_deep:
        raw={'schema':'performance-deep-capture-v1','job_id':r.get('job_id',job.name),
             'capture_status':deep.get('capture_status','provided'),**deep}
        tmp=target/'deep_profile.json.tmp'
        tmp.write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(tmp,target/'deep_profile.json')
        report['deep_profile']='deep_profile.json'
    # Historical reports only provide actual attempt images for some cells; never fabricate missing inputs.
    for c in report['cells']:
        for e in c['attempts']:
            eid=e.get('event_id'); safe=re.sub(r'[^a-zA-Z0-9_-]','_',c['cell_id']); matches=[]
            for folder in ('problem_digits','problem_text'):
                matches.extend((job/folder).glob(f'attempt_{eid}_{safe}.png'))
            for i,blob in enumerate((images or {}).get((eid,c['cell_id']),[])):
                dest=target/'images'/f'attempt_{eid}_{safe}_{i}.png'; dest.parent.mkdir(exist_ok=True);dest.write_bytes(blob);e.setdefault('report_images',[]).append('images/'+dest.name)
            if not e.get('report_images'):
                for src in matches[:1]:
                    dest=target/'images'/src.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest);e.setdefault('report_images',[]).append('images/'+dest.name)
            if not e.get('report_images'):e['image_status']='not_captured'
    for name,content in [('manifest.json',json.dumps(report,ensure_ascii=False,indent=2)),('report.html',render(report))]:
        tmp=target/(name+'.tmp');tmp.write_text(content,encoding='utf-8');os.replace(tmp,target/name)
    (target/'export_timing.json').write_text(json.dumps({'export_ms':round((time.perf_counter()-start)*1000,3),'excluded_from_receipt_wall':True}),encoding='utf-8')
    return target/'report.html'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('job',type=Path);ap.add_argument('--deep',type=Path);args=ap.parse_args();print(generate(args.job,read(args.deep,{}) if args.deep else None) or 'SKIPPED: DEBUG / duration gate')
