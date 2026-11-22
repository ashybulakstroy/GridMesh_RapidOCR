"""Actual early/full latency and final-evidence regression against the current app."""
import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_parallel_columns import PHOTOS, signature


def worker(enabled,path):
    os.environ.update(OCR_E4_ENABLE='1',OCR_E4_ADAPTIVE=str(enabled),OCR_TABLE_MODE='cell',
                      OCR_TABLE_PARALLEL='1',OCR_ONNX_THREADS='2',OCR_TABLE_ONNX_THREADS='2')
    from App.pipeline import process
    from App.ocr.p1_numbers import parse_number
    rows=[]
    for photo,source,expected in PHOTOS:
        data=(ROOT/source).read_bytes()
        for warmup in (True,False):
            first=[]
            start=time.perf_counter()
            def publish(update):
                if update.get('e4_early_ready'):
                    first.append((time.perf_counter()-start)*1000)
            result=process(data,publish)
            full=(time.perf_counter()-start)*1000
            perf=result['performance'];info=result.get('e4_adaptive',{})
            evidence=sorted((key,o['model'],o['path'],o['number'],o['text'])
                            for key,cell in result['numeric_candidate_book']['cells'].items()
                            for o in cell['observations'])
            comparison=result.get('e4_early_comparison',{})
            trial=dict(photo=photo,mode=enabled,warmup=warmup,job_id=result['job_id'],
                       first_ms=first[0] if first else None,full_ms=full,
                       profile_ms=perf['wall_ms'],cpu_ms=perf['process_cpu_ms'],
                       math_match=result['math_match'],expected_total_ok=parse_number(result['total'])==parse_number(expected),
                       total=result['total'],computed_total=result['computed_total'],printed_vat=result.get('printed_vat'),
                       rows=[{'row_id':r['row_id'],'kind':r['kind'],
                              'cells':{f:c['text'] for f,c in r['cells'].items()}} for r in result['rows']],
                       evidence=evidence,adaptive=info,early_comparison=comparison,
                       matrix_status=result['numeric_matrix']['status'],
                       stats=result['geometry']['ocr_batch'],timings=result['timings'],
                       rec_calls=result['rec_trace']['rec_only_model_invocations'],
                       issues=perf.get('issues',[]),samples=perf['samples'])
            rows.append(trial)
            path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            print(photo,enabled,'warmup' if warmup else 'measured','first',trial['first_ms'],
                  'full',round(full,1),'math',trial['math_match'],flush=True)


def compare(a,b):
    return dict(photo=a['photo'],repeat=a['repeat'],names_equal=signature(a)==signature(b),
                numeric_equal=signature(a,True)==signature(b,True),total_equal=a['total']==b['total'],
                matrix_status_equal=a['matrix_status']==b['matrix_status'],
                candidate_evidence_equal=a['evidence']==b['evidence'])


def report(directory,trials,pairs):
    measured=[t for t in trials if not t['warmup']]
    summaries=[]
    for photo,_,_ in PHOTOS:
        bymode={}
        for mode in (0,1):
            group=[t for t in measured if t['photo']==photo and t['mode']==mode]
            first=[t['first_ms'] for t in group if t['first_ms'] is not None]
            bymode[str(mode)]=dict(runs=len(group),full_ms=statistics.median(t['full_ms'] for t in group),
                first_ms=statistics.median(first) if first else None,previews=len(first),
                cpu_ms=statistics.median(t['cpu_ms'] for t in group),
                math_matches=sum(t['math_match'] for t in group),
                expected_totals=sum(t['expected_total_ok'] for t in group),
                early_number_changes=sum(t['early_comparison'].get('changed_count',0) for t in group))
        a,b=bymode['0'],bymode['1']
        summaries.append(dict(photo=photo,modes=bymode,full_change_percent=100*(b['full_ms']/a['full_ms']-1),
                              first_advantage_vs_baseline_ms=a['full_ms']-b['first_ms'] if b['first_ms'] is not None else None))
    summary={'summaries':summaries,'pairs':pairs,'table_threads':'1+1','early_result_is_provisional':True}
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    E=lambda v:html.escape(str(v))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>E4: первый и финальный результат</title><style>body{font:15px system-ui;max-width:1500px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;width:100%;margin:1em 0}td,th{border:1px solid #ccd6df;padding:8px;text-align:left}th{background:#edf2f9}pre{white-space:pre-wrap}.notice{background:#fff3cc;padding:1em}</style><h1>Адаптация E4 Adaptive Recovery / Early Stop V1</h1>',
          '<p>Сравнение с текущим проектом, включая новые проверки цен и параллельное чтение названий/цифр 1+1. Варианты выполнялись по очереди в отдельных процессах, с прогревом каждого фото, порядок чередовался. Штатные тайминги, DEBUG и полная проверка цифр включены. Прогрев исключён из сводки.</p>',
          '<p class="notice">Первый результат является предварительным снимком. Время первого результата измерено до публикации progress в API; фактическое появление в браузере добавляет ожидание опроса (150 мс в этом режиме) и отрисовку. Полное время измерено до возврата готового результата и сохранения файлов.</p>',
          '<table><tr><th>Чек</th><th>Повторов на режим</th><th>Обычный финальный, мс</th><th>Первый предварительный, мс</th><th>Экспериментальный финальный, мс</th><th>Изменение полного времени</th><th>Предварительных снимков</th><th>Изменений чисел после снимка</th><th>Суммы до / после</th></tr>']
    for s in summaries:
        a,b=s['modes']['0'],s['modes']['1']
        vals=[s['photo'],a['runs'],round(a['full_ms'],1),round(b['first_ms'],1) if b['first_ms'] is not None else 'нет',
              round(b['full_ms'],1),f"{s['full_change_percent']:+.1f}%",b['previews'],b['early_number_changes'],f"{a['math_matches']} / {b['math_matches']}"]
        page.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in vals)+'</tr>')
    page.append('</table><p>Положительное изменение полного времени означает замедление. Названия, числа и набор наблюдений сопоставлены с обычным маршрутом. Совпадение маршрутов и сумм не заменяет визуальную проверку всех цифр.</p>')
    page.append('<h2>Каждый измеряемый запуск</h2><table><tr><th>Чек</th><th>Повтор</th><th>Режим</th><th>Первый, мс</th><th>Финальный, мс</th><th>v5-ячеек до снимка</th><th>Решений матрицы до снимка</th><th>Всего v5-ячеек</th></tr>')
    for t in measured:
        vals=[t['photo'],t['repeat'],'adaptive' if t['mode'] else 'baseline',
              round(t['first_ms'],1) if t['first_ms'] is not None else '—',round(t['full_ms'],1),
              t['adaptive'].get('probes_before_first'),t['adaptive'].get('early_solver_checks'),t['adaptive'].get('full_v5_cells')]
        page.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in vals)+'</tr>')
    page.append('</table><h2>Проверка регрессии финальных данных</h2><pre>'+E(json.dumps(pairs,ensure_ascii=False,indent=2))+'</pre>')
    page.append('<p><a href="trials.json">Сырые запуски, поля, кандидаты и этапы</a> · <a href="summary.json">Сводка JSON</a></p></html>')
    (directory/'report.html').write_text(''.join(page),encoding='utf-8')
    print('REPORT',directory/'report.html',flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--worker-mode',type=int,choices=[0,1])
    parser.add_argument('--worker-output',type=Path)
    args=parser.parse_args()
    if args.worker_mode is not None:return worker(args.worker_mode,args.worker_output)
    if args.repeats<1:parser.error('--repeats must be positive')
    directory=ROOT/'data/experiment_reports'/('e4_adaptive_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    directory.mkdir(parents=True)
    trials=[];pairs=[]
    for repeat in range(1,args.repeats+1):
        for mode in ((0,1) if repeat%2 else (1,0)):
            dest=directory/f'worker_{repeat}_{mode}.json'
            print('START',repeat,'adaptive',mode,flush=True)
            with (directory/f'worker_{repeat}_{mode}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-mode',str(mode),
                                '--worker-output',str(dest)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            rows=json.loads(dest.read_text(encoding='utf-8'))
            for t in rows:t['repeat']=repeat
            trials.extend(rows)
            (directory/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
            print('DONE',repeat,mode,[(t['photo'],round(t['first_ms'] or 0,1),round(t['full_ms'],1)) for t in rows if not t['warmup']],flush=True)
        for photo,_,_ in PHOTOS:
            a=next(t for t in trials if t['repeat']==repeat and t['photo']==photo and t['mode']==0 and not t['warmup'])
            b=next(t for t in trials if t['repeat']==repeat and t['photo']==photo and t['mode']==1 and not t['warmup'])
            pairs.append(compare(a,b))
        if any(not all(p[k] for k in ('names_equal','numeric_equal','total_equal','matrix_status_equal','candidate_evidence_equal')) for p in pairs):
            print('STOP: final data/evidence differs; inspect before any more measurements.',flush=True)
            break
    report(directory,trials,pairs)


if __name__=='__main__':main()
