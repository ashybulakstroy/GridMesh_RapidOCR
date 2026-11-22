"""Isolate table budgets 1+1 and 2+2; keep footer/recovery/OpenCV settings fixed."""
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


def worker(budget, output):
    # Executed in a fresh process: no cached session can retain the other budget.
    os.environ.update(OCR_TABLE_MODE='cell',OCR_TABLE_PARALLEL='1',
                      OCR_ONNX_THREADS='2',OCR_TABLE_ONNX_THREADS=str(budget))
    from App.pipeline import process
    from App.ocr.p1_numbers import parse_number
    from App.config import setting
    trials=[]
    for photo,path,expected in PHOTOS:
        data=(ROOT/path).read_bytes()
        for warmup in (True,False):
            start=time.perf_counter()
            result=process(data)
            wall=(time.perf_counter()-start)*1000
            perf=result['performance']
            background=[max(0,s['system_cpu_percent']-s['process_cpu_percent']/perf['cpu_count'])
                        for s in perf['cpu_history']]
            row=dict(photo=photo,budget=budget,warmup=warmup,job_id=result['job_id'],
                     wall_ms=wall,profile_wall_ms=perf['wall_ms'],cpu_ms=perf['process_cpu_ms'],
                     timings=result['timings'],stats=result['geometry']['ocr_batch'],
                     math_match=result['math_match'],total=result['total'],computed_total=result['computed_total'],
                     expected_total_ok=parse_number(result['total'])==parse_number(expected),
                     rows=[{'row_id':r['row_id'],'kind':r['kind'],
                            'cells':{f:c['text'] for f,c in r['cells'].items()}} for r in result['rows']],
                     vat=result.get('vat'),issues=perf.get('issues',[]),
                     background_cpu_percent=statistics.mean(background) if background else None,
                     memory_mb=perf.get('debug',{}).get('memory_mb'),
                     fixed_settings={k:setting(k,'') for k in ('OCR_ONNX_THREADS','OCR_OPENCV_THREADS','LOG_LEVEL','OCR_E4_ENABLE','OCR_E2_ENABLE')})
            actual={k:v['onnx_threads'] for k,v in row['stats']['column_timings'].items()}
            assert actual=={'names':budget//2,'numeric':budget//2}, actual
            trials.append(row)
            output.write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
            print(photo,budget,'warmup' if warmup else 'measured',round(wall,1),'ms',flush=True)


def report(output,trials,repeats):
    summaries=[]; pairs=[]
    measured=[r for r in trials if not r['warmup']]
    for photo,_,_ in PHOTOS:
        modes={}
        for budget in (2,4):
            rows=[r for r in measured if r['photo']==photo and r['budget']==budget]
            modes[str(budget)]={key:statistics.median(r[key] for r in rows)
                                for key in ('wall_ms','profile_wall_ms','cpu_ms')}
            modes[str(budget)].update(runs=len(rows),
                min_wall_ms=min(r['wall_ms'] for r in rows),max_wall_ms=max(r['wall_ms'] for r in rows),
                table_ms=statistics.median(r['stats']['table_wall_ms'] for r in rows),
                names_ms=statistics.median(r['stats']['names_wall_ms'] for r in rows),
                numeric_ms=statistics.median(r['stats']['numeric_wall_ms'] for r in rows),
                footer_lane_ms=statistics.median(r['stats']['footer_lane_wall_ms'] for r in rows),
                ocr_ms=statistics.median(r['timings']['ocr_s']*1000 for r in rows),
                math_matches=sum(r['math_match'] for r in rows),
                correct_totals=sum(r['expected_total_ok'] for r in rows))
        differences=[]
        for repeat in range(1,repeats+1):
            a=next(r for r in measured if r['photo']==photo and r['repeat']==repeat and r['budget']==2)
            b=next(r for r in measured if r['photo']==photo and r['repeat']==repeat and r['budget']==4)
            difference=100*(1-b['wall_ms']/a['wall_ms']);differences.append(difference)
            pairs.append(dict(photo=photo,repeat=repeat,wall_change_percent=difference,
                              numeric_equal=signature(a,True)==signature(b,True),
                              names_equal=signature(a)==signature(b),
                              total_equal=a['total']==b['total'],computed_total_equal=a['computed_total']==b['computed_total']))
        summaries.append(dict(photo=photo,configurations=modes,
                              reduction_percent=100*(1-modes['4']['wall_ms']/modes['2']['wall_ms']),
                              paired_reduction_median_percent=statistics.median(differences)))
    summary=dict(repeats=repeats,summaries=summaries,pairs=pairs,
                 note='Only table budget changes; footer/recovery ONNX=2, OpenCV from .env. Fresh isolated worker per budget and repeat; alternating order. Warmup excluded.')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    E=lambda v:html.escape(str(v))
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Потоки OCR: 1+1 и 2+2</title><style>body{font:15px system-ui;max-width:1500px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;width:100%;margin:1em 0}td,th{border:1px solid #ccd6df;padding:8px;text-align:left}th{background:#edf2f9}pre{white-space:pre-wrap}</style><h1>Параллельное распознавание: 2 = 1+1 против 4 = 2+2</h1>',
           f'<p>{repeats} парных запуска на фото после прогрева. Конфигурации запускаются по очереди в отдельных процессах, порядок чередуется. Для каждого бюджета и повтора создаётся новый процесс и выполняется прогрев каждого фото. Измерения не включают прогрев и запуск процесса.</p>',
           '<p>Меняется только бюджет таблицы: OCR_TABLE_ONNX_THREADS=2 или 4. Названия и цифры читаются одновременно. OCR_ONNX_THREADS=2 для итога и дальнейших проверок, OpenCV и механизмы исправлений остаются одинаковыми. Штатный профиль и DEBUG-журнал включены. Полное время — до возврата готового результата и сохранения файлов.</p>',
           '<table><tr><th>Чек</th><th>Полный цикл 1+1, мс</th><th>Полный цикл 2+2, мс</th><th>Сокращение времени при 2+2</th><th>Таблица 1+1 / 2+2, мс</th><th>CPU 1+1 / 2+2, мс</th><th>Суммы совпали 1+1 / 2+2</th></tr>']
    for s in summaries:
        a,b=s['configurations']['2'],s['configurations']['4']
        vals=[s['photo'],round(a['wall_ms'],1),round(b['wall_ms'],1),f"{s['reduction_percent']:.1f}%",
              f"{a['table_ms']:.1f} / {b['table_ms']:.1f}",f"{a['cpu_ms']:.1f} / {b['cpu_ms']:.1f}",
              f"{a['math_matches']}/{repeats} / {b['math_matches']}/{repeats}"]
        parts.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in vals)+'</tr>')
    parts.append('</table><p>Положительный процент означает ускорение при 2+2, отрицательный — замедление. Три повтора дают предварительную оценку; диапазон и парные различия сохранены в JSON. Равенство полей проверяется между конфигурациями и не заменяет сверку каждого поля с оригиналом.</p>')
    parts.append('<h2>Вывод по полному времени</h2><ul>')
    for s in summaries:
        change=s['reduction_percent']
        outcome=f'2+2 быстрее на {change:.1f}%' if change>0 else f'2+2 медленнее на {-change:.1f}%'
        parts.append('<li>'+E(s['photo']+': '+outcome)+'</li>')
    parts.append('</ul><p>Ускорение таблицы может не ускорить финальный результат: параллельно работает чтение итога и резервных кандидатов. Сравнивайте полный цикл.</p>')
    parts.append('<h2>Все измеряемые запуски</h2><table><tr><th>Фото</th><th>Повтор</th><th>Бюджет</th><th>Полный цикл, мс</th><th>OCR, мс</th><th>Названия, мс</th><th>Цифры, мс</th><th>Итог и кандидаты, мс</th><th>Фоновый CPU</th></tr>')
    for r in measured:
        vals=[r['photo'],r['repeat'],r['budget'],round(r['wall_ms'],1),r['timings']['ocr_s']*1000,
              r['stats']['names_wall_ms'],r['stats']['numeric_wall_ms'],r['stats']['footer_lane_wall_ms'],
              round(r['background_cpu_percent'],1) if r['background_cpu_percent'] is not None else 'нет замера']
        parts.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in vals)+'</tr>')
    parts.append('</table><h2>Сравнение названий, цифр и итогов</h2><pre>'+E(json.dumps(pairs,ensure_ascii=False,indent=2))+'</pre>')
    parts.append('<p><a href="trials.json">Все поля и исходные измерения JSON</a> · <a href="summary.json">Сводка JSON</a></p></html>')
    (output/'report.html').write_text(''.join(parts),encoding='utf-8')
    print('REPORT',output/'report.html',flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--worker-budget',type=int,choices=[2,4])
    parser.add_argument('--worker-output',type=Path)
    args=parser.parse_args()
    if args.worker_budget:
        return worker(args.worker_budget,args.worker_output)
    if args.repeats<1:parser.error('--repeats must be positive')
    output=ROOT/'data/experiment_reports'/('parallel_threads_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True)
    trials=[]
    for repeat in range(1,args.repeats+1):
        for budget in ((2,4) if repeat%2 else (4,2)):
            dest=output/f'worker_{repeat}_{budget}.json'
            print('START repeat',repeat,'table budget',budget,flush=True)
            with (output/f'worker_{repeat}_{budget}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-budget',str(budget),
                                '--worker-output',str(dest)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            rows=json.loads(dest.read_text(encoding='utf-8'))
            for row in rows:row['repeat']=repeat
            trials.extend(rows)
            (output/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
            print('DONE',repeat,budget,[(r['photo'],round(r['wall_ms'],1)) for r in rows if not r['warmup']],flush=True)
    report(output,trials,args.repeats)


if __name__=='__main__':main()
