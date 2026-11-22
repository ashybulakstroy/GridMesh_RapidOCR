"""Paired, warm, whole-application runs; preserve normal timing and OCR evidence."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from App.pipeline import process
from App.config import setting
from App.ocr.p1_numbers import parse_number

PHOTOS = [
    ('short_51040', 'App/tests/fixtures/receipt_short_51040.jpg', '51040'),
    ('fold_253309', 'App/tests/fixtures/receipt_253309.jpg', '253309'),
    ('long_252161', 'data/output/7a0d379e961547828dd608a2cbb74f4a/input.bin', '252161'),
]


def signature(trial, numeric=False):
    return [(r['row_id'], r['kind'], f,
             str(parse_number(value)) if numeric else value)
            for r in trial['rows'] for f,value in r['cells'].items()
            if (f in ('quantity','price','printed_total') if numeric else f == 'name')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    args=parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    output=ROOT/'data/experiment_reports'/('parallel_columns_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True)
    trials=[]
    saved={key:os.environ.get(key) for key in ('OCR_TABLE_MODE','OCR_TABLE_PARALLEL')}
    try:
        os.environ['OCR_TABLE_MODE']='cell'
        for photo,path,expected in PHOTOS:
            data=(ROOT/path).read_bytes()
            for repeat in range(args.repeats+1):
                for mode in (('sequential','parallel') if repeat%2==0 else ('parallel','sequential')):
                    os.environ['OCR_TABLE_PARALLEL']='1' if mode=='parallel' else '0'
                    start=time.perf_counter()
                    result=process(data)
                    elapsed=(time.perf_counter()-start)*1000
                    perf=result['performance']
                    rows=[{'row_id':r['row_id'],'kind':r['kind'],
                           'cells':{f:c['text'] for f,c in r['cells'].items()}}
                          for r in result['rows']]
                    history=perf['cpu_history']
                    background=[max(0,s['system_cpu_percent']-s['process_cpu_percent']/perf['cpu_count']) for s in history]
                    trial=dict(photo=photo,path=path,expected_total=expected,mode=mode,repeat=repeat,
                               warmup=repeat==0,job_id=result['job_id'],wall_ms=elapsed,
                               profile_wall_ms=perf['wall_ms'],cpu_ms=perf['process_cpu_ms'],
                               timings=result['timings'],stats=result['geometry']['ocr_batch'],
                               math_match=result['math_match'],total=result['total'],computed_total=result['computed_total'],
                               expected_total_ok=parse_number(result['total'])==parse_number(expected),
                               vat=result.get('vat'),rows=rows,
                               background_cpu_percent=statistics.mean(background) if background else None,
                               rss_observed_mb=perf.get('debug',{}).get('memory_mb'),
                               issues=perf.get('issues',[]))
                    trials.append(trial)
                    (output/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
                    print(photo,mode,repeat,round(elapsed,1),'ms',result['total'],result['computed_total'],
                          'overlap',trial['stats'].get('column_overlap_ms'),flush=True)
    finally:
        for key,value in saved.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    summaries=[]
    pairs=[]
    for photo,path,expected in PHOTOS:
        by_mode={}
        for mode in ('sequential','parallel'):
            measured=[t for t in trials if t['photo']==photo and t['mode']==mode and not t['warmup']]
            by_mode[mode]={
                'runs':len(measured),
                'wall_ms':statistics.median(t['wall_ms'] for t in measured),
                'profile_wall_ms':statistics.median(t['profile_wall_ms'] for t in measured),
                'ocr_ms':statistics.median(t['timings']['ocr_s']*1000 for t in measured),
                'table_ms':statistics.median(t['stats']['table_wall_ms'] for t in measured),
                'cpu_ms':statistics.median(t['cpu_ms'] for t in measured),
                'math_matches':sum(t['math_match'] for t in measured),
                'expected_totals':sum(t['expected_total_ok'] for t in measured),
            }
        summaries.append({'photo':photo,**by_mode,
                          'wall_reduction_percent':100*(1-by_mode['parallel']['wall_ms']/by_mode['sequential']['wall_ms'])})
        for repeat in range(1,args.repeats+1):
            a=next(t for t in trials if t['photo']==photo and t['repeat']==repeat and t['mode']=='sequential')
            b=next(t for t in trials if t['photo']==photo and t['repeat']==repeat and t['mode']=='parallel')
            pairs.append({'photo':photo,'repeat':repeat,'names_equal':signature(a)==signature(b),
                          'numeric_equal':signature(a,True)==signature(b,True),
                          'total_equal':parse_number(a['total'])==parse_number(b['total']),
                          'computed_total_equal':parse_number(a['computed_total'])==parse_number(b['computed_total'])})
    summary={'repeats':args.repeats,'summaries':summaries,'pairs':pairs,
             'settings':{k:setting(k,'') for k in ('LOG_LEVEL','OCR_ONNX_THREADS','OCR_OPENCV_THREADS','OCR_E4_ENABLE','OCR_E2_ENABLE')}}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    E=lambda x:html.escape(str(x))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Параллельные названия и цифры</title><style>body{font:15px system-ui;max-width:1400px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;width:100%;margin:1em 0}td,th{border:1px solid #ccd6df;padding:8px;text-align:left}th{background:#edf2f9}pre{white-space:pre-wrap}</style><h1>Время до финального результата: названия и цифры параллельно</h1>',
          f'<p>{args.repeats} парных запуска после отдельного прогрева обоих маршрутов на каждом фото. Порядок чередуется. Модели, проверка сумм, повторные сканирования и штатные измерения включены. В обоих маршрутах итог читается параллельно таблице. Все сравнения выполнены одним процессом на этой машине.</p>',
          '<p>Полный цикл: от вызова process до возвращения готового результата, включая сохранение файлов. В JSON также сохранено время штатного профиля, всех этапов, CPU и фоновой загрузки. Совпадение полей проверяется с последовательным маршрутом; оно не доказывает правильность исходного OCR.</p>',
          '<table><tr><th>Фото</th><th>Последовательно, мс</th><th>Параллельно, мс</th><th>Сокращение полного времени</th><th>OCR до / после, мс</th><th>Совпадение сумм до / после</th><th>Эталон итога до / после</th></tr>']
    for item in summaries:
        a,b=item['sequential'],item['parallel']
        values=[item['photo'],round(a['wall_ms'],1),round(b['wall_ms'],1),f"{item['wall_reduction_percent']:.1f}%",
                f"{a['ocr_ms']:.1f} / {b['ocr_ms']:.1f}",f"{a['math_matches']}/{args.repeats} / {b['math_matches']}/{args.repeats}",
                f"{a['expected_totals']}/{args.repeats} / {b['expected_totals']}/{args.repeats}"]
        page.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in values)+'</tr>')
    page.append('</table><h2>Проверка полей в каждом парном запуске</h2><pre>'+E(json.dumps(pairs,ensure_ascii=False,indent=2))+'</pre>')
    page.append('<h2>Сырые результаты</h2><p><a href="trials.json">Все запуски и распознанные поля</a> · <a href="summary.json">Сводка JSON</a></p><pre>'+E(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>')
    (output/'report.html').write_text(''.join(page),encoding='utf-8')
    print('REPORT',output/'report.html',flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
