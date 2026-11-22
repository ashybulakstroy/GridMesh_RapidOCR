"""Paired whole receipt comparison, fixed CPU settings; no model load in warm results."""
import argparse,json,os,statistics,time,html
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from App.pipeline import process
from App.config import setting
from App.ocr.p1_numbers import parse_number

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--photo',default='data/output/7a0d379e961547828dd608a2cbb74f4a/input.bin')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    data=(ROOT/args.photo).read_bytes()
    destination=ROOT/'data/experiment_reports/cell_vs_batch_7a0d379e'
    destination.mkdir(parents=True,exist_ok=True)
    trials=[]
    for repeat in range(args.repeats+1):
        for mode in (('cell','batch') if repeat%2==0 else ('batch','cell')):
            os.environ['OCR_TABLE_MODE']=mode
            start=time.perf_counter();result=process(data)
            rows=[{'row_id':r['row_id'],'kind':r['kind'],'cells':{f:c['text'] for f,c in r['cells'].items()}} for r in result['rows']]
            trial=dict(mode=mode,repeat=repeat,warmup=repeat==0,wall_s=time.perf_counter()-start,
                timings=result['timings'],math_match=result['math_match'],total=result['total'],computed_total=result['computed_total'],
                rows=rows,stats=result['geometry']['ocr_batch'],job_id=result['job_id'])
            perf=result['performance']
            background=[max(0,s['system_cpu_percent']-s['process_cpu_percent']/perf['cpu_count']) for s in perf['cpu_history']]
            trial['background_cpu_percent']=statistics.mean(background) if background else None
            trial['process_cpu_ms']=perf['process_cpu_ms']
            trial['unstable']=trial['background_cpu_percent'] is not None and trial['background_cpu_percent']>30
            trial['known_lower_rows_ok']=all(
                parse_number(next(r for r in result['rows'] if r['row_id']==rid)['cells'][field]['text'])==expected
                for rid,field,expected in [(15,'quantity',5),(15,'price',2000),(16,'quantity',20),(16,'price',60)])
            trials.append(trial)
            (destination/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
            print(mode,repeat,trial['timings'],trial['total'],trial['computed_total'],flush=True)
    summary={}
    for mode in ('cell','batch'):
        selected=[t for t in trials if t['mode']==mode and not t['warmup']]
        summary[mode]=dict(repeats=len(selected),median_wall_s=statistics.median(t['wall_s'] for t in selected),
            median_ocr_s=statistics.median(t['timings']['ocr_s'] for t in selected),
            median_table_ms=statistics.median(t['stats']['table_wall_ms'] for t in selected),
            stable_trials=sum(not t['unstable'] for t in selected),known_lower_rows_ok=sum(t['known_lower_rows_ok'] for t in selected),
            median_process_cpu_ms=statistics.median(t['process_cpu_ms'] for t in selected),
            math_matches=sum(t['math_match'] for t in selected),totals=sorted({t['total'] for t in selected}),
            computed_totals=sorted({t['computed_total'] for t in selected}))
    pairs=[]
    for repeat in range(1,args.repeats+1):
        a=next(t for t in trials if t['repeat']==repeat and t['mode']=='cell')
        b=next(t for t in trials if t['repeat']==repeat and t['mode']=='batch')
        numeric=lambda t:[(r['row_id'],f,parse_number(v)) for r in t['rows'] for f,v in r['cells'].items() if f in ('quantity','price','printed_total')]
        names=lambda t:[r['cells'].get('name') for r in t['rows']]
        pairs.append(dict(repeat=repeat,numeric_fields_equal=numeric(a)==numeric(b),names_equal=names(a)==names(b)))
    summary['paired_fields']=pairs
    summary['photo']=args.photo
    summary['settings']={k:setting(k,d) for k,d in [('OCR_ONNX_THREADS','2'),('OCR_OPENCV_THREADS','2'),('OCR_E4_ENABLE','0'),('OCR_REC_BATCH_NUM','8')]}
    summary['settings']['inter_op']=1
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    page='<html lang="ru"><meta charset="utf-8"><title>Поячеечный и пакетный OCR</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Поячеечный и пакетный OCR</h1><p>Фото: '+html.escape(args.photo)+'</p><p>Пять парных повторов после прогрева; порядок чередуется. Геометрия исключена из времени OCR. В поячеечном режиме итог читается параллельно, подготовка ROI без лимита. В пакетном сохранены прежние бюджеты. Это сравнение полных маршрутов, не изолированной операции REC.</p>'
    page+='<table><tr><th>Режим</th><th>OCR, median с</th><th>Таблица, median мс</th><th>Полный цикл, median с</th><th>Совпадений суммы</th><th>Проверка строк 15–16</th><th>Стабильных повторов</th></tr>'
    for mode in ('cell','batch'):
        v=summary[mode]
        page+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [mode,v['median_ocr_s'],round(v['median_table_ms'],1),round(v['median_wall_s'],3),v['math_matches'],v['known_lower_rows_ok'],v['stable_trials']])+'</tr>'
    page+='</table><p>Фоновая загрузка выше 30% общей мощности CPU отмечается как нестабильность. Совпадение суммы не доказывает правильность всех полей. Для строк 15–16 проверены известные значения: 5 × 2000 и 20 × 60.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>'
    (destination/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
