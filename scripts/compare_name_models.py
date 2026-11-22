"""Paired real-receipt comparison of name recognizers."""
import os,sys,json,time,statistics,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
raise SystemExit('Архивный тест: OCR_NAME_MODEL больше не переключает модель. Результаты прежнего сравнения сохранены в data/experiment_reports/name_models_da7ede65/.')
from App.pipeline import process
from App.config import setting
PHOTO=ROOT/'data/output/da7ede65b96a4c05b9db2037c3217fea/input.bin'
OUT=ROOT/'data/experiment_reports/name_models_da7ede65';OUT.mkdir(parents=True,exist_ok=True)
models=['v5_cyrillic_mobile','v6_tiny_cyrillic_optional'];trials=[]
for repeat in range(6):
 for model in (models if repeat%2==0 else models[::-1]):
    os.environ['OCR_NAME_MODEL']=model
    os.environ['LOG_LEVEL']='DEBUG'
    start=time.perf_counter();r=process(PHOTO.read_bytes());elapsed=time.perf_counter()-start
    debug=r['performance']['debug'];cells=[x for x in debug['ocr_cells'] if x['field']=='name']
    item={'model':model,'repeat':repeat,'warmup':repeat==0,'wall_s':elapsed,'timings':r['timings'],
       'name_wall_ms':sum(x['wall_ms'] for x in cells),'names':[{'row':x['row_id'],'name':x['cells']['name']['text']} for x in r['rows']],
       'total':r['total'],'computed_total':r['computed_total'],'math_match':r['math_match'],'job_id':r['job_id'],
       'name_models':{k:v for k,v in r['rec_trace']['per_cell'].items() if k.endswith(':name')},
       'issues':r['performance']['issues'],'process_cpu_ms':r['performance']['process_cpu_ms']}
    trials.append(item);(OUT/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
    print(model,repeat,round(elapsed,3),round(item['name_wall_ms'],1),item['total'],flush=True)
summary={}
for model in models:
 x=[t for t in trials if t['model']==model and not t['warmup']]
 summary[model]={'wall_s':statistics.median(t['wall_s'] for t in x),'ocr_s':statistics.median(t['timings']['ocr_s'] for t in x),'name_ms':statistics.median(t['name_wall_ms'] for t in x),'process_cpu_ms':statistics.median(t['process_cpu_ms'] for t in x),'correct_sum':sum(t['math_match'] for t in x),'repeats':len(x)}
summary['settings']={k:setting(k,d) for k,d in [('OCR_ONNX_THREADS','2'),('OCR_OPENCV_THREADS','2'),('OCR_TABLE_MODE','cell'),('OCR_E4_ENABLE','0')]}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
a=next(t for t in trials if t['model']==models[0] and t['repeat']==1);b=next(t for t in trials if t['model']==models[1] and t['repeat']==1)
esc=lambda x:html.escape(str(x))
rows=''.join('<tr><td>'+esc(x['row'])+'</td><td>'+esc(x['name'])+'</td><td>'+esc(y['name'])+'</td></tr>' for x,y in zip(a['names'],b['names']))
page='<html lang="ru"><meta charset="utf-8"><title>Сравнение моделей названий</title><style>body{font:16px system-ui;margin:30px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style><h1>V5 Cyrillic и V6 Tiny Cyrillic</h1><p>Один чек, по пять повторов после прогрева, порядок чередуется. Время названий — сумма времени вызовов ячеек, включая подготовку внутри read_cell. Геометрия не включена в OCR; CPU-время включает все потоки. Это сравнение конфигураций на одном чеке, без эталонной разметки точности. Многострочный fallback с детектором может применять V5; на этом чеке так прочитана строка 15.</p><pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><table><tr><th>Строка</th><th>V5</th><th>V6 Tiny</th></tr>'+rows+'</table></html>'
(OUT/'report.html').write_text(page,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
