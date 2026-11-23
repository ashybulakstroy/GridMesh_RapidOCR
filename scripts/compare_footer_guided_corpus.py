"""Whole-app paired regression with standard telemetry enabled."""
import os,sys,json,time,statistics,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from App.pipeline import process
from App.ocr.p1_numbers import parse_number
out=ROOT/'data/experiment_reports/footer_line_guided';out.mkdir(parents=True,exist_ok=True)
photos=[('target_90665','data/output/11e98825dd84492ab54622f0e1cd6305/input.bin'),
 ('fold_253309','App/tests/fixtures/receipt_253309.jpg'),('short_51040','App/tests/fixtures/receipt_short_51040.jpg'),
 ('problem_29464','App/tests/fixtures/receipt_29464.jpg'),('fold_36981','App/tests/fixtures/receipt_fold_36981.jpg'),
 ('problem_32596','App/tests/fixtures/receipt_32596.jpg')]
trials=[];pairs=[]
def signature(r):
 return [(row['row_id'],row['kind'],f,str(parse_number(c['text'])) if f in ('quantity','price','printed_total') else c['text'])
         for row in r['rows'] for f,c in row['cells'].items()]
stop=False
for name,path in photos:
 for repeat in range(4):
  pair={}
  for mode in ((0,1) if repeat%2==0 else (1,0)):
   os.environ['OCR_FOOTER_FOCUS_ENABLE']=str(mode)
   start=time.perf_counter();r=process((ROOT/path).read_bytes());elapsed=(time.perf_counter()-start)*1000
   events=[e for e in r['rec_trace']['events'] if 'receipt_total' in e['cells'] and e['stage'].startswith('printed_footer_')]
   t={'photo':name,'repeat':repeat,'mode':mode,'ms':elapsed,'footer_ms':sum(e['wall_ms'] for e in events),
      'footer_calls':len(events),'total':r['total'],'printed_vat':r['printed_vat'],'math_match':r['math_match'],
      'signature':signature(r),'focus':r['footer_focus'],'attempts':r['footer_ocr_attempts'],
      'timings':r['timings'],'performance':r['performance'],'job':r['job_id']}
   trials.append(t);pair[mode]=t
   (out/'corpus_trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
   print(name,repeat,mode,round(elapsed),len(events),r['total'],r['footer_focus'],flush=True)
  p={'photo':name,'repeat':repeat,'fields_equal':pair[0]['signature']==pair[1]['signature'],
     'total_equal':parse_number(pair[0]['total'])==parse_number(pair[1]['total']),
     'vat_equal':parse_number(pair[0]['printed_vat'])==parse_number(pair[1]['printed_vat']),
     'math_equal':pair[0]['math_match']==pair[1]['math_match']}
  pairs.append(p)
  if not all(p[k] for k in ('fields_equal','total_equal','vat_equal','math_equal')):
   print('STOP: regression',p,flush=True);stop=True;break
 if stop:break
summary=[]
for name,_ in photos:
 modes={}
 for mode in (0,1):
  ts=[t for t in trials if t['photo']==name and t['mode']==mode and t['repeat']>0]
  if ts:modes[str(mode)]={k:statistics.median(t[k] for t in ts) for k in ('ms','footer_ms','footer_calls')}
 if modes:summary.append({'photo':name,'modes':modes})
payload={'regression':stop,'pairs':pairs,'summary':summary}
(out/'corpus_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
E=lambda x:html.escape(str(x))
page=['<!doctype html><meta charset="utf-8"><title>Поиск итога по линиям: проверка корпуса</title><style>body{font:15px system-ui;margin:30px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>Локализация строки итога по наблюдаемым линиям</h1><p>Прогрев и три измеряемых повтора на каждом проблемном чеке. Порядок режимов чередуется, штатная телеметрия и DEBUG включены. Поля, названия, итог, НДС и контроль сумм сравниваются попарно. Время потоков перекрывается; сокращение OCR итога не равно сокращению полного цикла.</p><table><tr><th>Чек</th><th>Режим</th><th>Полный цикл, мс</th><th>OCR итога, мс</th><th>OCR-вызовы итога</th></tr>']
for s in summary:
 for mode,m in s['modes'].items():page.append('<tr>'+''.join('<td>'+E(x)+'</td>' for x in (s['photo'],'Новый' if mode=='1' else 'Прежний',round(m['ms'],2),round(m['footer_ms'],2),m['footer_calls']))+'</tr>')
page.append('</table><h2>Проверка регрессии</h2><pre>'+E(json.dumps(pairs,ensure_ascii=False,indent=2))+'</pre><p><a href="corpus_trials.json">Все запуски, штатные тайминги, поля и причины fallback</a></p>')
(out/'corpus_report.html').write_text(''.join(page),encoding='utf-8')
print('SUMMARY',json.dumps(payload,ensure_ascii=False),flush=True)
