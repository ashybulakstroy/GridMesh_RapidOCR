"""Paired RapidOCR cell vs batch[6], real Benchmark2 ROIs, three measured runs."""
import sys,os,json,time,statistics,html,platform
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark2.prepare import sha
from benchmark2.preprocess import precompute,join_parts
from benchmark2.runner import metric
from rapidocr import RapidOCR

samples=json.loads((ROOT/'data/cells_manifest.json').read_text(encoding='utf-8'))
spec=next(x for x in json.loads((ROOT/'config/models.json').read_text(encoding='utf-8')) if x['id']=='v6_tiny_cyrillic_optional')
model=ROOT.parent/spec['model_path']
if not model.is_file():model=ROOT/spec['model_path']
assert sha(model)==spec['model_sha256']
for s in samples:assert sha(ROOT/s['file'])==s['crop_sha256']
directory=ROOT/'results'/('tiny_cell_batch6_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
directory.mkdir(parents=True)
engines={}
for mode in ('cell','batch6'):
 engines[mode]=RapidOCR(params={'Global.use_det':False,'Global.use_cls':False,
   'Rec.model_path':str(model),'Rec.lang_type':'cyrillic','Rec.rec_img_shape':spec['rec_img_shape'],
   'Rec.rec_batch_num':6 if mode=='batch6' else 1,
   'EngineConfig.onnxruntime.intra_op_num_threads':2,'EngineConfig.onnxruntime.inter_op_num_threads':1})
def trial(mode,repeat):
 start=time.perf_counter();pieces=precompute(samples,ROOT)
 segmented=time.perf_counter();flat=[]
 for s in samples:
  for im in pieces[s['id']]:flat.append((s['id'],im))
 arrays=[(sid,np.ascontiguousarray(np.asarray(im.convert('RGB'))[:,:,::-1])) for sid,im in flat]
 ready=time.perf_counter();texts={s['id']:[] for s in samples};calls=[]
 step=1 if mode=='cell' else 6
 for i in range(0,len(arrays),step):
  chunk=arrays[i:i+step];a=time.perf_counter();result=engines[mode].recognize_txt([x[1] for x in chunk]);b=time.perf_counter()
  if len(result.txts)!=len(chunk):raise RuntimeError('OCR returned incorrect count')
  for (sid,_),txt in zip(chunk,result.txts):texts[sid].append(str(txt))
  calls.append({'size':len(chunk),'ms':(b-a)*1000})
 done=time.perf_counter()
 out=[{'id':s['id'],'type':s['type'],'expected':s['expected'],'text':join_parts(texts[s['id']]),
       'image':s['file'],'metric':metric(s['expected'],join_parts(texts[s['id']]))} for s in samples]
 row={'mode':mode,'repeat':repeat,'warmup':repeat==0,'segmentation_ms':(segmented-start)*1000,
      'packing_ms':(ready-segmented)*1000,'ocr_ms':sum(c['ms'] for c in calls),
      'total_ms':(done-start)*1000,'segments':len(arrays),'calls':calls,'cells':out}
 print(mode,repeat,round(row['total_ms'],1),'ms calls',len(calls),flush=True)
 return row
trials=[]
for i in range(4):
 for mode in (('cell','batch6') if i%2==0 else ('batch6','cell')):
  trials.append(trial(mode,i));(directory/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
 if any(a['text']!=b['text'] for a,b in zip(next(t for t in trials if t['mode']=='cell' and t['repeat']==i)['cells'],next(t for t in trials if t['mode']=='batch6' and t['repeat']==i)['cells'])):
  print('STOP: different OCR output; inspect before additional runs',flush=True);break
measured=[t for t in trials if not t['warmup']]
summary={}
for mode in ('cell','batch6'):
 ts=[x for x in measured if x['mode']==mode]
 if not ts:continue
 summary[mode]={key:statistics.median(t[key] for t in ts) for key in ('segmentation_ms','packing_ms','ocr_ms','total_ms')}
 ns=[c for t in ts for c in t['cells'] if c['type']=='name']
 summary[mode]['name_cer']=sum(c['metric']['edit_distance'] for c in ns)/sum(len(c['expected']) for c in ns)
 summary[mode]['runs']=len(ts);summary[mode]['calls']=[len(t['calls']) for t in ts]
summary['model_sha256']=sha(model);summary['threads']='2 intra / 1 inter per session';summary['environment']=platform.platform()
(directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
E=lambda x:html.escape(str(x))
page=['<!doctype html><meta charset="utf-8"><title>PP-OCRv6 Tiny cell и batch 6</title><style>body{font:15px system-ui;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:8px}img{max-width:260px}pre{white-space:pre-wrap}</style><h1>PP-OCRv6 Tiny Cyrillic: cell и batch[6]</h1><p>17 ячеек из трёх проблемных чеков, исходные ROI и одинаковое разбиение на строки. Честное попарное сравнение: подготовка изображений, упаковка и вызов OCR включены. Разогрев обоих режимов, затем три измеряемых повтора с чередованием порядка. Пакет содержит до шести строк OCR, которые могут относиться к разным ячейкам и чекам. Это отдельный REC-бенчмарк, не полное время обработки чека. Эталоны предварительные.</p><pre>'+E(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><table><tr><th>Ячейка и фото</th><th>Эталон</th><th>Cell</th><th>Batch[6]</th></tr>']
if measured:
 a=next(x for x in measured if x['mode']=='cell');b=next(x for x in measured if x['mode']=='batch6')
 for x,y in zip(a['cells'],b['cells']):
  page.append('<tr><td>'+E(x['id'])+'<br><img src="../../'+E(x['image'])+'"></td><td>'+E(x['expected'])+'</td><td>'+E(x['text'])+'</td><td>'+E(y['text'])+'</td></tr>')
page.append('</table><p><a href="trials.json">Все попытки, этапы и результаты</a> · <a href="summary.json">Сводка</a></p>')
(directory/'report.html').write_text(''.join(page),encoding='utf-8')
print('REPORT',directory/'report.html',json.dumps(summary,ensure_ascii=False),flush=True)
