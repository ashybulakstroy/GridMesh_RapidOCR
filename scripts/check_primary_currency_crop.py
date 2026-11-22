"""Paired problematic-receipt regression; normal application timings stay enabled."""
import os, sys, json, time, html, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from App.pipeline import process
from App.ocr.p1_numbers import parse_number
from App.ocr.numeric_roi import prepare_numeric_input
import cv2

out=ROOT/'data/experiment_reports/primary_currency_crop'
out.mkdir(parents=True,exist_ok=True)
photos=[('4810','data/output/1cf81ccb658e4614ab900cecdc746e5c/original.png'),
        ('fold_253309','App/tests/fixtures/receipt_253309.jpg'),
        ('short_51040','App/tests/fixtures/receipt_short_51040.jpg')]
trials=[]
for name,path in photos:
    previous=None
    for repeat in range(4):
        pair={}
        for mode in ([0,1] if repeat%2==0 else [1,0]):
            os.environ['OCR_PRICE_CURRENCY_CROP']=str(mode)
            start=time.perf_counter();r=process((ROOT/path).read_bytes());elapsed=(time.perf_counter()-start)*1000
            rows=[{'row':row['row_id'],'kind':row['kind'],'cells':{f:c['text'] for f,c in row['cells'].items()},
                   'price_roi':row['cells']['price'].get('numeric_roi')} for row in r['rows']]
            t=dict(photo=name,repeat=repeat,mode=mode,warmup=repeat==0,ms=elapsed,math=r['math_match'],
                   total=r['total'],computed=r['computed_total'],rows=rows,job=r['job_id'],
                   timings=r['timings'],stats=r['geometry']['ocr_batch'])
            trials.append(t);pair[mode]=t
            (out/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
            print(name,repeat,mode,round(elapsed),t['math'],t['total'],flush=True)
        def numbers(t):
            return [(row['row'],f,str(parse_number(v))) for row in t['rows'] for f,v in row['cells'].items() if f!='name']
        if numbers(pair[0])!=numbers(pair[1]) or (pair[0]['math'] and not pair[1]['math']):
            print('STOP: numeric change requires inspection',flush=True)
            break

os.environ['OCR_PRICE_CURRENCY_CROP']='1'
image=cv2.imread(str(ROOT/'App/tests/fixtures/price_4810_currency.png'))
crop,info=prepare_numeric_input(image,'price')
cv2.imwrite(str(out/'4810_before.png'),image);cv2.imwrite(str(out/'4810_ocr_input.png'),crop)
E=lambda x:html.escape(str(x))
page=['<!doctype html><meta charset="utf-8"><title>Вырезание валюты перед OCR</title><style>body{font:16px system-ui;margin:30px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}img{image-rendering:pixelated;max-width:600px}</style><h1>Валюта исключается из входа RapidOCR</h1><p>Три парных измерения после прогрева. Штатные тайминги включены. Совпадение чисел и сумм проверяет регрессию, но не заменяет независимый эталон всех цифр.</p><img src="4810_before.png"> → <img src="4810_ocr_input.png"><pre>'+E(json.dumps(info,ensure_ascii=False,indent=2))+'</pre><table><tr><th>Чек</th><th>Режим</th><th>Медиана полного времени, мс</th><th>Контроль сумм</th><th>Цены</th></tr>']
for name,_ in photos:
    for mode in (0,1):
        group=[t for t in trials if t['photo']==name and t['mode']==mode and not t['warmup']]
        if not group:continue
        vals=[name,'Новый' if mode else 'Прежний',round(statistics.median(t['ms'] for t in group),1),
              f"{sum(t['math'] for t in group)}/{len(group)}",', '.join(row['cells']['price'] for row in group[-1]['rows'])]
        page.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in vals)+'</tr>')
page.append('</table><p><a href="trials.json">Все поля, ROI и штатные тайминги каждого запуска</a></p>')
(out/'report.html').write_text(''.join(page),encoding='utf-8')
