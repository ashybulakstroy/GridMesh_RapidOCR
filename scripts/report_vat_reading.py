"""Render saved whole-application VAT regression runs without rerunning OCR."""
import html
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'data/experiment_reports/vat_reading_fix'
before=json.loads((out/'before.json').read_text(encoding='utf-8'))
after=json.loads((out/'after.json').read_text(encoding='utf-8'))
E=lambda x:html.escape(str(x))
page=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка чтения НДС</title><style>body{font:16px system-ui;max-width:1350px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>E4 VAT Reading Fix V1: проверка в основном проекте</h1><p>НДС берётся из реально прочитанной ячейки строки налога. Пустая «Сумма» больше не перекрывает количество/цену. После коррекции E4 источник обновляется с сохранением модели исходного наблюдения. Новые модели и OCR-вызовы не добавляются. Это исправление выбора и отображения свидетельства, а не повышение оптической точности модели.</p><p>Три измерения после прогрева каждого фото, текущие настройки 1+1 и Adaptive включены. Штатные тайминги сохранены в before.json / after.json. Режимы запускались отдельными процессами, сначала прежний, затем новый; небольшие различия времени могут быть шумом.</p><table><tr><th>Чек</th><th>НДС до → после</th><th>Полный цикл до → после, мс</th><th>Изменение времени</th><th>Все строки идентичны</th><th>Контроль сумм до / после</th></tr>']
summaries=[]
for name in dict.fromkeys(t['photo'] for t in before):
 a=[t for t in before if t['photo']==name and t['repeat']]
 b=[t for t in after if t['photo']==name and t['repeat']]
 same=all(x['result']['rows']==y['result']['rows'] for x,y in zip(a,b))
 ma,mb=median(t['ms'] for t in a),median(t['ms'] for t in b)
 s=dict(photo=name,before_ms=ma,after_ms=mb,change_percent=100*(mb/ma-1),rows_identical=same,
        before_vat=a[-1]['result']['printed_vat'],after_vat=b[-1]['result']['printed_vat'],
        source=b[-1]['result']['vat_evidence']['source'])
 summaries.append(s)
 vals=[name,f"{s['before_vat']} → {s['after_vat']}",f'{ma:.1f} → {mb:.1f}',f"{s['change_percent']:+.1f}%",same,
       f"{sum(t['result']['math_match'] for t in a)}/3 · {sum(t['result']['math_match'] for t in b)}/3"]
 page.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in vals)+'</tr>')
page.append('</table><h2>Чек Еврокуб: НДС 23200, итог 168200</h2><p>Реальный браузер на 8018 проверен: поле «Напечатанный НДС» содержит 23200, источник quantity, контроль сумм проходит, ошибок JavaScript нет. Исходные строки совпадают с прежним результатом. Согласованная арифметика не считается независимым чтением другой моделью.</p><img src="browser_final.png"><h2>Все поля: до → после</h2><table><tr><th>Чек / строка / поле</th><th>До</th><th>После</th></tr>')
for name in dict.fromkeys(t['photo'] for t in before):
 a=next(t['result'] for t in before if t['photo']==name and t['repeat']==3)
 b=next(t['result'] for t in after if t['photo']==name and t['repeat']==3)
 for r1,r2 in zip(a['rows'],b['rows']):
  for field,c in r1['cells'].items():
   page.append('<tr>'+''.join('<td>'+E(v)+'</td>' for v in (f"{name} / {r1['row_id']} / {field}",c['text'],r2['cells'][field]['text']))+'</tr>')
page.append('</table><p>Проверки: 23 теста эксперимента и 41 тест интеграции/повторных чтений/подготовки цены. Корпус из трёх проблемных чеков не доказывает отсутствие регрессий на любых изображениях.</p><p><a href="before.json">До: результаты и тайминги</a> · <a href="after.json">После: результаты и тайминги</a> · <a href="browser_result.json">Результат браузерной проверки</a></p></html>')
(out/'report.html').write_text(''.join(page),encoding='utf-8')
(out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summaries,ensure_ascii=False,indent=2))
