from pathlib import Path
import json, html, statistics, sys
from collections import defaultdict
from benchmark.core.measure import levenshtein, canonical_text

def main(folder):
 folder=Path(folder); rows=[json.loads(x) for x in (folder/'raw.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; ok=[x for x in rows if x.get('status')=='OK']; E=lambda x:html.escape(str(x)); groups=defaultdict(list)
 for r in ok: groups[(r['model_id'],r['mode'])].append(r)
 lines=['''<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR — ошибки по ячейкам</title><style>body{font:15px system-ui;margin:2em auto;max-width:1600px;padding:0 1em;color:#182435}table{border-collapse:collapse;width:100%;margin:1em 0 2em;font-size:13px}td,th{border:1px solid #ccd6df;padding:7px;text-align:left;vertical-align:top}th{background:#e8eff7}tr.bad{background:#fff0ef}.warn{background:#fff2c4;padding:1em}details{margin:1em 0}code{white-space:nowrap}</style><h1>OCR: результаты по каждой ячейке</h1><p><a href="report.html">Основной отчёт: медианы по каждому чеку и режиму</a></p><p>3 чека × 4 модели × 2 режима × 3 забега = 72 попытки; прогрев отдельно. Время включает чтение изображения, подготовку ROI и OCR в независимом бенчмарке, не полное время приложения.</p><p class="warn">CER = расстояние Левенштейна после нормализации пробелов / длина нормализованного эталона; меньше — лучше, возможно более 100%. Показаны все названия, включая черновые и модели без поддержки кириллицы: такие значения предварительные и не равнозначны подтверждённой оценке. Числовая оценка использует текущий парсер стенда, включая обработку символа валюты; часть расхождений может быть вызвана разметкой ROI и подготовкой изображения до OCR.</p><h2>Сводка по модели и режиму</h2><table><tr><th>Модель</th><th>Режим</th><th>Забегов</th><th>Числовые поля: верно / всего</th><th>CER всех названий</th><th>Медиана времени одного чека, мс</th></tr>''']
 for (model,mode),rs in sorted(groups.items()):
  correct=sum(r['quality']['numeric_correct'] for r in rs); total=sum(r['quality']['numeric_total'] for r in rs); edits=chars=0
  for r in rs:
   for c in r['quality']['cell_metrics']:
    if c['field'].startswith('row:') and c['field'].endswith(':name'):
     edits+=levenshtein(c['expected'],c['observed']); chars+=len(canonical_text(c['expected']))
  lines.append(f'<tr><td>{E(model)}</td><td>{E(mode)}</td><td>{len(rs)}</td><td>{correct}/{total} ({100*correct/total:.2f}%)</td><td>{100*edits/max(1,chars):.2f}%</td><td>{statistics.median(r["receipt_total_wall_ms"] for r in rs):.2f}</td></tr>')
 lines.append('</table><p>Сводная медиана выше рассчитана по всем трём размерам чеков; для сравнения скорости одинаковых чеков используйте основной отчёт.</p>')
 for (model,mode),rs in sorted(groups.items()):
  lines.append(f'<h2>{E(model)} — {E(mode)}</h2>')
  for r in sorted(rs,key=lambda x:(x['receipt_id'],x['repeat'])):
   lines.append(f'<details><summary>{E(r["receipt_id"])} — забег {r["repeat"]}, {r["receipt_total_wall_ms"]:.1f} мс, чисел {r["quality"]["numeric_correct"]}/{r["quality"]["numeric_total"]}</summary><table><tr><th>Ячейка</th><th>Эталон</th><th>Распознано</th><th>Результат / CER</th><th>Статус эталона</th></tr>')
   for c in r['quality']['cell_metrics']:
    if c['type']=='text' and not(c['field'].startswith('row:') and c['field'].endswith(':name')): continue
    if c['type']=='numeric': result='совпало' if c['correct'] else 'ОШИБКА / проверить'; bad=not c['correct']; status='черновой эталон'
    else:
     d=levenshtein(c['expected'],c['observed']); n=len(canonical_text(c['expected'])); result=f'CER {100*d/max(1,n):.2f}% ({d}/{n})'; bad=d>0; status=c['status']
    lines.append('<tr'+(' class="bad"' if bad else '')+'>'+''.join('<td>'+E(v)+'</td>' for v in [c['field'],c['expected'],c['observed'],result,status])+'</tr>')
   lines.append('</table></details>')
 lines.append('</html>'); (folder/'cell_details.html').write_text('\n'.join(lines),encoding='utf-8'); print('written',folder/'cell_details.html','records',len(rows),'ok',len(ok))
if __name__=='__main__': main(sys.argv[1])
