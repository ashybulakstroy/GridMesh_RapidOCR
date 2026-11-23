"""Self-contained portable HTML comparison with transparent ms and transcripts."""
from __future__ import annotations
import html,json,statistics
from pathlib import Path
from collections import defaultdict

def esc(v):return html.escape('' if v is None else str(v),quote=True)
def med(values):return round(statistics.median(values),3) if values else None
def fmt(v,suf=''):return '—' if v is None else (f'{v:.2f}{suf}' if isinstance(v,float) else str(v)+suf)

def build_report(dest,inputs,specs,model_init,rows,metadata,complete=True):
    rows_done=[r for r in rows if r['phase']=='measured']
    groups=defaultdict(list)
    for r in rows_done:groups[r['model_id']].append(r)
    cards=[];summary=[]
    for spec in specs:
        items=groups.get(spec['id'],[])
        ok=[r for r in items if r['status']=='OK']
        failures=[r for r in items if r['status']!='OK']
        name_ok=[r for r in ok if r['cell_type']=='name']
        good= len(ok)>0
        run_times=[]
        for p in (2,3):
            rp=[r for r in items if r['pass']==p]
            if len(rp)==17 and all(r['status']=='OK' for r in rp):
                run_times.append(round(sum(r['ocr_ms'] for r in rp),3))
        name_times=[r['ocr_ms'] for r in name_ok]
        score={
            'id':spec['id'],'name':spec['name'],'measurements':len(ok),'expected_measurements':34,
            'error_count':len(failures),'median_cell_ms':med([r['ocr_ms'] for r in ok]),
            'median_name_ms':med(name_times),'run_17_ms':med(run_times),
            'exact_names':sum(bool(r.get('exact')) for r in name_ok),'total_names':len(name_ok),
            'cer_names':round(sum(r.get('edit_distance',0) for r in name_ok)/max(1,sum(len(r['expected']) for r in name_ok)),4) if name_ok else None,
            'host_cpu_percent_median':med([r['host_cpu_percent'] for r in ok if r.get('host_cpu_percent') is not None]),
            'proc_cpu_percent_median':med([r['process_cpu_percent_one_core'] for r in ok if r.get('process_cpu_percent_one_core') is not None]),
            'model_init_ms': next((m['init_ms'] for m in model_init if m['id']==spec['id']),None),
            'status':next((m['status'] for m in model_init if m['id']==spec['id']),'PENDING')
        }
        summary.append(score)
    available=[x for x in summary if x['run_17_ms'] is not None]
    fastest=min((x['run_17_ms'] for x in available),default=None)
    qualified=[x for x in summary if x['run_17_ms'] is not None and x['measurements']==34 and x['error_count']==0]
    if qualified:
        fastest_item=min(qualified,key=lambda x:x['run_17_ms'])
        accurate_items=[x for x in qualified if x['cer_names'] is not None]
        best_accuracy=min(accurate_items,key=lambda x:(x['cer_names'],x['run_17_ms'])) if accurate_items else None
        insight=(f'Самый быстрый полный запуск: {esc(fastest_item["name"])} — '
                 f'{fmt(fastest_item["run_17_ms"]," мс")} на 17 ячеек. ')
        if best_accuracy:
            insight+=(f'Лучшая предварительная точность названий: {esc(best_accuracy["name"])} — '
                      f'CER {fmt(best_accuracy["cer_names"]*100," %")}, '
                      f'{fmt(best_accuracy["run_17_ms"]," мс")} на 17 ячеек. ')
            if best_accuracy['id']!=fastest_item['id']:
                insight+=f'Разница времени между ними: {fmt(best_accuracy["run_17_ms"]-fastest_item["run_17_ms"]," мс")}. '
        insight+='Выбор для приложения требует проверки на дополнительных чеках; 11 названий недостаточно для окончательной оценки.'
    else:
        insight='Пока нет ни одного участника с двумя полными измеренными проходами по 17 ячейкам. Победитель не определён.'
    table_rows=''
    for x in summary:
        best= x['run_17_ms']==fastest and fastest is not None
        table_rows+='<tr'+(' class="best"' if best else '')+'>' + ''.join('<td>'+esc(val)+'</td>' for val in (
            x['name'],x['status'],fmt(x['model_init_ms'],' мс'),fmt(x['median_name_ms'],' мс'),
            fmt(x['run_17_ms'],' мс'),f"{x['exact_names']}/{x['total_names']}",
            (fmt(100*x['cer_names'],' %') if x['cer_names'] is not None else '—'),
            fmt(x['host_cpu_percent_median'],' %'),fmt(x['proc_cpu_percent_median'],' %'),
            f"{x['measurements']}/34",x['error_count'] ))+'</tr>'
    details=''
    for s in specs:
        sid=s['id'];own=[r for r in rows_done if r['model_id']==sid]
        if not own:continue
        details+=f'<h3>{esc(s["name"])}</h3><table><thead><tr><th>Ячейка</th><th>Оригинал</th><th>Эталон (предв.)</th><th>Распознано (проходы 2/3)</th><th>Время, мс</th><th>Ошибки / примечания</th></tr></thead><tbody>'
        for item in inputs:
            matched=sorted((r for r in own if r['cell_id']==item['id']),key=lambda r:r['pass'])
            if not matched:continue
            texts='<div>'.join(f'#{r["pass"]}: {esc(r["text"])}' for r in matched)
            times=' / '.join(fmt(r.get('ocr_ms')) for r in matched)
            issues='; '.join(esc(r['error']) for r in matched if r.get('error')) or f"редакций: {', '.join(str(r.get('edit_distance','—')) for r in matched)}"
            path='../../'+item['file']
            details+=f'<tr><td>{esc(item["id"])}<small>{esc(item["width"])}×{esc(item["height"])}px</small></td><td><img loading="lazy" class="crop" src="{esc(path)}"></td><td>{esc(item["expected"])}</td><td class="ocrtext">{texts}</td><td>{esc(times)}</td><td>{issues}</td></tr>'
        details+='</tbody></table>'
    info=json.dumps(summary,ensure_ascii=False,indent=2)
    css='''
:root{color-scheme:light;--ink:#102034;--muted:#607288;--line:#d9e0e8;--blue:#145da0;--bg:#f5f7fa}
*{box-sizing:border-box}body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 6px}h2{font-size:19px;margin:32px 0 12px}h3{font-size:17px;margin:24px 0 8px}
p{margin:8px 0}.meta{color:var(--muted)}.stats{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}.stats article{background:white;border:1px solid var(--line);border-radius:10px;padding:14px 18px;min-width:170px}
.stats strong{font-size:23px;display:block}.stats small{color:var(--muted)}
table{width:100%;border-collapse:collapse;min-width:990px;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}th{background:#e9eff6;position:sticky;top:0;z-index:1}tbody tr:nth-child(even){background:#f9fbfd}tbody tr.best{background:#e7f4ff}td small{display:block;color:var(--muted)}
.scroll{overflow:auto;background:white;border-radius:10px;border:1px solid var(--line)}.crop{max-width:240px;max-height:115px;object-fit:contain;image-rendering:auto}.ocrtext{min-width:200px;overflow-wrap:anywhere}.warning{background:#fff8e6;border:1px solid #f1dca9;padding:12px;border-radius:8px}
a{color:var(--blue)}.controls{display:flex;gap:10px;margin:14px 0}.controls button{padding:7px 11px;border:1px solid var(--line);border-radius:6px;background:white;cursor:pointer}
'''
    html_doc=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark2 · Кириллический OCR</title><style>{css}</style></head>
<body><main><h1>Benchmark2 — сравнение OCR для чеков Торгсофт</h1>
<p class="meta">Запуск: {esc(metadata['run_id'])} · 6 движков · 17 исходных ячеек · три прохода (первый прогрев) · один CPU, без GPU</p>
<p class="warning">{'Измерения завершены' if complete else 'Предварительный отчёт: часть моделей ещё не выполнена'}. Эталонные тексты проверены по фото, но НЕ утверждены пользователем. Время и точность на этом ПК, а не универсальный показатель. Многострочные названия разбиваются одинаково для всех моделей; входные оригинальные ROI никогда не масштабируются.</p>
<div class="stats"><article><strong>{len(available)}/6</strong><small>моделей с полными замерами</small></article><article><strong>{fmt(fastest,' мс')}</strong><small>быстрейшие 17 ячеек, медиана двух проходов</small></article><article><strong>{sum(1 for r in rows_done if r['status']=='OK')}/{6*2*17}</strong><small>успешных измеренных ячеек</small></article></div>
<h2>Анализ результатов</h2><p>{insight}</p><h2>Итог: миллисекунды и точность</h2><div class="scroll"><table><thead><tr><th>Модель</th><th>Статус</th><th>Загрузка</th><th>Медиана названия</th><th>17 ячеек</th><th>Точные названия</th><th>CER названий</th><th>CPU всего</th><th>CPU процесса*</th><th>Результаты</th><th>Ошибки</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<p class="meta">* CPU процесса: 100% = одно логическое ядро; показатель может быть выше 100%. CPU всего — периодические снимки загрузки компьютера; на очень коротких операциях приблизителен. Время загрузки модели и подготовки изображений отделено от времени OCR.</p>
<h2>Все распознанные тексты и изображения</h2><div class="scroll">{details or '<p>Нет измеренных результатов</p>'}</div>
<h2>Методика</h2><p>Каждая модель получает те же 17 ROI, извлечённые из трёх оригинальных JPEG с проверкой SHA256. Первые 17 ячеек прогревают модель, затем два прохода измеряют время, CPU, память, текст и посимвольную дистанцию Левенштейна. Дополнительных ускоряющих настроек CPU или модели нет. Ошибки не заменяются результатом другой модели.</p>
<p class="meta">Исходные данные: <code>inputs.json</code>; все проходы: <code>raw.jsonl</code> и <code>raw.csv</code>; среда: <code>environment.json</code>; статус: <code>status.json</code>.</p></main></body></html>'''
    (dest/'report.html').write_text(html_doc,encoding='utf-8')
    (dest/'summary.json').write_text(info,encoding='utf-8')
    return summary
