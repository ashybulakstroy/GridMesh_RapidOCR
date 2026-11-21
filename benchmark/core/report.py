"""Offline HTML report using ONLY local recorded measurements."""
import collections,csv,html,json,statistics
from pathlib import Path
from .measure import summarize


def write_report(folder):
    folder=Path(folder)
    raw=folder/'raw.jsonl'
    records=[json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines() if line.strip()] if raw.exists() else []
    ok=[r for r in records if r.get('status')=='OK']
    with (folder/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['receipt_id','model_id','mode','repeat','status','receipt_total_wall_ms','ocr_loop_wall_ms','sum_cell_ocr_ms','sum_batch_ocr_ms','ocr_calls','processed_rois','processed_fields','cpu_ms','cpu_core_equivalents','peak_rss_sampled_mb','process_cpu_percent_one_core_basis','numeric_correct','numeric_total','text_cer','text_quality_status','error']
        wr=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');wr.writeheader()
        for r in records:
            row=dict(r);row.update({k:v for k,v in r.get('quality',{}).items() if k in cols});wr.writerow(row)
    groups=collections.defaultdict(list)
    for r in ok:groups[(r['receipt_id'],r['model_id'],r['mode'])].append(r)
    aggregates=[]
    for (receipt,model,mode),rows in sorted(groups.items()):
        sm=lambda key:summarize([float(r[key]) for r in rows if r.get(key) is not None])
        text_cers=[r['quality']['text_cer'] for r in rows if r.get('quality',{}).get('text_cer') is not None]
        aggregates.append({'receipt_id':receipt,'model_id':model,'mode':mode,'trials':len(rows),'receipt_total_wall_ms':sm('receipt_total_wall_ms'),'ocr_loop_wall_ms':sm('ocr_loop_wall_ms'),
                           'sum_cell_ocr_ms':sm('sum_cell_ocr_ms'),'sum_batch_ocr_ms':sm('sum_batch_ocr_ms'),
                           'numeric_accuracy':round(sum(r['quality']['numeric_correct'] for r in rows)/max(1,sum(r['quality']['numeric_total'] for r in rows)),6),
                           'text_cer':summarize(text_cers)})
    comparisons=[]
    keyed={(a['receipt_id'],a['model_id'],a['mode']):a for a in aggregates}
    for (receipt,model,mode),row in sorted(keyed.items()):
        if mode!='single_cell':continue
        batch=keyed.get((receipt,model,'batch'))
        if not batch:continue
        va=row['receipt_total_wall_ms']['median'];vb=batch['receipt_total_wall_ms']['median']
        comparisons.append({'receipt_id':receipt,'model_id':model,'single_wall_median_ms':va,'batch_wall_median_ms':vb,
                            'measured_speedup_single_over_batch':round(va/vb,4) if vb else None,'time_saved_ms':round(va-vb,3),
                            'single_sum_calls_median_ms':row['sum_cell_ocr_ms']['median'],'batch_sum_calls_median_ms':batch['sum_batch_ocr_ms']['median']})
    (folder/'errors.json').write_text(json.dumps([r for r in records if r.get('status')!='OK'],ensure_ascii=False,indent=2),encoding='utf-8')
    result={'source':'LOCAL_ACTUAL_MEASUREMENTS_ONLY','records':len(records),'successful_trials':len(ok),'failed_trials':len(records)-len(ok),'aggregates':aggregates,'comparisons':comparisons}
    (folder/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def n(x):return 'NOT_MEASURED' if x is None else str(x)
    def cell(x):return '<td>'+html.escape(n(x))+'</td>'
    trs=[]
    for a in aggregates:
        trs.append('<tr>'+''.join(cell(v) for v in [a['receipt_id'],a['model_id'],a['mode'],a['trials'],a['numeric_accuracy'],a['text_cer']['median'] if a['text_cer'] else None,a['sum_cell_ocr_ms']['median'] if a['sum_cell_ocr_ms'] else None,
                        a['sum_batch_ocr_ms']['median'] if a['sum_batch_ocr_ms'] else None,a['ocr_loop_wall_ms']['median'],a['receipt_total_wall_ms']['median'],a['receipt_total_wall_ms']['p95']])+'</tr>')
    comp=[]
    for a in comparisons:
        comp.append('<tr>'+''.join(cell(a[k]) for k in ['receipt_id','model_id','single_wall_median_ms','batch_wall_median_ms','time_saved_ms','measured_speedup_single_over_batch','single_sum_calls_median_ms','batch_sum_calls_median_ms'])+'</tr>')
    failures=''.join('<li>'+html.escape(r.get('model_id','')+' / '+r.get('receipt_id','')+' / '+r.get('mode','')+' / '+r.get('error','ERROR'))+'</li>' for r in records if r.get('status')!='OK')
    page="""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Локальный OCR-бенчмарк</title><style>body{font:15px Arial,sans-serif;max-width:1400px;margin:30px auto;color:#182435;padding:0 18px}table{border-collapse:collapse;font-size:13px;width:100%;margin:12px 0 28px}td,th{border:1px solid #dbe2ec;padding:9px;text-align:left;vertical-align:top}th{background:#edf2f9}h1,h2{color:#173861}.fine{color:#556477}.note{border-left:3px solid #607d9c;padding-left:12px}</style><h1>OCR Benchmark — фактические результаты текущего ПК</h1><p class="note">В таблице только измерения из локального <code>raw.jsonl</code>. Никаких сторонних бенчмарков. Медиана/P95 — по реально выполненным повторам; ошибки не скрываются.</p><h2>Время и качество по чекам</h2><table><thead><tr><th>Чек</th><th>Модель</th><th>Режим</th><th>Забеги</th><th>Точность чисел</th><th>CER текста</th><th>Σ вызовов cell, мс</th><th>Σ batch, мс</th><th>OCR loop, мс</th><th>Весь чек (median), мс</th><th>Весь чек (P95), мс</th></tr></thead><tbody>"""+"".join(trs)+"""</tbody></table><h2>Прямое сравнение single_cell vs batch</h2><table><thead><tr><th>Чек</th><th>Модель</th><th>Single весь чек, мс</th><th>Batch весь чек, мс</th><th>Разница, мс</th><th>Ускорение x</th><th>Σ cell OCR, мс</th><th>Σ batch OCR, мс</th></tr></thead><tbody>"""+"".join(comp)+"""</tbody></table><p class="fine">Важно: сумма OCR-вызовов НЕ равна полному времени. Время декодирования изображения, кропов и Python overhead измерено отдельно. OCR-вызовы действительно выполняются по одному в режиме single_cell. Числовые ROI автоматически обрезаны по цифрам; многострочные имена разбиты на строки и склеены с устранением переноса. CER текста считается только для проверенных полей; approved_by_user=false означает, что финальное утверждение владельцем ещё не выставлено.</p><h2>Ошибки / прерванные попытки</h2><ul>"""+failures+"""</ul></html>"""
    manifest_path=folder/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if any(r.get('crop_source') == 'gridmesh_total' for r in records):
        page = page.replace('<h2>Время и качество', '<p class="note"><b>Grid Mesh + итог (по разметке).</b> Товарные ячейки сопоставлены по центрам ручных ROI. Итог — отдельная прямоугольная синтетическая ячейка по ручной ROI; автоматический поиск итога и его кривизна не проверялись. Полное время включает повторное построение сетки в каждом забеге.</p><h2>Время и качество')
        if manifest.get('geometry_timing_policy'):
            page=page.replace('Полное время включает повторное построение сетки в каждом забеге.', 'Grid Mesh строится только при отсутствии сохранённых координат, до измерений. Время OCR исключает геометрию и подготовку вырезок. Весь чек включает чтение фото, загрузку координат и подготовку вырезок.')
        total_rows=[]
        for (receipt,model,mode),rows in sorted(groups.items()):
            metrics=[m for r in rows for m in r['quality']['cell_metrics'] if m['field']=='printed_total']
            total_rows.append('<tr>'+''.join(cell(v) for v in [receipt,model,mode,
                metrics[0]['expected'] if metrics else None,
                ' | '.join(sorted({m['observed'] for m in metrics})),
                f"{sum(m['correct'] for m in metrics)}/{len(metrics)}"] )+'</tr>')
        page=page.replace('</html>','<h2>Печатный итог отдельно</h2><table><tr><th>Чек</th><th>Модель</th><th>Режим</th><th>Эталон</th><th>Распознано</th><th>Верно / повторов</th></tr>'+''.join(total_rows)+'</table></html>')
    from .quality_report import quality_sections
    page=page.replace('</html>',quality_sections(records)+'</html>')
    (folder/'report.html').write_text(page,encoding='utf-8')
    return result
