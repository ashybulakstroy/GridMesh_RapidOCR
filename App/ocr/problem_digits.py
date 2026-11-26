"""Debug-only photographic evidence; never run OCR or infer corrected digits."""
import html
import json
from pathlib import Path
import cv2
import math
from .p1_numbers import parse_number


def _review_reasons(record, *, math_match=True):
    """Explain recorded review triggers, without guessing why a model erred."""
    reasons=[]
    if record.get('reason')=='auto_corrected':
        reasons.append('Значение было автоматически исправлено.')
    if record.get('reason')=='vat_not_verified':
        reasons.append('Отдельная проверка НДС не дала числового результата.')
    cell=record.get('diagnostic_cell') or {}
    if cell and parse_number(cell.get('text')) is None:
        reasons.append('Выбранный текст не удалось прочитать как число.')
    if cell.get('numeric_issue'):
        reasons.append(str(cell['numeric_issue']))
    if cell.get('confidence') is not None and float(cell['confidence'])<.8:
        reasons.append('У выбранного чтения низкая уверенность OCR (меньше 0,8).')
    if not math_match and record.get('field') not in ('receipt_total','printed_vat'):
        reasons.append('Сумма строк не совпала с итогом чека; ячейка включена для общей проверки. Это не доказывает ошибку этой ячейки.')
    attempts=record.get('recognition_attempts',[])
    if any(not a.get('success',True) for a in attempts):
        reasons.append('Одна из попыток OCR завершилась ошибкой.')
    if any(any(float(c)<.8 for c in a.get('confidences',[])) for a in attempts):
        reasons.append('У одной из попыток низкая уверенность OCR (меньше 0,8).')
    if any(any(s in a.get('stage','') for s in ('retry','recovery','second_pass')) for a in attempts):
        reasons.append('Для поля выполнялось повторное чтение или восстановление.')
    values=set()
    for a in attempts:
        if a.get('success',True) and len(a.get('cells',[]))<=1:
            for text in a.get('texts',[]):
                value=parse_number(text)
                if value is not None:values.add(value)
    for obs in record.get('observations',[]):
        value=parse_number(obs.get('number'))
        if value is not None:values.add(value)
    for attempt in record.get('footer_attempts',[]):
        value=parse_number(attempt.get('total'))
        if value is not None:values.add(value)
    if len(values)>1:
        reasons.append('Попытки распознавания дали разные числовые значения.')
    if any(obs.get('number') is None for obs in record.get('observations',[])):
        reasons.append('Одно из сохранённых наблюдений не удалось прочитать как число.')
    return reasons or ['Поле включено в отчёт для проверки; более точная причина в записи не сохранена.']


def render_problem_report(directory, payload, *, math_match=True):
    """Render saved records; works with both current and older manifests."""
    E=lambda x:html.escape(str(x))
    page=['<!doctype html><meta charset="utf-8"><title>Цифры для проверки</title><style>body{font:16px system-ui;margin:25px}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}table{border-collapse:collapse}img{max-width:400px;max-height:160px}pre{white-space:pre-wrap}figure{margin:0 0 18px}figcaption{overflow-wrap:anywhere}.reason{background:#fff6df;padding:10px}.ocr-result{display:block;font-weight:600}</style><h1>Цифры для проверки</h1><p>Попадание в отчёт означает необходимость проверки, а не обязательно ошибку распознавания.</p><p>'+E(payload.get('note',''))+'</p><p>'+E(payload.get('coordinate_note',''))+'</p><table><tr><th>Строка / поле</th><th>Результат</th><th>Фото</th><th>OCR-наблюдения</th></tr>']
    for r in payload.get('cells',[]):
        by_image={a.get('input_image'):a for a in r.get('recognition_attempts',[]) if a.get('input_image')}
        pictures=''
        for label,path in r.get('images',{}).items():
            event=by_image.get(path)
            result=''
            if event:
                texts=event.get('texts')
                if len(event.get('cells',[]))>1:
                    shown='Общий пакет; результаты относятся ко всему пакету: '+(', '.join(map(str,texts)) if texts else 'нет сохранённого текста')
                else:
                    shown=', '.join(map(str,texts)) if texts else ('Ошибка OCR' if not event.get('success',True) else 'Текст не сохранён')
                result='<span class="ocr-result">Распознано: '+E(shown)+'</span>'
            pictures+='<figure><img src="'+E(path)+'"><figcaption>'+E(label)+result+'</figcaption></figure>'
        label='Печатный итог чека' if r['field']=='receipt_total' else 'Напечатанный НДС' if r['field']=='printed_vat' else f"{r['row_id']} / {r['field']}"
        values=(f"Автоматическое исправление: {r['before']} → {r['after']}" if r['reason']=='auto_corrected'
                else 'Отдельная проверка напечатанного НДС не дала числового результата.' if r['reason']=='vat_not_verified'
                else f"Выбранное значение: {r.get('after')}. Требуется проверка.")
        reasons=''.join('<li>'+E(reason)+'</li>' for reason in _review_reasons(r,math_match=math_match))
        attempts='<h3>Последовательность распознавания</h3><table><tr><th>Попытка</th><th>Модель / этап</th><th>Время, мс</th><th>Прочитано</th></tr>'
        for index,e in enumerate(r.get('recognition_attempts',[]),1):
            texts=e.get('texts')
            shown=', '.join(map(str,texts)) if texts else ('Ошибка OCR' if not e.get('success',True) else 'Текст не сохранён')
            if len(e.get('cells',[]))>1:shown+=' (общий пакет)'
            attempts+='<tr>'+''.join('<td>'+E(v)+'</td>' for v in (index,str(e.get('model_id','—'))+' / '+str(e.get('stage','—')),e.get('wall_ms','—'),shown))+'</tr>'
        attempts+='</table><p>'+E(r.get('timing_note',''))+'</p>'
        if not r.get('recognition_attempts'):attempts+='<p>Отдельные попытки OCR не сохранены.</p>'
        if r['field']=='receipt_total':
            attempts='<p><strong>Выбранный итог: '+E(r.get('after','—'))+'</strong></p>'+attempts
        raw='<details><summary>Подробные сохранённые наблюдения</summary><pre>'+E(json.dumps(r.get('observations',[]),ensure_ascii=False,indent=2))+'</pre></details>'
        page.append('<tr><td>'+E(label)+'</td><td>'+E(values)+'</td><td>'+pictures+'</td><td><div class="reason"><strong>Почему поле проверяется:</strong><ul>'+reasons+'</ul></div>'+attempts+raw+'</td></tr>')
    (Path(directory)/'report.html').write_text(''.join(page)+'</table>',encoding='utf-8')


def save_problem_digits(job_dir, rows, crops, book, corrections, original_crops=None, primary_inputs=None, footer_context=None,
                        rec_trace=None, attempt_inputs=None, vat_context=None, math_match=True):
    cells={(r['row_id'],f):c for r in rows for f,c in r['cells'].items()}
    changes={(x['row_id'],x['field']):x for x in corrections.get('changes',[])}
    keys=set(changes)
    events=(rec_trace or {}).get('events',[])
    for key,cell in cells.items():
        if key[1] not in ('quantity','price','printed_total'):continue
        if (parse_number(cell.get('text')) is None or cell.get('numeric_issue') or cell.get('auto_corrected') or not math_match):
            keys.add(key)
    for event in events:
        if (not event.get('success',True) or any(x in event.get('stage','') for x in ('retry','recovery','second_pass','e5_tesseract'))
                or (len(event.get('cells',[])) == 1 and event.get('texts') and any(parse_number(text) is None for text in event['texts']))):
            for key in event.get('cells',[]):
                if ':' in key:
                    rid,field=key.split(':',1)
                    if rid.isdigit() and field in ('quantity','price','printed_total'):keys.add((int(rid),field))
                elif key=='receipt_total':keys.add((None,'receipt_total'))
    vat_observations=book.get('vat',[])
    vat_source=(vat_context or {}).get('source') or {}
    vat_check_not_run=(vat_context or {}).get('source_status') == 'not_run'
    vat_missing=(not vat_check_not_run and any(row.get('kind')=='tax' for row in rows)
                 and not any(x.get('number') is not None for x in vat_observations))
    if vat_missing:
        keys.add((None,'printed_vat'))
    if (len({x['number'] for x in vat_observations if x.get('number') is not None})>1
            or any(x.get('number') is None for x in vat_observations)):
        if vat_source.get('row_id') is not None and vat_source.get('field'):
            keys.add((vat_source['row_id'],vat_source['field']))
        else:keys.add((None,'printed_vat'))
    footer_observations=book.get('footer',[])
    if (len({x['number'] for x in footer_observations if x.get('number') is not None})>1
            or any(x.get('number') is None for x in footer_observations)
            or (footer_context is not None and footer_context.get('total') is None)):
        keys.add((None,'receipt_total'))
    for bucket in book.get('cells',{}).values():
        observations=bucket.get('observations',[])
        if len({x['number'] for x in observations if x.get('number') is not None})>1:
            keys.add((bucket['row_id'],bucket['field']))
        elif any(x.get('number') is None for x in observations):
            keys.add((bucket['row_id'],bucket['field']))
    keys={key for key in keys if key[1] in ('quantity','price','printed_total','receipt_total','printed_vat')}
    if not keys:
        return {'count':0, 'report':None, 'manifest':None, 'log_entries':[]}
    directory=Path(job_dir)/'problem_digits'
    directory.mkdir(exist_ok=True)
    records=[]
    for rid,field in sorted(keys,key=lambda x:(x[0] is None,x[0] or 0,x[1])):
        if field not in ('quantity','price','printed_total','receipt_total','printed_vat'):continue
        change=changes.get((rid,field))
        cell=cells.get((rid,field),{})
        reason=('auto_corrected' if change else
                'vat_not_verified' if field=='printed_vat' and vat_missing and not vat_observations else
                'requires_review')
        record={'row_id':rid,'field':field,'reason':reason,
                'before':change.get('before') if change else None,
                'after':change.get('after') if change else cell.get('text'),
                'source_bbox':cell.get('source_bbox'), 'numeric_roi':cell.get('numeric_roi'),
                'observations':book.get('cells',{}).get(f'{rid}:{field}',{}).get('observations',[]),
                'images':{}}
        record['diagnostic_cell']={key:cell.get(key) for key in ('text','confidence','numeric_issue') if key in cell}
        trace_key='receipt_total' if field=='receipt_total' else 'printed_vat' if field=='printed_vat' else f'{rid}:{field}'
        record['recognition_attempts']=[{**e,'duration_scope':'shared_batch' if len(e.get('cells',[]))>1 else 'single_cell_or_detector_pipeline'}
                                        for e in events if trace_key in e.get('cells',[])]
        record['timing_note']='Время из rec_trace; общий пакет не делится между ячейками. DET+REC включает поиск текста. Отсутствующее время не заменяется нулём.'
        for event in record['recognition_attempts']:
            captured=(attempt_inputs or {}).get(event.get('event_id'))
            if isinstance(captured,dict):captured=captured.get(trace_key)
            if captured is not None and captured.size:
                filename=f"attempt_{event['event_id']}_{rid}_{field}.png"
                if not cv2.imwrite(str(directory/filename),captured):raise OSError(filename)
                event['input_image']=filename
                record['images'][f"attempt_{event['event_id']}: {event['model_id']} · {event['wall_ms']} мс"]=filename
        if (rid,field)==(vat_source.get('row_id'),vat_source.get('field')):
            record['vat_observations']=vat_observations
        if field=='printed_vat':
            record['observations']=vat_observations
            record['images']['full_receipt']='../original.png'
        if field=='receipt_total':
            record['images']['full_receipt']='../original.png'
            record['observations']=footer_observations
            context=footer_context or {}
            if not change:record['after']=context.get('total')
            record['footer_attempts']=context.get('attempts',[])
            record['footer_regions']=[]
            source=context.get('image')
            attempts=[{'strategy':'selected_footer_lines','lines':context.get('lines',[])}]+context.get('attempts',[])
            for i,attempt in enumerate(attempts):
                lines=attempt.get('lines',[])
                if not lines and attempt.get('source_roi_bbox'):
                    a,b,c,d=attempt['source_roi_bbox']
                    lines=[{**attempt,'bbox':[[a,b],[c,b],[c,d],[a,d]]}]
                for j,line in enumerate(lines):
                    box=line.get('bbox')
                    if source is None or not source.size or not box:continue
                    x0=max(0,math.floor(min(p[0] for p in box)));x1=min(source.shape[1],math.ceil(max(p[0] for p in box)))
                    y0=max(0,math.floor(min(p[1] for p in box)));y1=min(source.shape[0],math.ceil(max(p[1] for p in box)))
                    if x1<=x0 or y1<=y0:continue
                    filename=f'footer_attempt_{i}_line_{j}.png'
                    if not cv2.imwrite(str(directory/filename),source[y0:y1,x0:x1]):raise OSError(filename)
                    label=f"source_region_{i}_{j}: {attempt.get('strategy','detector')} / {line.get('text','')}"
                    record['images'][label]=filename
                    record['footer_regions'].append({'image':filename,'strategy':attempt.get('strategy'),
                        'text':line.get('text'),'total':attempt.get('total'), 'source_bbox':[x0,y0,x1,y1],
                        'note':'Вырезка исходного фото по сохранённым координатам; не повторный OCR и не реконструкция преобразований.'})
        elif field!='printed_vat':
            image=crops.get((rid,field))
            if image is not None and image.size:
                filename=f'row_{rid}_{field}_cell.png'
                if not cv2.imwrite(str(directory/filename),image):raise OSError(filename)
                record['images']['gridmesh_cell']=filename
                # Reconstruct the exact primary region from its recorded
                # cell-local bounds; no new detection or OCR is run here.
                roi=cell.get('numeric_roi') or {}
                bbox=roi.get('ocr_roi_bbox')
                if bbox is None:
                    x,y=roi.get('offset',[0,0]);w,h=roi.get('size',[image.shape[1],image.shape[0]])
                    bbox=[x,y,x+w,y+h]
                x0,y0,x1,y1=map(int,bbox)
                if 0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]:
                    filename=f'row_{rid}_{field}_recorded_roi.png'
                    if not cv2.imwrite(str(directory/filename),image[y0:y1,x0:x1]):raise OSError(filename)
                    record['images']['recorded_roi']=filename
            primary=(primary_inputs or {}).get((rid,field))
            if primary is not None and primary.size:
                filename=f'row_{rid}_{field}_primary_ocr_input.png'
                if not cv2.imwrite(str(directory/filename),primary):raise OSError(filename)
                record['images']['primary_ocr_input']=filename
            original=(original_crops or {}).get((rid,field))
            if original is not None and original.size:
                filename=f'row_{rid}_{field}_original.png'
                if not cv2.imwrite(str(directory/filename),original):raise OSError(filename)
                record['images']['original_pixels']=filename
            if not record['images']:record['image_status']='crop_unavailable'
        records.append(record)
    payload={'coordinate_note':'gridmesh_cell: полная ячейка, может содержать валюту. primary_ocr_input: фактический вход первого основного OCR до внутреннего масштабирования движком. recorded_roi: реконструкция по итоговым метаданным, может относиться к повторному чтению. source_bbox: фото; numeric_roi: пиксели ячейки. Не показаны все резервные попытки OCR.',
             'note':'Коррекция E4 и конфликт моделей — признаки для проверки, а не независимый эталон цифр.', 'cells':records}
    (directory/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    render_problem_report(directory,payload,math_match=math_match)
    log_entries=[]
    for record in records:
        label=('Итог чека' if record['field']=='receipt_total' else
               'Напечатанный НДС: отдельная проверка не дала результата' if record['field']=='printed_vat' and record['reason']=='vat_not_verified' else
               'Напечатанный НДС' if record['field']=='printed_vat' else f"Строка {record['row_id']} / {record['field']}")
        attempts=record['recognition_attempts']
        descriptions=[f"{x['model_id']} · {x['stage']} · {x['wall_ms']} мс"+
                      (' (общий пакет)' if x['duration_scope']=='shared_batch' else '') for x in attempts]
        log_entries.append({'type':'debug','stage':'Проблемные цифры','row_id':record['row_id'],
            'field':record['field'],'attempts':attempts,
            'message':label+'; '+('; '.join(descriptions) if descriptions else 'Время отдельных попыток не сохранено')})
    return {'count':len(records),'report':'problem_digits/report.html','manifest':'problem_digits/manifest.json',
            'log_entries':log_entries}
