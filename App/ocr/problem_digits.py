"""Debug-only photographic evidence; never run OCR or infer corrected digits."""
import html
import json
from pathlib import Path
import cv2
import math


def save_problem_digits(job_dir, rows, crops, book, corrections, original_crops=None, primary_inputs=None, footer_context=None):
    cells={(r['row_id'],f):c for r in rows for f,c in r['cells'].items()}
    changes={(x['row_id'],x['field']):x for x in corrections.get('changes',[])}
    keys=set(changes)
    footer_observations=book.get('footer',[])
    if (len({x['number'] for x in footer_observations if x.get('number') is not None})>1
            or any(x.get('number') is None for x in footer_observations)):
        keys.add((None,'receipt_total'))
    for bucket in book.get('cells',{}).values():
        observations=bucket.get('observations',[])
        if len({x['number'] for x in observations if x.get('number') is not None})>1:
            keys.add((bucket['row_id'],bucket['field']))
        elif any(x.get('number') is None for x in observations):
            keys.add((bucket['row_id'],bucket['field']))
    directory=Path(job_dir)/'problem_digits'
    directory.mkdir(exist_ok=True)
    records=[]
    for rid,field in sorted(keys,key=lambda x:(x[0] is None,x[0] or 0,x[1])):
        if field not in ('quantity','price','printed_total','receipt_total'):continue
        change=changes.get((rid,field))
        cell=cells.get((rid,field),{})
        record={'row_id':rid,'field':field,'reason':'auto_corrected' if change else 'conflicting_or_invalid_ocr',
                'before':change.get('before') if change else None,
                'after':change.get('after') if change else cell.get('text'),
                'source_bbox':cell.get('source_bbox'), 'numeric_roi':cell.get('numeric_roi'),
                'observations':book.get('cells',{}).get(f'{rid}:{field}',{}).get('observations',[]),
                'images':{}}
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
        else:
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
    E=lambda x:html.escape(str(x))
    page=['<!doctype html><meta charset="utf-8"><title>Проблемные цифры</title><style>body{font:16px system-ui;margin:25px}td,th{border:1px solid #ccc;padding:10px}table{border-collapse:collapse}img{max-width:400px;max-height:160px}pre{white-space:pre-wrap}</style><h1>Проблемные цифровые ячейки</h1><p>'+E(payload['note'])+'</p><p>'+E(payload['coordinate_note'])+'</p><table><tr><th>Строка / поле</th><th>До → после</th><th>Фото</th><th>OCR-наблюдения</th></tr>']
    for r in records:
        pictures=''.join('<figure><img src="'+E(p)+'"><figcaption>'+E(label)+'</figcaption></figure>' for label,p in r['images'].items())
        label='Печатный итог чека' if r['field']=='receipt_total' else f"{r['row_id']} / {r['field']}"
        values=f"{r['before']} → {r['after']}" if r['reason']=='auto_corrected' else f"Выбранное значение: {r['after']}. Разные варианты OCR — см. справа."
        page.append('<tr><td>'+E(label)+'</td><td>'+E(values)+'</td><td>'+pictures+'</td><td><pre>'+E(json.dumps(r['observations'],ensure_ascii=False,indent=2))+'</pre></td></tr>')
    (directory/'report.html').write_text(''.join(page)+'</table>',encoding='utf-8')
    return {'count':len(records),'report':'problem_digits/report.html','manifest':'problem_digits/manifest.json'}
