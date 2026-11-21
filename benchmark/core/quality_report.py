"""Separate text and numeric views of recorded OCR measurements."""
import collections
import html


def quality_sections(records):
    groups=collections.defaultdict(list)
    for r in records:
        if r.get('quality'):
            groups[(r.get('stage','—'),r['model_id'],r['receipt_id'],r['mode'],
                    r.get('intra_op_threads',r.get('onnx_intra_threads','—')),
                    r.get('opencv_threads','—'))].append(r)
    def table(headers,rows):
        cell=lambda tag,v:f'<{tag}>'+html.escape(str(v))+f'</{tag}>'
        return '<table><tr>'+''.join(cell('th',x) for x in headers)+'</tr>'+''.join(
            '<tr>'+''.join(cell('td',v) for v in row)+'</tr>' for row in rows)+'</table>'
    categories=[('Количество товаров',lambda f:f.startswith('row:') and f.endswith(':quantity')),
                    ('Цена товара',lambda f:f.startswith('row:') and f.endswith(':unit_price')),
                    ('НДС: числовые поля',lambda f:f.startswith('vat:')),
                    ('Напечатанный итог',lambda f:f=='printed_total')]
    text_rows=[];numeric_rows=[];details=[]
    for key,rs in sorted(groups.items(),key=lambda item:str(item[0])):
        metrics=[c for r in rs for c in r['quality']['cell_metrics']]
        names=[c for c in metrics if c['field'].startswith('row:') and c['type']=='text']
        scored=[c for c in names if c.get('char_edits') is not None]
        if scored:
            chars=sum(len(' '.join(str(c['expected']).split())) for c in scored)
            text_rows.append([*key,len(rs),f"{len(scored)}/{len(names)}",
                              round(sum(c['char_edits'] for c in scored)/max(1,chars),6),
                              f"{sum(c['char_edits']==0 for c in scored)}/{len(scored)}"])
        elif names and not all(c.get('status')=='UNSUPPORTED_SCRIPT' for c in names):
            text_rows.append([*key,len(rs),f'0/{len(names)}','Не оценено: эталон не проверен','—'])
        fields=[]
        for field in sorted({c['field'] for c in metrics}):
            samples=[c for c in metrics if c['field']==field]
            fields.append([field,samples[0]['expected'],' | '.join(sorted({c['observed'] for c in samples})),
                           ' | '.join(sorted({str(c.get('char_edits',c.get('correct'))) for c in samples}))])
        details.append('<details><summary>'+html.escape(' / '.join(map(str,key)))+'</summary>'+table(['Поле','Эталон','Распознано','Ошибок символов / верно'],fields)+'</details>')
        # numeric_only never appears as a successful text benchmark.
        nums=[c for c in metrics if c['type']=='numeric']
        counts=[]
        for _,predicate in categories:
            selected=[c for c in nums if predicate(c['field'])]
            counts.append(f"{sum(c['correct'] for c in selected)}/{len(selected)}")
        numeric_rows.append([*key,len(rs),*counts,
                             round(sum(c['correct'] for c in nums)/len(nums),6) if nums else '—',
                             f"{sum(r['quality'].get('arithmetic_matches_printed_total',False) for r in rs)}/{len(rs)}"
                             if all('arithmetic_matches_printed_total' in r['quality'] for r in rs) else 'Не измерено'])
    common=['Этап','Модель','Чек','Режим','Intra-op','OpenCV','Повторов']
    return ('<section id="text-benchmark"><h2>Текстовый бенчмарк — названия товаров</h2>'
            '<p>Смешанная кириллица и латиница, артикулы, цифры и знаки. CER считается только по названиям, без строки НДС. '
            'Числовые модели в этот блок не включаются. Счётчики суммируются по измеряемым повторам.</p>'+
            (table(common+['Оценено названий','CER названий','Точных совпадений'],text_rows) if text_rows else '<p>Текстовых измерений пока нет.</p>')+
            '</section><section id="numeric-benchmark"><h2>Числовой бенчмарк — товары и итоги</h2>'
            '<p>Здесь сравниваются только числа, независимо от результатов названий. В ячейках — верно / проверено. '
            'Арифметическая согласованность сама по себе не доказывает совпадение с эталоном.</p>'+
            table(common+[name for name,_ in categories]+['Точность чисел','Арифметика итога верна'],numeric_rows)+
            '</section><section><h2>Подробности распознавания полей</h2>'+''.join(details)+'</section>')
