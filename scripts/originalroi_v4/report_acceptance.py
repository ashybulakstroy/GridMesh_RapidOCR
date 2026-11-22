"""Render actual V4 acceptance measurements without running OCR."""
import json,html,statistics,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'data/experiment_reports/OriginalROI_CandidateMatrix_V4/acceptance'


def main():
    initial=json.loads((DIRECTORY/'problem_comparison.json').read_text(encoding='utf-8'))
    repeats_path=DIRECTORY/'problem_repeats.json'
    repeats=json.loads(repeats_path.read_text(encoding='utf-8')) if repeats_path.exists() else []
    esc=lambda x:html.escape(str(x))
    table=[];summary=[]
    for item in initial:
        samples=[item]+[r for r in repeats if r['photo']==item['photo']]
        a=item['results']['baseline'];b=item['results']['v4']
        before=statistics.median(r['results']['baseline']['timings']['total_s'] for r in samples)
        after=statistics.median(r['results']['v4']['timings']['total_s'] for r in samples)
        changes=sum(len(r['changed']) for r in samples)
        auto=sum(len(r['results']['v4']['corrections'].get('changes',[])) for r in samples)
        numbers_identical=all(r['results']['baseline']['numbers']==r['results']['v4']['numbers'] for r in samples)
        link=os.path.relpath(ROOT/'data/output'/b['job_id']/'original.png',DIRECTORY).replace('\\','/')
        label=a['total']+' ₸' if '/input.bin' in item['photo'].replace('\\','/') else Path(item['photo']).name
        vals=[a['computed_total'],b['computed_total'],b['total'],a['math_match'],b['math_match'],len(samples),f'{before:.3f}',f'{after:.3f}',changes,auto,b['matrix']]
        table.append('<tr><td><a href="'+esc(link)+'">'+esc(label)+'</a></td>'+''.join('<td>'+esc(v)+'</td>' for v in vals)+'</tr>')
        summary.append(dict(photo=item['photo'],repeats=len(samples),baseline_median_s=before,v4_median_s=after,selected_numbers_identical=numbers_identical,auto_corrections=auto,baseline_math_match=a['math_match'],v4_math_match=b['math_match']))
    bad=[r for r in summary if not r['baseline_math_match']]
    before=sum(r['baseline_median_s'] for r in summary);after=sum(r['v4_median_s'] for r in summary)
    result=dict(photos=len(summary),problem_photos=len(bad),paired_trials=sum(r['repeats'] for r in summary),automatic_corrections=sum(r['auto_corrections'] for r in summary),unchanged_selected_numbers=all(r['selected_numbers_identical'] for r in summary),problem_cases_fixed=sum(r['v4_math_match'] for r in bad),sum_medians_baseline_s=before,sum_medians_v4_s=after,rows=summary)
    (DIRECTORY/'acceptance_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    headings=['Фото / прочитанный итог','Расчёт до','Расчёт V4','OCR итога V4','Суммы совпали до','Суммы совпали V4','Парных повторов','До, median с','V4, median с','Изменений чисел','Автоправок','Матрица']
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка OriginalROI Candidate Matrix V4</title><style>body{font:15px system-ui;max-width:1500px;margin:30px auto;padding:15px;color:#17293a}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #cbd4df;padding:9px;text-align:left}th{background:#edf3fa}.note{background:#fff1d4;padding:15px}.scroll{overflow:auto}</style><h1>OriginalROI Candidate Matrix V4 — реальные проблемные фото</h1>'''
    page+=f'<p class="note">Улучшение точности выбранных чисел не подтверждено: исправлено {result["problem_cases_fixed"]} из {len(bad)} случаев с несовпадением итога. Автоправок: {result["automatic_corrections"]}. Проверено {len(summary)} фото, {result["paired_trials"]} парных прогонов. V4 оставлена за OCR_E4_ENABLE=0.</p>'
    page+='<p>Контроль — pipeline.py из резервной копии до интеграции V4. Условия: текущие локальные модели, intra_op=2, inter_op=1, OpenCV=2, одинаковые фото и бюджеты 12/16 мс. Порядок до/V4 чередовался; модели повторно использовались. Для шести проблемных фото — по три парных повтора; для трёх контрольных — по одному. Это ограниченная проверка на текущем ПК, а не доказательство точности на любых чеках.</p>'
    page+='<div class="scroll"><table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headings)+'</tr>'+''.join(table)+'</table></div>'
    page+=f'<p>Сумма медиан обработки девяти фото: до {before:.3f} с, V4 {after:.3f} с ({(after/before-1)*100:+.1f}%). Загрузка вторичных моделей может влиять на холодные запуски.</p>'
    page+='<h2>Визуально проверенные примеры</h2><ul><li>Фото 62 060 ₸: на оригинале напечатано «62 060,00»; обе версии оставили OCR итога 6206000, при сумме строк 62060.</li><li>Фото 58 174 ₸: на оригинале 5 товарных строк и НДС, а обработка возвращает 4 товара и НДС. Повторное чтение существующих ячеек не восстанавливает пропущенную строку сетки.</li><li>Фото 252 161 ₸: строки 15 и 16 по-прежнему 5 × 2000 и 20 × 60; улучшение предыдущего исправления сохранено.</li></ul><p>Совпадение суммы само по себе не считается доказательством правильности всех полей. Полная точность всех цифр не заявляется без посимвольной разметки каждого фото. Исходные JSON, время, модельные кандидаты и номера заданий доступны рядом с отчётом.</p></html>'
    (DIRECTORY/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
    print(DIRECTORY/'report.html')

if __name__=='__main__':main()
