"""Render saved before/INFO/DEBUG acceptance measurements, without OCR."""
import html,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'data/experiment_reports/gridmesh_debug_performance_v2'


def main():
    baseline=json.loads((DEST/'baseline.json').read_text(encoding='utf-8'))
    instrumented=json.loads((DEST/'instrumented.json').read_text(encoding='utf-8'))
    groups={'До установки, INFO':[r for r in baseline if not r['warmup']],
            'После установки, INFO':[r for r in instrumented if not r['warmup'] and r['level']=='INFO'],
            'После установки, DEBUG':[r for r in instrumented if not r['warmup'] and r['level']=='DEBUG']}
    summary={}
    reference=baseline[1]
    table=[]
    for name,rows in groups.items():
        value=dict(repeats=len(rows),median_wall_s=statistics.median(r['wall_s'] for r in rows),
            median_profile_s=statistics.median(r['timings']['total_s'] for r in rows),
            median_ocr_s=statistics.median(r['timings']['ocr_s'] for r in rows),
            wall_range_s=[min(r['wall_s'] for r in rows),max(r['wall_s'] for r in rows)],
            selected_fields_identical=all(r['rows']==reference['rows'] and r['total']==reference['total'] for r in rows),
            all_math_match=all(r['math_match'] for r in rows))
        summary[name]=value
        table.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [name,len(rows),round(value['median_wall_s'],3),value['median_ocr_s'],value['selected_fields_identical'],value['all_math_match']])+'</tr>')
    summary['debug_vs_info_wall_percent']=(summary['После установки, DEBUG']['median_wall_s']/summary['После установки, INFO']['median_wall_s']-1)*100
    summary['settings']={'intra_op':2,'inter_op':1,'opencv':2,'table_mode':'cell','e4':True,'photo':'7a0d379e961547828dd608a2cbb74f4a'}
    details=[]
    for row in instrumented:
        if row['warmup']:continue
        link='../../output/'+row['job_id']+'/performance_profile.json'
        details.append('<li><a href="'+link+'">'+row['level']+' / повтор '+str(row['repeat'])+' / performance_profile.json</a></li>')
    page='<html lang="ru"><meta charset="utf-8"><title>DEBUG Performance V2 — приёмка</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px}pre{white-space:pre-wrap}</style><h1>DEBUG Performance V2 — приёмка на реальном чеке</h1><p>Контрольный чек: 252161. По пять повторов после прогрева; после установки INFO и DEBUG чередуются. Выбранные тексты и числовые поля сравниваются с результатом до установки.</p><table><tr><th>Режим</th><th>Повторов</th><th>Полный цикл, median с</th><th>OCR, median с</th><th>Все поля совпали</th><th>Сумма совпала</th></tr>'+''.join(table)+'</table><p>Небольшая разница времени находится в диапазоне разброса этих запусков: это не доказательство ускорения DEBUG. Время ячеек включает подготовку вызова и декодирование. Потоки перекрываются, их длительности нельзя складывать в wall-time. RAM — максимум по контрольным точкам, не точный пик.</p><h2>Сохранённые профили</h2><ul>'+''.join(details)+'</ul><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>'
    (DEST/'acceptance_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
