r"""Run OCR on user's Windows .venv, compare legacy and focused footer paths.

Usage: .venv\Scripts\python.exe scripts\footer_line_guided_v1\compare.py --rounds 3
"""
from __future__ import annotations
import argparse
from decimal import Decimal
from html import escape
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from App.pipeline import process
from App.ocr.p1_numbers import parse_number


def perform(image_bytes, mode):
    os.environ['OCR_FOOTER_FOCUS_ENABLE']='1' if mode=='focused' else '0'
    started=perf_counter()
    result=process(image_bytes)
    trace=result.get('rec_trace',{}).get('events',[])
    primary=[event for event in trace if 'receipt_total' in event.get('cells',[])
             and event.get('stage','').startswith('printed_footer_')]
    attempts=result.get('footer_ocr_attempts',[])
    focus=result.get('footer_focus') or {}
    return {'mode':mode,'total':result.get('total'),
            'math_match':result.get('math_match'),
            'printed_vat':result.get('printed_vat'),
            'first_ready_ms':(result.get('e4_adaptive') or {}).get('first_elapsed_ms'),
            'total_wall_ms':round((perf_counter()-started)*1000,2),
            'ocr_primary_footer_calls':len(primary),
            'ocr_primary_footer_call_ms':round(sum(float(x.get('wall_ms') or 0) for x in primary),3),
            'ocr_primary_events':primary,
            'foot_attempt_count':len(attempts),
            'foot_attempts':[{k:v for k,v in x.items() if k in ('strategy','total','confidence','source_roi_bbox','status','wall_ms')} for x in attempts],
            'footer_focus':focus}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--photo',type=Path,default=HERE/'fixtures'/'receipt_90665.png')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--expected-total',default='90665')
    parser.add_argument('--report',type=Path,default=HERE/'compare_results.json')
    args=parser.parse_args()
    if args.rounds<1 or args.rounds>30:parser.error('rounds: 1..30')
    photo=args.photo.read_bytes()
    runs=[]
    # Warm up both models/paths; then alternate to avoid one-sided cache effects.
    for mode in ('baseline','focused'):
        perform(photo,mode)
    for i in range(args.rounds):
        order=('baseline','focused') if i%2==0 else ('focused','baseline')
        for mode in order:
            result=perform(photo,mode)
            runs.append(result)
            print(f"{mode:8s} {result['total']!s:>12s} "
                  f"calls={result['ocr_primary_footer_calls']:<2} "
                  f"footerOCR={result['ocr_primary_footer_call_ms']:7.2f}ms "
                  f"wall={result['total_wall_ms']:8.2f}ms "
                  f"first={result['first_ready_ms']}ms")
    expected=parse_number(args.expected_total)
    report={'photo':str(args.photo),'expected_total':str(expected),
            'source_commit':'5c1e94980ec8eb32ff96b5267dd83f34c055aa30',
            'rounds':args.rounds,'runs':runs,'summary':{}}
    for mode in ('baseline','focused'):
        subset=[x for x in runs if x['mode']==mode]
        report['summary'][mode]={
            'median_total_wall_ms':median(x['total_wall_ms'] for x in subset),
            'median_footer_ocr_ms':median(x['ocr_primary_footer_call_ms'] for x in subset),
            'median_footer_calls':median(x['ocr_primary_footer_calls'] for x in subset),
            'all_totals_match_photo':all(parse_number(x['total'])==expected for x in subset),
            'all_matrix_math_match':all(x['math_match'] for x in subset),
            'median_first_ready_ms':median(x['first_ready_ms'] for x in subset if x['first_ready_ms'] is not None)
                 if any(x['first_ready_ms'] is not None for x in subset) else None,
        }
    base=report['summary']['baseline'];new=report['summary']['focused']
    report['summary']['delta']={'footer_ocr_ms':round(new['median_footer_ocr_ms']-base['median_footer_ocr_ms'],2),
                                'total_wall_ms':round(new['median_total_wall_ms']-base['median_total_wall_ms'],2),
                                'footer_calls':new['median_footer_calls']-base['median_footer_calls']}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Human-readable comparison with every run and the OCR fallback diagnosis.
    summary=report['summary']; rows_html=[]
    for row in runs:
        focus=row['footer_focus'] or {}
        rows_html.append('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
            row['mode'],row['total'],row['ocr_primary_footer_calls'],
            row['ocr_primary_footer_call_ms'],row['total_wall_ms'],
            row['first_ready_ms'],focus.get('status','—'),
            'yes' if focus.get('accepted') else 'no')) + '</tr>')
    summary_html=[]
    for key in ('median_footer_calls','median_footer_ocr_ms','median_total_wall_ms',
                'median_first_ready_ms','all_totals_match_photo'):
        summary_html.append(f'<tr><td>{escape(key)}</td><td>{escape(str(summary["baseline"].get(key)))}</td>'
                            f'<td>{escape(str(summary["focused"].get(key)))}</td></tr>')
    html=f"""<!doctype html><html lang='ru'><meta charset='utf-8'>
<title>Эксперимент Footer Focus V1</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:30px auto;padding:0 22px;color:#172b45}}
table{{width:100%;border-collapse:collapse;margin:20px 0}}th,td{{padding:10px;border:1px solid #ced8e4}}
th{{background:#eef4ff}}.small{{color:#66788f}}</style>
<h1>Footer Focus V1 · сравнение OCR печатного итога</h1>
<p>Контрольный итог на фотографии: <strong>{escape(str(expected))} ₸</strong></p>
<h2>Медиана по {args.rounds} запускам</h2><table><tr><th>Метрика</th><th>Базовый</th><th>Эксперимент</th></tr>
{''.join(summary_html)}</table>
<p>Разница focused − baseline: OCR итога <strong>{summary['delta']['footer_ocr_ms']} мс</strong>,
полное время <strong>{summary['delta']['total_wall_ms']} мс</strong>, вызовы OCR <strong>{summary['delta']['footer_calls']}</strong>.</p>
<h2>Проверка каждого запуска</h2><table><tr><th>Режим</th><th>Итог</th><th>Вызовы</th><th>OCR итога, мс</th>
<th>Весь чек, мс</th><th>Первый, мс</th><th>Статус ROI</th><th>Accepted</th></tr>
{''.join(rows_html)}</table>
<p class='small'>Результат OCR является кандидатом; математическое совпадение не заменяет проверки фотографии.
Длительности OCR-вызовов могут перекрываться с OCR других частей чека.</p>
</html>"""
    html_path=args.report.with_suffix('.html')
    html_path.write_text(html,encoding='utf-8')
    print('Saved HTML:',html_path)
    print('\nSummary:',json.dumps(report['summary'],ensure_ascii=False,indent=2))
    print('Saved:',args.report)
    if not new['all_totals_match_photo']:
        print('WARNING: focused OCR did not consistently read the photographed footer.')
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
