#!/usr/bin/env python3
"""Run the experiment on existing project receipts and write a JSON+CSV report.

Must be invoked *after* --apply, with the host repo available on PYTHONPATH.
The script neither fabricates ground truth nor reports accuracy without labels.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULTS=(
    'App/tests/receipt_5730.png',
    'App/tests/receipt_575866.png',
    'App/tests/receipt_597275.png',
    'App/tests/receipt_six_columns.jpg',
    'App/tests/fixtures/receipt_long_grid.png',
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--photos',nargs='*',help='Пути фото относительно корня проекта; по умолчанию 5 тестов')
    parser.add_argument('--output',type=Path,help='Относительный путь внутри data/experiment_reports')
    args=parser.parse_args()
    project=args.project.resolve()
    if not (project/'App/pipeline.py').is_file():
        raise SystemExit('Ошибка: укажите корень проекта с App/pipeline.py')
    sys.path.insert(0,str(project))
    os.environ["OCR_E4_ENABLE"]="1"
    from App.pipeline import process
    destination=(project/'data'/'experiment_reports'/'OriginalROI_CandidateMatrix_V4'/
                 datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.output:
        destination=(project/'data'/'experiment_reports'/args.output).resolve()
        if not destination.is_relative_to((project/'data'/'experiment_reports').resolve()):
            raise SystemExit('Папка отчётов должна быть внутри data/experiment_reports')
    destination.mkdir(parents=True,exist_ok=True)
    report=[]
    for photo in args.photos or DEFAULTS:
        filename=(project/photo).resolve()
        if not filename.is_relative_to(project) or not filename.is_file():
            report.append({'photo':str(photo),'error':'file_not_found_or_outside_project'})
            continue
        begin=time.perf_counter()
        try:
            result=process(filename.read_bytes())
            matrix=result.get('numeric_matrix',{})
            original=result.get('original_roi_experiment',{})
            ocr=result.get('geometry',{}).get('ocr_batch',{})
            row={'photo':str(photo),'status':result.get('status'),
                 'math_match':result.get('math_match'),
                 'computed_total':result.get('computed_total'),
                 'printed_total':result.get('total'),
                 'geometry_s':result.get('timings',{}).get('geometry_s'),
                 'ocr_s':result.get('timings',{}).get('ocr_s'),
                 'wall_s':round(time.perf_counter()-begin,3),
                 'cache_count':original.get('cache_cells'),
                 'cache_bytes':original.get('cache_bytes'),
                 'cache_prepare_ms':original.get('cache_prepare_ms'),
                 'original_rec_calls':original.get('rec_calls'),
                 'e2_models':result.get('geometry',{}).get('numeric_experiment',{}).get('model_ensemble',{}).get('models',{}),
                 'e2_model_calls':result.get('geometry',{}).get('numeric_experiment',{}).get('model_ensemble',{}).get('model_calls',0),
                 'original_rec_wall_ms':original.get('rec_wall_ms'),
                 'footer_variants':matrix.get('footer_variants'),
                 'vat_variants':matrix.get('vat_variants'),
                 'vat_ambiguous':matrix.get('vat_ambiguous'),
                 'vat_evidence_source':result.get('vat_evidence',{}).get('source_status'),
                 'vat_probe_calls':result.get('vat_evidence',{}).get('probes'),
                 'critical_ambiguity':matrix.get('critical_ambiguity'),
                 'matrix_risk_level':matrix.get('risk_level'),
                 'footer_ambiguous':matrix.get('footer_ambiguous'),
                 'matrix_status':matrix.get('status'),
                 'matrix_nodes':matrix.get('nodes_visited'),
                 'matrix_review':matrix.get('review_required'),
                 'suggestions':len(matrix.get('recommended_changes',[])),
                 'numeric_roi_skipped':ocr.get('numeric_roi_skipped'),
                 'experiment_file':result.get('numeric_experiment_file')}
            report.append(row)
        except Exception as exc:
            report.append({'photo':str(photo), 'wall_s':round(time.perf_counter()-begin,3),
                           'error':f'{type(exc).__name__}: {exc}'})
    (destination/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for row in report for k in row))
    with (destination/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in report:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})
    print(f'Отчёты: {destination}')
    for row in report:
        print(row['photo'], ':',row.get('matrix_status',row.get('error')),
              '| total:',row.get('printed_total'), '| wall_s:',row.get('wall_s'))
    return 1 if any('error' in row for row in report) else 0


if __name__=='__main__':
    raise SystemExit(main())
