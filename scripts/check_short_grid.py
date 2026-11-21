"""Compare the short-grid optimization against the pre-change geometry code."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import App.geometry.geometry_lab.physical_grid_graph as graph
import App.geometry.geometry_lab.physical_grid_graph_local as local
import App.geometry.geometry_lab.physical_grid_graph_v3 as detector
from App.ocr.p1_numbers import parse_number
from App.pipeline import process


def old_functions(path, names, namespace):
    source = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT).decode('utf-8')
    functions = {node.name:node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    for name in names:
        exec(ast.get_source_segment(source, functions[name]), namespace)


def value(text):
    number = parse_number(text)
    return format(number.normalize(), 'f') if number is not None else None


def run(photo):
    result = process(photo)
    stage = next(item for item in result['performance']['samples'] if item['name']=='Поиск физической сетки')
    rows = []
    for row in result['rows']:
        fields = {}
        for field in ('quantity', 'price', 'printed_total'):
            if field in row['cells']:
                cell = row['cells'][field]
                fields[field] = {'text':cell['text'], 'value':value(cell['text']),
                                 'source_bbox':cell.get('source_bbox')}
        rows.append({'kind':row['kind'], 'fields':fields})
    directory = ROOT/result['job_directory']
    return {'job_id':result['job_id'], 'search_ms':stage['wall_ms'],
            'total_ms':result['performance']['wall_ms'], 'method':result['geometry']['method'],
            'printed_total':result['total'], 'computed_total':result['computed_total'],
            'math_match':result['math_match'], 'vat':result['vat'], 'rows':rows,
            'skipped':result['geometry']['ocr_batch']['numeric_roi_skipped'],
            'overlay_sha256':hashlib.sha256((directory/'rectified.png').read_bytes()).hexdigest()}


def main():
    directory = ROOT/'data/output/digit_1/short_grid_51040'
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(vars(local))
    old_functions('App/geometry/geometry_lab/detection.py', ['_candidate_lines','_group_and_fit'], env)
    old_functions('App/geometry/geometry_lab/physical_grid_graph.py', ['connected_fragments'], env)
    old_functions('App/geometry/geometry_lab/physical_grid_graph_local.py',
                  ['table_local_verticals','trace_seed','detect_physical_graph'], env)
    current = (detector.v2, detector.connected_fragments, detector._complete_short_grid)
    def select(mode):
        if mode == 'before':
            detector.v2 = lambda image, ridge=None: env['detect_physical_graph'](image)
            detector.connected_fragments = env['connected_fragments']
            detector._complete_short_grid = lambda data, report: False
        else:
            detector.v2, detector.connected_fragments, detector._complete_short_grid = current

    sources = {
        'short_51040':'App/tests/fixtures/receipt_short_51040.jpg',
        'fold_36981':'App/tests/fixtures/receipt_fold_36981.jpg',
        'long_538936':'App/tests/fixtures/receipt_long_grid.png',
        'receipt_32596':'App/tests/fixtures/receipt_32596.jpg',
        'receipt_29464':'App/tests/fixtures/receipt_29464.jpg',
        'six_columns':'App/tests/receipt_six_columns.jpg',
        'tax_below_grid':'data/output/5dbab730f56d4a959d6b0bd0e32d4ad4/input.bin',
    }
    records = []
    try:
        for name, path in sources.items():
            photo = (ROOT/path).read_bytes()
            count = 5 if name == 'short_51040' else 1
            if name == 'short_51040':
                for mode in ('before','after'):
                    select(mode)
                    run(photo)  # Warm the models and detector before measurements.
            for repeat in range(count):
                for mode in ('before','after'):
                    select(mode)
                    result = run(photo)
                    records.append({'photo':name,'source':path,'repeat':repeat,'mode':mode,**result})
                    print(name,repeat,mode,'search_ms',result['search_ms'],
                          'total',result['computed_total'],'match',result['math_match'],flush=True)
                    (directory/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        detector.v2, detector.connected_fragments, detector._complete_short_grid = current


if __name__ == '__main__':
    main()
