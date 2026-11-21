"""Compare the latest fold and compact-number changes on saved receipt photos."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import App.geometry.service as geometry
import App.ocr.numeric_roi as numeric_roi
from App.ocr.p1_numbers import parse_number
from App.pipeline import process


def canonical(text):
    value = parse_number(text)
    return format(value, 'f') if value is not None else None


def snapshot(result):
    rows = []
    for row in result['rows']:
        fields = {}
        for name in ('quantity', 'price', 'printed_total'):
            if name in row['cells']:
                cell = row['cells'][name]
                fields[name] = {'text':cell['text'], 'value':canonical(cell['text']),
                                'confidence':cell.get('confidence'),
                                'roi':cell.get('numeric_roi'),
                                'source_bbox':cell.get('source_bbox')}
        rows.append({'row_id':row['row_id'], 'kind':row['kind'], 'fields':fields})
    return {'job_id':result['job_id'], 'rows':rows, 'total':result['total'],
            'computed_total':result['computed_total'], 'math_match':result['math_match'],
            'vat':result['vat'], 'fold_lines':result['geometry'].get('fold_recovered_horizontal', 0),
            'batch':result['geometry']['ocr_batch']}


def main():
    out = ROOT / 'data/output/digit_1/numeric_regression_2026_11_20'
    out.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT/'data/output/idea_2/corpus_comparison.json').read_text(encoding='utf-8'))
    candidates = [(r['hash'], r['job']) for r in catalog if r['mode'] == 'old']
    candidates.extend([('1120', 'a6554fd1ae10408eaab92afc41017191'),
                       ('long', '5e781cb12b4d487cbe68c031adab0c7b'),
                       ('fold', '9ba2114e651441408fd620c8d4d762cb')])
    real_fold = geometry.recover_fold
    real_compact = numeric_roi._compact_number_bounds
    seen = set()
    comparisons = []
    for label, job in candidates:
        path = ROOT/'data/output'/job/'input.bin'
        photo = path.read_bytes()
        digest = hashlib.sha256(photo).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        pair = {}
        for mode in ('before', 'after'):
            geometry.recover_fold = (lambda image, data: (data, 0)) if mode == 'before' else real_fold
            numeric_roi._compact_number_bounds = (lambda gray: None) if mode == 'before' else real_compact
            try:
                pair[mode] = snapshot(process(photo))
            except Exception as exc:
                pair[mode] = {'error':f'{type(exc).__name__}: {exc}'}
        item = {'label':label, 'source_job':job, 'sha256':digest, **pair}
        comparisons.append(item)
        print(f"{label}: before={pair['before'].get('computed_total')} "
              f"after={pair['after'].get('computed_total')} "
              f"fold={pair['after'].get('fold_lines')} "
              f"errors={[v['error'] for v in pair.values() if 'error' in v]}", flush=True)
        (out/'results.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding='utf-8')
    geometry.recover_fold = real_fold
    numeric_roi._compact_number_bounds = real_compact


if __name__ == '__main__':
    main()
