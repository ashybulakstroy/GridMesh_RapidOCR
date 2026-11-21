"""Measure thread settings in isolated processes using unmodified pipeline telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = {
    'small': 'App/tests/receipt_5730.png',
    'medium': 'App/tests/fixtures/receipt_29464.jpg',
    'long': 'App/tests/fixtures/receipt_long_grid.png',
}


def canonical(value):
    from App.ocr.p1_numbers import parse_number
    number = parse_number(value)
    return str(number.normalize()) if number is not None else None


def quality(result):
    return {
        'rows': [dict(kind=row['kind'], **{
            field: canonical(row['cells'][field]['text'])
            for field in ('quantity', 'price', 'printed_total') if field in row['cells']
        }) for row in result['rows']],
        'total': canonical(result['total']),
        'computed_total': canonical(result['computed_total']),
        'math_match': result['math_match'],
        'vat': {key: result['vat'].get(key) for key in ('status', 'amount', 'rate_percent')},
    }


def worker(args):
    import cv2
    import psutil
    from App.pipeline import process
    import App.ocr.engine as engines
    records = []
    order = list(SOURCES)
    for repeat in range(-1, args.repeats):
        for name in order[repeat % len(order):] + order[:repeat % len(order)]:
            result = process((ROOT / SOURCES[name]).read_bytes())
            performance = result['performance']
            assert performance['version'] == 'ocr-performance-v1'
            assert performance['samples'], 'Native timing telemetry must be enabled'
            records.append({
                'source': name, 'warmup': repeat == -1, 'repeat': repeat,
                'job_id': result['job_id'], 'quality': quality(result),
                'performance': performance, 'timings': result['timings'],
                'batch': result['geometry']['ocr_batch'],
            })
            print(f'{args.tag} {name} {repeat}: {performance["wall_ms"]:.0f} ms; '
                  f'match={result["math_match"]}', flush=True)
    sessions = {}
    for key in ('_engine', '_footer_engine', '_english_engine'):
        model = getattr(engines, key)
        if model is None:
            continue
        for stage in ('text_rec', 'text_det'):
            component = getattr(model, stage, None)
            if component is None:
                continue
            session = component.session.session
            opts = session.get_session_options()
            sessions[f'{key}.{stage}'] = {
                'intra': opts.intra_op_num_threads, 'inter': opts.inter_op_num_threads,
                'providers': session.get_providers(),
            }
            assert opts.intra_op_num_threads == args.onnx
    payload = {
        'tag': args.tag, 'onnx': args.onnx, 'opencv': args.opencv,
        'opencv_actual': cv2.getNumThreads(), 'sessions': sessions, 'records': records,
        'machine': {'platform': platform.platform(), 'cpu_logical': psutil.cpu_count(),
                    'cpu_physical_reported': psutil.cpu_count(logical=False),
                    'python': sys.version, 'memory_gb': psutil.virtual_memory().total / 2**30},
    }
    assert payload['opencv_actual'] == args.opencv
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--onnx', type=int, default=2)
    parser.add_argument('--opencv', type=int, default=2)
    parser.add_argument('--tag', default='baseline')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default=str(ROOT / 'data/output/cpu_threads_experiment'))
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configs = [(2,2), (1,1), (4,4), (2,4), (4,2), (6,6), (8,8), (10,10), (2,2)]
    env_hash = hashlib.sha256((ROOT/'.env').read_bytes()).hexdigest()
    for index, (onnx, opencv) in enumerate(configs):
        tag = f'{index:02d}_onnx{onnx}_cv{opencv}'
        env = os.environ.copy()
        env.update(OCR_ONNX_THREADS=str(onnx), OCR_OPENCV_THREADS=str(opencv))
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--onnx', str(onnx), '--opencv', str(opencv), '--tag', tag,
                   '--repeats', str(args.repeats), '--output', str(output / f'{tag}.json')]
        print(f'START {tag}', flush=True)
        with (output / f'{tag}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f'DONE {tag}', flush=True)
    assert hashlib.sha256((ROOT/'.env').read_bytes()).hexdigest() == env_hash
    manifest = {'sources': {name: {'path': path, 'sha256': hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
                            for name,path in SOURCES.items()}, 'env_sha256': env_hash,
                'profiles': [f'{i:02d}_onnx{a}_cv{b}' for i,(a,b) in enumerate(configs)]}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
