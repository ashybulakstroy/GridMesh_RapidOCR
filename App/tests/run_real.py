import concurrent.futures
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run(name):
    from App.pipeline import process
    try:
        data = process((ROOT/'App'/'tests'/name).read_bytes())
        (ROOT/'data/output'/f'{Path(name).stem}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        return {k:data[k] for k in ('status','geometry','total','computed_total','timings','rows')}
    except Exception as exc:
        return {'error':str(exc)}

if __name__=='__main__':
    names=['receipt_575866.png','receipt_wheels.png']
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,names))
    (ROOT/'data/output/real_results.json').write_text(json.dumps(dict(zip(names,results)),ensure_ascii=False,indent=2),encoding='utf-8')
    for n,r in zip(names,results):
        print(n, r.get('error') or (r['status'],r['total'],r['computed_total'],r['timings'],r['geometry']['warnings']),flush=True)
