"""Measure the real control receipt before/after DEBUG instrumentation."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from App.pipeline import process
from App.config import setting

PHOTO='data/output/7a0d379e961547828dd608a2cbb74f4a/input.bin'
DEST=ROOT/'data/experiment_reports/gridmesh_debug_performance_v2'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['baseline','instrumented'],required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    levels=['INFO'] if args.phase=='baseline' else ['INFO','DEBUG']
    records=[]
    data=(ROOT/PHOTO).read_bytes()
    for repeat in range(args.repeats+1):
        for level in (levels if repeat%2==0 else levels[::-1]):
            os.environ['LOG_LEVEL']=level
            start=time.perf_counter();result=process(data)
            perf=result['performance'];debug=perf.get('debug')
            fingerprint=[{'row_id':r['row_id'],'kind':r['kind'],'cells':{f:c['text'] for f,c in r['cells'].items()}} for r in result['rows']]
            record=dict(repeat=repeat,warmup=repeat==0,level=level,job_id=result['job_id'],wall_s=time.perf_counter()-start,
                timings=result['timings'],process_cpu_ms=perf['process_cpu_ms'],total=result['total'],computed_total=result['computed_total'],math_match=result['math_match'],
                rows=fingerprint,issues=perf.get('issues'),has_debug=debug is not None)
            if debug:
                record['debug_summary']={key:len(debug[key]) for key in ['ocr_cells','ocr_batches','ocr_lanes','geometry_substeps','issues']}
                record['memory_mb']=debug['memory_mb']
            records.append(record)
            (DEST/(args.phase+'.json')).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
            print(level,repeat,record['timings'],record['math_match'],record.get('debug_summary'),flush=True)
    print('Saved:',DEST/(args.phase+'.json'))

if __name__=='__main__':main()
