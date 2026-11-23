"""Benchmark2: 3 passes including warm-up, unmodified OCR cells, transparent diagnostics."""
from __future__ import annotations
import csv, datetime as dt, json, math, os, platform, statistics, sys, time, traceback
from pathlib import Path
from PIL import Image
from .prepare import ROOT, BASE, prepare, sha
from .models import prepare_models
from .adapters import make_adapter
from .preprocess import precompute, join_parts
from .report import build_report


def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)

def pctstring(done,total):
    pct=int(done/max(1,total)*100)
    return f"[{('█'*int(pct//4)).ljust(25,'░')}] {pct:3d}%"

def levenshtein(a,b):
    old=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        new=[i]+[0]*len(b)
        for j,d in enumerate(b,1):new[j]=min(new[j-1]+1,old[j]+1,old[j-1]+int(c!=d))
        old=new
    return old[-1]

def canonical(s):
    # Preserve case, punctuation, Cyrillic/Latin distinction. Normalize only whitespace.
    return ' '.join(str(s).split())

def metric(expected,observed):
    a=canonical(expected);b=canonical(observed)
    dist=levenshtein(a,b)
    return {'edit_distance':dist,'exact':a==b,'cer':dist/max(1,len(a))}

def report_env():
    import importlib.metadata as md
    names=['rapidocr','onnxruntime','tesserocr','winrt-Windows.Media.Ocr','opencv-python','numpy','Pillow','psutil']
    packages={}
    for name in names:
        try:packages[name]=md.version(name)
        except md.PackageNotFoundError:packages[name]='NOT_INSTALLED'
    return {'platform':platform.platform(),'processor':platform.processor(),'python':sys.version,
            'logical_cpus':os.cpu_count(),'packages':packages,'base_project':str(BASE)}

def run(*,offline=False,auto_prepare=True,progress_cb=None):
    try:import psutil
    except ImportError:psutil=None
    started=time.perf_counter()
    specs=json.loads((ROOT/'config/models.json').read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/'config/benchmark.json').read_text(encoding='utf-8'))
    if not all(s.get('cyrillic') for s in specs):raise ValueError('Non-Cyrillic model forbidden')
    if int(cfg['total_passes'])!=3 or int(cfg['warmup_passes'])!=1:raise ValueError('3 passes incl 1 warmup required')
    run_id=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    dest=ROOT/'results'/run_id; dest.mkdir(parents=True,exist_ok=False)
    (ROOT/'results'/'latest.txt').write_text(run_id,encoding='utf-8')
    samples=json.loads((ROOT/'data/samples.json').read_text(encoding='utf-8'))
    if len(samples)!=17 or len({s['id'] for s in samples})!=17:raise ValueError('Must have 17 unique ROIs')
    total=len(samples)*len(specs)*3
    status={'state':'preparing','started':dt.datetime.now().isoformat(timespec='seconds'),'run_id':run_id,
            'completed':0,'total':total,'percent':0,'model':'Preparing','pass':None,
            'elapsed_sec':0,'eta_sec':None,'cpu_percent':None,'process_cpu_percent_one_core':None,
            'results_dir':str(dest),'message':'Checking originals and input cells'}
    def update(**kwargs):
        status.update(kwargs)
        elapsed=time.perf_counter()-started
        status['elapsed_sec']=round(elapsed,2)
        done=status['completed']
        status['percent']=round(100*done/total,2)
        status['eta_sec']=round(elapsed/done*(total-done),1) if done else None
        write_json(dest/'status.json',status)
        write_json(ROOT/'results'/'status.json',status)
        if progress_cb:progress_cb(dict(status))
    update()
    try:
        if auto_prepare:
            prepare(download_missing=not offline,fetch_tessdata=True)
            prepare_models(download_missing=not offline)
        manifest=json.loads((ROOT/'data/cells_manifest.json').read_text(encoding='utf-8'))
        if [s['id'] for s in samples]!=[s['id'] for s in manifest]:
            raise ValueError('Cell manifest mismatch. Run python -m benchmark2.prepare')
        for s in manifest:
            if sha(ROOT/s['file'])!=s['crop_sha256']:raise ValueError('Crop SHA256 mismatch: '+s['id'])
        crops=precompute(manifest,ROOT)
        environment=report_env()
        write_json(dest/'environment.json',environment)
        write_json(dest/'inputs.json',manifest)
        metadata={'run_id':run_id,'config':cfg,'models':specs,'environment':environment,
                  'ground_truth_approved_by_user':False,'timestamps':'local',
                  'model_loading_excluded_from_ocr_ms':True,'passes':{'warmup':[1],'measured':[2,3]}}
        write_json(dest/'manifest.json',metadata)
        allrows=[];models_results=[]
        raw=dest/'raw.jsonl'
        total_cpu_sample=psutil.cpu_percent(interval=None) if psutil else None
        def save_row(row):
            allrows.append(row)
            with raw.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        for spec in specs:
            update(state='loading',model=spec['name'],pass_=None,message='Loading model / creating engine')
            model_start=time.perf_counter();adapter=None;load_error=None
            try:adapter=make_adapter(spec,ROOT,BASE)
            except Exception as exc:load_error=f'{type(exc).__name__}: {exc}'
            load_ms=(time.perf_counter()-model_start)*1000
            mdl={'id':spec['id'],'name':spec['name'],'init_ms':round(load_ms,3),
                 'status':'READY' if adapter else 'UNAVAILABLE','error':load_error,
                 'model_sha256':getattr(adapter,'model_sha256',None),
                 'model_path':getattr(adapter,'model_path',None)}
            models_results.append(mdl)
            write_json(dest/'model_initialization.json',models_results)
            for pass_no in (1,2,3):
                pass_type='warmup' if pass_no==1 else 'measured'
                for s in manifest:
                    update(state='running',model=spec['name'],pass_number=pass_no,
                           pass_type=pass_type,cell=s['id'],message='OCR '+s['id'])
                    row={'model_id':spec['id'],'model_name':spec['name'],'pass':pass_no,
                         'phase':pass_type,'cell_id':s['id'],'cell_type':s['type'],
                         'width_px':s['width'],'height_px':s['height'],
                         'segments':len(crops[s['id']]),'expected':s['expected'],
                         'source':s['source'],'file':s['file'],'ocr_ms':None,'cpu_ms':None,
                         'process_cpu_percent_one_core':None,'host_cpu_percent':None,
                         'rss_mb':None,'confidence':None,'text':'','segment_texts':[],
                         'status':'SKIPPED' if adapter is None else 'ERROR', 'error':load_error}
                    if adapter is not None:
                        t0=time.perf_counter_ns();c0=time.process_time_ns()
                        try:
                            parts=[];scores=[]
                            for im in crops[s['id']]:
                                text,score=adapter.recognize(im)
                                parts.append(text);scores.append(score)
                            row['text']=join_parts(parts)
                            row['segment_texts']=parts
                            valid_scores=[s for s in scores if s is not None]
                            row['confidence']=round(sum(valid_scores)/len(valid_scores),6) if valid_scores else None
                            row['status']='OK';row['error']=None
                        except Exception as exc:
                            row['error']=f'{type(exc).__name__}: {exc}'
                        finally:
                            wall=(time.perf_counter_ns()-t0)/1e6
                            cpu=(time.process_time_ns()-c0)/1e6
                            row['ocr_ms']=round(wall,3);row['cpu_ms']=round(cpu,3)
                            row['process_cpu_percent_one_core']=round(100*cpu/wall,1) if wall else None
                            if psutil:
                                row['host_cpu_percent']=round(psutil.cpu_percent(interval=None),1)
                                row['rss_mb']=round(psutil.Process().memory_info().rss/1048576,1)
                        if row['status']=='OK':row.update(metric(s['expected'],row['text']))
                    save_row(row)
                    status['completed']+=1
                    update(cpu_percent=row.get('host_cpu_percent'),
                           process_cpu_percent_one_core=row.get('process_cpu_percent_one_core'),
                           last_ocr_ms=row.get('ocr_ms'),last_text=row.get('text'),
                           last_status=row['status'])
            try:
                if adapter is not None:adapter.close()
            except Exception as exc:mdl['close_error']=str(exc)
            build_report(dest,manifest,specs,models_results,allrows,metadata,complete=False)
        elapsed=round(time.perf_counter()-started,2)
        status['completed']=total
        update(state='finished',message='All benchmark models processed',total_elapsed_sec=elapsed)
        build_report(dest,manifest,specs,models_results,allrows,metadata,complete=True)
        write_csv(dest/'raw.csv',allrows)
        return dest
    except Exception as exc:
        update(state='error',message=f'{type(exc).__name__}: {exc}',error_trace=traceback.format_exc())
        raise

def write_csv(path,rows):
    fields=['model_id','model_name','pass','phase','cell_id','cell_type','width_px','height_px',
            'segments','ocr_ms','cpu_ms','process_cpu_percent_one_core','host_cpu_percent','rss_mb',
            'status','confidence','expected','text','edit_distance','exact','cer','error']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
