"""Failure-only per-job grid diagnostics; no detector decisions changed."""
from contextvars import ContextVar
from functools import wraps
from collections import deque
from pathlib import Path
from time import perf_counter
import hashlib, json, logging, os, sys, traceback, zipfile, platform, ast, re

# GRID_ANOMALY_V5_BIDIRECTIONAL
from App import grid_abort_compact_v2 as _compact

_CTX=ContextVar('grid_abort_diagnostics_v1',default=None)
LOG=logging.getLogger(__name__)
# Never copy .env or arbitrary environment values.
PREFIXES=('OCR_GRID_', 'OCR_LOCALGRID_', 'OCR_SOURCE_GRID_', 'OCR_PAPER_', 'OCR_OPENCV_', 'OCR_ONNX_', 'OCR_TABLE_', 'OCR_HEADER_', 'OCR_RECTIFY_', 'OCR_MODEL_', 'OCR_REC_', 'OCR_DET_')
DENY=('TOKEN','PASSWORD','SECRET','API_KEY','CREDENTIAL','AUTH','URL','PROXY','PATH','DIR','EXE')
NAMES=set('data graph physical automatic_data baseline previous report physical_report diagnostic diagnostics stats audit accepted safe safety geometry scan source_extension rect result mapping _grid_gate _grid_trace _grid_attempt _grid_candidate _ga _gg _gr _state _mode _retry failures options _grid_fixed _grid_route _v12_error _candidate _direct _fixed _premesh _combo _combo_scan _combo_recovery H W shape top bottom left right nodes intersections vs hs vv hh horizontal vertical hypotheses scored clean picked rejected edges parts supplied_parts min_rows prefer_multiscale scales retry rectify number_col force_number detailed rail_support anchor levels rails paper_bbox bbox roi reason invalid_spacing auto_vertical_count _texts _header _candidate header names numeric _bands _typical _giant _direct _gr _go _gc _ga _gg _combo_original _combo_scan _combo_report _combo_fixed _combo_used'.split())
SOURCE_FILES=('App/pipeline.py','App/config.py','App/geometry/service.py','App/geometry/short_receipt_grid_v1.py','App/geometry/grid_anomaly_v3.py','App/geometry/detector_routing_v1.py','App/geometry/local_grid_runtime.py','App/geometry/lab_adapter_gridrefine_v3.py','App/geometry/source_grid_v16.py','App/geometry/paper_first_grid_v12.py','App/geometry/grid_edge_validator_v12.py','App/geometry/grid_completeness.py','App/geometry/geometry_lab/physical_grid_graph_v3.py','App/geometry/geometry_lab/physical_grid_graph_local.py','App/geometry/geometry_lab/physical_grid_graph.py','App/geometry/geometry_lab/detection.py','App/geometry/geometry_lab/algorithms.py','App/geometry/geometry_lab/structure_diagnostics.py','App/geometry/geometry_lab/metrics.py','App/geometry/autogrid_gridrefine_v3/detector.py','App/geometry/autogrid_gridrefine_v3/fitter.py')

def allowed(name):
    return isinstance(name,str) and (name=='LOG_LEVEL' or name.startswith(PREFIXES)) and not any(x in name.upper() for x in DENY)

def _safe_text(x):
    s=str(x)
    try:
        from App.config import ROOT
        s=s.replace(str(ROOT),'[PROJECT]').replace(str(ROOT).replace('\\','/'),'[PROJECT]')
    except Exception: pass
    return s[:16000]

def native(x,depth=0):
    import numpy as np
    if depth>12:return {'omitted':'depth_limit'}
    if x is None or isinstance(x,(bool,int,str)):return _safe_text(x) if isinstance(x,str) else x
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.ndarray):
        if x.size>20000:return {'omitted':'large_array','shape':list(x.shape),'dtype':str(x.dtype)}
        return native(x.tolist(),depth+1)
    if isinstance(x,dict):
        return {str(k):native(v,depth+1) for k,v in list(x.items())[:1500] if str(k) not in ('source_image','image','map_x','map_y','cell_mesh')}
    if isinstance(x,(tuple,list,deque)):
        out=[native(v,depth+1) for v in list(x)[:1500]]
        if len(x)>1500:out.append({'omitted_items':len(x)-1500})
        return out
    if hasattr(x,'eval') and hasattr(x,'position'):
        out={'rule_class':type(x).__name__}
        for k,v in vars(x).items():
            if not k.startswith('__'):out[k]=native(v,depth+1)
        # Exact coefficient/point fields + sampled source curve for inspection.
        try:
            t=np.linspace(-.5,.5,65);out['normalized_samples']=native(t,depth+1);out['evaluated_samples']=native(x.eval(t),depth+1)
        except Exception as e:out['sample_error']=type(e).__name__
        return out
    if hasattr(x,'info') and hasattr(x,'xs') and hasattr(x,'ys'):
        return {'method':x.method,'xs':native(x.xs,depth+1),'ys':native(x.ys,depth+1),'info':native(x.info,depth+1)}
    if type(x).__name__=='RouteState':return native(vars(x),depth+1)
    if isinstance(x,BaseException):return {'type':type(x).__name__,'message':_safe_text(x)}
    return {'type':type(x).__name__}

class State:
    def __init__(self,rectify, v2=False):
        self.v2=v2
        if v2:_compact.init(self)
        self.start=perf_counter();self.dir=None;self.active=False;self.started=False;self.attempt=0;self.rectify=rectify;self.events=deque(maxlen=256);self.count=0;self.config={};self.maps=deque(maxlen=3);self.errors=[];self.exporting=False;self.source_images=deque(maxlen=2)
    def add(self,kind,label,detail):
        self.count+=1;self.events.append({'seq':self.count,'attempt':self.attempt,'elapsed_ms':round((perf_counter()-self.start)*1000,3),'kind':kind,'algorithm':label,'detail':detail})

def _quiet(fn,*args):
    try:return fn(*args)
    except Exception as e:
        s=_CTX.get()
        if s is not None and len(s.errors)<20:s.errors.append(type(e).__name__+': '+_safe_text(e))

def bind(job_dir):
    s=_CTX.get()
    if s is not None:s.dir=Path(job_dir)

def begin():
    s=_CTX.get()
    if s is not None:
        s.active=True;s.started=True;s.attempt+=1;s.add('phase','geometry',{'status':'start'})

def complete():
    s=_CTX.get()
    if s is not None and s.active:s.add('phase','geometry',{'status':'confirmed'});s.active=False

def config_read(fn):
    @wraps(fn)
    def wrapped(name,default):
        value=fn(name,default);s=_CTX.get()
        if s is not None and not s.exporting and allowed(name):
            def add():
                item={'effective':native(value),'default':native(default),'source':'process_env' if name in os.environ else 'dotenv_or_default','phase':'geometry' if s.active else 'other','attempt':s.attempt}
                bucket=s.config.setdefault(name,[])
                if not bucket or bucket[-1]!=item:bucket.append(item)
            _quiet(add)
        return value
    return wrapped

def _snapshot(s,local):
    if s.v2:return _compact.snapshot(s,local)
    out={}
    for k,v in local.items():
        if k in NAMES or k.startswith(('_grid_','_combo_','_premesh','_v12_')):
            if callable(v):continue
            if k in ('parts','supplied_parts') and isinstance(v,dict):out[k]={a:{'count':len(b),'segments':native(b)} for a,b in v.items()};continue
            out[k]=native(v)
            if hasattr(v,'map_x') and hasattr(v,'info'):
                if all(v is not x[1] for x in s.maps):s.maps.append((k,v))
    im=local.get('image');gray=local.get('gray')
    if im is not None and hasattr(im,'shape'):
        out['input_shape']=list(im.shape)
        if getattr(im,'ndim',0)==3 and all(im is not i for i in s.source_images):s.source_images.append(im)
    elif gray is not None and hasattr(gray,'shape'):out['input_shape']=list(gray.shape)
    return out

def ret(value,local,label):
    s=_CTX.get()
    if s is not None and s.active:
        _quiet(lambda:s.add('return',label,{'locals':_snapshot(s,local),'result':_compact.result(s,value) if s.v2 else native(value)}))
        # A returned Result is also worth retaining for failure-only NPZ export.
        def keep():
            v=value[0] if isinstance(value,tuple) and value else value
            if hasattr(v,'map_x') and all(v is not x[1] for x in s.maps):s.maps.append((label,v))
        _quiet(keep)
    return value

def caught(local,label):
    s=_CTX.get()
    if s is not None and s.active:
        exc=sys.exc_info()[1]
        _quiet(lambda:s.add('caught_exception',label,{'error':native(exc),'locals':_snapshot(s,local)}))

def observe(label):
    def deco(fn):
        @wraps(fn)
        def wrap(*args,**kwargs):
            s=_CTX.get()
            if s is None or (not s.active and not label.endswith((':_process',':_process_route_body'))):return fn(*args,**kwargs)
            start=perf_counter();_quiet(lambda:s.add('enter',label,{'kwargs':native({k:v for k,v in kwargs.items() if not callable(v)})}))
            try:return fn(*args,**kwargs)
            except Exception as e:
                def log():
                    if not s.active:return
                    tb=e.__traceback__; chosen=None
                    while tb:
                        if tb.tb_frame.f_code.co_name==fn.__name__:chosen=tb
                        if tb.tb_next is None:break
                        tb=tb.tb_next
                    frame=chosen or tb
                    s.add('exception',label,{'error':native(e),'locals':_snapshot(s,frame.tb_frame.f_locals) if frame else {}})
                _quiet(log);raise
            finally:_quiet(lambda:s.add('exit',label,{'wall_ms':round((perf_counter()-start)*1000,3)}))
        return wrap
    return deco

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def export(s,exc):
    import numpy as np
    import cv2
    from App.config import ROOT,setting
    import shutil
    s.active=False;s.exporting=True
    p=s.dir/'problem_grid';p.mkdir(exist_ok=True)
    # Capture complete live deep buffer before outer process deactivates it.
    try:
        from App.ocr.performance_deep.live_capture import current
        c=current()
        if c is not None:
            (p/'deep_profile.json').write_text(json.dumps(c.export(),ensure_ascii=False,indent=2,default=lambda v:native(v)),encoding='utf-8')
    except Exception as e:s.errors.append('deep_profile: '+_safe_text(e))
    files=[];defaults={}
    # Include active experimental modules without collecting unrelated project files.
    extra=[]
    for f in (ROOT/'App/geometry').glob('*.py'):
        if any(tag in f.name.lower() for tag in ('combo','combined','overlay','routing','anomaly','premesh','speed')):extra.append(f.relative_to(ROOT).as_posix())
    for rel in dict.fromkeys((*SOURCE_FILES,*extra,'App/grid_abort_compact_v2.py','App/grid_abort_diagnostics_v1.py')):
        src=ROOT/rel
        if not src.is_file():continue
        raw=src.read_bytes();files.append({'file':rel,'sha256':hashlib.sha256(raw).hexdigest()})
        # Actual source and hardcoded thresholds, not guesses from .env.
        if rel not in ('App/config.py','App/pipeline.py'):
            dest=p/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        try:
            source=raw.decode('utf-8-sig'); tree=ast.parse(source)
            if rel=='App/pipeline.py':
                fragments=[]
                for n in tree.body:
                    if isinstance(n,ast.FunctionDef) and n.name in ('_process','_process_route_body'):
                        # Include actual geometry control flow, stop before main product OCR.
                        lines=source.splitlines(); end=n.end_lineno
                        if n.name=='_process_route_body':
                            end=next((i for i in range(n.lineno-1,end) if '_route.finish_geometry()' in lines[i]),end-1)+1
                        fragments.append('\n'.join(lines[n.lineno-1:end]))
                (p/'pipeline_geometry_source.txt').write_text('\n\n'.join(fragments),encoding='utf-8')
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id.endswith('setting') and len(n.args)>=2 and all(isinstance(a,ast.Constant) for a in n.args[:2]) and allowed(n.args[0].value):
                    name,default=n.args[0].value,n.args[1].value
                    defaults.setdefault(name,[]).append({'file':rel,'line':n.lineno,'default':default,'effective_at_export':setting(name,default)})
        except (SyntaxError,UnicodeError,ValueError) as e:s.errors.append(rel+': '+str(e))
    for i,im in enumerate(s.source_images):
        try:
            if not cv2.imwrite(str(p/f'candidate_source_{i:02d}.png'),im):raise ValueError('imwrite returned false')
        except Exception as e:s.errors.append('source image: '+_safe_text(e))
    _map_seen={}
    for i,(label,r) in enumerate(s.maps):
        try:
            m=r.info.get('cell_mesh') if r.method=='original_cells' else r
            if m is None:continue
            map_name=f'candidate_map_{i:02d}.npz'
            if s.v2:
                dig=hashlib.sha256()
                for a in (m.map_x,m.map_y,np.asarray(m.xs),np.asarray(m.ys)):
                    dig.update(str((a.shape,a.dtype)).encode());dig.update(np.ascontiguousarray(a).tobytes())
                key=dig.hexdigest(); map_name=_map_seen.get(key,map_name)
                if key not in _map_seen:
                    np.savez_compressed(p/map_name,map_x=m.map_x,map_y=m.map_y,xs=m.xs,ys=m.ys);_map_seen[key]=map_name
                dump(p/f'candidate_graph_{i:02d}.json',{'label':label,'map_file':map_name,'rect':_compact.plain(s,r)})
            else:
                np.savez_compressed(p/map_name,map_x=m.map_x,map_y=m.map_y,xs=m.xs,ys=m.ys)
                dump(p/f'candidate_graph_{i:02d}.json',{'label':label,'rect':native(r)})
        except Exception as e:s.errors.append('map: '+_safe_text(e))
    try:
        import subprocess
        git_revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,timeout=2).stdout.strip() or None
    except (OSError,subprocess.TimeoutExpired):git_revision=None
    dump(p/'algorithm_settings.json',{'actual_reads':s.config,'configured_defaults_and_effective_at_export':defaults,'source_hashes':files,'git_revision':git_revision,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads(),'logical_cpu':os.cpu_count(),'rectify_requested':s.rectify}})
    for f in s.dir.iterdir():
        if f.is_file() and (f.name in ('input.bin','error.json','performance_profile.json','processing_log.json','detector_routing.json','grid_anomaly.json','grid_anomaly_initial.png','rectified.png','original.png') or f.name.startswith('routing_trial_')):
            shutil.copy2(f,p/f.name)
    if s.v2:
        (p/'objects.json').write_text(json.dumps(s.objects,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    dump(p/'grid_diagnostics.json',{**(_compact.metadata(s) if s.v2 else {'schema':'grid-abort-diagnostics-v1'}),'job_id':s.dir.name,'status':'aborted_geometry','error':native(exc),'traceback':_safe_text(''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))), 'attempts':s.attempt,'events':list(s.events),'events_dropped':max(0,s.count-len(s.events)),'capture_errors':s.errors,'notes':['Graph coordinates refer to detector input shape; source_offset identifies local crop where available.','No synthetic reconstruction is claimed. Missing candidates are recorded as null.','Event buffer and array/sequence limits are explicit; only last 3 available maps and last 2 detector images are retained.']})
    target=s.dir/'grid_diagnostics.zip';tmp=s.dir/'grid_diagnostics.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in sorted(p.rglob('*')):
            if f.is_file():z.write(f,Path('problem_grid')/f.relative_to(p))
    os.replace(tmp,target)

def job(fn):
    @wraps(fn)
    def wrap(*args,**kwargs):
        from App.config import setting
        if setting('LOG_LEVEL','INFO').strip().upper()!='DEBUG':return fn(*args,**kwargs)
        v2=setting('OCR_GRID_ABORT_DIAGNOSTICS_V2','0')=='1'
        if not v2 and setting('OCR_GRID_ABORT_DIAGNOSTICS_V1','1')!='1':return fn(*args,**kwargs)
        s=State(kwargs.get('rectify',False),v2);token=_CTX.set(s)
        if v2:s.config['OCR_GRID_ABORT_DIAGNOSTICS_V2']=[{'effective':'1','default':'0','phase':'job_start','attempt':0}]
        try:return fn(*args,**kwargs)
        except Exception as exc:
            if s.started and s.active and s.dir is not None:
                try:export(s,exc)
                except Exception:LOG.exception('Grid diagnostics export failed; original error preserved')
            raise
        finally:_CTX.reset(token)
    return wrap
