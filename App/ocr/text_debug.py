"""Diagnostic only: no OCR retries or mutation of recognized values."""
import json, re, hashlib, html, time
from pathlib import Path
import cv2
import numpy as np
from ..config import setting

def enabled(): return setting('OCR_TEXT_DEBUG_ENABLE','1') == '1'

def bands(image):
    if image is None or not image.size: return []
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    _, m=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    m[:2,:]=0; m[-2:,:]=0; m[:,:2]=0; m[:,-2:]=0
    # Remove near-full-width rules for diagnostic count only, not OCR.
    m=cv2.subtract(m,cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((1,max(3,int(g.shape[1]*.8))),np.uint8)))
    active=np.count_nonzero(m,axis=1)>=max(2,g.shape[1]*.015)
    edges=np.diff(np.r_[False,active,False].astype(np.int8))
    return [[int(a),int(b)] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=4]

def reasons(item, image, events, threshold=.8):
    t=str(item.get('text') or '').strip(); out=[]
    if not t: out.append('empty_name')
    if float(item.get('confidence') or 0)<threshold: out.append('low_confidence')
    if re.search(r'[-/–]$',t): out.append('unfinished_suffix')
    if any(re.search('[A-Za-z]',w) and re.search('[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]',w) for w in re.findall(r'[^\W\d_]+',t)): out.append('mixed_script_word')
    b=bands(image)
    if len(b)>=2 and len(item.get('blocks') or [])<len(b): out.append('possible_missing_line')
    if len(b)>=2 and len(t)<12: out.append('short_multiline_name')
    if any(not e.get('success',True) for e in events): out.append('ocr_exception')
    if any(e.get('model_id')=='v5_cyrillic_mobile' for e in events): out.append('v5_route_or_fallback')
    if any(any(float(c)<threshold for c in e.get('confidences',[])) for e in events): out.append('low_attempt_confidence')
    return out,b

def save_text_debug(job_dir, rows, requests, source, rect, columns, trace, inputs, dropped):
    if not enabled(): return None
    start=time.perf_counter(); threshold=float(setting('OCR_TEXT_DEBUG_MIN_CONF','0.80'))
    if not 0<=threshold<=1: raise ValueError('OCR_TEXT_DEBUG_MIN_CONF must be 0..1')
    cells={c.row_id:c for c in requests if c.column_id=='name'}
    records=[]
    for row in rows:
        if row.get('kind')!='product': continue
        rid=row['row_id']; c=cells.get(rid)
        if c is None: continue
        item=row['cells']['name']; key=f'{rid}:name'
        events=[e for e in trace.get('events',[]) if key in e.get('cells',[])]
        flags,b=reasons(item,c.image,events,threshold)
        records.append({'row_id':rid,'flags':flags,'estimated_ink_bands':b,'final':item,'events':events,'offset':c.offset,'source_bbox':c.source_bbox})
    suspects=sum(bool(r['flags']) for r in records)
    if not suspects and setting('OCR_TEXT_DEBUG_FORCE','0')!='1': return None
    from .experiment_original import OriginalNumericCache
    from .experiment_models import load_catalog
    from importlib import metadata
    directory=Path(job_dir)/'problem_text'; directory.mkdir(exist_ok=True)
    cache=OriginalNumericCache(source,rect,max_bytes=16*1024*1024)
    errors=[]; max_bytes=32*1024*1024; written=0
    def save(name,img):
        nonlocal written
        if img is None or not img.size: return None
        if written+img.nbytes>max_bytes:
            errors.append('output_limit:'+name); return None
        ok,data=cv2.imencode('.png',img)
        if not ok: errors.append('encode_failed:'+name); return None
        (directory/name).write_bytes(data.tobytes()); written+=img.nbytes
        return {'file':name,'shape':list(img.shape),'dtype':str(img.dtype),'sha256_pixels':hashlib.sha256(img.tobytes()).hexdigest()}
    for r in records:
        rid=r['row_id']; c=cells[rid]; key=f'{rid}:name'
        r['cell_image']=save(f'row_{rid:03d}_cell.png',c.image)
        cache.store(rid,'name',columns['name'])
        r['source_polygon_image']=save(f'row_{rid:03d}_source.png',cache.crops.get((rid,'name')))
        for e in r['events']:
            img=inputs.get(e['event_id']); img=img.get(key) if isinstance(img,dict) else img
            # Keep event metadata local to this report, not the existing numerical trace.
            e=dict(e)
            e['input']=save(f'row_{rid:03d}_event_{e["event_id"]:05d}.png',img)
            r.setdefault('attempts',[]).append(e)
        r.pop('events')
    versions={}
    for name in ('rapidocr','onnxruntime','opencv-python','numpy'):
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]='not_installed'
    payload={'schema':'text-ocr-debug-v1','suspects':suspects,'count':len(records),'threshold':threshold,'records':records,'capture_dropped':dropped,'errors':errors+cache.errors,'versions':versions,'model_catalog':load_catalog(),'configuration':{k:setting(k,v) for k,v in {'OCR_TABLE_MODE':'cell','OCR_ONNX_THREADS':'2','OCR_TABLE_ONNX_THREADS':setting('OCR_ONNX_THREADS','2'),'OCR_NAME_ENGLISH_FALLBACK':'0','OCR_TEXT_DEBUG_ENABLE':'1'}.items()},'notes':['Flags are heuristic; no ground truth or CER is inferred.','Actual public OCR API inputs, not internal ONNX tensors. DET+REC internal line crops are opaque. Source polygon images are diagnostic reconstructions.']}
    def esc(x): return html.escape(str(x))
    def pic(v): return f'<a href="{esc(v["file"])}"><img style="max-width:450px;max-height:180px" src="{esc(v["file"])}"></a>' if v else 'not captured'
    chunks=['<!doctype html><meta charset=utf-8><title>Text OCR debug</title><style>body{font:15px Arial;margin:24px}section{border:1px solid #aaa;padding:12px;margin:12px 0}img{display:block;margin:5px 0}pre{white-space:pre-wrap}</style><h1>Text OCR debug</h1>',f'<p>Suspects: {suspects} / {len(records)} · dropped: {dropped}</p>']
    for r in records:
        chunks.append(f'<section><h2>Row {r["row_id"]}: {esc(r["final"].get("text"))}</h2><p>{esc(r["flags"])}</p><p>Cell / source polygon</p>'+pic(r['cell_image'])+pic(r['source_polygon_image']))
        for e in r.get('attempts',[]): chunks.append(f'<p>{esc(e["model_id"])} | {esc(e["mode"])} | {esc(e.get("texts"))} | {esc(e.get("confidences"))} | {e["wall_ms"]} ms</p>'+pic(e['input']))
        chunks.append('</section>')
    chunks.append('<p><a href="manifest.json">JSON manifest</a></p>')
    (directory/'report.html').write_text('\n'.join(chunks),encoding='utf-8')
    payload['save_wall_ms']=round((time.perf_counter()-start)*1000,3)
    (directory/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o)),encoding='utf-8')
    return {'count':len(records),'suspects':suspects,'report':str(directory/'report.html'),'save_wall_ms':payload['save_wall_ms'],'capture_dropped':dropped}
