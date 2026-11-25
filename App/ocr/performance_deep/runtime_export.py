"""Adapter for App.pipeline; import only for DEBUG. Existing trace images are reused."""
from pathlib import Path
from .performance_report import generate, build, read, normalized
from .live_capture import current

def export(job_dir, tracker, job_ms=2000, stage_ms=1000):
    p=read(Path(job_dir)/'performance_profile.json',{});r=read(Path(job_dir)/'receipt.json',{})
    if not r:
        r={'job_id':Path(job_dir).name, 'error':read(Path(job_dir)/'error.json',{}).get('error')}
    if not build(p,r,job_ms=job_ms,stage_ms=stage_ms):return None
    c=current();deep=c.export() if c else {'warnings':['Live capture unavailable']}
    report=build(p,r,deep,job_ms,stage_ms);images={}
    import cv2
    for cell in report['cells']:
        cid=cell['cell_id']
        for e in cell['attempts']:
            eid=e.get('event_id'); raw=tracker.inputs.get(eid); text=tracker.text_inputs.get(eid,{})
            if isinstance(raw,dict):raw=next((v for k,v in raw.items() if normalized(k)==cid),None)
            elif len(e.get('cells',[]))!=1:raw=None
            if raw is None:raw=next((v for k,v in text.items() if normalized(k)==cid),None)
            if raw is not None:
                try:
                    ok,buf=cv2.imencode('.png',raw)
                    if ok:images[(eid,cid)]=[buf.tobytes()]
                except Exception:deep.setdefault('warnings',[]).append(f'PNG failed: {eid}/{cid}')
    if tracker.text_dropped:deep.setdefault('warnings',[]).append(f'Text input capture dropped: {tracker.text_dropped}')
    return generate(job_dir,deep,images,job_ms,stage_ms,receipt=r)
