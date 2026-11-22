"""Per-job inference accounting with explicit distinction between REC-only and DET+REC.

One ONNX batch call is a single model invocation but one recognition exposure for
*each* member cell. Detector pipelines may spawn internal REC calls that RapidOCR
hides; these are reported as opaque instead of inventing an exact count.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT = ContextVar('ocr_e4_rec_tracker',default=None)
_SCOPE = ContextVar('ocr_e4_rec_scope',default=None)

class RecTracker:
    def __init__(self):
        self.events=[]
    def record(self,keys,stage,model,mode,ms,*,success=True,note=None):
        keys=[str(k) for k in (keys if isinstance(keys,(list,tuple,set)) else [keys])]
        keys=list(dict.fromkeys(keys or ['unattributed']))
        event={'cells':keys,'stage':str(stage),'model_id':str(model),'mode':str(mode),
               'wall_ms':round(float(ms),3),'success':bool(success)}
        if note:event['note']=str(note)
        self.events.append(event)
    def export(self):
        per=defaultdict(lambda:{'rec_only_exposures':0,'detector_pipeline_calls':0,
                                'by_model':{},'stages':{}})
        rec_calls=0; detector=0
        for e in self.events:
            if e['mode']=='REC_only':rec_calls+=1
            else:detector+=1
            for key in e['cells']:
                item=per[key]
                name=('rec_only_exposures' if e['mode']=='REC_only' else 'detector_pipeline_calls')
                item[name]+=1
                item['by_model'][e['model_id']]=item['by_model'].get(e['model_id'],0)+1
                item['stages'][e['stage']]=item['stages'].get(e['stage'],0)+1
        return {'version':'e4-rec-trace-v1','rec_only_model_invocations':rec_calls,
                'rec_only_cell_exposures':sum(len(e['cells']) for e in self.events if e['mode']=='REC_only'),
                'detector_pipeline_invocations':detector,'per_cell':dict(per),
                'events':self.events,
                'scope_note':'REC-only exposures counted exactly for instrumented paths; DET+REC internal recognition count is opaque.'}

def activate(tracker):return _CURRENT.set(tracker)
def deactivate(token):_CURRENT.reset(token)

@contextmanager
def rec_scope(keys,stage):
    if not isinstance(keys,(list,tuple,set)):
        keys=[keys]
    keys=[f'{k[0]}:{k[1]}' if isinstance(k,tuple) else str(k) for k in keys]
    token=_SCOPE.set((keys,stage))
    try:yield
    finally:_SCOPE.reset(token)

def record_rec(*,model='v5_cyrillic_mobile',mode='REC_only',ms=0,success=True,note=None,keys=None,stage=None):
    tracker=_CURRENT.get()
    if tracker is None:return
    current=_SCOPE.get()
    keys=keys if keys is not None else (current[0] if current else ['unattributed'])
    stage=stage if stage is not None else (current[1] if current else 'unscoped')
    tracker.record(keys,stage,model,mode,ms,success=success,note=note)
