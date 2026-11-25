"""Per-job buffered DEBUG instrumentation; no image encoding or disk I/O until export."""
from contextvars import ContextVar
from functools import wraps
from threading import Lock
from time import perf_counter_ns, thread_time_ns
from contextlib import contextmanager
from collections import Counter, defaultdict
import statistics

_CURRENT=ContextVar('performance_deep_capture',default=None)
class Capture:
    def __init__(self):
        self.start=perf_counter_ns(); self.lock=Lock(); self.spans=[]; self.counters=Counter(); self.details=[]; self.dropped=0
    def add(self, item):
        with self.lock:
            if len(self.spans)<20000:self.spans.append(item)
            else:self.dropped+=1
    def export(self):
        with self.lock: spans=list(self.spans); counts=dict(self.counters); details=list(self.details); dropped=self.dropped
        stats=defaultdict(list)
        for s in spans:stats[s['name']].append(s['wall_ms'])
        grid_names={s['name'] for s in spans if s['group']=='grid'}
        summary={k:{'calls':len(v),'sum_call_ms':round(sum(v),3),'median_ms':statistics.median(v),'p90_ms':sorted(v)[max(0,(9*len(v)+9)//10-1)],'min_ms':min(v),'max_ms':max(v)} for k,v in stats.items()}
        return {'spans':spans,'grid_counters':{'counts':{k:v for k,v in counts.items() if not k.startswith('ocr_')},'details':[x for x in details if not x['name'].startswith('ocr_')],'functions':{k:v for k,v in summary.items() if k in grid_names},'repeated_exact_seed_signatures':{k:v for k,v in counts.items() if k.startswith('trace_seed:') and v>1}},'ocr_decisions':[x for x in details if x['name'].startswith('ocr_')],'warnings':[f'Dropped records: {dropped}'] if dropped else []}
def activate(enabled):return _CURRENT.set(Capture() if enabled else None)
def deactivate(token):_CURRENT.reset(token)
def current():return _CURRENT.get()
@contextmanager
def span(group,name,**detail):
    c=current()
    if c is None:yield;return
    a=perf_counter_ns(); b=thread_time_ns(); status='ok'
    try:yield
    except BaseException:status='error';raise
    finally:
        end=perf_counter_ns();c.add({'group':group,'name':name,'start_ms':(a-c.start)/1e6,'end_ms':(end-c.start)/1e6,'wall_ms':(end-a)/1e6,'thread_cpu_ms':(thread_time_ns()-b)/1e6,'status':status,'detail':detail})
def timed(group,name=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args,**kwargs):
            if current() is None:return fn(*args,**kwargs)
            detail={}
            if fn.__name__=='trace_seed' and len(args)>=3:
                try:
                    seed=tuple(round(float(x),2) for x in args[1]);detail={'seed':list(seed),'sample_count':len(args[2])}
                    # Exact input signature; not proof of same physical line.
                    count('trace_seed:'+str(seed), seed=list(seed))
                except (TypeError,ValueError,IndexError):pass
            with span(group,name or fn.__qualname__,**detail):return fn(*args,**kwargs)
        return wrapper
    return deco
def count(name,n=1,**detail):
    c=current()
    if c is None:return
    with c.lock:
        c.counters[name]+=n
        if detail:
            if len(c.details)<10000:c.details.append({'name':name,**detail})
            else:c.dropped+=1

def annotate_attempt(event, *, fallback_reason=None, selection_reason=None, currency=None):
    # Call at the actual branch, not inferred from result. Existing trace dict only.
    if current() is None:return
    if fallback_reason is not None:event['fallback_reason']=str(fallback_reason)
    if selection_reason is not None:event['selection_reason']=str(selection_reason)
    if currency is not None:event['currency_roi']=dict(currency)

_REASON=ContextVar('performance_ocr_reason',default=None)
@contextmanager
def reason(value):
    if current() is None:yield;return
    token=_REASON.set(value)
    try:yield
    finally:_REASON.reset(token)
def attempt_reason():return _REASON.get()
