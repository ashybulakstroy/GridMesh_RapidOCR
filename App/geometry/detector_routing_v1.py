"""Job-local routing state; no global toggles and no OCR changes."""
from contextvars import ContextVar
from dataclasses import dataclass, field

@dataclass
class RouteState:
    mode: str
    min_horizontal: int = 10
    attempted: bool = False
    candidate_used: bool = False
    before_ocr: bool = True
    events: list = field(default_factory=list)

_state = ContextVar('grid_route_v1', default=None)

def start(mode, min_horizontal=10):
    if mode not in ('legacy','auto','multiscale_first'):
        raise ValueError('OCR_GRID_ROUTE_V1 must be legacy, auto or multiscale_first')
    if min_horizontal < 3: raise ValueError("min_horizontal must be >=3")
    state=RouteState(mode, min_horizontal)
    return state,_state.set(state)

def end(token):
    _state.reset(token)

def choose(data, report, image):
    state=_state.get()
    if state is None or state.mode=='legacy' or image.shape[1]>image.shape[0]:
        return False
    if state.attempted: return False
    if state.mode=='multiscale_first':
        return True
    if data is None: return False
    h=len(data.get('horizontal', [])); v=len(data.get('vertical', []))
    return h>=state.min_horizontal and v in (3,4)

def record(event, **details):
    state=_state.get()
    if state is not None:
        state.events.append(dict(event=event,**details))

def attempt():
    s=_state.get()
    if s is not None: s.attempted=True

def mark_candidate():
    s=_state.get()
    if s is not None: s.candidate_used=True

def used():
    s=_state.get()
    return bool(s and s.candidate_used)

def can_retry():
    s=_state.get()
    return bool(s and s.mode!='legacy' and s.candidate_used and s.before_ocr)

def finish_geometry():
    s=_state.get()
    if s is not None: s.before_ocr=False

def accept_candidate(data, min_rows=0):
    """Cheap guard only; real anomaly/header checks remain in pipeline."""
    if data is None: return False
    import numpy as np
    v=data.get('vertical',[]); h=data.get('horizontal',[])
    if len(v)!=5 or len(h)<max(3,min_rows):return False
    try:
        H,W=data['shape']; xs=np.linspace(v[0].position,v[-1].position,64)
        yy=np.stack([r.eval((xs-W/2)/W) for r in h]); ys=np.linspace(h[0].position,h[-1].position,64)
        xx=np.stack([r.eval((ys-H/2)/H) for r in v]);
        return bool(np.isfinite(xx).all() and np.isfinite(yy).all() and
                    np.min(np.diff(xx,axis=0))>=5 and np.min(np.diff(yy,axis=0))>=4)
    except (TypeError,ValueError,AttributeError,IndexError):return False
