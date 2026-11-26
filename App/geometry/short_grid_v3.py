"""Short-grid speculative route. Full source validation, unchanged legacy fallback."""
import numpy as np
import cv2
from time import perf_counter
from ..config import setting

def enabled():
    return setting('OCR_GRID_SHORT_ROUTE_V3','0')=='1' and setting('OCR_GRID_ANOMALY_V3','1')=='1'

def is_short(data):
    return enabled() and data is not None and 3<=len(data.get('horizontal',[]))<=9 and 2<=len(data.get('vertical',[]))<=4

def roi_box(data,shape):
    H,W=shape[:2]; h=data['horizontal'];v=data['vertical']
    x0=min(float(r.domain[0]) for r in h);x1=max(float(r.domain[1]) for r in h)
    y0=min(float(r.domain[0]) for r in v);y1=max(float(r.domain[1]) for r in v)
    if not np.isfinite([x0,x1,y0,y1]).all() or x1-x0<80 or y1-y0<30:return None
    # Independent detector may have lost either outer rail or multiple rows.
    dx=max(50,.50*(x1-x0));dy=max(70,.90*(y1-y0))
    a,b,c,d=max(0,int(x0-dx)),max(0,int(y0-dy)),min(W,int(np.ceil(x1+dx))),min(H,int(np.ceil(y1+dy)))
    return (a,b,c,d) if (c-a)*(d-b)<.75*W*H and c-a>=100 and d-b>=80 else None

def translate_parts(parts,box,shape):
    x,y,_,_=box;H,W=shape[:2];out={'h':[],'v':[]}
    for axis,du,dv,extent in [('h',x,y,W),('v',y,x,H)]:
        for _,(a,b,c,d),length in parts[axis]:
            a,b,c,d=a+du,b+dv,c+du,d+dv
            if length<max(18 if axis=='h' else 30,extent*.025):continue
            ref=b+(extent/2-a)*(d-b)/max(c-a,1)
            out[axis].append((float(ref),(a,b,c,d),length))
    return out

def _scan(image,graph):
    from .grid_speed_combined_v1 import scan_graph
    from .grid_anomaly_v3 import scan_above
    from .geometry_lab.algorithms import _dest
    return scan_graph(image,graph,scan_above,_dest)

def _safe_order(image,graph):
    from .detector_routing_v1 import accept_candidate
    from .geometry_lab.algorithms import intersection
    if not accept_candidate(graph,3):return False
    H,W=image.shape[:2];n=np.asarray([[intersection(v,h,image.shape) for v in graph['vertical']] for h in graph['horizontal']])
    return bool(np.isfinite(n).all() and (n[...,0]>=0).all() and (n[...,0]<W).all() and (n[...,1]>=0).all() and (n[...,1]<H).all() and (np.diff(n[...,0],axis=1)>=5).all() and (np.diff(n[...,1],axis=0)>=5).all())

def prove(image,graph,report,automatic,min_rows,box=None):
    """Recover before V2. Never infer rows from a count or a uniform pitch."""
    from .grid_anomaly_v3 import recover_graph
    from .geometry_lab.detection import Rule
    from .grid_anomaly_v5 import scan_side
    from .geometry_lab.algorithms import intersection
    from .detector_routing_v1 import record
    t=perf_counter(); original=graph; diag={}
    try:
        if not _safe_order(image,graph):raise ValueError('invalid_trial_graph')
        scan=_scan(image,graph);diag['initial_top']=scan
        # Use the production Direct Anchor (including its existing V5 hook).
        if scan.get('status')=='suspect':
            try:graph,detail=recover_graph(image,graph,scan,rule_factory=Rule);diag['direct']=detail
            except (ValueError,TypeError,IndexError,KeyError,cv2.error) as exc:diag['direct_error']=str(exc)
        # For unconfirmed or still incomplete graphs use installed V5.2, if enabled.
        if graph is original and (scan.get('status')!='no_signal' or len(graph['horizontal'])<min_rows):
            from .grid_anomaly_v52 import recover,enabled as v52_on
            if v52_on():graph,detail=recover(image,graph,automatic=automatic);diag['v52']=detail
        if not _safe_order(image,graph) or len(graph['horizontal'])<max(3,min_rows):raise ValueError('insufficient_or_crossing_recovered_grid')
        final=_scan(image,graph);diag['final_top']=final
        anchor=final.get('anchor',{});fast=final.get('fast_screen') and final.get('top_support',0)>=.60 and final.get('bottom_support',0)>=.60 and final.get('max_coverage',1)<.25
        if final.get('status')!='no_signal' or not (anchor.get('status')=='aligned' or fast):raise ValueError('top_not_independently_confirmed')
        bottom,detail=scan_side(image,graph,'bottom',Rule);diag['bottom']=detail
        if bottom or detail.get('status')!='no_signal':raise ValueError('bottom_not_confirmed')
        # ROI is a proposal only. Original source coordinates and evidence used above.
        if box is not None:
            x0,y0,x1,y1=box
            nodes=np.array([[intersection(v,h,image.shape) for v in graph['vertical']] for h in graph['horizontal']])
            if (nodes[...,0].min()<x0+6 or nodes[...,0].max()>x1-6 or nodes[...,1].min()<y0+6 or nodes[...,1].max()>y1-6):raise ValueError('roi_truncation_risk')
        # Match all strong original AutoGrid horizontal hints, using the five observed rails.
        # Weak hints are not a row prior and cannot force creation of a line.
        from .grid_anomaly_v51 import curve_proof,_gray
        H,W=image.shape[:2];gray=_gray(image);rails=graph['vertical'];xs=np.linspace(rails[0].position,rails[-1].position,100);u=(xs-W/2)/W
        for h in (automatic or {}).get('horizontal',[]):
            if not curve_proof(gray,h,rails)['accepted']:continue
            if not any(np.percentile(abs(h.eval(u)-r.eval(u)),90)<=10 for r in graph['horizontal']):raise ValueError('missing_supported_autogrid_hint')
        diag['accepted']=True;diag['wall_ms']=(perf_counter()-t)*1000
        record('short_grid_proof',**diag)
        report=dict(report,short_grid_v3=diag)
        if graph is not original:
            report.pop('nodes',None);report['horizontal']=len(graph['horizontal']);report['edges_before_repair']=report.pop('edges',[])
        return graph,report,True
    except (ValueError,TypeError,IndexError,KeyError,AttributeError,OverflowError,ImportError,cv2.error) as exc:
        diag.update(accepted=False,reason=str(exc),wall_ms=(perf_counter()-t)*1000);record('short_grid_proof',**diag)
        return original,report,False

def trial(image,gray,response,dark,automatic,min_rows,core,core_impl,segments):
    from .detector_routing_v1 import record
    if setting('OCR_GRID_SHORT_ROI_V3','0')=='1':
        box=roi_box(automatic,gray.shape)
        if box:
            start=perf_counter()
            try:
                x,y,X,Y=box;parts=translate_parts(segments(gray[y:Y,x:X]),box,gray.shape)
                candidate,report=core_impl(gray,response,dark,None,{'status':'NOT_RUN'},supplied_parts=parts)
                candidate,report,ok=prove(image,candidate,report,automatic,min_rows,box)
                record('short_grid_roi',accepted=ok,bbox=list(box),wall_ms=(perf_counter()-start)*1000)
                if ok:return candidate,report,True
            except (ValueError,TypeError,IndexError,KeyError,AttributeError,OverflowError,cv2.error) as exc:
                record('short_grid_roi',accepted=False,bbox=list(box),reason=str(exc),wall_ms=(perf_counter()-start)*1000)
    candidate,report=core(gray,response,dark,None,{'status':'NOT_RUN'})
    return prove(image,candidate,report,automatic,min_rows)
