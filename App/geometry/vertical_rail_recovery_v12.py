"""Source-evidenced fifth rail recovery for 4-rail Torgsoft receipt grids.

No inferred/equidistant rails. A missing rail is accepted only when the *actual*
photo contains a consistent long vertical stroke throughout the grid, at many
observed row separators, with substantial source-pixel support.
"""
from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np


def _coverage(intervals, top, bottom, gap=9):
    out=[]
    for a,b in sorted((max(float(top),float(a)),min(float(bottom),float(b))) for a,b in intervals):
        if b<=a: continue
        if out and a<=out[-1][1]+gap:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return float(sum(b-a for a,b in out)/max(1.,bottom-top))


def _candidate_rails(image, top, bottom, known_x):
    """Find independent LSD evidence, do NOT build a fifth line from width ratios."""
    H,W=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if raw is None:return []
    mid=(top+bottom)/2
    segments=[]
    for xa,ya,xb,yb in raw.reshape(-1,4):
        xa,ya,xb,yb=map(float,(xa,ya,xb,yb))
        if ya>yb:xa,xb=xb,xa;ya,yb=yb,ya
        dy=yb-ya;dx=xb-xa
        if dy<max(19.,(bottom-top)*.025) or abs(dx)>dy*.23:continue
        # Ignore isolated paper-edge/letter strokes outside the printed table.
        if yb<top+(bottom-top)*.02 or ya>bottom-(bottom-top)*.02:continue
        slope=dx/dy
        intercept=(xa+xb)*.5 + slope*(mid-(ya+yb)*.5)
        # Need an INTERNAL rail. Exterior completion has a different safety case.
        if not min(known_x)+12<intercept<max(known_x)-12:continue
        if min(abs(intercept-x) for x in known_x)<max(10.,W*.014):continue
        segments.append((intercept,xa,ya,xb,yb))
    if not segments:return []
    # Dense/sparse one-dimensional intercept clustering; long rails dominate.
    bins=np.zeros(W,dtype=float)
    for x,xa,ya,xb,yb in segments:
        k=int(round(x))
        if 1<=k<W-1:
            bins[max(0,k-4):min(W,k+5)]+=min(yb-ya,bottom-top)
    from scipy.ndimage import gaussian_filter1d
    smooth=gaussian_filter1d(bins,3)
    from scipy.signal import find_peaks
    peaks,_=find_peaks(smooth,prominence=max(25,float(smooth.max())*.13),distance=max(12,int(W*.023)))
    ranked=sorted(peaks,key=lambda ix:smooth[ix],reverse=True)
    candidates=[]
    for ix in ranked[:16]:
        if min(abs(float(ix)-x) for x in known_x)<max(11.,W*.017):continue
        selected=[s for s in segments if abs(s[0]-ix)<max(9.,W*.017)]
        if not selected:continue
        coverage=_coverage(((s[2],s[4]) for s in selected),top,bottom)
        # Sum of fragment length cannot replace span coverage.
        if coverage<.68:continue
        candidates.append({'x':float(ix),'coverage':coverage,'segments':selected})
    return candidates


def _fit_rail(image,segments,top,bottom):
    H,W=image.shape[:2]
    points=[]
    for _,xa,ya,xb,yb in segments:
        for y,x in ((ya,xa),(yb,xb),((ya+yb)/2,(xa+xb)/2)):
            if top-8<=y<=bottom+8:points.append((y,x))
    if len(points)<9:return None
    ys=np.array([p[0] for p in points],float)
    xs=np.array([p[1] for p in points],float)
    norm=(ys-H/2)/H
    keep=np.ones(len(xs),bool)
    for _ in range(4):
        if int(keep.sum())<9 or np.ptp(ys[keep])<.58*(bottom-top):return None
        coef=np.polyfit(norm[keep],xs[keep],1)
        error=np.abs(np.polyval(coef,norm)-xs)
        new=error<max(5.,W*.014)
        if np.array_equal(new,keep):break
        keep=new
    if keep.sum()<9:return None
    if np.ptp(ys[keep])<.58*(bottom-top):return None
    return coef, float(np.mean(np.abs(np.polyval(coef,norm[keep])-xs[keep])))


def _source_bridge_score(image,coef,row_y,top,bottom):
    """At each observed row crossing, verify vertical ink ABOVE and BELOW."""
    H,W=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
    # Vertical morphology suppresses characters and horizontal separators.
    vh=max(11,min(23,round((bottom-top)/42)))
    ink=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((vh,1),np.uint8))
    good=0;details=[]
    for y in row_y:
        if y<top+12 or y>bottom-12:continue
        px=int(round(np.polyval(coef,(y-H/2)/H)))
        if not 6<=px<W-6:continue
        a=ink[max(0,int(y)-13):max(0,int(y)-3),px-5:px+6]
        b=ink[min(H,int(y)+4):min(H,int(y)+14),px-5:px+6]
        up=bool(a.size and np.count_nonzero(a)>=4)
        dn=bool(b.size and np.count_nonzero(b)>=4)
        good+=up and dn
        details.append([round(float(y),1),int(up),int(dn)])
    return float(good/max(1,len(details))),details


def recover_missing_fifth_rail(image,graph,*,rule_factory=None,intersection_fn=None):
    """Return (new_graph_or_None, diagnostics). Only 4->5 and source-evidenced."""
    start=perf_counter()
    diag={'version':'vertical_rail_recovery_v8','status':'rejected'}
    def fail(reason):
        diag.update(reason=reason,elapsed_ms=round((perf_counter()-start)*1000,2))
        return None,diag
    if graph is None or len(graph.get('vertical',[]))!=4 or len(graph.get('horizontal',[]))<4:
        return fail('not_a_four_rail_graph')
    if rule_factory is None:
        from .geometry_lab.detection import Rule as rule_factory
    if intersection_fn is None:
        from .geometry_lab.algorithms import intersection as intersection_fn
    try:
        H,W=image.shape[:2]
        nodes=np.asarray([[intersection_fn(v,h,image.shape) for v in graph['vertical']]
                          for h in graph['horizontal']],dtype=float)
        if not np.isfinite(nodes).all():return fail('nonfinite_four_rail_nodes')
        if np.any(np.diff(nodes[:,:,0],axis=1)<5) or np.any(np.diff(nodes[:,:,1],axis=0)<4):
            return fail('original_four_rails_cross')
        known_x=np.median(nodes[:,:,0],axis=0).tolist()
        row_y=np.median(nodes[:,:,1],axis=1)
        top=float(row_y[0]);bottom=float(row_y[-1])
        if bottom-top<110:return fail('grid_too_short_for_reliable_recovery')
        candidates=_candidate_rails(image,top,bottom,known_x)
        diag['candidate_count']=len(candidates)
        shortlist=[]
        for candidate in candidates:
            fit=_fit_rail(image,candidate['segments'],top,bottom)
            if fit is None:continue
            coef,rms=fit
            bridge,details=_source_bridge_score(image,coef,row_y,top,bottom)
            item={'x':round(candidate['x'],1),'coverage':round(candidate['coverage'],3),
                  'rms_px':round(rms,2),'row_bridge_fraction':round(bridge,3),
                  'segment_count':len(candidate['segments'])}
            shortlist.append((item,coef))
        diag['candidates']=[x[0] for x in shortlist]
        shortlist=[(x,c) for x,c in shortlist if x['rms_px']<=max(3.3,W*.01)
                   and x['row_bridge_fraction']>=.60]
        if len(shortlist)!=1:
            return fail('no_unique_source_supported_rail')
        best,coef=shortlist[0]
        recovered=rule_factory('v',np.array(coef,dtype=float),
                               float(np.polyval(coef,0)),float(best['coverage']),
                               (float(top),float(bottom)))
        vertical=sorted([*graph['vertical'],recovered],key=lambda r:r.position)
        new={**graph,'vertical':vertical}
        corrected=np.asarray([[intersection_fn(v,h,image.shape) for v in vertical]
                              for h in graph['horizontal']],dtype=float)
        if not np.isfinite(corrected).all() or np.any(np.diff(corrected[:,:,0],axis=1)<=6):
            return fail('new_rail_crosses_existing_graph')
        diag.update(status='recovered',source_evidence=best,
                    inserted_at=int(next(i for i,v in enumerate(vertical) if v is recovered)),
                    elapsed_ms=round((perf_counter()-start)*1000,2))
        return new,diag
    except (cv2.error,ValueError,TypeError,IndexError,KeyError,np.linalg.LinAlgError) as exc:
        return fail(type(exc).__name__+': '+str(exc))
