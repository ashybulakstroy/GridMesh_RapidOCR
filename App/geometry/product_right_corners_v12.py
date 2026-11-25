"""Observed right-corner refinement for *existing* upper product separators.

V12 does not insert rows, guess a target from expected row height, or
extrapolate a free quadratic from the left. Physical horizontal LSD edges
near the right two columns are required for each individual correction.
Reject unsafe/ambiguous evidence; later independent whole-paper audit remains
mandatory. Source-image coordinates, not OCR result, govern the adjustment.
"""
from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np


def _intersect(v,h,shape):
    H,W=shape[:2]; x=float(v.position);y=float(h.position)
    for _ in range(10):
        x=float(v.eval((y-H/2)/H));y=float(h.eval((x-W/2)/W))
    return x,y


def _profile(image, top, bottom):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    lo=max(0,int(np.floor(top)));hi=min(gray.shape[0],int(np.ceil(bottom)))
    if hi-lo<30:return []
    enhanced=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(gray[lo:hi])
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(enhanced)[0]
    out=[]
    if lines is not None:
        for xa,ya,xb,yb in lines.reshape(-1,4):
            xa=float(xa);xb=float(xb);ya=float(ya+lo);yb=float(yb+lo)
            length=abs(xb-xa)
            if length<14 or abs(yb-ya)>length*.17:continue
            out.append((xa,ya,xb,yb,length))
    return out


def _adjust_rule(rule, width, left, right, shift, rule_factory):
    # Keep left 60% exactly; approach right printed evidence smoothly.
    x0=left+.61*(right-left);x1=left+.87*(right-left)
    xx=np.linspace(left,right,131)
    t=(xx-width/2)/width
    u=np.clip((xx-x0)/(x1-x0),0.,1.)
    target=np.polyval(rule.coeff,t)+shift*u*u*(3-2*u)
    # Mesh/offset_graph use polynomial coefficients. Approximate a smooth
    # piecewise rule in that native representation, and verify the error.
    fit=np.polynomial.Polynomial.fit(t,target,8,domain=[t[0],t[-1]]).convert()
    co=np.asarray(fit.coef[::-1],float)
    if np.max(np.abs(np.polyval(co,t)-target))>.75:return None
    if np.max(np.abs(np.polyval(co,t[:28])-np.polyval(rule.coeff,t[:28])))>1.:return None
    return rule_factory('h',co,float(np.polyval(co,0)),float(rule.support),tuple(rule.domain))


def refine_product_right_corners(image,graph,*,rule_factory=None,max_upper_rows=7):
    """Refine only source-backed upper row ends; fail-closed otherwise.

    Applies to upper existing product rows after recovering header boundaries.
    Never shifts newly recovered header rules; never invents a missing row.
    Independent full-paper audit must approve the modified graph downstream.
    """
    began=perf_counter()
    info={'version':'product_right_corners_v12','status':'unchanged',
          'refined':0,'decisions':[]}
    def finish(g,status=None):
        if status:info['status']=status
        info['elapsed_ms']=round((perf_counter()-began)*1000,2)
        return g,info
    if image is None or not image.size or not graph:return finish(graph,'invalid_input')
    vs=list(graph.get('vertical') or []); hs=list(graph.get('horizontal') or [])
    if len(vs)!=5 or len(hs)<5:return finish(graph,'unsupported_graph')
    try:
        from copy import copy
        if rule_factory is None:
            from .geometry_lab.detection import Rule
            rule_factory=Rule
        H,W=image.shape[:2]
        nodes=np.array([[_intersect(v,h,image.shape) for v in vs] for h in hs])
        if not np.isfinite(nodes).all():return finish(graph,'bad_geometry')
        left=float(nodes[0,0,0]);right=float(nodes[0,-1,0]);span=right-left
        if span<90:return finish(graph,'narrow_table')
        hi=min(len(hs)-1,max_upper_rows+1)
        start=max(0,float(nodes[1,:,1].min()-15))
        end=min(H,float(nodes[hi,:,1].max()+15))
        segments=_profile(image,start,end)
        revised=list(hs)
        for i in range(2,hi+1):
            row=nodes[i]
            candidates=[]
            for xa,ya,xb,yb,length in segments:
                xm=(xa+xb)/2; ym=(ya+yb)/2
                if not left+.79*span<=xm<=right+7:continue
                expected=float(hs[i].eval((xm-W/2)/W))
                error=ym-expected
                if abs(error)>12.:continue
                candidates.append((xm,error,length))
            # Require at least two segments with independent ink support;
            # otherwise preserve the original (never claim it was verified).
            if len(candidates)<2:
                info['decisions'].append({'row':i,'status':'no_right_support','segments':len(candidates)})
                continue
            # Prefer segments closest to the right rail; avoid left-column text.
            candidates.sort(key=lambda c:c[0],reverse=True)
            close=[v for v in candidates if v[0]>=candidates[0][0]-40]
            if len(close)<2:
                info['decisions'].append({'row':i,'status':'insufficient_right_segments'})
                continue
            offsets=np.asarray([v[1] for v in close],float)
            if np.ptp(offsets)>7.5:
                info['decisions'].append({'row':i,'status':'ambiguous_ink','spread_px':round(float(np.ptp(offsets)),2)})
                continue
            weights=np.asarray([v[2] for v in close],float)
            if weights.sum()<33.:
                info['decisions'].append({'row':i,'status':'weak_support'})
                continue
            correction=float(np.average(offsets,weights=weights))
            if abs(correction)<2.0:
                info['decisions'].append({'row':i,'status':'already_aligned','offset_px':round(correction,2)})
                continue
            new_rule=_adjust_rule(hs[i],W,float(row[0,0]),float(row[-1,0]),correction,rule_factory)
            if new_rule is None:
                info['decisions'].append({'row':i,'status':'polynomial_approximation_rejected'})
                continue
            # No crossing of the other paper-observed rows at any physical rail.
            yy=np.asarray([float(new_rule.eval((x-W/2)/W)) for x in row[:,0]])
            if np.any(yy<nodes[i-1,:,1]+6) or np.any(yy>nodes[i+1,:,1]-6):
                info['decisions'].append({'row':i,'status':'topology_rejected'})
                continue
            revised[i]=new_rule
            info['refined']+=1
            info['decisions'].append({'row':i,'status':'refined','offset_px':round(correction,2),
               'right_y_before':round(float(row[-1,1]),2),'right_y_after':round(float(yy[-1]),2),
               'right_segments':len(close),'support_length_px':round(float(weights.sum()),1)})
        return finish({**graph,'horizontal':revised},'source_corners_refined' if info['refined'] else 'unchanged')
    except (TypeError,ValueError,IndexError,cv2.error,np.linalg.LinAlgError,FloatingPointError) as exc:
        info['error']=type(exc).__name__+': '+str(exc)
        return finish(graph,'exception_rejected')
