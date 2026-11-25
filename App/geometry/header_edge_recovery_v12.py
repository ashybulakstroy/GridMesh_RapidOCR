"""V12: recover *source-observed* printed header boundaries above an incomplete grid.

Not an extrapolation of expected row count. Each proposed boundary must have
independent horizontal printed strokes (LSD on an enhanced original crop) in
multiple physical columns, persistent vertical rails, and non-crossing topology.
No VAT/НДС requirement. Does NOT modify pixels or OCR observations.
"""
from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np


def _intersect(v, h, shape):
    H, W = shape[:2]
    x=float(v.position); y=float(h.position)
    for _ in range(10):
        x=float(v.eval((y-H/2)/H))
        y=float(h.eval((x-W/2)/W))
    return x,y


def _coverage(intervals, a, b, gap=5):
    merged=[]
    for lo, hi in sorted(intervals):
        lo=max(a,lo); hi=min(b,hi)
        if hi<=lo:continue
        if merged and lo<=merged[-1][1]+gap:
            merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:merged.append((lo,hi))
    return sum(hi-lo for lo,hi in merged)/max(1.,b-a)


def _bridge(image, xcoords, head_y, base_y):
    """Require observed printed vertical rails in the space above first row."""
    lo=max(0,int(head_y)+7);hi=min(image.shape[0],int(base_y)-7)
    if hi-lo<22:return {'supported':0,'coverage':[], 'reason':'no_bridge_span'}
    gray=cv2.cvtColor(image[lo:hi],cv2.COLOR_BGR2GRAY) if image.ndim==3 else image[lo:hi]
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV,31,9)
    vk=max(9,min(19,int((hi-lo)*.22)))
    vl=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((vk,1),np.uint8))
    ratios=[]
    for x in xcoords:
        xx=int(round(x))
        if xx<6 or xx>=image.shape[1]-6:
            ratios.append(0.0);continue
        ink=np.any(vl[:,xx-5:xx+6]>0,axis=1)
        ratios.append(float(ink.mean()))
    return {'supported':int(sum(v>=.22 for v in ratios)),
            'coverage':[round(v,3) for v in ratios],
            'y_range':[int(lo),int(hi)]}


def _shape_constrained_fit(source_rule, xx, yy, xlines, width):
    """Fit the observed ROW OFFSET relative to the adjacent printed separator.

    A free quadratic fitted on the left/centre can curve upward 10+ px on an
    unsupported right edge. Instead, inherit source-observed curvature from
    the immediately adjacent physical rule and fit only a robust affine offset.
    The small residual tilt is bounded unless direct image evidence supports
    stronger movement; do not invent an independent polynomial bend.
    """
    xx=np.asarray(xx,dtype=np.float64)
    yy=np.asarray(yy,dtype=np.float64)
    xlines=np.asarray(xlines,dtype=np.float64)
    span=float(xlines[-1]-xlines[0])
    if span<70 or xx.size<6 or not np.isfinite(xx).all() or not np.isfinite(yy).all():
        return None
    mid=float(np.median(xlines))
    norm=(xx-width/2.)/width
    baseline=np.asarray(source_rule.eval(norm),float)
    difference=yy-baseline
    center=float(np.median(difference))
    # Weight source fragments at their observed endpoints only; iteratively
    # suppress glyph strokes and other non-separator outliers.
    X=np.column_stack(((xx-mid)/span,np.ones(len(xx))))
    fit=np.asarray([0.,center],float)
    for _ in range(7):
        residual=difference-X@fit
        weights=np.minimum(1.,3./np.maximum(np.abs(residual),1e-6))
        try:
            fit=np.linalg.lstsq(X*weights[:,None],difference*weights,rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
    # Do not extrapolate a novel curve beyond the photographed strokes.
    # A small tilt remains possible; strong independent bending requires a
    # separate physically verified reconstruction and must fail closed here.
    max_drift=min(5.,max(2.,span*.015))
    slope=float(np.clip(fit[0],-max_drift,max_drift))
    offset=float(np.clip(fit[1],center-3.,center+3.))
    pred=baseline+slope*(xx-mid)/span+offset
    rms=float(np.sqrt(np.mean(np.minimum(np.abs(pred-yy),14.)**2)))
    # Convert the measured offset back to the exact source_rule normalization:
    # t=(x-width/2)/width.
    affine=np.poly1d([slope*width/span,offset+slope*(width/2.-mid)/span])
    shape=np.poly1d(np.asarray(source_rule.coeff,dtype=float))+affine
    return np.asarray(shape.c,dtype=float),rms,{'offset_median_px':round(center,3),
        'offset_px':round(offset,3),'tilt_across_table_px':round(slope,3),
        'fit_model':'adjacent_observed_curve_plus_robust_affine_offset'}


def _right_ink_anchor(fragments, coefficients, line_rails, image_width):
    """Reconcile right of proposed curve with actual source LSD strokes.

    Fails closed when right-side physical evidence is absent. This does NOT
    claim that all printed lines are correct; full row audit stays mandatory.
    """
    rails=np.asarray(line_rails,dtype=float)
    xl,xr=float(rails[0]),float(rails[-1]);span=xr-xl
    base=np.poly1d(np.asarray(coefficients,float))
    right_start=xl+.70*span
    candidates=[]
    for f in fragments:
        xa,ya,xb,yb=(float(f[k]) for k in ('x0','y0','x1','y1'))
        middle=(xa+xb)/2.
        length=abs(xb-xa)
        if middle<right_start or middle>xr+5 or length<14.:
            continue
        dy=yb-ya
        if abs(dy)>length*.18:continue
        measured=(ya+yb)/2.
        expected=float(base((middle-image_width/2.)/image_width))
        deviation=measured-expected
        if not 1.5<=abs(deviation)<=17.:continue
        candidates.append((middle,deviation,length))
    if not candidates:
        return None,{'status':'unverified_right','right_support_segments':0}
    # Favor direct evidence close to last physical rail, reject conflicting
    # right anchors: text strokes can otherwise move a genuine separator.
    candidates.sort(key=lambda a:a[0],reverse=True)
    best=candidates[0]
    agreeing=[v for v in candidates if abs(v[1]-best[1])<5.]
    disagreed=[v for v in candidates if abs(v[1]-best[1])>=7. and abs(v[0]-best[0])<20.]
    if disagreed:
        return None,{'status':'ambiguous_right','right_support_segments':len(candidates)}
    evidence_length=sum(p[2] for p in agreeing)
    if evidence_length<14.:
        return None,{'status':'weak_right','right_support_segments':len(agreeing)}
    correction=float(np.median([v[1] for v in agreeing]))
    anchor_x=float(np.average([v[0] for v in agreeing],weights=[v[2] for v in agreeing]))
    # Preserve the left half and transition continuously using smoothstep.
    x0=xl+.62*span  # preserve observed left and central separators
    x1=max(x0+20, anchor_x)
    return {'start_x':float(x0),'end_x':float(x1),'correction_px':correction,
            'anchor_x':anchor_x,'support_length_px':round(evidence_length,2),
            'segments':len(agreeing),'status':'right_observed'},dict(status='right_observed',
            corrected_by_px=round(correction,2), anchor_global_x=None,
            support_length_px=round(evidence_length,2),support_segments=len(agreeing))


def _apply_smooth_right_correction(coefficients, anchor, image_width):
    """Cubic smoothstep fitted to observed right rail; no left-side drift.

    First evaluate a smooth piecewise transition, then fit a true polynomial
    Rule.coeff across the observed table span for GridMesh compatibility.
    """
    base=np.asarray(coefficients,dtype=float)
    def evaluation(t):
        x=np.asarray(t,dtype=float)*image_width+image_width/2.
        u=np.clip((x-anchor['start_x'])/max(1.,anchor['end_x']-anchor['start_x']),0.,1.)
        v=np.polyval(base,t)+anchor['correction_px']*(u*u*(3.-2.*u))
        return float(v) if np.ndim(v)==0 else v
    return evaluation


def _right_anchor_polynomial(coefficients,anchor,width,rails):
    """Approximate observed right-side bend as a genuine polynomial Rule.

    GridMesh's offset_graph() and V3 Mesh use Rule.coeff, not custom eval.
    Derive a degree-7 polynomial over the *observed table span* and require
    <=0.8px approximation error; otherwise refuse the local rescue.
    """
    xx=np.linspace(float(rails[0]),float(rails[-1]),101)
    ts=(xx-width/2.)/width
    values=_apply_smooth_right_correction(coefficients,anchor,width)(ts)
    poly=np.polynomial.Polynomial.fit(ts,values,7,domain=[ts.min(),ts.max()]).convert()
    fit=np.asarray(poly.coef[::-1],float)
    mismatch=float(np.max(np.abs(np.polyval(fit,ts)-values)))
    if mismatch>0.8 or not np.isfinite(fit).all():return None,mismatch
    return fit,mismatch


def recover_header_boundaries(image, graph, *, rule_factory=None):
    """Add observed row boundaries above first grid row, or keep unchanged.

    Returns graph, diagnostics; the calling full-height audit and header OCR
    MUST still approve the result before it can become a receipt grid.
    """
    started=perf_counter()
    report={'version':'header_edge_recovery_v12','status':'not_recovered',
            'observations':[], 'recovered':0}
    def done(value, reason):
        report['status']=reason
        report['elapsed_ms']=round((perf_counter()-started)*1000,3)
        return value,report
    if image is None or not image.size or graph is None:
        return done(graph,'missing_image_or_grid')
    vv=list(graph.get('vertical') or []);hh=list(graph.get('horizontal') or [])
    if len(vv)!=5 or len(hh)<4:
        return done(graph,'unsupported_grid_shape')
    H,W=image.shape[:2]
    try:
        top=np.asarray([_intersect(v,hh[0],image.shape) for v in vv],float)
        nextrow=np.asarray([_intersect(v,hh[1],image.shape) for v in vv],float)
        if not np.isfinite(top).all() or np.any(np.diff(top[:,0])<=5):
            return done(graph,'invalid_first_row')
        span=float(top[-1,0]-top[0,0])
        if span<90:return done(graph,'narrow_grid')
        step=float(np.median(nextrow[:,1]-top[:,1]))
        if step<=9:return done(graph,'invalid_row_spacing')
        # Search extends a few *measured* row heights above candidate, with
        # enough room for a two-band printed table header.
        depth=max(110,min(215,round(step*3.9)))
        x0=max(0,int(top[0,0])-12);x1=min(W,int(top[-1,0])+13)
        y1=max(0,int(np.min(top[:,1]))-10)
        y0=max(0,y1-depth)
        if y1-y0<65:return done(graph,'no_upstream_search_region')
        roi=image[y0:y1,x0:x1]
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi
        enh=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(gray)
        detected=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(enh)[0]
        if detected is None:return done(graph,'no_source_horizontal_lines')
        fragments=[]
        for a,b,c,d in detected.reshape(-1,4):
            dx=float(c-a);dy=float(d-b)
            if abs(dx)<max(15.,span*.043) or abs(dy)>abs(dx)*.18:
                continue
            xa=float(a+x0);xb=float(c+x0);ya=float(b+y0);yb=float(d+y0)
            left,right=sorted((xa,xb))
            if right < top[0,0]-4 or left > top[-1,0]+4:continue
            mx=(xa+xb)*.5;my=(ya+yb)*.5
            reference=float(np.interp(mx,top[:,0],top[:,1]))
            d0=my-reference
            if d0>=-16 or d0 < -depth-25:continue
            fragments.append({'delta':d0,'x0':xa,'y0':ya,'x1':xb,'y1':yb,
                              'left':left,'right':right,'length':abs(dx)})
        if not fragments:return done(graph,'no_header_source_segments')
        # Weighted peak votes in displacement from the already observed grid
        # curve. This tolerates bends while separating distinct physical lines.
        deltas=np.asarray([f['delta'] for f in fragments])
        weights=np.asarray([min(110.,f['length']) for f in fragments])
        centers=np.arange(-depth-24,-15,dtype=float)
        votes=np.asarray([float(np.sum(weights*np.exp(-.5*((deltas-c)/6.)**2)))
                          for c in centers])
        # Distinct rows must be separated; non-max suppression avoids double
        # accepting opposite sides of a single thin printed separator.
        peaks=[]
        for i in np.argsort(votes)[::-1]:
            center=float(centers[i])
            if votes[i]<55:break
            if all(abs(center-old)>18 for old in peaks):
                peaks.append(center)
            if len(peaks)>=8:break
        if not peaks:return done(graph,'weak_source_peaks')
        accepted=[]
        xlines=np.asarray(top[:,0],float)
        for center in peaks:
            parts=[f for f in fragments if abs(f['delta']-center)<=11.]
            if not parts:continue
            ints=[(p['left'],p['right']) for p in parts]
            cov=_coverage(ints,float(xlines[0]),float(xlines[-1]))
            col=[_coverage(ints,float(a),float(b)) for a,b in zip(xlines[:-1],xlines[1:])]
            supported=int(sum(v>=.13 for v in col))
            if cov<.48 or supported<3 or max(col[:2])<.17 or max(col[2:])<.17:
                continue
            # V12: keep the VERIFIED source curve's shape across all columns.
            # A separate quadratic on fragments ending before the right rail
            # overfits the extrapolation and misses the right-hand separator.
            xx=np.asarray([x for p in parts for x in (p['x0'],p['x1'])],float)
            yy=np.asarray([y for p in parts for y in (p['y0'],p['y1'])],float)
            fitted=_shape_constrained_fit(hh[0],xx,yy,xlines,W)
            if fitted is None:continue
            coef,rms,fit_detail=fitted
            if rms>7.2:continue
            # V12: test separate right-side evidence. Where not available,
            # reject the line rather than extrapolating the V12 curve.
            anchor,anchor_diag=_right_ink_anchor(
                [f for f in fragments if abs(f['delta']-center)<=20.],coef,xlines,W)
            if anchor is None:
                continue
            predicted=_apply_smooth_right_correction(coef,anchor,W)((xlines-W/2)/W)
            if np.min(top[:,1]-predicted)<13:continue
            polyfit,approx_error=_right_anchor_polynomial(coef,anchor,W,xlines)
            if polyfit is None:continue
            coef=polyfit
            bridge=_bridge(image,xlines,float(np.median(predicted)),float(np.median(top[:,1])))
            if bridge['supported']<4:continue
            obs={'delta':round(center,2),'median_y':round(float(np.median(predicted)),2),
                 'coverage':round(cov,3),'columns':[round(v,3) for v in col],
                 'supported_columns':supported,'fit_rms_px':round(rms,3),
                 'vertical_bridge':bridge,
                 'segments':len(parts),'curve_fit':fit_detail,'right_anchor':anchor_diag,
                 'right_polynomial_error_px':round(approx_error,3)}
            report['observations'].append(obs)
            accepted.append((float(np.median(predicted)),coef,obs))
        accepted.sort(key=lambda t:t[0])
        # Deduplicate near-identical physical printed strokes.
        chosen=[]
        for item in accepted:
            if chosen and item[0]-chosen[-1][0]<17:
                continue
            chosen.append(item)
        if not chosen:return done(graph,'no_supported_top_lines')
        if len(chosen)>5:return done(graph,'ambiguous_many_top_lines')
        if rule_factory is None:
            from .geometry_lab.detection import Rule
            rule_factory=Rule
        extra=[]
        for median_y,coef,obs in chosen:
            c=np.asarray(coef,float)
            position=float(np.polyval(c,0.))
            extra.append(rule_factory('h',c,position,float(obs['coverage']),
                                      (float(xlines[0]),float(xlines[-1]))))
        # Keep every existing measured rule unchanged; add only source-backed
        # upstream lines. An independent audit + OCR header check follow.
        combined={**graph,'horizontal':extra+hh}
        report['recovered']=len(extra)
        report['median_y']=[v['median_y'] for _,_,v in chosen]
        return done(combined,'source_recovered_pending_independent_audit')
    except (ValueError,TypeError,IndexError,cv2.error,FloatingPointError,np.linalg.LinAlgError) as exc:
        report['error']=type(exc).__name__+': '+str(exc)
        return done(graph,'exception_rejected')
