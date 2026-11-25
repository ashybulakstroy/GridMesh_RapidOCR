"""Automatic table ROI + bounded printed grid, no GT/ROI inputs during inference.

This is an exploratory detector, intended as a lab plugin, never as a silent
replacement for production OCR. All coordinates are in original image pixels.
"""
from __future__ import annotations
from App.ocr.performance_deep.live_capture import timed as _deep_timed

from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Candidate:
    box: tuple[int, int, int, int]
    source: str
    count: int = 0
    score: float = 0.0
    diagnostics: dict | None = None


@_deep_timed("grid", "App/geometry/autogrid/detector.py:_h_v_segments")
def _h_v_segments(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    masks={a:np.zeros((h,w),np.uint8) for a in ('h','v')}
    segments={'h':[],'v':[]}
    if raw is not None:
        for a,b,c,d in raw.reshape(-1,4):
            a,b,c,d=map(float,(a,b,c,d))
            dx,dy=c-a,d-b
            if abs(dx)>2.1*abs(dy) and abs(dx)>max(20,w*.025):
                axis='h'
            elif abs(dy)>2.1*abs(dx) and abs(dy)>max(21,h*.014):
                axis='v'
            else: continue
            segments[axis].append((a,b,c,d))
            cv2.line(masks[axis],(round(a),round(b)),(round(c),round(d)),255,2,cv2.LINE_8)
    return gray,segments,masks


@_deep_timed("grid", "App/geometry/autogrid/detector.py:_morph_masks")
def _morph_masks(gray):
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
    hmask=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,38),np.uint8))
    vmask=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((29,1),np.uint8))
    return {'h':hmask,'v':vmask}


def _components(masks,src,h,w):
    combined=cv2.max(masks['h'],masks['v'])
    combined=cv2.morphologyEx(combined,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    cnt,labels,stats,cen=cv2.connectedComponentsWithStats(combined,8)
    out=[]
    for x,y,ww,hh,area in stats[1:]:
        if ww<max(175,w*.24) or hh<max(105,h*.072): continue
        if ww>w*.94 or hh>h*.78:continue
        if ww*hh==0:continue
        # reject low-information background structures; score actual grid later
        if not (y+hh>h*.20 and y<h*.87): continue
        out.append(Candidate((int(x),int(y),int(x+ww),int(y+hh)),src))
    return out


def _cluster_groups(peaks,scores,maximum_gap=100):
    groups=[]
    for p in sorted(peaks):
        if groups and p-groups[-1][-1]<=maximum_gap:groups[-1].append(int(p))
        else:groups.append([int(p)])
    return groups


def _wave_peaks(v, smooth=4.2, distance=15, height=8):
    arr=gaussian_filter1d(v.astype(np.float64),smooth)
    indexes,props=find_peaks(arr,distance=distance,prominence=max(2,0.035*arr.max()))
    return [(int(k),float(arr[k])) for k in indexes if arr[k]>max(height,0.13*arr.max())]


def _box_peaks(box,masks):
    x0,y0,x1,y1=box
    # local LSD stroke lengths rather than connected component area
    H=masks['h'][y0:y1,x0:x1]
    V=masks['v'][y0:y1,x0:x1]
    if H.size==0:return [],[],0,0
    ww=x1-x0;hh=y1-y0
    hp=_wave_peaks(H.sum(axis=1)/255.,smooth=4,distance=max(16,int(hh*.027)),height=max(5,ww*.12))
    vp=_wave_peaks(V.sum(axis=0)/255.,smooth=4,distance=max(16,int(ww*.03)),height=max(5,hh*.12))
    hp=[(y0+y,s) for y,s in hp]
    vp=[(x0+x,s) for x,s in vp]
    return hp,vp,float((H>0).mean()),float((V>0).mean())


@_deep_timed("grid", "App/geometry/autogrid/detector.py:_rank_candidates")
def _rank_candidates(gray,masks,more):
    h,w=gray.shape
    proposals=_components(masks,'lsd_connected',h,w)+_components(more,'morph_connected',h,w)
    # Add union for two disconnected halves at matching y, where there are row lines
    orig=list(proposals)
    for i,a in enumerate(orig):
        x0,y0,x1,y1=a.box
        for b in orig[i+1:]:
            xx0,yy0,xx1,yy1=b.box
            inter=max(0,min(y1,yy1)-max(y0,yy0))
            if inter<.67*min(y1-y0,yy1-yy0):continue
            dx=max(x0,xx0)-min(x1,xx1)
            if dx>50:continue
            bb=(min(x0,xx0),min(y0,yy0),max(x1,xx1),max(y1,yy1))
            if (bb[2]-bb[0])>w*.92 or (bb[3]-bb[1])>h*.78:continue
            proposals.append(Candidate(bb,'merged'))
    bybox={}
    for c in proposals:
        bx=c.box
        # +8px pad for whole-line coverage, but no header/footer flood fill
        x0=max(0,bx[0]-5);x1=min(w,bx[2]+5)
        y0=max(0,bx[1]-5);y1=min(h,bx[3]+5)
        c.box=(x0,y0,x1,y1)
        if c.box not in bybox: bybox[c.box]=c
    candidates=list(bybox.values())
    for c in candidates:
        x0,y0,x1,y1=c.box;ww=x1-x0;hh=y1-y0
        hp,vp,h_density,v_density=_box_peaks(c.box,masks)
        mhp,mvp,_,_=_box_peaks(c.box,more)
        # Prefer genuine intersecting grids, with multiple columns and rows.
        # Avoid a long vertical stripe crossing the whole paper/background.
        nh=len(hp);nv=len(vp)
        safe_h=min(nh,14);safe_v=min(nv,8)
        intersects=safe_h*safe_v
        # grid peaks in both detectors and geometrically reasonable table size
        size_bonus=np.sqrt((ww/w)*(hh/h))
        c.score=(max(0,safe_h-1)*max(0,safe_v-1)*1.5 + min(len(mhp),14)*min(len(mvp),8)*.2 + intersects*.2)*float(size_bonus)
        c.score+=max(0, min(nh,12)-2)*2 + max(0,min(nv,8)-2)*2
        if ww>w*.85:c.score*=.55
        if hh>h*.60:c.score*=.53
        if h_density<.045:c.score*=.75
        if len(mvp)>=3:c.score*=1.15
        c.count=nh+nv
        c.diagnostics={'h_peaks':[(int(y),round(s,2)) for y,s in hp],
                       'v_peaks':[(int(x),round(s,2)) for x,s in vp],
                       'morph_h_peaks':len(mhp),'morph_v_peaks':len(mvp),
                       'h_density':round(h_density,4),'v_density':round(v_density,4)}
    candidates.sort(key=lambda c:c.score,reverse=True)
    return candidates




def _strong_vertical_pair_proposals(image, segments, limit=10):
    """Independent fallback when connected-component ROI selected wrong region.

    Generate bounding proposals from two independently observed long, mutually
    overlapping vertical strokes spanning much of the image width. Barcode
    stripes are too close together to qualify. No known table size, GT, manual
    rectangle or OCR content enters the search.
    """
    h, w = image.shape[:2]
    verticals = []
    for xa, ya, xb, yb in segments.get('v', []):
        ylo, yhi = min(ya, yb), max(ya, yb)
        length = yhi-ylo
        if length < h*.095 or ylo > h*.88 or yhi < h*.18:
            continue
        verticals.append(((xa+xb)/2, ylo, yhi, length))
    ranked = []
    for i, a in enumerate(verticals):
        for b in verticals[i+1:]:
            left, right = sorted((a,b), key=lambda line: line[0])
            width = right[0]-left[0]
            if width < w*.48 or width > w*.92:
                continue
            overlap = min(a[2], b[2])-max(a[1], b[1])
            if overlap < max(70, h*.075):
                continue
            top_delta = abs(a[1]-b[1])
            bottom_delta = abs(a[2]-b[2])
            # Two mutually aligned exterior rails are much stronger evidence
            # than unrelated background edges; no prescribed 4/6-column count.
            score = (overlap/(1+top_delta/180+bottom_delta/200)
                     + width/10-abs(a[3]-b[3])*.15)
            x0 = max(0, int(left[0]-20))
            x1 = min(w, int(right[0]+20))
            y0 = max(0, int(min(a[1], b[1])-20))
            y1 = min(h, int(max(a[2], b[2])+20))
            if x1-x0 >= w*.48 and y1-y0>=h*.11:
                ranked.append((score,(x0,y0,x1,y1)))
    ranked.sort(key=lambda k:k[0],reverse=True)
    candidates, seen = [], set()
    for score, box in ranked:
        signature=tuple(round(v/15) for v in box)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(Candidate(box, 'paired_vertical_rails', score=score))
        if len(candidates)>=limit:
            break
    return candidates


@_deep_timed("grid", "App/geometry/autogrid/detector.py:_recover_by_observed_vertical_rails")
def _recover_by_observed_vertical_rails(image, segments):
    """Try small ranked candidate set; accept only independently detected grid."""
    from .fitter import fit_grid
    proposals = _strong_vertical_pair_proposals(image,segments)
    for candidate in proposals:
        # Keep the original stringent threshold if it already finds the lines.
        # Relaxed support is an alternate fallback, never a replacement for a
        # better-supported existing result on the very same ROI.
        standard=fit_grid(image,candidate.box,relaxed_horizontal_support=False,anchor_right=True)
        relaxed=fit_grid(image,candidate.box,relaxed_horizontal_support=True,anchor_right=True)
        fitted=max((standard,relaxed), key=lambda r:
                   (r['status']=='CURVES_FITTED',r['horizontal_boundaries'],
                    r['vertical_boundaries']))
        # Reject cropped two-line proposals and support-deficient tables.
        # All accepted boundaries must have actually produced a fitted curve.
        if (fitted['status']=='CURVES_FITTED'
                and fitted['horizontal_boundaries']>=3
                and fitted['vertical_boundaries']>=4
                and fitted['vertical_boundaries']<=8
                and all(x is not None for x in fitted['h_curves']+fitted['v_curves'])):
            return candidate, fitted
    return None, None

def detect(image):
    """Return (overlay, JSON-serializable metrics); does not consult GT or prior ROIs."""
    tic=perf_counter()
    height,width=image.shape[:2]
    gray,segments,lsd_masks=_h_v_segments(image)
    morph_masks=_morph_masks(gray)
    candidates=_rank_candidates(gray,lsd_masks,morph_masks)
    best=candidates[0] if candidates else None
    overlay=image.copy()
    from .fitter import fit_grid
    if best is None:
        recovered_roi, recovered_result = _recover_by_observed_vertical_rails(image, segments)
        if recovered_roi is None:
            return overlay,{'status':'NO_DETECTION','auto_roi':None,
                            'horizontal_boundaries':0,'vertical_boundaries':0,
                            'elapsed_seconds':round(perf_counter()-tic,4),
                            'candidates':[]}
        best, result = recovered_roi, recovered_result
        roi_fallback=True
    else:
        # The primary component-based ROI is unchanged for previously working
        # photos; fallback activates only on an actual rejected initial grid.
        result=fit_grid(image,best.box)
        roi_fallback=False
        if result['status']!='CURVES_FITTED':
            recovered_roi, recovered_result = _recover_by_observed_vertical_rails(image, segments)
            if recovered_roi is not None:
                best, result = recovered_roi, recovered_result
                roi_fallback=True
    # If a substantial unaccounted strip remains at the left of the first
    # detected vertical, rerun with a wider search region. This is evidence
    # recovery, not a synthesized border: a new vertical must itself be
    # observed by LSD and satisfy the same peak-strength criterion.
    recovered=False
    left_positions=result.get('v_positions',[])
    x0,y0,x1,y1=best.box
    gap=left_positions[0]-x0 if left_positions else 0
    if x0>0 and gap>max(36,.075*(x1-x0)):
        for extra in [max(10,round((x1-x0)*.025)),
                      max(18,round((x1-x0)*.04)),
                      max(25,round((x1-x0)*.055))]:
            newbox=(max(0,x0-extra),y0,x1,y1)
            trial=fit_grid(image,newbox)
            if (len(trial.get('v_positions',[]))>len(left_positions)
                    and trial['horizontal_boundaries']>=result['horizontal_boundaries']):
                result=trial
                x0,y0,x1,y1=newbox
                recovered=True
                break
    overlay=result.pop('overlay')
    # Draw automatically inferred region in orange for diagnostic (not OCR pixels).
    cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,165,255),2,cv2.LINE_AA)
    res={'status':result['status'], 'auto_roi':[x0,y0,x1,y1],
         'auto_roi_expanded_after_vertical_evidence':recovered,
         'roi_detector':best.source,'roi_proposal_score':round(best.score,2),
         'vertical_pair_roi_fallback_used':roi_fallback,
         'geometry_review_required':any(abs(a.get('correction_px',0))>20
             for a in result.get('right_horizontal_ink_anchors', [])),
         'candidate_boxes':[{'box':list(c.box),'source':c.source,'score':round(c.score,2),
                             'num_peaks':c.count} for c in candidates[:7]],
         'elapsed_seconds':round(perf_counter()-tic,4),
         'input_shape':[height,width]}
    res.update(result)
    return overlay,res
