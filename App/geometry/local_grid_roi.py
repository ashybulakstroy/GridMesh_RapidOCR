"""Fast geometry-only ROI locator and exact coordinate conversion for GridMesh V3.

Stand-alone: importing this module does not import/modify production App.
No OCR, receipt identifier, user-supplied ground truth, or fixed table coordinates.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np

@dataclass
class Location:
    bbox: tuple[int,int,int,int] | None
    elapsed_ms: float
    reason: str
    diagnostics: dict


def locate_table(image: np.ndarray, *, target_width: int = 590) -> Location:
    """Find a dense, consistent set of parallel horizontal printed lines.

    Low-resolution LSD is a *proposal only*, never accepted as a grid. The
    downstream physical detector must verify five rails and actual ink edges.
    """
    start=time.perf_counter()
    H,W=image.shape[:2]
    if H < 160 or W < 180:
        return Location(None,(time.perf_counter()-start)*1000,'image_too_small',{})
    scale=min(1.0,target_width/W)
    width=max(16,round(W*scale));height=max(16,round(H*scale))
    small=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA) if scale<1 else image
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY) if small.ndim==3 else small
    # Avoid CLAHE here: it can make barcode, trousers and photo texture dominate.
    found=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    segments=[]
    if found is not None:
        for x0,y0,x1,y1 in found.reshape(-1,4):
            dx=float(x1-x0);dy=float(y1-y0)
            if abs(dx)<max(26,width*.09) or abs(dy)>abs(dx)*.19:continue
            if x0>x1:x0,y0,x1,y1=x1,y1,x0,y0
            x0/=scale;x1/=scale;y0/=scale;y1/=scale
            # Global center is unsuitable for small off-center table proposals;
            # group by the mean local line level, preserving the slope.
            segments.append({'x0':float(x0),'x1':float(x1),
                             'y':float((y0+y1)*.5),'length':float(x1-x0),
                             'slope':float((y1-y0)/(x1-x0))})
    if len(segments)<3:
        return Location(None,(time.perf_counter()-start)*1000,'too_few_horizontal_segments',
                        {'segments':len(segments),'image_hw':[H,W],'scale':scale})
    # Collapse split segments from one physical boundary; do not invent lines.
    groups=[]
    for s in sorted(segments,key=lambda a:a['y']):
        nearby=[g for g in groups if abs(g['y']-s['y']) <= max(5.0,H*.006)]
        nearby=[g for g in nearby if s['x1']>=g['min_x']-max(35,W*.1)
                and s['x0']<=g['max_x']+max(35,W*.1)]
        if nearby:
            g=min(nearby,key=lambda g:abs(g['y']-s['y']))
            g['members'].append(s)
            weight=max(s['length'],1.0)
            g['y']=(g['y']*g['length']+s['y']*weight)/(g['length']+weight)
            g['length']+=weight
            g['min_x']=min(g['min_x'],s['x0']);g['max_x']=max(g['max_x'],s['x1'])
        else:
            groups.append({'y':s['y'],'members':[s],'min_x':s['x0'],
                           'max_x':s['x1'],'length':s['length']})
    groups.sort(key=lambda g:g['y'])
    # Consecutive row boundaries in a receipt table tend to be clustered;
    # large gaps separate the upper paper edge and unrelated bottom features.
    runs=[];run=[]
    gap_limit=max(58.0,min(140.0,H*.085))
    for g in groups:
        if run and (g['y']-run[-1]['y']>gap_limit):
            runs.append(run);run=[]
        run.append(g)
    if run:runs.append(run)
    proposals=[]
    for run in runs:
        # Reject 1-2 unrelated rules, but keep 3 lines for shortest receipts.
        if len(run)<3:continue
        for left in range(len(run)):
            for right in range(left+2,len(run)):
                subset=run[left:right+1]
                # A very tall span is unlikely to be a tiny physical grid;
                # long receipts can be found by the production fallback.
                span=subset[-1]['y']-subset[0]['y']
                if span<18 or span>H*.65:continue
                strong=[g for g in subset if (g['max_x']-g['min_x'])>W*.08]
                if len(strong)<3:continue
                mids=[(g['min_x']+g['max_x'])/2 for g in strong]
                mid=float(np.median(mids))
                common=[g for g in strong if abs((g['min_x']+g['max_x'])/2-mid)<W*.18]
                if len(common)<3:continue
                # Prefer 4+ distinct physical levels over longer isolated edges.
                # Avoid rewarding fragment multiplicity or vast vertical spans.
                score=(len(common)*5.0
                      + sum(min(g['max_x']-g['min_x'],W*.35)/W for g in common)*3
                      - span/H*2.5)
                proposals.append((score,subset,common))
    if not proposals:
        return Location(None,(time.perf_counter()-start)*1000,'no_coherent_horizontal_group',
                        {'segments':len(segments),'levels':len(groups),'image_hw':[H,W],
                         'scale':round(scale,4)})
    proposals.sort(key=lambda p:p[0],reverse=True)
    score,selected,common=proposals[0]
    # Extend beyond central row fragments: include left number column, price
    # and VAT, and a lower footer border where one exists.
    left=min(g['min_x'] for g in selected)
    right=max(g['max_x'] for g in selected)
    top=min(g['y'] for g in selected)
    bottom=max(g['y'] for g in selected)
    span_x=right-left;span_y=bottom-top
    x_pad=max(35.0,span_x*.46,W*.055)
    y_pad=max(26.0,span_y*.23)
    x0=max(0,int(np.floor(left-x_pad)));x1=min(W,int(np.ceil(right+x_pad)))
    y0=max(0,int(np.floor(top-y_pad)));y1=min(H,int(np.ceil(bottom+y_pad)))
    # Tight ROI performance is only worthwhile when at least 30% is removed.
    area=(x1-x0)*(y1-y0)
    if x1-x0<70 or y1-y0<55 or area>W*H*.70:
        return Location(None,(time.perf_counter()-start)*1000,'roi_not_small_enough',
                        {'bbox':[x0,y0,x1,y1],'ratio':round(area/(W*H),3)})
    return Location((x0,y0,x1,y1),(time.perf_counter()-start)*1000,'ok',
                    {'image_hw':[H,W],'scaled_hw':[height,width],
                     'scale':round(scale,4),'segments':len(segments),
                     'levels':len(groups),'candidate_group_count':len(proposals),
                     'selected_levels':[round(g['y'],1) for g in selected],
                     'score':round(score,3),'roi_area_fraction':round(area/(W*H),4),
                     'group_bounds':[round(left),round(top),round(right),round(bottom)]})


def offset_rule(rule, bbox:tuple[int,int,int,int], image_shape:tuple[int,int], *, rule_factory=None, polyline_factory=None):
    """Convert a rule evaluated at ROI-normalized coordinates to GLOBAL-normalized.

    A local polynomial `x = P((y - roi_h/2) / roi_h)` must be composed
    with the affine transform of the independent variable; simply adding
    offsets to its constant coefficient is geometrically WRONG.
    """
    x0,y0,x1,y1=bbox
    H,W=image_shape[:2]
    h=y1-y0;w=x1-x0
    if h<1 or w<1:raise ValueError('empty ROI')
    axis=getattr(rule,'axis',None)
    if axis not in ('h','v'):raise ValueError('unsupported rule axis')
    if hasattr(rule,'points'):
        pts=np.asarray(rule.points,float).copy()
        pts[:,0]+=x0;pts[:,1]+=y0
        if polyline_factory is None:
            from App.geometry.geometry_lab.structure_diagnostics import PolylineRule
            polyline_factory=PolylineRule
        return polyline_factory(pts,axis,(H,W))
    if not hasattr(rule,'coeff'):raise ValueError('rule has no polynomial')
    if axis=='v':
        a=H/h;b=(H/2-y0-h/2)/h;shift_across=x0;shift_along=y0
        old_extent=h;new_extent=H
    else:
        a=W/w;b=(W/2-x0-w/2)/w;shift_across=y0;shift_along=x0
        old_extent=w;new_extent=W
    polynomial=np.poly1d(np.asarray(rule.coeff,float))
    transformed=polynomial(np.poly1d([a,b]))+shift_across
    coeff=np.atleast_1d(np.asarray(transformed.c,float))
    domain=(float(rule.domain[0]+shift_along),float(rule.domain[1]+shift_along))
    support=float(max(0.0,(domain[1]-domain[0])/new_extent))
    if rule_factory is None:
        from App.geometry.geometry_lab.detection import Rule
        rule_factory=Rule
    return rule_factory(axis,coeff,float(np.polyval(coeff,0)),support,domain)


def offset_graph(graph:dict,bbox:tuple[int,int,int,int],image_shape:tuple[int,int],**factories)->dict:
    if graph is None:raise ValueError('graph is None')
    return {'vertical':[offset_rule(r,bbox,image_shape,**factories) for r in graph['vertical']],
            'horizontal':[offset_rule(r,bbox,image_shape,**factories) for r in graph['horizontal']],
            'shape':image_shape[:2]}


def roi_overlay(image:np.ndarray,location:Location)->np.ndarray:
    overlay=image.copy()
    if location.bbox is not None:
        x0,y0,x1,y1=location.bbox
        cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,130,255),3)
        cv2.putText(overlay,f'ROI {x1-x0}x{y1-y0}',(x0,max(y0-10,30)),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(0,110,220),2,cv2.LINE_AA)
    return overlay
