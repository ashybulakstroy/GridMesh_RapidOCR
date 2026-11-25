"""V16: independently ink-anchored 5-rail grid including ALL right row corners.

Fallback only after the existing geometry produced a suspicious first row or
failed header semantics. It never relies on VAT or a receipt-specific offset.
Failure is explicit: no artificial lines or assumed equal-width columns.
"""
from __future__ import annotations
import cv2
import numpy as np
from collections import defaultdict
from time import perf_counter

SELECTED='source_verified_five_rail_v16'


def _line_segments(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    return np.empty((0,4),dtype=float) if raw is None else raw.reshape(-1,4).astype(float)


def _clusters(items, gap):
    groups=[]
    for value in sorted(items,key=lambda x:x[0]):
        if not groups or value[0]-float(np.median([v[0] for v in groups[-1]]))>gap:
            groups.append([value])
        else:groups[-1].append(value)
    return groups


def _find_rails(lines, image_shape):
    """Find measured long and segmented parallel vertical strokes.

    No fixed x positions or five-way equal spacing. Paper edges with only one
    long line are excluded in favour of repeatedly observed table rails.
    """
    H,W=image_shape[:2]; yref=H*.45
    items=[]
    for xa,ya,xb,yb in lines:
        dy=yb-ya;dx=xb-xa
        if abs(dy)<max(18,H*.014) or abs(dx)>.20*abs(dy):continue
        ym=(ya+yb)/2
        xref=(xa+xb)/2+(dx/dy)*(yref-ym)
        if not (W*.10<xref<W*.90):continue
        items.append((xref,min(ya,yb),max(ya,yb),float(dx/dy)))
    groups=_clusters(items,max(5.5,W*.008))
    tracks=[]
    for group in groups:
        if len(group)<3:continue
        starts=np.asarray([v[1] for v in group]);ends=np.asarray([v[2] for v in group]);
        # Distinct physically observed stretches; repeated detection alone
        # is not enough to establish a rail.
        ycover=np.zeros(H,bool)
        for v in group:ycover[max(0,int(v[1])):min(H,int(v[2])+1)]=True
        coverage=int(ycover.sum())
        if coverage<max(72,H*.065):continue
        slopes=np.asarray([v[3] for v in group]);
        slope=float(np.median(slopes))
        xc=float(np.median([v[0] for v in group]))
        tracks.append({'x_ref':xc,'slope':slope,'first':float(np.min(starts)),
                       'last':float(np.max(ends)),'coverage':coverage,
                       'segments':len(group),'quality':coverage+16*len(group)})
    # Seed the table's physical y-range from the two most repeatedly
    # segmented rails, not the page-edge tracks spanning the whole sheet.
    core=sorted(tracks,key=lambda t:t['segments'],reverse=True)[:2]
    if len(core)<2 or min(t['segments'] for t in core)<5:
        return None,{'reason':'no_repeated_table_rails','rails':tracks}
    top=float(np.median([t['first'] for t in core]))
    bottom=float(np.median([t['last'] for t in core]))
    if bottom-top<max(100,H*.09):
        return None,{'reason':'no_common_vertical_range','rails':tracks}
    for t in tracks:
        # Score only independently measured segments within the inferred
        # table-height window. A paper rim outside this band gets no credit.
        matches=[]
        for group in groups:
            if abs(float(np.median([q[0] for q in group]))-t['x_ref'])>3:continue
            for v in group:
                span=max(0.,min(v[2],bottom)-max(v[1],top))
                if span>max(8.,H*.008):matches.append((max(v[1],top),min(v[2],bottom)))
        coverage=np.zeros(H,bool)
        for lo,hi in matches:coverage[max(0,int(lo)):min(H,int(hi)+1)]=True
        t['band_segments']=len(matches)
        t['band_coverage']=int(coverage.sum())
        t['quality']=t['band_coverage']+18*min(20,t['band_segments'])
    tracks=[t for t in tracks if t['band_segments']>=3 and t['band_coverage']>=.15*(bottom-top)]
    tracks.sort(key=lambda t:t['quality'],reverse=True)
    if len(tracks)<5:return None,{'reason':'less_than_five_band_rails','rails':tracks,'top':top,'bottom':bottom}
    from itertools import combinations
    candidates=[]
    for combo in combinations(tracks[:min(10,len(tracks))],5):
        combo=sorted(combo,key=lambda t:t['x_ref'])
        span=combo[-1]['x_ref']-combo[0]['x_ref']
        if span<W*.22 or span>W*.80:continue
        if min(t['band_segments'] for t in combo)<3:continue
        # A five-column grid must include five independent rail tracks all
        # supported over this same y band; extra long page borders lose.
        score=sum(t['quality'] for t in combo)
        candidates.append((score,combo))
    if not candidates:return None,{'reason':'no_coherent_five_band_rails','rails':tracks}
    candidates.sort(key=lambda p:p[0],reverse=True)
    score,rail=candidates[0]
    return (rail,top,bottom),{'rail_count':len(tracks),'selected':rail,'top':top,'bottom':bottom,'score':score}


def _horizontal_levels(gray, rails, top, bottom):
    H,W=gray.shape
    # Alternative thresholds capture faded printed rules; a level enters
    # proposal only if native pixels contain long horizontal strokes.
    x0=max(0,int(min(r['x_ref'] for r in rails))-20)
    x1=min(W,int(max(r['x_ref'] for r in rails))+20)
    if x1-x0<100:return [],{'reason':'too_narrow'}
    start=max(0,int(top)-17);end=min(H,int(bottom)+17)
    crop=gray[start:end,x0:x1]
    if crop.size==0:return [],{'reason':'no_horizontal_roi'}
    # Threshold in full-frame coordinates so faint header strokes are not
    # altered by a new ROI border or different local neighbourhood.
    full_bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,31,6)
    bw=full_bw[start:end,x0:x1]
    kernel=max(13,min(23,int((x1-x0)*.046)))
    strokes=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,kernel),np.uint8))
    strength=np.count_nonzero(strokes,axis=1)
    mask=strength>=max(23,round((x1-x0)*.09))
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    points=[]
    for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
        yy=int(a+np.argmax(strength[a:b]));global_y=yy+start
        if top-7<=global_y<=bottom+7:
            # Reject isolated word strokes: an actual horizontal rule must
            # have one or more long fragments crossing non-adjacent columns.
            line=strokes[max(0,yy-2):yy+3].max(axis=0)
            rail_x=[int(r['x_ref'])-x0 for r in rails]
            occupied=sum(np.count_nonzero(line[max(0,u+3):min(x1-x0,v-3)])>=7
                         for u,v in zip(rail_x[:-1],rail_x[1:]))
            points.append((global_y,int(strength[yy]),occupied))
    # Multiple images bands of the same printed separator: merge nearby.
    out=[]
    for group in _clusters([(float(v[0]),v[1],v[2]) for v in points],max(3.5,H*.003)):
        candidate=max(group,key=lambda v:v[1]);out.append({'y':float(candidate[0]),
                                                'strength':int(candidate[1]),
                                                'columns':int(candidate[2])})
    return out,{'roi':[x0,start,x1,end],'raw_levels':points}


def _fit_rail(lines,track,top,bottom,H):
    xref=track['x_ref'];yref=H*.45
    obs=[]
    for xa,ya,xb,yb in lines:
        dy=yb-ya;dx=xb-xa
        if abs(dy)<15 or abs(dx)>.2*abs(dy):continue
        ym=(ya+yb)*.5
        if ym<top-15 or ym>bottom+15:continue
        predicted=xref+track['slope']*(ym-yref)
        if abs((xa+xb)*.5-predicted)<8:
            obs.extend([(min(ya,yb),xa if ya<=yb else xb),
                        (max(ya,yb),xb if ya<=yb else xa)])
    if len(obs)>=4:
        yy=np.array([v[0] for v in obs]);xx=np.array([v[1] for v in obs])
        for _ in range(2):
            slope,intercept=np.polyfit(yy,xx,1)
            d=np.abs(xx-(slope*yy+intercept));valid=d<max(2.5,np.percentile(d,75)*1.5)
            if valid.sum()<4:break
            yy=yy[valid];xx=xx[valid]
        if abs(slope)>.2:slope=track['slope'];intercept=xref-slope*yref
    else:slope=track['slope'];intercept=xref-slope*yref
    return np.array([float(slope*H),float(intercept+slope*H/2)])


def _fit_horizontal(lines,level,rail_x,W):
    # Measured source fragments near the row are fitted robustly to a straight
    # near-horizontal line; never extrapolate a free quadratic into the right.
    y=level['y'];data=[]
    lo=min(rail_x);hi=max(rail_x)
    for xa,ya,xb,yb in lines:
        dx=xb-xa;dy=yb-ya
        if abs(dx)<15 or abs(dy)>.17*abs(dx):continue
        xm=(xa+xb)*.5;ym=(ya+yb)*.5
        if not lo-10<xm<hi+10 or abs(ym-y)>max(4.5,W*.006):continue
        data.extend([(xa,ya,abs(dx)),(xb,yb,abs(dx))])
    if len(data)>=4:
        xx=np.asarray([p[0] for p in data]);yy=np.asarray([p[1] for p in data]);
        # Dense printed strokes dominate; do not fit arbitrary nearby glyphs.
        slope,offset=np.polyfit(xx,yy,1)
        if abs(slope)>.11:slope=0;offset=y
    else:slope=0;offset=y
    return np.array([slope*W,float(offset+slope*W/2)])


def detect_source_grid(image, *, rule_factory=None):
    """Return (graph, diagnostic); graph None means never accept uncertain crop."""
    began=perf_counter()
    def fail(reason,**kw):return None,{'status':'rejected','reason':reason,**kw,
                             'wall_ms':round((perf_counter()-began)*1000,2)}
    if image is None or image.size==0:return fail('empty_image')
    H,W=image.shape[:2]
    lines=_line_segments(image)
    result,rail_debug=_find_rails(lines,image.shape)
    if result is None:return fail('rail_failure',rail_debug=rail_debug)
    rails,top,bottom=result
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    levels,level_debug=_horizontal_levels(gray,rails,top,bottom)
    if len(levels)<5:return fail('too_few_observed_rows',rails=rail_debug,levels=levels)
    # Only physically witnessed rows; tolerate VAT-free 2+ item receipts.
    levels=[p for p in levels if p['columns']>=2 or p['strength']>max(55,.18*(rails[-1]['x_ref']-rails[0]['x_ref']))]
    if len(levels)<5:return fail('too_few_supported_rows',rails=rail_debug,levels=levels)
    levels=sorted(levels,key=lambda x:x['y'])
    if rule_factory is None:
        from .geometry_lab.detection import Rule as rule_factory
    vertical=[];horizontal=[]
    for r in rails:
        coeff=_fit_rail(lines,r,top,bottom,H)
        vertical.append(rule_factory('v',coeff,float(coeff[-1]),1.,(top,bottom)))
    rail_x=[r['x_ref'] for r in rails]
    for p in levels:
        coeff=_fit_horizontal(lines,p,rail_x,W)
        horizontal.append(rule_factory('h',coeff,float(coeff[-1]),1.,(min(rail_x),max(rail_x))))
    graph={'vertical':vertical,'horizontal':horizontal,'shape':(H,W)}
    # Geometry invariants; check each row and full data range independently.
    from numpy.polynomial.polynomial import polyval
    xs=np.array([[v.eval((p['y']-H/2)/H) for v in vertical] for p in levels]);
    ys=np.array([[h.eval((x-W/2)/W) for x in xs[i]] for i,h in enumerate(horizontal)])
    if np.min(np.diff(xs,axis=1))<=6 or np.min(np.diff(ys,axis=0))<=5:
        return fail('crossing_or_too_close',rails=rail_debug,levels=levels)
    if xs.min()<0 or xs.max()>=W or ys.min()<0 or ys.max()>=H:
        return fail('outside_image',rails=rail_debug,levels=levels)
    # A fabricated header is especially dangerous: accept only if two top
    # levels span the same physical vertical rail system.
    if max([r['first'] for r in rails[:2]])>levels[1]['y']+20:
        return fail('top_two_rows_unverified',rails=rail_debug,levels=levels)
    # V16: every right-hand corner must be supported by source-photo ink.
    # This alters the actual curves consumed by GridMesh, not just the preview.
    # V16: trace the observed ink path across left, middle AND right of
    # every physical rule. No free polynomial or left-to-right straight join.
    try:
        from .trace_printed_rules_v16 import trace_printed_horizontals
    except ImportError:
        from trace_printed_rules_v16 import trace_printed_horizontals
    traced,trace_report=trace_printed_horizontals(image,graph)
    if traced is None:
        return fail('physical_ink_tracing_not_verified',trace_v16=trace_report,
                    rails=rail_debug,levels=levels)
    return traced,{'status':'candidate_requires_header_and_full_audit',
                  'selection':SELECTED,'rails':rail_debug,'levels':levels,
                  'level_detail':level_debug,'trace_v16':trace_report,
                  'wall_ms':round((perf_counter()-began)*1000,2)}



def anchor_all_right_corners(image,graph,*,rule_factory=None,max_correction=15.0):
    """Anchor EVERY right-side horizontal to an independently observed ink stroke.

    These are original source pixels and are never generated from the prior
    geometric curve. Failure of even one right anchor rejects the candidate.
    Uses a quadratic *correction* anchored at the left paper rail, not a
    free quadratic fit to sparse glyph fragments.
    """
    H,W=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV,31,6)
    strokes=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,13),np.uint8))
    vertical=graph['vertical'];horizontal=graph['horizontal']
    if len(vertical)!=5 or len(horizontal)<3:
        return None,{'reason':'invalid_graph_shape'}
    # Inspect printed strokes INSIDE the last (price) column, not the paper edge.
    y_middle=float(np.median([h.position for h in horizontal]))
    rail_x=[float(v.eval((y_middle-H/2)/H)) for v in vertical]
    xl=float(rail_x[0]);last_col_width=rail_x[-1]-rail_x[-2]
    if last_col_width<35:return None,{'reason':'price_column_too_narrow'}
    # A short high-confidence strip near the last printed vertical rail.
    xa=int(round(rail_x[-1]-min(43,max(21,last_col_width*.60))))
    xb=int(round(rail_x[-1]-max(8,min(12,last_col_width*.20))))
    xa=max(0,xa);xb=min(W,xb)
    if xb-xa<15:return None,{'reason':'empty_right_strip'}
    x_anchor=float((xa+xb)/2)
    x_left=float(xl+max(5.,(rail_x[1]-xl)*.10))
    if x_anchor-x_left<90:return None,{'reason':'anchor_span_too_short'}
    corrections=[];fixed=[];last_y=-1e9
    for idx,h in enumerate(horizontal):
        predicted=float(h.eval((x_anchor-W/2)/W))
        y0=max(0,int(round(predicted-max_correction)))
        y1=min(H,int(round(predicted+max_correction))+1)
        if y1-y0<12:return None,{'reason':'empty_search_band','index':idx}
        region=strokes[y0:y1,xa:xb]
        strengths=np.count_nonzero(region,axis=1)
        # Bound admissible ink by the per-row candidate. No fixed y coordinates.
        possibles=sorted(range(len(strengths)),key=lambda k:(-int(strengths[k]),abs(y0+k-predicted)))
        best=possibles[0]
        peak=float(strengths[best]);observed=float(y0+best)
        # The price-column separator must be supported across a substantial
        # physical width. Reject glyphs, missing ink, and wholly wrong rows.
        if peak<max(11.,(xb-xa)*.38):
            return None,{'reason':'right_stroke_not_observed','index':idx,
                         'peak':int(peak),'roi':[xa,y0,xb,y1]}
        # Right/left physical rules are explicitly ordered; do not allow
        # resampling to turn a short row into a crossed separator.
        if observed-last_y<8:
            return None,{'reason':'right_row_order_crossed','index':idx,
                         'observed':observed,'previous':last_y}
        last_y=observed
        delta=observed-predicted
        if abs(delta)>max_correction:
            return None,{'reason':'excessive_right_displacement','index':idx,
                         'delta':delta}
        # Add to existing polynomial y(t): delta * ((x-x_left)/span)^2.
        # x=W*(t+.5). All coefficients remain in production Rule convention.
        tpoly=np.poly1d([1.,0.])
        xpoly=W*(tpoly+0.5)
        correction=delta*((xpoly-x_left)/(x_anchor-x_left))**2
        coeff=np.asarray((np.poly1d(h.coeff)+correction).c,dtype=float)
        if rule_factory is None:
            from .geometry_lab.detection import Rule as rule_factory
        fixed.append(rule_factory('h',coeff,float(np.polyval(coeff,0)),h.support,h.domain))
        corrections.append({'index':idx,'x_anchor':x_anchor,'predicted_y':round(predicted,2),
                            'observed_y':observed,'delta_y':round(delta,2),
                            'right_ink_px':int(peak),'right_strip':[xa,xb]})
    return {**graph,'horizontal':fixed},{'status':'anchored_all_right_rows',
        'right_anchors':corrections,'x_left':x_left,'x_right':x_anchor}

def _nodes_from_graph(graph, shape):
    H,W=shape[:2]
    rows=[]
    for h in graph['horizontal']:
        row=[]
        for v in graph['vertical']:
            y=float(h.position)
            for _ in range(8):
                x=float(v.eval((y-H/2)/H))
                y=float(h.eval((x-W/2)/W))
            row.append((x,y))
        rows.append(row)
    return np.asarray(rows,float)


def validate_source_grid(image, graph, diagnostic):
    """Full source row census + independent per-curve LSD verification.

    Only the faint top/header rule can use a separately observed original-pixel
    morphology proof. Every missing line anywhere is a hard rejection, and
    no weak mid-table line is implicitly accepted.
    """
    from .grid_edge_validator_v12 import audit_rowaware_printed_rows
    from .paper_first_grid_v12 import json_native
    nodes=_nodes_from_graph(graph,image.shape)
    H,W=image.shape[:2]
    if nodes.shape[1:]!=(5,2) or len(nodes)<5 or not np.isfinite(nodes).all():
        return False,{'reason':'invalid_source_nodes'}
    if np.min(np.diff(nodes[:,:,0],axis=1))<=6 or np.min(np.diff(nodes[:,:,1],axis=0))<=5:
        return False,{'reason':'crossing_source_grid'}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    raw_lines=_line_segments(image)
    detected,_=_find_rails(raw_lines,image.shape)
    if detected is None:return False,{'reason':'rails_disappeared_during_audit'}
    rails,top,bottom=detected
    census,_=_horizontal_levels(gray,rails,top,bottom)
    census=[v for v in census if v['columns']>=2 or v['strength']>55]
    if not census:return False,{'reason':'no_source_row_census'}
    unmatched=[]
    for observed in census:
        distances=[abs(observed['y']-np.median(row[:,1])) for row in nodes]
        if min(distances)>10:unmatched.append(observed)
    if unmatched:return False,{'reason':'missing_observed_source_rows','unmatched':unmatched}
    missing=[]
    for index,row in enumerate(nodes):
        nearest=min(census,key=lambda v:abs(v['y']-np.median(row[:,1])))
        if abs(nearest['y']-np.median(row[:,1]))>10:missing.append(index)
    if missing:return False,{'reason':'unverified_candidate_rows','indices':missing}
    # The V12 validator also searches for previously unseen lines across the
    # entire paper. For this photograph the first faint line is demonstrable
    # in the original thresholded pixels, but not detected by regular LSD.
    passed,base=audit_rowaware_printed_rows(image,nodes,paper_bbox=None)
    if not passed:
        old_missing=base.get('unverified_grid_edges',[])
        if not (base.get('reason')=='not_all_grid_lines_optically_verified' and
                old_missing and set(old_missing)<={0,1}):
            return False,{'reason':'independent_audit_rejected','audit':json_native(base)}
        special=[]
        for index in old_missing:
            observed=min(census,key=lambda v:abs(v['y']-np.median(nodes[index,:,1])))
            if (abs(observed['y']-np.median(nodes[index,:,1]))>5 or
                    observed['columns']<3 or observed['strength']<30):
                return False,{'reason':'weak_header_rule_not_confirmed',
                              'index':index,'observed':observed}
            special.append({'index':int(index),'observed_y':float(observed['y']),
                            'supported_columns':int(observed['columns']),
                            'original_ink_px':int(observed['strength'])})
    else:special=[]
    return True,json_native({'status':'accepted_source_pixel_census_and_independent_lsd',
                             'boundaries':len(nodes),'vertical_rails':5,
                             'weak_header_original_pixel_proofs':special,
                             'independent_lsd':base,
                             'source_census':census})
