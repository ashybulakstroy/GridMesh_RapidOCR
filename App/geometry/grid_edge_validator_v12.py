"""V12 full-height source audit, with addressable enhanced-source verification.

A document rim may be rejected as a missing row only if it is connected to
printed grid rails; unaccounted actual table lines must still cause fallback.
No receipt-specific coordinates, OCR, model-dependent shortcuts or GT imports.
"""
from __future__ import annotations
# GRID_ABORT_DIAGNOSTICS_V1
from App import grid_abort_diagnostics_v1 as _grid_diag
from time import perf_counter
import cv2
import numpy as np


def merge_intervals(pieces,gap):
    merged=[]
    for a,b in sorted(pieces):
        if b<=a:continue
        if merged and a<=merged[-1][1]+gap:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return merged


def coverage(pieces,a,b,gap=6):
    return sum(max(0.,min(b,y)-max(a,x)) for x,y in merge_intervals(pieces,gap))/max(1.,b-a)


def vertical_rail_bridge(image,nodes,line_y,grid_y):
    """Check continuous vertical rails BETWEEN a disputed line and the table."""
    H,W=image.shape[:2]
    dist=abs(line_y-grid_y)
    lo=max(0,int(min(line_y,grid_y)+dist*.17))
    hi=min(H,int(max(line_y,grid_y)-dist*.17))
    if hi-lo<35:return {'supported':5,'reason':'insufficient_bridge_pixels'}
    gray=cv2.cvtColor(image[lo:hi],cv2.COLOR_BGR2GRAY) if image.ndim==3 else image[lo:hi]
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    height=max(15,min(39,round((hi-lo)*.19)))
    vmask=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((height,1),np.uint8))
    first=nodes[0] if line_y<grid_y else nodes[-1]
    other=nodes[1] if line_y<grid_y else nodes[-2]
    x=first[:,0]+np.clip((lo+hi)*.5-first[:,1],-300,300)*(
        (first[:,0]-other[:,0])/np.maximum(1.,abs(first[:,1]-other[:,1])))
    ratios=[]
    for loc in x:
        col=round(float(loc))
        ratios.append(1. if not 7<=col<W-7 else float((vmask[:,col-6:col+7].max(axis=1)>0).mean()))
    return {'supported':sum(v>=.22 for v in ratios),
            'coverage':[round(v,3) for v in ratios],'y_range':[lo,hi]}


@_grid_diag.observe('App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows')
def audit_rowaware_printed_rows(image,nodes,*,paper_bbox=None):
    """Independent full-height LSD scan; every printed row must be verified.

    Each LSD fragment is matched against the actual curved candidate row at the
    fragment's own x coordinate (rather than median tilt of all rows).
    """
    started=perf_counter()
    info={'status':'rejected','method':'independent_lsd_curve_per_row_v12'}
    try:
        n=np.asarray(nodes,float);H,W=image.shape[:2]
        if n.ndim!=3 or n.shape[1:]!=(5,2) or len(n)<3 or not np.isfinite(n).all():
            info['reason']='invalid_nodes';return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L60')
        ys=np.median(n[:,:,1],axis=1);xs=np.median(n[:,:,0],axis=0)
        if np.min(np.diff(ys))<=4 or np.min(np.diff(xs))<=5:
            info['reason']='crossing_grid';return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L63')
        lo,hi=float(xs[0]),float(xs[-1]);span=hi-lo
        x0=max(0,int(lo-22));x1=min(W,int(hi+22))
        y0=0 if paper_bbox is None else max(0,int(paper_bbox[1]))
        y1=H if paper_bbox is None else min(H,int(paper_bbox[3]))
        if span<70 or y1-y0<30:
            info['reason']='invalid_region';return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L69')
        img=image[y0:y1,x0:x1]
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        found=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
        by_row=[[] for _ in n];unknown=[]
        if found is not None:
            for a,b,c,d in found.reshape(-1,4):
                xa,xb,ya,yb=float(a+x0),float(c+x0),float(b+y0),float(d+y0)
                dx=xb-xa;dy=yb-ya
                if abs(dx)<max(16.,span*.042) or abs(dy)>abs(dx)*.16:continue
                left=max(lo,min(xa,xb));right=min(hi,max(xa,xb))
                if right-left<8:continue
                mx=(xa+xb)*.5;my=(ya+yb)*.5
                errors=[abs(my-float(np.interp(mx,row[:,0],row[:,1]))) for row in n]
                best=int(np.argmin(errors))
                if errors[best]<=11.5:by_row[best].append((left,right))
                else:unknown.append((mx,my,left,right))
        grid_evidence=[]
        for index,segments in enumerate(by_row):
            intervals=merge_intervals(segments,max(4,span*.016))
            fractions=[coverage(intervals,a,b) for a,b in zip(xs[:-1],xs[1:])]
            width=coverage(intervals,lo,hi)
            grid_evidence.append({'index':index,'y':round(float(ys[index]),2),
                'coverage':round(width,3),'columns':[round(float(v),3) for v in fractions],
                'supported_columns':int(sum(v>=.16 for v in fractions))})
        missing=[v['index'] for v in grid_evidence
                 if v['coverage']<.50 or v['supported_columns']<3
                 or max(v['columns'][:2])<.12 or max(v['columns'][2:])<.12]
        info.update(grid_evidence=grid_evidence,unverified_grid_edges=missing,
                    horizontal_boundaries=len(n),vertical_boundaries=5,
                    scan_bbox=[x0,y0,x1,y1])
        if missing:
            # V12: additional, independently located physical strokes on the SAME
            # photographed pixels. CLAHE may reveal faint ink but is never by
            # itself proof of an extra/different printed line. Require an
            # existing original-pixel segment, broad enhanced coverage, and
            # vertical-rail intersection evidence for EVERY recovered edge.
            enhanced=cv2.createCLAHE(clipLimit=3.,tileGridSize=(8,8)).apply(gray)
            seen=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(enhanced)[0]
            detected_by_row=[[] for _ in n]
            if seen is not None:
                for a,b,c,d in seen.reshape(-1,4):
                    xa,xb,ya,yb=float(a+x0),float(c+x0),float(b+y0),float(d+y0)
                    dx=xb-xa;dy=yb-ya
                    if abs(dx)<max(16.,span*.042) or abs(dy)>abs(dx)*.16:continue
                    left=max(lo,min(xa,xb));right=min(hi,max(xa,xb))
                    if right-left<8:continue
                    mx=(xa+xb)*.5;my=(ya+yb)*.5
                    errors=[abs(my-float(np.interp(mx,row[:,0],row[:,1]))) for row in n]
                    best=int(np.argmin(errors))
                    if errors[best]<=9.:detected_by_row[best].append((left,right))
            # Vertical supports at the source intersections cannot be created
            # by contrast enhancement: use the original grayscale photo.
            binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV,31,9)
            rail_ink=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((9,1),np.uint8))
            verified=[]; enhanced_details=[]
            for index in missing:
                segs=merge_intervals(detected_by_row[index],max(4,span*.016))
                enhanced_cov=coverage(segs,lo,hi)
                cols=[coverage(segs,a,b) for a,b in zip(xs[:-1],xs[1:])]
                crossings=[]
                for xx,yy in n[index]:
                    px=int(round(float(xx)))-x0
                    py=int(round(float(yy)))-y0
                    if px<5 or px>=rail_ink.shape[1]-5 or py<7 or py>=rail_ink.shape[0]-7:
                        crossings.append(0.)
                    else:
                        crossing=rail_ink[py-6:py+7,px-4:px+5]
                        crossings.append(float(np.any(crossing>0,axis=1).mean()))
                primary=grid_evidence[index]
                valid=(primary['coverage']>=.06 and bool(by_row[index])
                       and enhanced_cov>=.70
                       and sum(v>=.16 for v in cols)>=3
                       and max(cols[:2])>=.16 and max(cols[2:])>=.16
                       and sum(v>=.30 for v in crossings)>=4)
                enhanced_details.append({'index':int(index),'raw_coverage':primary['coverage'],
                   'enhanced_coverage':round(float(enhanced_cov),3),
                   'enhanced_columns':[round(float(v),3) for v in cols],
                   'intersection_support':[round(v,3) for v in crossings],
                   'accepted':bool(valid)})
                if valid:verified.append(index)
            info['enhanced_original_pixel_validation']=enhanced_details
            info['source_verified_weak_edges']=[int(x) for x in verified]
            missing=[x for x in missing if x not in verified]
            info['unverified_grid_edges']=missing
        if missing:
            info['reason']='not_all_grid_lines_optically_verified';return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L156')
        groups=[]
        for segment in sorted(unknown,key=lambda z:z[1]):
            if groups and abs(segment[1]-np.median([s[1] for s in groups[-1]]))<=5.5:
                groups[-1].append(segment)
            else:groups.append([segment])
        extras=[];partial=[]
        for group in groups:
            inter=merge_intervals([(s[2],s[3]) for s in group],max(5,span*.02))
            ratio=coverage(inter,lo,hi)
            cols=[coverage(inter,a,b) for a,b in zip(xs[:-1],xs[1:])]
            y=float(np.median([s[1] for s in group]))
            item={'y':round(y,2),'coverage':round(ratio,3),'columns':[round(float(v),3) for v in cols]}
            if ratio>=.55 and sum(v>=.20 for v in cols)>=3:extras.append(item)
            elif ratio>=.36 and max(cols[:2])>=.16 and max(cols[2:])>=.16:partial.append(item)
        # V12 independent negative audit. The original-only audit cannot see
        # the pale upper/header line once it has been omitted from the graph:
        # search enhanced SOURCE pixels for additional *unmatched* horizontal
        # strokes, then require multiple column supports. Do not use known row
        # count or VAT labels to infer a missing boundary.
        enhanced=cv2.createCLAHE(clipLimit=3.,tileGridSize=(8,8)).apply(gray)
        other=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(enhanced)[0]
        unknown_enh=[]
        if other is not None:
            center=(lo+hi)/2
            for a,b,c,d in other.reshape(-1,4):
                xa,xb,ya,yb=float(a+x0),float(c+x0),float(b+y0),float(d+y0)
                dx=xb-xa;dy=yb-ya
                if abs(dx)<max(16.,span*.042) or abs(dy)>abs(dx)*.16:continue
                left=max(lo,min(xa,xb));right=min(hi,max(xa,xb))
                if right-left<8:continue
                mx=(xa+xb)*.5;my=(ya+yb)*.5
                errs=[abs(my-float(np.interp(mx,row[:,0],row[:,1]))) for row in n]
                nearest=int(np.argmin(errs))
                if errs[nearest]<=11.5:continue
                row=n[nearest]
                projected=my+float(np.interp(center,row[:,0],row[:,1])-
                                   np.interp(mx,row[:,0],row[:,1]))
                unknown_enh.append((projected,my,left,right))
        groups_enh=[]
        for segment in sorted(unknown_enh,key=lambda k:k[0]):
            if (groups_enh and segment[0]-float(np.median([v[0] for v in groups_enh[-1]]))<=11):
                groups_enh[-1].append(segment)
            else:groups_enh.append([segment])
        enhanced_extras=[]
        for group in groups_enh:
            segments=merge_intervals([(p[2],p[3]) for p in group],max(5,span*.02))
            wide=coverage(segments,lo,hi)
            cols=[coverage(segments,a,b) for a,b in zip(xs[:-1],xs[1:])]
            if (wide>=.53 and sum(v>=.16 for v in cols)>=3 and
                    max(cols[:2])>=.16 and max(cols[2:])>=.16):
                enhanced_extras.append({'y':round(float(np.median([p[1] for p in group])),2),
                   'coverage':round(float(wide),3),
                   'columns':[round(float(v),3) for v in cols],
                   'source':'independent_clahe_lsd'})
        info.update(unmatched_source_edges=extras,suspicious_partial_rows=partial,
                    enhanced_unmatched_source_edges=enhanced_extras)
        # Do NOT ignore unaccounted lines merely because they are outside the
        # current grid. Only accept directly observed *paper rims*, close to
        # the verified paper envelope, far from the table and NOT connected to
        # continuing vertical rails. Otherwise reject and use full fallback.
        unidentified=sorted(extras+partial+enhanced_extras,key=lambda item:item['y'])
        if not unidentified:
            info.update(status='accepted',reason='all_physical_rows_verified');return _grid_diag.ret((True,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L219')
        groups=[]
        for item in unidentified:
            if groups and item['y']-groups[-1][-1]['y']<=13.:
                groups[-1].append(item)
            else:groups.append([item])
        classifications=[]
        for group in groups:
            level=float(np.median([v['y'] for v in group]))
            is_inside=ys[0]<=level<=ys[-1]
            nearest=float(ys[0] if level<ys[0] else ys[-1])
            gap=abs(level-nearest)
            min_gap=max(95.,float(np.median(np.diff(ys)))*2.1)
            if paper_bbox is None:
                info['reason']='unmatched_without_verified_paper';return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L233')
            ptop,pbot=float(paper_bbox[1]),float(paper_bbox[3])
            rim_margin=max(55., min(95.,(pbot-ptop)*.065))
            near_rim=(level-ptop<=rim_margin if level<ys[0]
                      else pbot-level<=rim_margin if level>ys[-1] else False)
            if is_inside or gap<min_gap or not near_rim:
                info['reason']='unmatched_possible_table_separator'
                info['unmatched_failed']={'y':round(level,2),'inside':bool(is_inside),
                    'gap_px':round(gap,2),'near_paper_rim':bool(near_rim)}
                return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L242')
            bridge=vertical_rail_bridge(image,n,level,nearest)
            classifications.append({'y':round(level,2),'gap_px':round(gap,2),
                                    'paper_edge':'top' if level<ys[0] else 'bottom',
                                    'bridge':bridge,'strokes':len(group)})
            if bridge['supported']>=2:
                info['reason']='unmatched_line_connected_to_grid'
                info['unmatched_failed']=classifications[-1]
                return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L250')
        if len({x['paper_edge'] for x in classifications}) != len(classifications):
            info['reason']='multiple_unconnected_lines_same_paper_rim';return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L252')
        info.update(status='accepted_with_observed_paper_rims',
                    reason='verified_grid_plus_independent_unconnected_paper_rims',
                    excluded_document_rules=classifications)
        return _grid_diag.ret((True,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L256')
    except (ValueError,TypeError,IndexError,cv2.error,FloatingPointError) as exc:
        _grid_diag.caught(locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L257')
        info['reason']='audit_error:'+type(exc).__name__;return _grid_diag.ret((False,info), locals(), 'App/geometry/grid_edge_validator_v12.py:audit_rowaware_printed_rows:L258')
    finally:
        info['elapsed_ms']=round((perf_counter()-started)*1000,3)
