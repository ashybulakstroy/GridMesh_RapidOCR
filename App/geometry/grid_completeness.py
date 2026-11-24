"""Independent printed-row completeness audit for candidate five-rail grids.

Checks source-image horizontal line segments over the entire paper envelope,
not just the candidate crop. A topology-valid partial grid is rejected if
another independently observed, width-spanning row separator exists.
No OCR, VAT semantics, expected row count, or inferred lines.
"""
from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np


def _merge_intervals(items, gap):
    out=[]
    for a,b in sorted(items):
        if b<=a:continue
        if out and a<=out[-1][1]+gap:
            out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out


def audit_printed_rows(image, nodes, *, paper_bbox=None,
                       min_width_coverage=.65, match_px=8.5):
    """Return (accept, diagnostics). Nodes are (rows, five rails, x/y).

    All independent strong segments must correspond 1:1 to a graph row.
    Conservative: if observed rows cannot be verified, select SMART.
    """
    began=perf_counter()
    info={'status':'rejected','method':'independent_source_lsd_row_coverage',
          'paper_scan':paper_bbox is not None}
    try:
        nodes=np.asarray(nodes,dtype=np.float64)
        if nodes.ndim!=3 or nodes.shape[1:]!=(5,2) or len(nodes)<3 or not np.isfinite(nodes).all():
            info['reason']='invalid_node_coordinates';return False,info
        H,W=image.shape[:2]
        xs=np.median(nodes[:,:,0],axis=0)
        lo=float(xs[0]);hi=float(xs[-1]);span=hi-lo
        if not (span>=70 and np.min(np.diff(xs))>5):
            info['reason']='invalid_horizontal_span';return False,info
        # Normalize curved row tilt from measured grid; use the same relative
        # line shape for independent candidate rows, not the grid row positions.
        xc=(lo+hi)/2
        slopes=[]
        predicted=[]
        for row in nodes:
            center=float(np.interp(xc,row[:,0],row[:,1]))
            predicted.append(center)
            slopes.append(np.interp(xs,row[:,0],row[:,1])-center)
        tilt=np.median(np.asarray(slopes),axis=0)
        ymin=0 if paper_bbox is None else max(0,int(paper_bbox[1]))
        ymax=H if paper_bbox is None else min(H,int(paper_bbox[3]))
        if (min(predicted)<ymin-4 or max(predicted)>ymax+4):
            info['reason']='grid_outside_paper';return False,info
        margin=max(14,round(span*.035))
        x0=max(0,int(np.floor(lo))-margin);x1=min(W,int(np.ceil(hi))+margin)
        if x1-x0<80 or ymax-ymin<30:
            info['reason']='empty_audit_roi';return False,info
        img=image[ymin:ymax,x0:x1]
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        detected=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
        fragments=[]
        if detected is not None:
            for x,y,X,Y in detected.reshape(-1,4):
                x+=x0;X+=x0;y+=ymin;Y+=ymin
                dx=float(X-x);dy=float(Y-y)
                if abs(dx)<max(16,span*.043) or abs(dy)>abs(dx)*.15:continue
                xm=(x+X)/2
                level=float((y+Y)/2-np.interp(xm,xs,tilt))
                start=max(lo,min(x,X));end=min(hi,max(x,X))
                if end-start>=6:
                    fragments.append({'level':level,'span':(float(start),float(end))})
        groups=[]
        tolerance=max(3.5,min(6.0,H*.0027))
        for frag in sorted(fragments,key=lambda q:q['level']):
            if not groups or frag['level']-groups[-1][-1]['level']>tolerance:
                groups.append([frag])
            else:groups[-1].append(frag)
        strong=[]; partial=[]
        for group in groups:
            merged=_merge_intervals([f['span'] for f in group],max(3,span*.01))
            covered=sum(b-a for a,b in merged)
            fraction=covered/span
            column_coverage=[]
            for a,b in zip(xs[:-1],xs[1:]):
                covered_col=sum(max(0.,min(b,v)-max(a,u)) for u,v in merged)
                column_coverage.append(float(covered_col/max(b-a,1)))
            supported=sum(v>=.22 for v in column_coverage)
            if (paper_bbox is not None and
                    (float(np.median([f['level'] for f in group])) < ymin+max(38,(ymax-ymin)*.035) or
                     float(np.median([f['level'] for f in group])) > ymax-max(38,(ymax-ymin)*.035))):
                continue  # photographed paper rim is not a printed table separator
            elif_partial = (fraction >= .38 and supported >= 2 and max(column_coverage[:2])>=.18 and max(column_coverage[2:])>=.18)
            if (fraction>=min_width_coverage and supported>=3 and
                    max(column_coverage[:2])>=.22 and max(column_coverage[2:])>=.22):
                strong.append({'y':round(float(np.median([f['level'] for f in group])),2),
                               'coverage':round(float(fraction),3),
                               'columns':[round(v,3) for v in column_coverage]})
            elif elif_partial:
                partial.append({'y':round(float(np.median([f['level'] for f in group])),2),
                                'coverage':round(float(fraction),3),
                                'columns':[round(v,3) for v in column_coverage]})
        info.update(scan_bbox=[x0,ymin,x1,ymax],predicted_rows=[round(v,2) for v in predicted],
                    observed_rows=strong, expected_grid_lines=len(predicted),
                    strong_line_count=len(strong), partial_rows=partial)
        # Explicit one-to-one matching; a line cannot validate two grid rows.
        unmatched=[]; matched=set()
        for edge in strong:
            choices=sorted((abs(edge['y']-y),i) for i,y in enumerate(predicted) if i not in matched)
            if choices and choices[0][0]<=match_px:
                matched.add(choices[0][1])
            else:unmatched.append(edge)
        info['unmatched_source_edges']=unmatched
        info['unverified_grid_edges']=[i for i in range(len(predicted)) if i not in matched]
        if unmatched:
            info['reason']='unexplained_full_width_line';return False,info
        # A partially visible separator outside the candidate is also evidence
        # that the local detector may have omitted the top or bottom rows.
        partial_unmatched=[p for p in partial
                           if min(abs(p['y']-y) for y in predicted)>max(match_px,11)]
        info['suspicious_partial_rows']=partial_unmatched
        if partial_unmatched:
            info['reason']='suspicious_partial_unmatched_line';return False,info
        if len(matched)<len(predicted):
            info['reason']='not_all_grid_lines_optically_verified';return False,info
        if len(strong)!=len(predicted):
            info['reason']='observed_row_count_differs';return False,info
        info.update(status='accepted',reason='all_observed_full_width_lines_accounted_for')
        return True,info
    except (ValueError, TypeError, cv2.error, IndexError) as exc:
        info['reason']='audit_exception:'+type(exc).__name__
        return False,info
    finally:
        info['elapsed_ms']=round((perf_counter()-began)*1000,3)
