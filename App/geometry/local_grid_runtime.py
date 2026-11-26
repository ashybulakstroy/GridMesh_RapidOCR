"""PaperROI/AutoGrid V4: propose a compact Fast ROI inside the observed paper.

Fast path: paper candidate -> table-line proposal -> contained ROI -> standard
AutoGrid GridRefine V3 on ORIGINAL-RESOLUTION ROI -> global rules -> independent
full-paper printed-row audit. Any uncertainty returns None so the unchanged
production AutoGrid on the FULL image runs in service.build_grid().

This is an experiment, not a guarantee of optical correctness. No OCR changes.
"""
from __future__ import annotations
# GRID_ABORT_DIAGNOSTICS_V1
from App import grid_abort_diagnostics_v1 as _grid_diag

from time import perf_counter
import numpy as np

from .local_grid_roi import locate_table, offset_graph
from .paper_locator import locate_paper
from .grid_completeness import audit_printed_rows
from .geometry_lab.algorithms import intersection
from .lab_adapter_gridrefine_v3 import detect_for_lab


SELECTED = 'local_roi_autogrid_paperguard_v4'


def contained(inner, outer, *, margin=1):
    """Strict rectangle inclusion; not proof that the grid itself is complete."""
    if inner is None or outer is None or len(inner) != 4 or len(outer) != 4:
        return False
    a,b,c,d = map(int, inner)
    x0,y0,x1,y1 = map(int, outer)
    return (x0 + margin <= a < c <= x1 - margin and
            y0 + margin <= b < d <= y1 - margin)


def make_fast_roi(location, paper_bbox, shape):
    """Shrink generous legacy Fast ROI using its *observed* line-group bounds.

    The original locator overpads horizontally by 46% on each side. Here the
    group itself must be contained in paper; its padding is restricted to the
    paper envelope. A crop too close to a line-group edge is rejected rather
    than silently treated as a complete grid.
    """
    H,W = shape[:2]
    if location is None or location.bbox is None:
        return None, 'table_not_located', {}
    if paper_bbox is None:
        return None, 'paper_not_located', {}
    bounds = (location.diagnostics or {}).get('group_bounds')
    if not bounds or len(bounds) != 4:
        # Never fabricate a compact crop when source-line evidence is missing.
        return (tuple(location.bbox), 'original_fast_roi',
                {'original_fast_roi':list(location.bbox)}) if contained(location.bbox,paper_bbox) else (None,'fast_roi_outside_paper',{})
    left,top,right,bottom = map(float,bounds)
    px0,py0,px1,py1 = map(int,paper_bbox)
    if not (0 <= left < right <= W and 0 <= top < bottom <= H):
        return None,'invalid_group_bounds',{}
    # Require physical line-group itself well inside the detected paper.
    if not contained((int(left),int(top),int(right),int(bottom)),paper_bbox,margin=8):
        return None,'line_group_outside_paper',{'group_bounds':list(bounds)}
    span_x=right-left;span_y=bottom-top
    pad_left=max(26.0,span_x*.17)
    pad_right=max(34.0,span_x*.23)
    pad_y=max(26.0,span_y*.14)
    # The target ROI is explicitly INSIDE the paper. Never crop the observed
    # line group: limit only the *extra padding*.
    x0=max(px0+2,int(np.floor(left-pad_left)))
    y0=max(py0+2,int(np.floor(top-pad_y)))
    x1=min(px1-2,int(np.ceil(right+pad_right)))
    y1=min(py1-2,int(np.ceil(bottom+pad_y)))
    roi=(x0,y0,x1,y1)
    if not contained(roi,paper_bbox,margin=1):
        return None,'fast_roi_outside_paper',{'fast_roi':list(roi)}
    if (right-left<70 or bottom-top<18 or x1-x0<110 or y1-y0<60):
        return None,'fast_roi_too_small',{'fast_roi':list(roi)}
    # Must have enough room on every side for a detector to find outer rails.
    min_x_pad=max(12.0,span_x*.035)
    min_y_pad=max(12.0,span_y*.05)
    pads=(left-x0,x1-right,top-y0,y1-bottom)
    if min(pads[0],pads[1])<min_x_pad or min(pads[2],pads[3])<min_y_pad:
        return None,'paper_clipped_required_margin',{'pads':list(map(float,pads))}
    area=(x1-x0)*(y1-y0)
    return roi,'compact_contained_roi',{
        'original_fast_roi':list(location.bbox),
        'group_bounds':list(bounds),
        'padding_px':[round(float(v),1) for v in pads],
        'area_pixels':area,
        'saved_area_percent':round(100*(1-area/max(1,(location.bbox[2]-location.bbox[0])*(location.bbox[3]-location.bbox[1]))),2),
    }


def validate_local_grid(graph, shape):
    """Five-column topology on the crop; observable row completeness is separate."""
    if not graph:return False,'no_graph',None
    vs=graph.get('vertical') or []
    hs=graph.get('horizontal') or []
    if len(vs)!=5 or len(hs)<3:return False,'unexpected_boundaries',None
    try:
        nodes=np.asarray([[intersection(v,h,shape) for v in vs] for h in hs],dtype=float)
        if nodes.shape!=(len(hs),5,2) or not np.isfinite(nodes).all():
            return False,'invalid_intersections',None
        x=nodes[:,:,0];y=nodes[:,:,1];H,W=shape[:2]
        if (np.diff(x,axis=1)<=6).any() or (np.diff(y,axis=0)<=4).any():
            return False,'crossing_or_duplicate_lines',None
        if x.min() < -3 or x.max()>W+3 or y.min()<-3 or y.max()>H+3:
            return False,'grid_outside_region',None
        if np.min(x[:,-1]-x[:,0])<W*.30:
            return False,'too_narrow',None
        margin=min(12,max(5,round(H*.05)))
        if y[0].min()<margin or H-y[-1].max()<margin:
            return False,'grid_touches_roi_top_or_bottom',None
        if x[:,0].min()<8 or W-x[:,-1].max()<8:
            return False,'grid_touches_roi_left_or_right',None
        return True,'topology_pass',nodes
    except (ValueError,TypeError,IndexError,FloatingPointError) as exc:
        return False,'invalid_geometry:'+type(exc).__name__,None


@_grid_diag.observe('App/geometry/local_grid_runtime.py:try_local_grid')
def try_local_grid(image):
    """Return (global grid, audit report, stats) or (None, rejected, stats).

    The calling service MUST fall back to its original _build_grid_core(image).
    """
    begun=perf_counter()
    stats={'route':'paper_roi_contains_fast_roi_then_local_autogrid_then_full_autogrid',
           'status':'fallback_to_smart','fallback_to_full':True,'selected':None}
    def reject(reason,**info):
        stats.update(status='fallback_to_smart',reason=reason,**info)
        stats['total_ms']=round((perf_counter()-begun)*1000,3)
        return None,{'status':'REJECTED','reason':reason},stats

    paper=locate_paper(image)
    stats.update(paper_ms=round(paper.elapsed_ms,3),
                 paper_bbox=list(paper.bbox) if paper.bbox else None,
                 paper_status=paper.reason,paper_metrics=paper.metrics)
    if paper.bbox is None:
        return _grid_diag.ret((reject('paper_not_confident')), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L136')
    # Existing Fast locator works on a downscaled full image. Its group bounds
    # are used to build a smaller crop INSIDE the paper rectangle.
    t=perf_counter()
    loc=locate_table(image)
    stats.update(locator_scope='full_image_proposal',locator_ms=round((perf_counter()-t)*1000,3),
                 original_fast_roi=list(loc.bbox) if loc.bbox else None,
                 roi_reason=loc.reason,roi_segments=loc.diagnostics.get('segments',0))
    if loc.bbox is None:
        return _grid_diag.ret((reject('fast_locator_failed')), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L145')

    t=perf_counter()
    roi,reason,detail=make_fast_roi(loc,paper.bbox,image.shape)
    stats.update(roi_compact_ms=round((perf_counter()-t)*1000,3),
                 roi_bbox=list(roi) if roi else None,
                 roi_geometry=detail,containment_ok=contained(roi,paper.bbox))
    if roi is None:
        return _grid_diag.ret((reject(reason)), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L153')
    if not stats['containment_ok']:
        return _grid_diag.ret((reject('fast_roi_outside_paper')), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L155')
    x0,y0,x1,y1=roi
    pixels=(x1-x0)*(y1-y0)
    stats['roi_pixels']=pixels
    stats['roi_area_fraction']=round(pixels/max(1,image.shape[0]*image.shape[1]),4)
    # No legacy V2 180k-pixel cutoff: contained ROI always gets a local AutoGrid
    # attempt; an unsuccessful/unsafe graph falls back to full-image AutoGrid.
    crop=image[y0:y1,x0:x1]
    if crop.size==0:
        return _grid_diag.ret((reject('empty_roi')), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L164')

    t=perf_counter()
    graph, local_report=detect_for_lab(crop)  # Same production AutoGrid GridRefine V3.
    stats.update(auto_grid_crop_ms=round((perf_counter()-t)*1000,3),
                 autogrid_roi_status=(local_report or {}).get('status'),
                 autogrid_roi_reason=(local_report or {}).get('reason'),
                 autogrid_roi_h=len(graph['horizontal']) if graph else None,
                 autogrid_roi_v=len(graph['vertical']) if graph else None)
    if graph is None:
        return _grid_diag.ret((reject('autogrid_on_roi_rejected')), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L174')
    ok,why,_=validate_local_grid(graph,crop.shape[:2])
    stats['local_geometry']=why
    if not ok:
        return _grid_diag.ret((reject('autogrid_on_roi_'+why)), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L178')
    try:
        global_graph=offset_graph(graph,roi,image.shape[:2])
        global_ok, global_why,nodes=validate_local_grid(global_graph,image.shape[:2])
        stats['global_geometry']=global_why
        if not global_ok:
            return _grid_diag.ret((reject('global_'+global_why)), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L184')
        # Re-check that the OBSERVED grid itself stays inside paper, not just
        # the ROI rectangle. Every extra printed separator must be accounted for.
        audit_ok,audit=audit_printed_rows(image,nodes,paper_bbox=paper.bbox)
        stats['completeness']=audit
        stats['completeness_ms']=audit.get('elapsed_ms')
        if not audit_ok:
            return _grid_diag.ret((reject('completeness_'+str(audit.get('reason')))), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L191')
    except Exception as exc:
        _grid_diag.caught(locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L192')
        return _grid_diag.ret((reject('local_coordinate_or_audit_error:'+type(exc).__name__)), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L193')

    stats.update(status='accepted',selected='roi_autogrid_paperguard_v4',fallback_to_full=False)
    stats['total_ms']=round((perf_counter()-begun)*1000,3)
    return _grid_diag.ret((global_graph,{
        'status':'OBSERVED_NOT_GROUND_TRUTH',
        'selection':SELECTED,
        'version':'local_grid_paperguard_v4',
        'autogrid':(local_report or {}).get('autogrid',{}),
        'completeness_audit':audit,
        'edges':[],
    },stats), locals(), 'App/geometry/local_grid_runtime.py:try_local_grid:L197')
