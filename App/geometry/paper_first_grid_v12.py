"""Paper-first source geometry for Torgsoft, V12 (no mandatory VAT).

Does NOT invent lines or accept a grid because of a mathematical receipt total.
If a paper envelope exists, search it at native resolution before any full-frame
AutoGrid. Any ambiguous/missing separator causes rejection and legacy fallback.
"""
from __future__ import annotations
# GRID_ABORT_DIAGNOSTICS_V1
from App import grid_abort_diagnostics_v1 as _grid_diag
from time import perf_counter
from collections import defaultdict
import cv2
import numpy as np

SELECTED = 'paper_first_autogrid_v12'


def _rect(box, shape):
    if box is None or len(box) != 4: return None
    H,W=shape[:2]
    x0,y0,x1,y1=[int(v) for v in box]
    x0=max(0,x0);y0=max(0,y0);x1=min(W,x1);y1=min(H,y1)
    return (x0,y0,x1,y1) if (x1-x0>=105 and y1-y0>=130) else None


def _source_levels(image, bbox, min_fraction=.17):
    """Source-pixel horizontal-ink census. Never infer missing separators.

    This is intentionally permissive (a census, not final proof): its purpose is
    to REJECT partial local grids. A later independent curve-aware audit must
    still verify accepted grids. Coordinates are always global.
    """
    roi=_rect(bbox,image.shape)
    if roi is None:return []
    x0,y0,x1,y1=roi
    gray=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY) if image.ndim==3 else image[y0:y1,x0:x1]
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
    k=max(15, min(40,int(gray.shape[1]*.07)))
    rules=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,k),dtype=np.uint8))
    strength=np.count_nonzero(rules,axis=1)
    mask=strength>=max(32,round(gray.shape[1]*min_fraction))
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    pairs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    levels=[]
    for start,end in pairs:
        if end<=start:continue
        at=int(start+np.argmax(strength[start:end]))
        if at<9 or at>len(strength)-9:continue  # paper edges
        levels.append({'y':at+y0,'ink_px':int(strength[at])})
    # Consecutive pixels from one physical stroke have already been grouped.
    return levels


def _table_band_from_evidence(paper, levels, image_shape):
    """Optional second ROI from ALL observed paper lines, not densest subset.

    A physical header can lie well above the first robust horizontal stroke.
    The generous upper margin protects it. Isolated photo/page rims are not used
    as table start or end. When evidence is weak, refuse this secondary crop.
    """
    if paper is None or len(levels)<6:return None
    ys=sorted({int(k['y']) for k in levels})
    H,W=image_shape[:2]
    # Coherent run, allowing tall multiline product rows.
    gap=max(72,min(130,round((paper[3]-paper[1])*.085)))
    groups=[];cur=[]
    for y in ys:
        if cur and y-cur[-1]>gap:
            groups.append(cur);cur=[]
        cur.append(y)
    if cur:groups.append(cur)
    groups=[g for g in groups if len(g)>=5]
    if not groups:return None
    group=max(groups,key=lambda g:(len(g),g[-1]-g[0]))
    x0,y0,x1,y1=paper
    # Extend upward to include headers and first item even when first horizontal
    # grid separator was weak. No fixed receipt coordinates or row count.
    step=float(np.median(np.diff(group))) if len(group)>1 else 32.
    top_pad=max(120,round(step*4.8))
    bottom_pad=max(70,round(step*2.2))
    top=max(y0, group[0]-top_pad);bottom=min(y1,group[-1]+bottom_pad)
    candidate=_rect((x0,top,x1,bottom),image_shape)
    if candidate is None or candidate==paper:return None
    if len(group)<max(6,round(len(levels)*.65)):return None
    return candidate


def _validate_nodes(graph, image_shape):
    from .geometry_lab.algorithms import intersection
    if not graph:return None,'missing_graph'
    vertical=graph.get('vertical') or []; horizontal=graph.get('horizontal') or []
    if not (5<=len(vertical)<=9) or len(horizontal)<3:
        return None,'insufficient_grid_boundaries'
    H,W=image_shape[:2]
    try:
        nodes=np.asarray([[intersection(v,h,image_shape) for v in vertical] for h in horizontal],float)
        if nodes.shape!=(len(horizontal),len(vertical),2) or not np.isfinite(nodes).all():
            return None,'invalid_nodes'
        if np.any(np.diff(nodes[:,:,0],axis=1)<=5) or np.any(np.diff(nodes[:,:,1],axis=0)<=4):
            return None,'crossing_grid'
        if nodes[:,:,0].min()<0 or nodes[:,:,0].max()>=W or nodes[:,:,1].min()<0 or nodes[:,:,1].max()>=H:
            return None,'grid_outside_photo'
        if np.median(nodes[:,-1,0]-nodes[:,0,0])<70:
            return None,'implausibly_narrow_table'
    except (IndexError,TypeError,ValueError,cv2.error,FloatingPointError) as exc:
        return None,'geometry_exception_'+type(exc).__name__
    return nodes,'valid_topology'


def _has_excess_rows(levels, nodes, paper_bbox):
    """Conservative pre-gate: lots of observed lines cannot be a 3-row grid."""
    n=len(nodes); visible=[v for v in levels if paper_bbox[1]+12<v['y']<paper_bbox[3]-12]
    # Do not equate an observed ink level to a product row: only reject when
    # evidence is vastly greater than the graph, independent of VAT presence.
    if len(visible)>=9 and n <= max(3,int(len(visible)*.35)):
        return True,{'reason':'many_visible_rules_missing','source_levels':len(visible),
                     'candidate_grid_lines':n}
    return False,{'source_levels':len(visible),'candidate_grid_lines':n}


def _audit_nodes(image, nodes, paper_bbox):
    """Independent full-paper audit; generic row count, VAT-free.

    For five rails reuse V6 audited real-pixel curve matching. For a different
    number of rails, FAIL CLOSED until that geometry is explicitly validated:
    the production full-grid path can still handle 6-column tables.
    """
    if nodes.shape[1]!=5:
        return False,{'reason':'rail_count_requires_independent_audit',
                      'rail_count':int(nodes.shape[1])}
    try:
        from .grid_edge_validator_v12 import audit_rowaware_printed_rows
        return audit_rowaware_printed_rows(image,nodes,paper_bbox=paper_bbox)
    except ImportError as exc:
        return False,{'reason':'source_auditor_missing','error':str(exc)}


@_grid_diag.observe('App/geometry/paper_first_grid_v12.py:try_paper_grid')
def try_paper_grid(image, *, retry=False):
    """Return globally normalized graph or None; no OCR, no VAT assumptions.

    retry=False: native-resolution paper ROI (not whole photo).
    retry=True: alternative table-band ROI inside the independently detected
    paper, used when subsequent header OCR rejects a geometrically valid graph.
    Either unsuccessful attempt falls back to the original production route.
    """
    began=perf_counter()
    stats={'version':'paper_first_v12','route':'header_retry' if retry else 'paper_first',
           'status':'rejected','fallback_to_full':True}
    def reject(reason):
        stats.update(reason=reason,elapsed_ms=round((perf_counter()-began)*1000,2))
        return None,{'status':'REJECTED','reason':reason},json_native(stats)
    if image is None or not image.size:return _grid_diag.ret((reject('empty_source')), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L150')
    from .paper_locator import locate_paper
    from .lab_adapter_gridrefine_v3 import detect_for_lab
    from .local_grid_roi import offset_graph
    paper=locate_paper(image)
    stats.update(paper_ms=round(float(paper.elapsed_ms),2),
                 paper_bbox=list(paper.bbox) if paper.bbox else None,paper_reason=paper.reason)
    bbox=_rect(paper.bbox,image.shape)
    if bbox is None:return _grid_diag.ret((reject('paper_not_confident')), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L158')
    if retry:
        t=perf_counter();levels=_source_levels(image,bbox)
        stats['paper_ink_census_ms']=round((perf_counter()-t)*1000,2)
        stats['paper_observed_rule_count']=len(levels)
        candidate=_table_band_from_evidence(bbox,levels,image.shape)
        if candidate is None:return _grid_diag.ret((reject('no_independent_full_table_band')), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L164')
    else:
        candidate=bbox
        levels=None
    stats['roi_bbox']=list(candidate)
    x0,y0,x1,y1=candidate
    t=perf_counter()
    try:
        graph,raw_report=detect_for_lab(image[y0:y1,x0:x1])
    except Exception as exc:
        _grid_diag.caught(locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L173')
        stats['autogrid_roi_ms']=round((perf_counter()-t)*1000,2)
        stats['autogrid_error']=type(exc).__name__+': '+str(exc)
        return _grid_diag.ret((reject('paper_autogrid_exception')), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L176')
    stats['autogrid_roi_ms']=round((perf_counter()-t)*1000,2)
    stats['local_edges']={ 'vertical':len(graph['vertical']) if graph else 0,
                           'horizontal':len(graph['horizontal']) if graph else 0}
    if graph is None:return _grid_diag.ret((reject('paper_autogrid_rejected')), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L180')
    # V12: only four genuine rails were fitted: seek the missing fifth in the
    # source pixels. Do not infer its position from the expected column widths.
    if len(graph.get('vertical',())) == 4 and len(graph.get('horizontal',())) >= 4:
        from .vertical_rail_recovery_v12 import recover_missing_fifth_rail
        recovered, recovery_info = recover_missing_fifth_rail(
            image[y0:y1,x0:x1], graph)
        stats['vertical_rail_recovery'] = recovery_info
        if recovered is None:
            return _grid_diag.ret((reject('four_rails_without_supported_fifth')), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L189')
        graph = recovered
        stats['recovered_edges']={'vertical':len(graph['vertical']),
                                  'horizontal':len(graph['horizontal'])}
    # V12: find physical header and first-item boundaries omitted by AutoGrid.
    # This runs only in an error-recovery path, never for already good V4 jobs.
    from .header_edge_recovery_v12 import recover_header_boundaries
    graph,header_info=recover_header_boundaries(image[y0:y1,x0:x1], graph)
    stats['header_boundary_recovery']=header_info
    # V12: correct *existing* first product separators at right using their
    # own source-observed strokes, independent from header reconstruction.
    # A subsequent full-paper audit must accept the whole graph.
    from .product_right_corners_v12 import refine_product_right_corners
    graph,corner_info=refine_product_right_corners(image[y0:y1,x0:x1],graph)
    stats['product_right_corners']=corner_info
    stats['post_header_edges']={'vertical':len(graph['vertical']),
                                'horizontal':len(graph['horizontal'])}
    local_nodes,why=_validate_nodes(graph,(y1-y0,x1-x0))
    if local_nodes is None:return _grid_diag.ret((reject('paper_'+why)), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L207')
    try:
        global_graph=offset_graph(graph,candidate,image.shape[:2])
        nodes,why=_validate_nodes(global_graph,image.shape)
        if nodes is None:return _grid_diag.ret((reject('global_'+why)), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L211')
        if levels is None:
            t=perf_counter();levels=_source_levels(image,bbox)
            stats['paper_ink_census_ms']=round((perf_counter()-t)*1000,2)
        stats['paper_observed_rule_count']=len(levels)
        excess,census=_has_excess_rows(levels,nodes,bbox)
        stats['ink_census']=census
        if excess:return _grid_diag.ret((reject('incomplete_local_grid_observed_rules')), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L218')
        audit_ok,audit=_audit_nodes(image,nodes,bbox)
        stats['source_completeness']=audit
        if not audit_ok:return _grid_diag.ret((reject('incomplete_local_grid_'+str(audit.get('reason')))), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L221')
    except Exception as exc:
        _grid_diag.caught(locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L222')
        stats['audit_error']=type(exc).__name__+': '+str(exc)
        return _grid_diag.ret((reject('audit_exception_'+type(exc).__name__)), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L224')
    stats.update(status='accepted',fallback_to_full=False,
                 elapsed_ms=round((perf_counter()-began)*1000,2))
    return _grid_diag.ret((global_graph,json_native({'status':'OBSERVED_NOT_GT', 'selection':SELECTED,
                         'version':'paper_first_v12','autogrid':(raw_report or {}).get('autogrid',{}),
                         'completeness_audit':audit,'edges':[]}),json_native(stats)), locals(), 'App/geometry/paper_first_grid_v12.py:try_paper_grid:L227')


# V12: diagnostics are transmitted as JSON, numpy scalars are not serializable.
def json_native(value):
    if isinstance(value,np.ndarray):return [json_native(x) for x in value.tolist()]
    if isinstance(value,np.generic):return json_native(value.item())
    if isinstance(value,dict):return {str(k):json_native(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_native(x) for x in value]
    if isinstance(value,(bool,int,float,str)) or value is None:return value
    return str(value)
