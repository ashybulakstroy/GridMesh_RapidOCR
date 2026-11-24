"""LocalGrid PaperGuard V3 for GridMesh 3.0.

Enabled automatically at installation via OCR_LOCALGRID_MODE=on. No OCR changes and no monkey-patches.
Runs within the normal build_grid() call and uses exactly the standard mesh
construction and safety check after it has produced source-image Rules.
"""
from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np

from .local_grid_roi import locate_table, offset_graph
from .paper_locator import locate_paper
from .grid_completeness import audit_printed_rows
from .geometry_lab.algorithms import intersection
from .geometry_lab.physical_grid_graph_local import detect_physical_graph as physical_v2
from .geometry_lab.physical_grid_graph_v3 import evidence, ridge_response


def validate_local_grid(graph, shape):
    """Check topology; not a guarantee against omitted receipt rows."""
    if graph is None:
        return False, 'no_graph'
    vs = graph.get('vertical') or []
    hs = graph.get('horizontal') or []
    # This fast route is intentionally limited to standard Torgsoft 5 rails.
    # Other layouts fall back to the unchanged production detector.
    if len(vs) != 5 or len(hs) < 3:
        return False, 'wrong_number_of_boundaries'
    try:
        nodes = np.asarray([[intersection(v, h, shape) for v in vs] for h in hs], np.float64)
        if nodes.shape != (len(hs), 5, 2) or not np.isfinite(nodes).all():
            return False, 'nonfinite_intersections'
        xs, ys = nodes[:, :, 0], nodes[:, :, 1]
        if (np.diff(xs, axis=1) <= 6).any() or (np.diff(ys, axis=0) <= 4).any():
            return False, 'crossing_or_duplicate_lines'
        H, W = shape[:2]
        if xs.min() < -8 or xs.max() > W+8 or ys.min() < -8 or ys.max() > H+8:
            return False, 'outside_roi'
        if np.min(xs[:, -1]-xs[:, 0]) < W*.15:
            return False, 'narrow_grid'
        # Reject grid ending on crop boundary. A cropped row cannot be recovered
        # from this local search: use full-image fallback instead.
        margin = min(12, max(5, round(H*.055)))
        if ys[0].min() < margin or H-ys[-1].max() < margin:
            return False, 'touches_roi_top_or_bottom'
    except (ValueError, TypeError, IndexError, FloatingPointError) as exc:
        return False, f'invalid_geometry:{type(exc).__name__}'
    return True, 'basic_geometry_pass'


def try_local_grid(image):
    """Return (global graph or None, report, timings); never change originals.

    The caller must fall back to its normal full-image routing on None.
    Independent source pixel audit catches observed but omitted row rules;
    faint or completely invisible rules still require regression coverage.
    """
    started = perf_counter()
    stats = {'status': 'rejected', 'route': 'paper_then_preflight_local_v2_then_completeness_then_smart', 'fallback_to_full': True}
    # Step 1. Find the printed receipt's paper envelope before the table.
    paper = locate_paper(image)
    stats['paper_ms'] = round(paper.elapsed_ms, 3)
    stats['paper_bbox'] = list(paper.bbox) if paper.bbox else None
    stats['paper_status'] = paper.reason
    stats['paper_metrics'] = paper.metrics
    if paper.bbox is None:
        stats.update(status='fallback_to_smart', reason='paper_not_confident_skip_unverifiable_fast', v2_skipped=True)
        stats['total_ms'] = round((perf_counter()-started)*1000, 3)
        return None, {'status':'REJECTED', 'reason':stats['reason']}, stats
    # Paper contour is a proposal, not a trusted crop: if ambiguous, use
    # whole-image table ROI discovery, still audit every observed line.
    t = perf_counter()
    # Paper bounding boxes can clip the margins needed by the physical
    # detector. Locate on the full source and use paper only for the audit.
    loc=locate_table(image)
    stats['locator_scope']='full_image'
    stats['locator_ms'] = round((perf_counter()-t)*1000, 3)
    stats['roi_bbox'] = list(loc.bbox) if loc.bbox else None
    stats['roi_reason'] = loc.reason
    stats['roi_area_fraction'] = loc.diagnostics.get('roi_area_fraction')
    if loc.bbox is None:
        stats['total_ms'] = round((perf_counter()-started)*1000, 3)
        return None, {'status': 'REJECTED', 'reason':loc.reason}, stats
    x0, y0, x1, y1 = loc.bbox
    roi_pixels = (x1-x0)*(y1-y0)
    segments = loc.diagnostics.get('segments', 0)
    stats.update(roi_pixels=roi_pixels, roi_segments=segments)
    if roi_pixels > 180_000 or segments > 35:
        stats.update(status='fallback_to_smart', reason='fast_preflight_large_roi',
                     v2_skipped=True, fast_roi_limit_pixels=180_000, fast_segment_limit=35)
        stats['total_ms'] = round((perf_counter()-started)*1000, 3)
        return None, {'status':'REJECTED', 'reason':stats['reason']}, stats
    # Large local regions are costly and have shown partial-grid failures on
    # long receipts. Leave those to the full detector without a costly V2 run.
    if stats['roi_area_fraction'] is not None and stats['roi_area_fraction'] > .20:
        stats.update(status='fallback_to_smart', reason='roi_too_large')
        stats['total_ms'] = round((perf_counter()-started)*1000, 3)
        return None, {'status':'REJECTED', 'reason':'roi_too_large'}, stats
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        stats['total_ms'] = round((perf_counter()-started)*1000, 3)
        return None, {'status':'REJECTED', 'reason':'empty_roi'}, stats

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    t = perf_counter()
    response, dark = ridge_response(gray)
    stats['ridge_ms'] = round((perf_counter()-t)*1000, 3)

    # FAST FIRST: the local Physical V2 detector is the proven fast route.
    # If it is rejected for any reason, the caller runs the ORIGINAL smart
    # AutoGrid -> full Physical V3 sequence. No extra local V3 before it.
    t = perf_counter()
    graph, report = physical_v2(region, ridge=(response, dark))
    stats['v2_ms'] = round((perf_counter()-t)*1000, 3)
    ok, reason = validate_local_grid(graph, region.shape[:2])
    stats['v2_geometry'] = reason
    if ok:
        t = perf_counter()
        ink = [evidence(rule, graph['vertical'], region.shape[:2], response, dark)
               for rule in graph['horizontal']]
        ink_ok = all(x is not None and bool(x.get('accepted')) for x in ink)
        stats['v2_ink_ms'] = round((perf_counter()-t)*1000, 3)
        stats['v2_accepted_edges'] = sum(bool(x and x.get('accepted')) for x in ink)
        if ink_ok:
            global_graph = offset_graph(graph, loc.bbox, image.shape[:2])
            global_ok, global_reason = validate_local_grid(global_graph, image.shape[:2])
            if global_ok:
                # Step 3. Independent source-image line scan over the entire
                # paper, NOT only the ROI returned by fast locator.
                nodes=np.asarray([[intersection(v,h,image.shape) for v in global_graph['vertical']]
                                  for h in global_graph['horizontal']],dtype=float)
                audit_ok, audit = audit_printed_rows(image,nodes,paper_bbox=paper.bbox)
                stats['completeness'] = audit
                stats['completeness_ms'] = audit.get('elapsed_ms')
                if audit_ok:
                    stats.update(status='accepted', selected='local_v2_paperguard',
                                 fallback_to_full=False)
                    stats['total_ms'] = round((perf_counter()-started)*1000, 3)
                    return global_graph, {
                        'status':'OBSERVED_NOT_VERIFIED',
                        'selection':'local_roi_v2_paperguard',
                        'version':'local_grid_paperguard_v3',
                        'edges':ink,
                        'local_report':report,
                        'completeness_audit':audit,
                    }, stats
                stats['v2_global_check'] = 'completeness_audit_'+str(audit.get('reason'))
            else:
                stats['v2_global_check'] = global_reason
        else:
            stats['v2_geometry'] = 'horizontal_ink_evidence_rejected'
    stats['status'] = 'fallback_to_smart'
    stats['reason'] = stats.get('v2_global_check') or stats.get('v2_geometry') or report.get('reason')
    stats['total_ms'] = round((perf_counter()-started)*1000, 3)
    return None, {'status':'REJECTED', 'reason':'local_graph_invalid',
                  'last_report_status':report.get('status')}, stats
