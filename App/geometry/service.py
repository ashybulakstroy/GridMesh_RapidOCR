"""Automatic observed-grid geometry, with explicit safety warnings."""
# GRID_ABORT_DIAGNOSTICS_V1
from App import grid_abort_diagnostics_v1 as _grid_diag
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import cv2
import numpy as np
from time import perf_counter
from ..ocr.debug_performance import get_debug
from .lab_adapter_gridrefine_v3 import detect_for_lab
from .fold import recover_fold, recover_left_edges
from .geometry_lab.algorithms import mesh, Result, intersection
from .geometry_lab.metrics import safety
from .geometry_lab.physical_grid_graph_v3 import detect_physical_graph, evidence, ridge_response



def _refine_vertical_by_ink(image, rule, horizontal):
    """Refit a recovered vertical only with distributed printed-stroke evidence."""
    from .geometry_lab.detection import Rule
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 9)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((13, 1), np.uint8))
    top = max(0, round(horizontal[0].position)); bottom = min(height, round(horizontal[-1].position))
    points = []
    for y in range(top+12, bottom-12, 24):
        predicted = float(rule.eval((y-height/2)/height))
        left = max(0, int(predicted)-16); right = min(width, int(predicted)+17)
        score = np.count_nonzero(ink[y-9:y+10,left:right], axis=0)
        if not len(score): continue
        adjusted = score - .10*np.abs(np.arange(left,right)-predicted)
        x = left + int(np.argmax(adjusted))
        if score[x-left] >= 8: points.append((y,x))
    if len(points) < 8 or points[-1][0]-points[0][0] < (bottom-top)*.7:
        return rule
    y,x = np.array(points,dtype=float).T
    coeff = np.polyfit((y-height/2)/height,x,2)
    residual = np.abs(np.polyval(coeff,(y-height/2)/height)-x)
    keep = residual <= 3
    if keep.sum() < 8 or keep.mean() < .7: return rule
    coeff = np.polyfit((y[keep]-height/2)/height,x[keep],2)
    probe = np.linspace(top,bottom,30)
    if np.max(np.abs(np.polyval(coeff,(probe-height/2)/height)-rule.eval((probe-height/2)/height))) > 16:
        return rule
    return Rule(rule.axis, coeff, float(np.polyval(coeff,0)), rule.support, rule.domain)


def _recover_observed_horizontal(image, physical, automatic):
    """Restore internal candidate boundaries only with original-pixel evidence."""
    if automatic is None or len(physical['vertical']) != 5:
        return physical, 0
    horizontal = list(physical['horizontal'])
    if len(horizontal) < 3:
        return physical, 0
    candidates = [h for h in automatic['horizontal']
                  if horizontal[0].position + 8 < h.position < horizontal[-1].position - 8
                  and all(abs(h.position - known.position) >= 8 for known in horizontal)]
    if not candidates:
        return physical, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response, dark = ridge_response(gray)
    width = image.shape[1]
    sample = (np.linspace(physical['vertical'][0].position,
                          physical['vertical'][-1].position, 100) - width / 2) / width
    recovered = 0
    for candidate in candidates:
        observed = evidence(candidate, physical['vertical'], gray.shape, response, dark)
        if not observed or not observed['accepted']:
            continue
        if any(np.min(np.abs(candidate.eval(sample) - h.eval(sample))) < 6 for h in horizontal):
            continue
        horizontal.append(candidate)
        recovered += 1
    if recovered:
        horizontal.sort(key=lambda rule: rule.position)
        return {**physical, 'horizontal': horizontal}, recovered
    return physical, 0


def _repair_extreme_horizontal_curves(image, data):
    """Repair only extreme curves using long observed horizontal ink segments.

    Search is bounded by the adjacent detected rules and only triggered by
    severe curvature. Missing or ambiguous evidence leaves the grid unchanged.
    """
    from .geometry_lab.detection import Rule
    H,W=image.shape[:2]
    vs=data['vertical']; rules=list(data['horizontal'])
    if len(vs)!=5 or len(rules)<4:
        return data,0,0
    left=max(0,round(vs[1].position)); right=min(W,round(vs[-1].position))
    width=right-left
    if width<100:
        return data,0,0
    xs=np.linspace(left,right,40); norm=(xs-W/2)/W
    suspicious=[]
    for i in range(1,len(rules)-1):
        ys=rules[i].eval(norm)
        linear=np.linspace(ys[0],ys[-1],len(ys))
        if np.max(np.abs(ys-linear)) > max(12,width*.035):
            # GRID_ANOMALY_V51_FULL_COMPLETENESS: only proven curves skip old repair.
            from ..config import setting as _v51_setting
            _confirmed=False
            if data.get('v51_recovery_proven') and _v51_setting('OCR_GRID_ANOMALY_V51_CURVES','0')=='1':
                try:
                    from .grid_anomaly_v51 import verified_curvature
                    _confirmed,_proof=verified_curvature(image,data,i)
                    from . import detector_routing_v1 as _v51_route
                    _v51_route.record('v51_curvature',index=i,accepted=_confirmed,proof=_proof)
                except (ValueError,TypeError,IndexError,KeyError,cv2.error,OverflowError) as _exc:
                    _confirmed=False
            if not _confirmed:suspicious.append(i)
    if not suspicious:
        return data,0,0
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    repaired=0; unresolved=0
    for idx in suspicious:
        prev,old,nxt=rules[idx-1:idx+2]
        top=max(0,int(np.ceil(np.max(prev.eval(norm))))+6)
        bottom=min(H,int(np.floor(np.min(nxt.eval(norm))))-6)
        if bottom-top<22:
            unresolved+=1; continue
        crop=gray[top:bottom,left:right]
        binary=cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV,31,9)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(18,round(width*.06)),1))
        strokes=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        lines=cv2.HoughLinesP(strokes,1,np.pi/360,threshold=max(18,round(width*.06)),
                              minLineLength=max(24,round(width*.08)),maxLineGap=8)
        groups=[]
        if lines is not None:
            for x1,y1,x2,y2 in lines.reshape(-1,4):
                dx=float(x2-x1)
                if abs(dx)<max(24,width*.08) or abs(y2-y1)>abs(dx)*.08:continue
                slope=float(y2-y1)/dx
                mid_y=top+y1+slope*(width/2-x1)
                if not top+4<mid_y<bottom-4:continue
                group=next((g for g in groups if abs(g['mid_y']-mid_y)<4),None)
                if group is None:
                    group={'mid_y':mid_y,'segments':[]};groups.append(group)
                group['segments'].append((left+x1,top+y1,left+x2,top+y2))
        replacements=[]
        for group in groups:
            segments=group['segments']
            points=np.array([(x,y) for x1,y1,x2,y2 in segments for x,y in ((x1,y1),(x2,y2))],dtype=float)
            if np.ptp(points[:,0])<width*.45:continue
            coeff=np.polyfit((points[:,0]-W/2)/W,points[:,1],1)
            if np.max(np.abs(np.polyval(coeff,(points[:,0]-W/2)/W)-points[:,1]))>3:continue
            ys=np.polyval(coeff,norm)
            if np.min(ys-prev.eval(norm))<6 or np.min(nxt.eval(norm)-ys)<6:continue
            replacements.append(Rule('h',coeff,float(np.polyval(coeff,0)),0.0,(left,right)))
        replacements.sort(key=lambda rule:rule.position)
        if len(replacements)!=2 or np.min(replacements[1].eval(norm)-replacements[0].eval(norm))<8:
            unresolved+=1;continue
        # The corrupt rule is replaced by the two independently visible lines.
        rules[idx:idx+1]=replacements
        repaired+=1
        break  # Indices change after insertion; defer any other suspect to review.
    unresolved+=max(0,len(suspicious)-repaired-unresolved)
    if repaired:
        return {**data,'horizontal':rules},repaired,unresolved
    return data,0,unresolved

@_grid_diag.observe('App/geometry/service.py:_build_grid_core')
def _build_grid_core(image, *, rectify=False, phase=None, pre_detected=None):
    _debug=get_debug()
    _grid_t0=perf_counter() if _debug is not None else None
    if pre_detected is not None:
        data, report = pre_detected
        is_local_autogrid = (report or {}).get('selection') in ('local_roi_autogrid_paperguard_v4', 'local_table_roi_autogrid_paperguard_v5', 'local_table_roi_autogrid_paperguard_v6', 'paper_first_autogrid_v12', 'source_verified_five_rail_v16')
        automatic_data = data if is_local_autogrid else None
        recovered_horizontal = 0
        method = 'AutoGrid GridRefine V3' if is_local_autogrid else 'Physical Grid Graph V3'
        auto_vertical_count = len(data['vertical']) if is_local_autogrid else None
        invalid_spacing = False
    else:
        data, report = detect_for_lab(image)
        if _debug is not None:
            _debug.add_geometry('autogrid_gridrefine_v3',(perf_counter()-_grid_t0)*1000,
                                {'found':data is not None,'status':report.get('status'),
                                 'reason':report.get('reason'),
                                 'detector_status':report.get('autogrid',{}).get('status'),
                                 'horizontal_boundaries':report.get('autogrid',{}).get('horizontal_boundaries'),
                                 'vertical_boundaries':report.get('autogrid',{}).get('vertical_boundaries'),
                                 'roi':report.get('autogrid',{}).get('auto_roi')})
        automatic_data = data
        recovered_horizontal = 0
        method = 'AutoGrid GridRefine V3'
        auto_vertical_count = len(data['vertical']) if data is not None else None
        # Reject grids that Grid Mesh cannot represent before choosing a detector.
        # In particular, near-duplicate false lines can pass the detector's looser
        # spacing threshold while violating the mesh's minimum row spacing.
        invalid_spacing = data is not None and (any(np.diff([v.position for v in data['vertical']]) < 6) or
                                                   any(np.diff([h.position for h in data['horizontal']]) < 5))
        if invalid_spacing:
            data = None
        if data is None or not 5 <= len(data['vertical']) <= 9 or len(data['horizontal']) < 3:
            _physical_t0=perf_counter() if _debug is not None else None
            if _debug is not None:
                _debug.add_geometry('geometry_fallback_trigger',0,
                    {'reason':'invalid_spacing' if invalid_spacing else
                              'autogrid_rejected' if data is None else 'insufficient_boundaries',
                     'autogrid_reason':report.get('reason'),
                     'detector_status':report.get('autogrid',{}).get('status')})
            # ADAPTIVE_DETECTOR_ROUTING_V1: keep all other service paths unchanged.
            from . import detector_routing_v1 as _route
            _prefer=_route.choose(data,report,image)
            _route.record('physical_route',prefer_multiscale=_prefer,autogrid_h=len(data['horizontal']) if data is not None else 0,autogrid_v=len(data['vertical']) if data is not None else 0)
            if _prefer:
                physical, physical_report = detect_physical_graph(
                    image, prefer_multiscale=True,
                    min_rows=len(data['horizontal']) if data is not None else 0, automatic=data)
            else:
                physical, physical_report = detect_physical_graph(image)
            if _debug is not None:
                _debug.add_geometry('physical_grid_graph_v3',(perf_counter()-_physical_t0)*1000,
                                    {'found':physical is not None,'status':physical_report.get('status'),
                                     'selection':physical_report.get('selection'),
                                     'reason':physical_report.get('reason'),
                                     'v2_status':physical_report.get('v2_status'),
                                     'horizontal_candidates':physical_report.get('horizontal_candidates'),
                                     'vertical_candidates':physical_report.get('vertical_candidates'),
                                     'local_hypotheses':physical_report.get('local_hypotheses')})
            # Retain the more complete horizontal observations only when the
            # independent graph confirms all four detected vertical boundaries
            # and supplies exactly one additional physical boundary.
            if data is not None and physical is not None and len(data['vertical']) == 4 and len(physical['vertical']) == 5 and len(data['horizontal']) >= 3:
                observed = [v.position for v in data['vertical']]
                graph = [v.position for v in physical['vertical']]
                tolerance = max(3.0, image.shape[1]*.008)
                matches = [min(range(len(graph)), key=lambda i: abs(graph[i]-x)) for x in observed]
                # Completing a column must not discard additional source-confirmed
                # row boundaries found by the physical detector (including header).
                # Otherwise two products can become the apparent header band.
                if (len(set(matches)) == 4 and all(abs(graph[i]-x) <= tolerance for x,i in zip(observed,matches))
                        and len(data['horizontal']) >= len(physical['horizontal'])):
                    missing = next(i for i in range(5) if i not in matches)
                    vertical = list(physical['vertical'])
                    vertical[missing] = _refine_vertical_by_ink(image, vertical[missing], data['horizontal'])
                    data = {**data, 'vertical': vertical}
                    report = {**report, 'vertical_completion': physical_report}
                    method = 'AutoGrid GridRefine V3 + Physical Grid Graph V3'
                else:
                    data, report, method = physical, physical_report, 'Physical Grid Graph V3'
            else:
                data, report, method = physical, physical_report, 'Physical Grid Graph V3'
    if data is not None and method == 'Physical Grid Graph V3':
        _ridge_t0=perf_counter() if _debug is not None else None
        data, recovered_horizontal = _recover_observed_horizontal(image, data, automatic_data)
        if _debug is not None:
            _debug.add_geometry('recover_observed_horizontal',(perf_counter()-_ridge_t0)*1000,
                                {'recovered':recovered_horizontal})
    if data is None:
        raise ValueError('Не удалось уверенно найти печатную сетку таблицы. Загрузите другое фото.')
    if not 5 <= len(data['vertical']) <= 9 or len(data['horizontal']) < 3:
        raise ValueError('Не удалось определить границы столбцов товарной таблицы. Требуется проверка фото.')
    _repair_t0=perf_counter() if _debug is not None else None
    data, repaired_curves, unresolved_curves = _repair_extreme_horizontal_curves(image, data)
    if _debug is not None:
        _debug.add_geometry('repair_extreme_horizontal_curves',(perf_counter()-_repair_t0)*1000,
                            {'repaired':repaired_curves,'unresolved':unresolved_curves})
    fold_recovered = 0
    if method == 'AutoGrid GridRefine V3':
        _fold_t0=perf_counter() if _debug is not None else None
        data, fold_recovered = recover_fold(image, data)
        if _debug is not None:
            _debug.add_geometry('recover_fold',(perf_counter()-_fold_t0)*1000,
                                {'recovered':fold_recovered})
    _left_t0=perf_counter() if _debug is not None else None
    data, left_recovered = recover_left_edges(image, data)
    if _debug is not None:
        _debug.add_geometry('recover_left_edges',(perf_counter()-_left_t0)*1000,
                            {'recovered':left_recovered})
    # GRID_SPEED_COMBINED_V1: repair only a speculative adaptive candidate.
    _combo_diag=None
    from ..config import setting as _combo_setting
    if method=='Physical Grid Graph V3' and _combo_setting('OCR_GRID_COMBO_PREMESH','1')=='1' and _combo_setting('OCR_GRID_ANOMALY_V3','1')=='1':
        from . import detector_routing_v1 as _combo_route
        if _combo_route.used() and not unresolved_curves:
            from .grid_speed_combined_v1 import premesh
            from .grid_anomaly_v3 import scan_above,recover_graph
            from .geometry_lab.algorithms import _dest
            from .geometry_lab.detection import Rule
            try:
                data,_combo_diag=premesh(image,data,scan_above,recover_graph,_dest,Rule)
            except cv2.error:
                _grid_diag.caught(locals(), 'App/geometry/service.py:_build_grid_core:L273')
                _combo_diag=None
            _combo_route.record('combo_premesh',accepted=_combo_diag is not None)
    H, W = image.shape[:2]
    vs, hs = data['vertical'], data['horizontal']
    if phase:
        phase('Создание сетки Grid Mesh')
    _mesh_t0=perf_counter() if _debug is not None else None
    if not rectify:
        intersections = np.array([[intersection(v,h,image.shape) for v in vs] for h in hs], dtype=float)
        if not np.isfinite(intersections).all(): raise ValueError('Некорректные пересечения линий сетки')
        left = max(0, int(np.floor(intersections[:,:,0].min()))-16)
        top = max(0, int(np.floor(intersections[:,:,1].min()))-16)
        right = min(W, int(np.ceil(intersections[:,:,0].max()))+17)
        bottom = min(H, int(np.ceil(intersections[:,:,1].max()))+17)
        if right-left < 30 or bottom-top < 30: raise ValueError('Некорректная область сетки')
        vx = np.median(intersections[:,:,0],axis=0); hy = np.median(intersections[:,:,1],axis=1)
        if np.any(np.diff(vx) <= 4) or np.any(np.diff(hy) <= 4):
            raise ValueError('Пересекающиеся границы сетки; проверьте фото')
        gy, gx = np.indices((bottom-top, right-left), dtype=np.float32)
        result = Result('original_cells', image[top:bottom,left:right].copy(), gx+left, gy+top,
                        [float(x-left) for x in vx], [float(y-top) for y in hy],
                        {'physical_grid':'observed_lines_source', 'source_grid':data,
                         'source_shape':image.shape, 'source_offset':(left,top),
                         'cell_mesh':mesh(image, data, sample_image=False), 'source_image':image})
    else:
        result = mesh(image, data)
        result.info['source_grid'] = data  # Grid Anomaly V3: preserve rules for direct repair
    if _debug is not None:
        _debug.add_geometry('grid_mesh_construction',(perf_counter()-_mesh_t0)*1000,
                            {'rectify_full_grid':bool(rectify)})
        _overlay_t0=perf_counter()
    check = safety(result.info['cell_mesh'] if not rectify else result, image.shape)
    review = bool(report.get('autogrid', {}).get('geometry_review_required'))
    warnings = []
    if repaired_curves:
        warnings.append(f'Восстановлены границы по видимым участкам линий: {repaired_curves}; проверьте сетку.')
    if unresolved_curves:
        warnings.append('Обнаружена сильно искривлённая граница строки; проверьте сетку.')
    if left_recovered:
        warnings.append(f'Уточнены левые участки линий по изображению: {left_recovered}; проверьте сетку.')
    if recovered_horizontal:
        warnings.append(f'Восстановлено границ строк по видимым линиям: {recovered_horizontal}; проверьте сетку и данные.')
    if invalid_spacing:
        warnings.append('Основной детектор нашёл слишком близкие границы; применена резервная сетка. Проверьте данные.')
    if 'vertical_completion' in report:
        warnings.append('Граница столбца восстановлена по наблюдаемым линиям; проверьте сетку и данные.')
    if review: warnings.append('Ненадёжные участки линий: проверьте границы ячеек.')
    if check.get('folded_fraction', 1) > 0 or check.get('source_valid_fraction', 0) < .995:
        warnings.append('Карта выпрямления выходит за границы фотографии или содержит складки.')
    overlay = result.image.copy()
    if not rectify:
        # GRID_OVERLAY_REUSE_V1
        from ..config import setting as _overlay_setting
        _reuse = _overlay_setting('OCR_GRID_OVERLAY_REUSE_V1','1') == '1'
        if _reuse:
            from .grid_overlay_reuse_v1 import polygon_reader
            _polygon = polygon_reader(result, intersections, cell_polygon)
        else:
            _polygon = lambda row,col: cell_polygon(result,row,col)
        for row in range(len(hs)):
            for col in range(len(vs)-1):
                poly = _polygon(max(0,min(row,len(hs)-2)), col)
                edge = poly[:17] if row < len(hs)-1 else poly[34:51]
                cv2.polylines(overlay, [np.rint(edge).astype(np.int32)], False, (255,100,0), 2, cv2.LINE_AA)
        for col in range(len(vs)):
            for row in range(len(hs)-1):
                poly = _polygon(row, max(0,min(col,len(vs)-2)))
                edge = poly[17:34] if col == len(vs)-1 else poly[51:68]
                cv2.polylines(overlay, [np.rint(edge).astype(np.int32)], False, (255,100,0), 2, cv2.LINE_AA)
    else:
        for x in result.xs:
            cv2.line(overlay, (round(x), round(result.ys[0])), (round(x), round(result.ys[-1])), (255, 100, 0), 2)
        for y in result.ys:
            cv2.line(overlay, (round(result.xs[0]), round(y)), (round(result.xs[-1]), round(y)), (255, 100, 0), 2)
    if _debug is not None:
        _debug.add_geometry('safety_and_overlay',(perf_counter()-_overlay_t0)*1000)
    if _combo_diag is not None and (check.get('status')!='MEASURED' or check.get('folded_fraction',1)>0 or check.get('source_valid_fraction',0)<.995):
        raise ValueError('combo premesh: unsafe repaired mesh')
    return _grid_diag.ret((result, overlay, {'combo_premesh':_combo_diag,'method': method, 'autogrid_vertical_count': auto_vertical_count,
                             'repaired_extreme_horizontal': repaired_curves,
                             'unresolved_extreme_horizontal': unresolved_curves,
                             'physical_detection_selection': report.get('selection'),
                             'fold_recovered_horizontal': fold_recovered, 'left_recovered_horizontal': left_recovered,
                             'ridge_refined_horizontal': sum(edge.get('refinement') == 'nearby_source_ridge_3px' for edge in report.get('edges', [])),
                             'rectification': result.method, 'safety': check, 'warnings': warnings}), locals(), 'App/geometry/service.py:_build_grid_core:L352')



# LOCALGRID_ROI_INTEGRATION_V1
@_grid_diag.observe('App/geometry/service.py:build_grid')
def build_grid(image, *, rectify=False, phase=None):
    """Fast local ROI path with a production-safe full-image fallback."""
    from ..config import setting
    if setting('OCR_LOCALGRID_MODE', 'on').strip().lower() != 'on':
        return _grid_diag.ret((_build_grid_core(image, rectify=rectify, phase=phase)), locals(), 'App/geometry/service.py:build_grid:L367')
    from .local_grid_runtime import try_local_grid
    from time import perf_counter as _local_clock
    start = _local_clock()
    stats = None
    try:
        graph, report, stats = try_local_grid(image)
    except Exception as exc:
        _grid_diag.caught(locals(), 'App/geometry/service.py:build_grid:L374')
        stats = {'status':'exception', 'reason':f'{type(exc).__name__}: {exc}',
                  'fallback_to_full':True}
        graph = None
        report = None
    debug = get_debug()
    if graph is not None:
        try:
            result, overlay, geometry = _build_grid_core(
                image, rectify=rectify, phase=phase, pre_detected=(graph,report))
            safety = geometry.get('safety') or {}
            if safety.get('status') != 'MEASURED':
                raise ValueError('local mesh safety was not measured')
            if (safety.get('folded_fraction', 0) > 0 or
                    safety.get('source_valid_fraction', 1) < .995 or
                    geometry.get('unresolved_extreme_horizontal', 0) > 0):
                raise ValueError('local mesh failed safety/geometry checks')
            geometry['method'] = 'LocalGrid PaperGuard V4 · AutoGrid on Fast ROI'
            geometry['local_grid_roi'] = stats
            if debug is not None and stats is not None:
                debug.add_geometry('local_grid_roi', stats.get('total_ms',(_local_clock()-start)*1000), stats)
            return _grid_diag.ret((result, overlay, geometry), locals(), 'App/geometry/service.py:build_grid:L395')
        except Exception as exc:
            # A valid detected grid might still fail mesh/safety construction.
            # Never fail a production job merely because the optional path failed.
            _grid_diag.caught(locals(), 'App/geometry/service.py:build_grid:L396')
            if stats is not None:
                stats['fallback_to_full'] = True
                stats['postprocess_error'] = f'{type(exc).__name__}: {exc}'
    if stats is not None:
        stats['status'] = 'fallback_to_smart'
        stats['fallback_to_full'] = True
    if debug is not None and stats is not None:
        debug.add_geometry('local_grid_roi', stats.get('total_ms',(_local_clock()-start)*1000), stats)
    result, overlay, geometry = _build_grid_core(image, rectify=rectify, phase=phase)
    geometry['local_grid_roi'] = stats
    return _grid_diag.ret((result, overlay, geometry), locals(), 'App/geometry/service.py:build_grid:L409')

@_deep_timed("grid", "App/geometry/service.py:cell_polygon")
def cell_polygon(result, row, column, *, nodes=None):
    """Sample the four observed source curves bounding a single grid cell."""
    data=result.info['source_grid']; H,W=result.info['source_shape'][:2]
    left,top=result.info['source_offset']; v0,v1=data['vertical'][column:column+2]
    h0,h1=data['horizontal'][row:row+2]
    if nodes is None:
        tl=np.array(intersection(v0,h0,(H,W))); tr=np.array(intersection(v1,h0,(H,W)))
        bl=np.array(intersection(v0,h1,(H,W))); br=np.array(intersection(v1,h1,(H,W)))
    else:
        tl,tr=nodes[row,column],nodes[row,column+1]
        bl,br=nodes[row+1,column],nodes[row+1,column+1]
    def horizontal(rule,a,b):
        x=np.linspace(a[0],b[0],17);return np.column_stack((x,rule.eval((x-W/2)/W)))
    def vertical(rule,a,b):
        y=np.linspace(a[1],b[1],17);return np.column_stack((rule.eval((y-H/2)/H),y))
    return np.vstack((horizontal(h0,tl,tr),vertical(v1,tr,br),horizontal(h1,br,bl),vertical(v0,bl,tl)))-[left,top]


def cell_crop(result, row, column):
    """Extract a cell using the physical grid mesh and return source corners."""
    if result.method == 'original_cells':
        mapping = result.info['cell_mesh']
        xs=list(map(round,mapping.xs)); ys=list(map(round,mapping.ys))
        x0,x1=xs[column]+1,xs[column+1]-1; y0,y1=ys[row]+1,ys[row+1]-1
        if x1-x0<4 or y1-y0<4: raise ValueError('Invalid cell dimensions')
        mx=mapping.map_x[y0:y1,x0:x1]; my=mapping.map_y[y0:y1,x0:x1]
        crop=cv2.remap(result.info['source_image'],mx,my,cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        corners=[(x0,y0),(x1-1,y0),(x1-1,y1-1),(x0,y1-1)]
        source=[[round(float(mapping.map_x[y,x]),2),round(float(mapping.map_y[y,x]),2)] for x,y in corners]
        return crop,(x0,y0),source
    xs=list(map(round,result.xs));ys=list(map(round,result.ys))
    x0,x1=xs[column]+1,xs[column+1]-1;y0,y1=ys[row]+1,ys[row+1]-1
    if x1-x0<4 or y1-y0<4: raise ValueError('Некорректный размер ячейки')
    corners=[(x0,y0),(x1-1,y0),(x1-1,y1-1),(x0,y1-1)]
    source=[[round(float(result.map_x[y,x]),2),round(float(result.map_y[y,x]),2)] for x,y in corners]
    return result.image[y0:y1,x0:x1],(x0,y0),source


# PAPER_FIRST_FULLGRID_V16: fallback V12 geometry plus independent V16 source rails.
# Normal, successful V4 receipts do not pay for an extra local search.
@_grid_diag.observe('App/geometry/service.py:build_grid_paper_rescue')
def build_grid_paper_rescue(image, *, rectify=False, phase=None, header_validator=None):
    from ..config import setting
    if setting('OCR_PAPER_RESCUE_V12', '1') != '1':
        return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_paper_rescue:L456')
    from .paper_first_grid_v12 import try_paper_grid, json_native
    for retry in (False, True):
        try:
            graph, report, stats = try_paper_grid(image, retry=retry)
            if graph is None:
                continue
            result, overlay, geometry = _build_grid_core(
                image, rectify=rectify, phase=phase, pre_detected=(graph,report))
            safe = geometry.get('safety') or {}
            if (safe.get('status') != 'MEASURED' or
                    safe.get('folded_fraction', 1) > 0 or
                    safe.get('source_valid_fraction', 0) < .995 or
                    geometry.get('unresolved_extreme_horizontal', 0) > 0):
                continue
            if header_validator is not None and not header_validator(result):
                # A complete grid whose first physical band is a product,
                # not a header, remains INVALID. Try the alternative ROI.
                continue
            geometry['method'] = 'PaperFirst V12 · source-verified local AutoGrid'
            geometry['paper_first_v12'] = json_native(stats)
            return _grid_diag.ret((result,overlay,geometry), locals(), 'App/geometry/service.py:build_grid_paper_rescue:L477')
        except Exception as exc:
            # An optional rescue can never break the existing geometry route.
            _grid_diag.caught(locals(), 'App/geometry/service.py:build_grid_paper_rescue:L478')
            pass  # Rescue failed: let the original error stand
    return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_paper_rescue:L481')


# PAPER_FIRST_FULLGRID_V16: independent physical-ink recovery. This route
# runs only when the usual first table band is implausible or a header fails.
@_grid_diag.observe('App/geometry/service.py:build_grid_source_rescue')
def _build_grid_source_rescue_before_short_v1(image, *, rectify=False, phase=None, header_validator=None):
    from ..config import setting
    if setting('OCR_SOURCE_GRID_V16','1') != '1':
        return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_source_rescue:L489')
    try:
        from .source_grid_v16 import detect_source_grid, validate_source_grid
        from .paper_first_grid_v12 import json_native
        graph,diagnostic=detect_source_grid(image)
        if graph is None:
            return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_source_rescue:L495')
        accepted,audit=validate_source_grid(image,graph,diagnostic)
        if not accepted:
            return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_source_rescue:L498')
        report={'status':'OBSERVED_NOT_GT','selection':'source_verified_five_rail_v16',
                'version':'source_grid_v16','ink_validation':audit}
        result,overlay,geometry=_build_grid_core(
            image,rectify=rectify,phase=phase,pre_detected=(graph,report))
        safe=geometry.get('safety') or {}
        if (safe.get('status')!='MEASURED' or
                safe.get('folded_fraction',1)>0 or
                safe.get('source_valid_fraction',0)<.995 or
                geometry.get('unresolved_extreme_horizontal',0)>0):
            return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_source_rescue:L508')
        if header_validator is not None and not header_validator(result):
            return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_source_rescue:L510')
        geometry['method']='SourceGrid V16 · independently confirmed source rails'
        geometry['source_grid_v16']=json_native({'detector':diagnostic,'audit':audit})
        return _grid_diag.ret((result,overlay,geometry), locals(), 'App/geometry/service.py:build_grid_source_rescue:L513')
    except Exception as exc:
        # Do not turn a failed optional rescue into corrupt production output.
        _grid_diag.caught(locals(), 'App/geometry/service.py:build_grid_source_rescue:L514')
        return _grid_diag.ret((None), locals(), 'App/geometry/service.py:build_grid_source_rescue:L516')

# SHORT_RECEIPT_GRID_RECOVERY_V1: only on established source-rescue path.
@_grid_diag.observe('App/geometry/service.py:short_receipt_grid_v1')
def build_grid_source_rescue(image, *, rectify=False, phase=None, header_validator=None):
    from ..config import setting
    previous = _build_grid_source_rescue_before_short_v1(image, rectify=rectify,
        phase=phase, header_validator=header_validator)
    if previous is not None or setting('OCR_SHORT_RECEIPT_GRID_V1','1') != '1' or header_validator is None:
        return previous
    if setting('OCR_SOURCE_GRID_V16','1') != '1':
        return None
    try:
        from .short_receipt_grid_v1 import rescue
        result = rescue(image, rectify=rectify, phase=phase,
                        header_validator=header_validator, build_core=_build_grid_core,
                        physical_detector=detect_physical_graph)
        return _grid_diag.ret(result, locals(), 'App/geometry/service.py:short_receipt_grid_v1:result')
    except Exception:
        _grid_diag.caught(locals(), 'App/geometry/service.py:short_receipt_grid_v1')
        return None  # preserve the original geometry failure, never force a result

