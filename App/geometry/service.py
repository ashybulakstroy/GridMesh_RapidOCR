"""Automatic observed-grid geometry, with explicit safety warnings."""
import cv2
import numpy as np
from .lab_adapter_gridrefine_v3 import detect_for_lab
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

def build_grid(image, *, rectify=False, phase=None):
    data, report = detect_for_lab(image)
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
        physical, physical_report = detect_physical_graph(image)
        # Retain the more complete horizontal observations only when the
        # independent graph confirms all four detected vertical boundaries
        # and supplies exactly one additional physical boundary.
        if data is not None and physical is not None and len(data['vertical']) == 4 and len(physical['vertical']) == 5 and len(data['horizontal']) >= 3:
            observed = [v.position for v in data['vertical']]
            graph = [v.position for v in physical['vertical']]
            tolerance = max(3.0, image.shape[1]*.008)
            matches = [min(range(len(graph)), key=lambda i: abs(graph[i]-x)) for x in observed]
            if len(set(matches)) == 4 and all(abs(graph[i]-x) <= tolerance for x,i in zip(observed,matches)):
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
        data, recovered_horizontal = _recover_observed_horizontal(image, data, automatic_data)
    if data is None:
        raise ValueError('Не удалось уверенно найти печатную сетку таблицы. Загрузите другое фото.')
    if not 5 <= len(data['vertical']) <= 9 or len(data['horizontal']) < 3:
        raise ValueError('Не удалось определить границы столбцов товарной таблицы. Требуется проверка фото.')
    H, W = image.shape[:2]
    vs, hs = data['vertical'], data['horizontal']
    if phase:
        phase('Создание сетки Grid Mesh')
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
    check = safety(result.info['cell_mesh'] if not rectify else result, image.shape)
    review = bool(report.get('autogrid', {}).get('geometry_review_required'))
    warnings = []
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
        for row in range(len(hs)):
            for col in range(len(vs)-1):
                poly = cell_polygon(result, max(0,min(row,len(hs)-2)), col)
                edge = poly[:17] if row < len(hs)-1 else poly[34:51]
                cv2.polylines(overlay, [np.rint(edge).astype(np.int32)], False, (255,100,0), 2, cv2.LINE_AA)
        for col in range(len(vs)):
            for row in range(len(hs)-1):
                poly = cell_polygon(result, row, max(0,min(col,len(vs)-2)))
                edge = poly[17:34] if col == len(vs)-1 else poly[51:68]
                cv2.polylines(overlay, [np.rint(edge).astype(np.int32)], False, (255,100,0), 2, cv2.LINE_AA)
    else:
        for x in result.xs:
            cv2.line(overlay, (round(x), round(result.ys[0])), (round(x), round(result.ys[-1])), (255, 100, 0), 2)
        for y in result.ys:
            cv2.line(overlay, (round(result.xs[0]), round(y)), (round(result.xs[-1]), round(y)), (255, 100, 0), 2)
    return result, overlay, {'method': method, 'autogrid_vertical_count': auto_vertical_count,
                             'ridge_refined_horizontal': sum(edge.get('refinement') == 'nearby_source_ridge_3px' for edge in report.get('edges', [])),
                             'rectification': result.method, 'safety': check, 'warnings': warnings}


def cell_polygon(result, row, column):
    """Sample the four observed source curves bounding a single grid cell."""
    data=result.info['source_grid']; H,W=result.info['source_shape'][:2]
    left,top=result.info['source_offset']; v0,v1=data['vertical'][column:column+2]
    h0,h1=data['horizontal'][row:row+2]
    tl=np.array(intersection(v0,h0,(H,W))); tr=np.array(intersection(v1,h0,(H,W)))
    bl=np.array(intersection(v0,h1,(H,W))); br=np.array(intersection(v1,h1,(H,W)))
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
