"""Opt-in, annotation-assisted GridMesh experiment, never an automatic footer detector."""
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .measure import is_numeric_field, is_name_field, tighten_numeric_crop, split_name_lines


def total_crop(image, annotation):
    """Independent one-cell mesh bounded by the manually annotated total ROI.

    Its boundaries are synthetic, not observed printed rules. With rectangular
    annotations the map is rectangular too; do not extrapolate table curvature.
    """
    from App.geometry.geometry_lab.algorithms import mesh
    from App.geometry.geometry_lab.detection import Rule
    entry = next(e for e in annotation['rois'] if e['field_id'] == 'printed_total')
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [round(v*s) for v, s in zip(entry['bbox'], (w,h,w,h))]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Invalid annotated total bounds')
    rule = lambda axis, p, domain: Rule(axis, np.array([float(p)]), float(p), 0., domain)
    data = {'vertical': [rule('v', x, (y0,y1)) for x in (x0,x1)],
            'horizontal': [rule('h', y, (x0,x1)) for y in (y0,y1)]}
    result = mesh(image, data)
    a,b = map(round,result.xs); c,d = map(round,result.ys)
    return result.image[c:d,a:b].copy(), [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


CACHE_VERSION = 1


def _coordinate_key(image, annotation, include_total):
    digest = hashlib.sha256()
    digest.update(str((CACHE_VERSION, image.shape, image.dtype.str, include_total)).encode())
    digest.update(image.tobytes())
    digest.update(json.dumps(annotation, sort_keys=True).encode())
    return digest.hexdigest()


def extract_gridmesh(image, annotation, *, include_total=False, cache_dir=None):
    """Persist per-field inverse maps; never redetect a valid cached receipt."""
    key = _coordinate_key(image, annotation, include_total)
    path = Path(cache_dir)/f'{key}.npz' if cache_dir is not None else None
    details = None
    maps = []
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            details = json.loads(str(saved['details']))
            maps = [(saved[f'x{i}'], saved[f'y{i}']) for i in range(len(details))]
    if details is None:
        started = time.perf_counter()
        from App.geometry.service import build_grid, cell_crop, cell_polygon
        result, _, _ = build_grid(image, rectify=False)
        mapping = result.info['cell_mesh']
        ox,oy = result.info['source_offset']; h,w = image.shape[:2]
        polygons = [(r,c,(cell_polygon(result,r,c)+[ox,oy]).astype(np.float32))
                    for r in range(len(result.ys)-1) for c in range(len(result.xs)-1)]
        details=[]
        for entry in annotation['rois']:
            field=entry['field_id']; x0,y0,x1,y1=entry['bbox']
            if field == 'printed_total' and include_total:
                _,corners=total_crop(image,annotation)
                a,b=corners[0]; c,d=corners[2]
                my,mx=np.mgrid[b:d,a:c].astype(np.float32)
                source='annotated_total_synthetic_mesh'
            else:
                center=((x0+x1)*w/2,(y0+y1)*h/2)
                found=[(r,c) for r,c,p in polygons if cv2.pointPolygonTest(p,center,False)>=0]
                if len(found)!=1:
                    raise ValueError(f'No unique GridMesh cell for {field}; no ROI fallback')
                r,c=found[0]; _,_,corners=cell_crop(result,r,c)
                xs=list(map(round,mapping.xs)); ys=list(map(round,mapping.ys))
                mx=mapping.map_x[ys[r]+1:ys[r+1]-1,xs[c]+1:xs[c+1]-1].copy()
                my=mapping.map_y[ys[r]+1:ys[r+1]-1,xs[c]+1:xs[c+1]-1].copy()
                source=f'observed_cell:{r}:{c}'
            maps.append((mx,my))
            details.append({'field_id':field,'source':source,'corners':corners})
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            payload={'details':np.array(json.dumps(details)),
                     'build_ms':np.array((time.perf_counter()-started)*1000)}
            for i,(mx,my) in enumerate(maps):
                payload[f'x{i}']=mx; payload[f'y{i}']=my
            temporary=path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,**payload)
            temporary.replace(path)
    cells=[]
    for entry,(mx,my) in zip(details,maps):
        field=entry['field_id']
        crop=cv2.remap(image,mx,my,cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        if is_numeric_field(field): cells.append((field,tighten_numeric_crop(crop)))
        elif is_name_field(field): cells.extend((field,p) for p in split_name_lines(crop))
        else: cells.append((field,crop))
    return cells,details
