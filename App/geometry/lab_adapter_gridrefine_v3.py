"""Optional bridge for Standalone_Torgsoft_Geometry_Lab_V3.

Does NOT replace, monkeypatch, or modify any V3 detector. Add this method as
an additional worker in the lab's method registry if desired. Pass BGR image.
"""
from __future__ import annotations
# GRID_ABORT_DIAGNOSTICS_V1
from App import grid_abort_diagnostics_v1 as _grid_diag
import numpy as np
from .autogrid_gridrefine_v3 import detect


def _converted_coeff(coeff, ref, extent):
    """Map poly((coordinate-ref)/500) to poly((coordinate-extent/2)/extent)."""
    original=np.poly1d(np.asarray(coeff,dtype=float))
    substitute=np.poly1d([float(extent)/500.,(float(extent)/2-float(ref))/500.])
    return np.asarray((original(substitute)).c, dtype=np.float64)


@_grid_diag.observe('App/geometry/lab_adapter_gridrefine_v3.py:detect_for_lab')
def detect_for_lab(image):
    """Return (V3-compatible observation, details); rules are measured, not GT."""
    from .geometry_lab.detection import Rule
    _,res=detect(image)
    if res['status']!='CURVES_FITTED':
        return _grid_diag.ret((None,{'status':'REJECTED','reason':res['status'],'autogrid':res}), locals(), 'App/geometry/lab_adapter_gridrefine_v3.py:detect_for_lab:L23')
    x0,y0,x1,y1=res['auto_roi'];H,W=image.shape[:2]
    xref=int(x0+.45*(x1-x0));yref=int(y0+.52*(y1-y0))
    horizontal=[];vertical=[]
    for coeff in res['h_curves']:
        c=_converted_coeff(coeff,xref,W)
        horizontal.append(Rule('h',c,float(np.polyval(c,0)),0.0,(x0,x1)))
    for coeff in res['v_curves']:
        c=_converted_coeff(coeff,yref,H)
        vertical.append(Rule('v',c,float(np.polyval(c,0)),0.0,(y0,y1)))
    # A supported horizontal or vertical curve is only a candidate; never
    # mislabel it as ground truth.
    if (len(horizontal)<3 or len(vertical)<2):
        return _grid_diag.ret((None,{'status':'REJECTED','reason':'insufficient_grid','autogrid':res}), locals(), 'App/geometry/lab_adapter_gridrefine_v3.py:detect_for_lab:L36')
    if any(np.diff([r.position for r in horizontal])<=4) or any(np.diff([r.position for r in vertical])<=5):
        return _grid_diag.ret((None,{'status':'REJECTED','reason':'crossing_or_unsorted','autogrid':res}), locals(), 'App/geometry/lab_adapter_gridrefine_v3.py:detect_for_lab:L38')
    return _grid_diag.ret(({'horizontal':horizontal,'vertical':vertical,'shape':(H,W)}, {'status':'OBSERVED','autogrid':res}), locals(), 'App/geometry/lab_adapter_gridrefine_v3.py:detect_for_lab:L39')


def run_v3_mesh(image):
    """Run V3 Grid Mesh on observed automatic lines; isolated experiment only."""
    from .geometry_lab.algorithms import mesh
    data,report=detect_for_lab(image)
    if data is None:return None,report
    try:
        return mesh(image,data),report
    except Exception as exc:
        return None,{'status':'REJECTED','reason':f'{type(exc).__name__}: {exc}','autogrid':report}

