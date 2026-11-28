from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from App.geometry import service
from App.geometry.geometry_lab.detection import Rule


@pytest.mark.parametrize('observed',[False,True])
def test_final_curvature_requires_actual_source_ink(observed):
    image=np.full((330,500,3),255,np.uint8)
    vertical=[Rule('v',np.array([float(x)]),float(x),1.,(30,300))
              for x in (40,80,260,340,460)]
    curved=Rule('h',np.array([500.,0.,150.]),150.,1.,(40,460))
    horizontal=[Rule('h',np.array([50.]),50.,1.,(40,460)),curved,
                Rule('h',np.array([280.]),280.,1.,(40,460))]
    if observed:
        xs=np.arange(40,461)
        points=np.column_stack((xs,curved.eval((xs-250)/500)))
        cv2.polylines(image,[np.rint(points).astype(np.int32)],False,(0,0,0),2)
    unresolved,proofs=service._remaining_extreme_curves(image,dict(vertical=vertical,horizontal=horizontal))
    assert len(proofs)==1
    assert unresolved==(0 if observed else 1)


@pytest.mark.parametrize('safe,header_ok,accepted',[
    ({'status':'MEASURED','folded_fraction':0,'source_valid_fraction':1},True,True),
    ({'status':'MEASURED','folded_fraction':0,'source_valid_fraction':1},False,False),
    ({'status':'MEASURED','folded_fraction':.01,'source_valid_fraction':1},True,False),
    ({'status':'MEASURED','folded_fraction':0,'source_valid_fraction':.9},True,False),
    ({'status':'UNKNOWN'},True,False),
])
def test_header_rescue_keeps_mesh_and_optical_header_gates(monkeypatch,safe,header_ok,accepted):
    graph={'vertical':[object()]*5}
    candidate=(SimpleNamespace(xs=range(5)),None,{'safety':safe})
    monkeypatch.setattr(service,'detect_physical_graph',lambda _:(graph,{}))
    monkeypatch.setattr(service,'_build_grid_core',lambda *a,**k:candidate)
    result=service.build_grid_physical_header_rescue(np.zeros((20,20,3),np.uint8),
                                                    header_validator=lambda _:header_ok)
    assert (result is not None)==accepted


def test_header_rescue_does_not_truncate_six_column_candidate(monkeypatch):
    monkeypatch.setattr(service,'detect_physical_graph',lambda _:({'vertical':[object()]*7},{}))
    def unexpected(*a,**k):
        pytest.fail('Six-column graph must not be forced into four-column rescue')
    monkeypatch.setattr(service,'_build_grid_core',unexpected)
    assert service.build_grid_physical_header_rescue(np.zeros((20,20,3),np.uint8),
                                                     header_validator=lambda _:True) is None
