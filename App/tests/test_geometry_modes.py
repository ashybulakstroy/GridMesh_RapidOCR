import numpy as np
from App.geometry import service
from App.geometry.geometry_lab.detection import Rule
import pytest


@pytest.mark.parametrize('auto_ys,physical_ys,expected',[
    ((60,110,170),(20,60,90,110,170),(20,60,90,110,170)),
    ((20,60,90,110,170),(60,110,170),(20,60,90,110,170)),
])
def test_vertical_completion_preserves_more_complete_row_boundaries(monkeypatch,auto_ys,physical_ys,expected):
    image=np.full((200,250,3),255,np.uint8)
    v=[Rule('v',np.array([float(x)]),float(x),1.,(20,180)) for x in (20,50,140,180,225)]
    def horizontal(ys):return [Rule('h',np.array([float(y)]),float(y),1.,(20,225)) for y in ys]
    automatic={'vertical':[v[i] for i in (0,1,3,4)],'horizontal':horizontal(auto_ys)}
    physical={'vertical':v,'horizontal':horizontal(physical_ys)}
    monkeypatch.setattr(service,'detect_for_lab',lambda _:(automatic,{'autogrid':{}}))
    monkeypatch.setattr(service,'detect_physical_graph',lambda _:(physical,{'selection':'confirmed_test_lines'}))
    monkeypatch.setattr(service,'_refine_vertical_by_ink',lambda _,rule,__:rule)
    result,_,_=service.build_grid(image)
    assert [rule.position for rule in result.info['source_grid']['horizontal']]==list(expected)


def test_straight_observed_grid_uses_original_pixels(monkeypatch):
    img = np.full((200, 250, 3), 210, dtype=np.uint8)
    v = [Rule('v', np.array([float(x)]), float(x), 1., (20,180)) for x in (20,50,140,180,225)]
    h = [Rule('h', np.array([float(y)]), float(y), 1., (20,225)) for y in (20,60,110,170)]
    monkeypatch.setattr(service, 'detect_for_lab', lambda _: ({'vertical':v,'horizontal':h,'shape':img.shape[:2]}, {'autogrid':{}}))
    result, _, info = service.build_grid(img)
    assert info['rectification'] == 'original_cells'
    x, y = round(result.xs[0]), round(result.ys[0])
    assert abs(result.map_x[y,x] - 20) < 1
    assert abs(result.map_y[y,x] - 20) < 1


def test_recover_only_horizontal_with_observed_ink(monkeypatch):
    img = np.full((200, 250, 3), 255, dtype=np.uint8)
    vertical = [Rule('v', np.array([float(x)]), float(x), 1., (20, 180))
                for x in (20, 50, 140, 180, 225)]
    def line(y):
        return Rule('h', np.array([float(y)]), float(y), 1., (20, 225))
    physical = {'vertical': vertical, 'horizontal': [line(y) for y in (20, 60, 170)]}
    automatic = {'horizontal': [line(y) for y in (20, 60, 110, 140, 170)]}
    monkeypatch.setattr(service, 'ridge_response', lambda gray: (gray, gray))
    monkeypatch.setattr(service, 'evidence',
                        lambda rule, *_: {'accepted': rule.position == 110})
    result, recovered = service._recover_observed_horizontal(img, physical, automatic)
    assert recovered == 1
    assert [h.position for h in result['horizontal']] == [20, 60, 110, 170]
    assert len(physical['horizontal']) == 3


def test_nearby_ridge_does_not_invent_line_without_ink():
    from App.geometry.geometry_lab.physical_grid_graph_v3 import refine_nearby_ridge
    shape = (200, 250)
    vertical = [Rule('v', np.array([float(x)]), float(x), 1., (20,180))
                for x in (20,50,140,180,225)]
    candidate = Rule('h', np.array([100.]), 100., 1., (20,225))
    observation = {'ink_coverage':.85, 'column_coverage':[1.,1.,1.,.4],
                   'intersection_evidence':[True]*5, 'accepted':False}
    empty = np.zeros(shape, dtype=np.float32)
    result, checked = refine_nearby_ridge(candidate,vertical,shape,empty,empty,observation)
    assert result is candidate
    assert checked is observation
    assert not checked['accepted']


def test_short_grid_fast_path_rejects_weak_edges_and_competing_lines():
    from App.geometry.geometry_lab.physical_grid_graph_v3 import _complete_short_grid
    horizontal = [Rule('h', np.array([float(y)]), float(y), 1., (20,225))
                  for y in (20,60,110,170)]
    edge = {'ink_coverage':.95, 'column_coverage':[.95]*4,
            'intersection_evidence':[True]*5, 'longest_unsupported_fraction':.02}
    data = {'horizontal':horizontal, 'vertical':[object()]*5}
    report = {'vertical_fallback':False, 'vertical_table_local_refit':True,
              'edges':[edge.copy() for _ in range(4)], 'rejected':[]}
    assert _complete_short_grid(data,report)
    report['edges'][1] = {**edge,'column_coverage':[.95,.95,.2,.95]}
    assert not _complete_short_grid(data,report)
    report['edges'][1] = edge.copy()
    report['rejected'] = [{'position':85,'ink_coverage':.7,'intersection_evidence':[True]*4}]
    assert not _complete_short_grid(data,report)

def test_left_folded_receipt_corner_follows_printed_line():
    import cv2
    from pathlib import Path
    image = cv2.imread(str(Path(__file__).parent / 'fixtures/receipt_253309.jpg'))
    result, _, info = service.build_grid(image)
    point = service.cell_polygon(result, 0, 0)[0] + result.info['source_offset']
    assert 154 <= point[0] <= 166
    assert 258 <= point[1] <= 268
    assert info['left_recovered_horizontal'] >= 3


def test_left_recovery_ignores_empty_image():
    from App.geometry.fold import recover_left_edges
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    v = [Rule('v', np.array([float(x)]), float(x), 1., (20, 280)) for x in (20, 60, 200, 270, 370)]
    h = [Rule('h', np.array([float(y)]), float(y), 1., (20, 370)) for y in (30, 80, 130, 180, 230)]
    data = {'vertical': v, 'horizontal': h}
    result, count = recover_left_edges(img, data)
    assert result is data
    assert count == 0
