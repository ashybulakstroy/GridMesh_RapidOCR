import numpy as np
from App.geometry import service
from App.geometry.geometry_lab.detection import Rule


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
