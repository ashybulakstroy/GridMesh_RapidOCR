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
