import numpy as np

from App.geometry import service, short_receipt_grid_v1


def test_short_rescue_requires_enabled_source_and_header_validator(monkeypatch):
    monkeypatch.setattr(service, '_build_grid_source_rescue_before_short_v1', lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(short_receipt_grid_v1, 'rescue', lambda *a, **k: calls.append(True))
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setenv('OCR_SHORT_RECEIPT_GRID_V1', '1')
    monkeypatch.setenv('OCR_SOURCE_GRID_V16', '1')
    assert service.build_grid_source_rescue(image, header_validator=None) is None
    monkeypatch.setenv('OCR_SOURCE_GRID_V16', '0')
    assert service.build_grid_source_rescue(image, header_validator=lambda _: True) is None
    assert not calls


def test_short_rescue_error_preserves_failure_and_is_recorded(monkeypatch):
    monkeypatch.setattr(service, '_build_grid_source_rescue_before_short_v1', lambda *a, **k: None)
    monkeypatch.setenv('OCR_SHORT_RECEIPT_GRID_V1', '1')
    monkeypatch.setenv('OCR_SOURCE_GRID_V16', '1')
    def fail(*args, **kwargs):
        raise RuntimeError('test failure')
    monkeypatch.setattr(short_receipt_grid_v1, 'rescue', fail)
    events = []
    monkeypatch.setattr(service._grid_diag, 'caught', lambda local, label: events.append(label))
    assert service.build_grid_source_rescue(np.zeros((20, 20, 3), dtype=np.uint8),
                                             header_validator=lambda _: True) is None
    assert events == ['App/geometry/service.py:short_receipt_grid_v1']
