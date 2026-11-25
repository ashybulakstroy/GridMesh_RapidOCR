from types import SimpleNamespace

import numpy as np

from App.ocr.experiment_rec_trace import RecTracker
from App.ocr.text_debug import reasons


def test_text_capture_is_bounded_and_can_be_disabled(monkeypatch):
    monkeypatch.setenv('OCR_TEXT_DEBUG_ENABLE', '1')
    tracker = RecTracker()
    image = np.zeros((4096, 4096), dtype=np.uint8)
    result = SimpleNamespace(txts=['текст'], scores=[.9])
    for _ in range(3):
        tracker.record(['1:name'], 'primary', 'v6', 'REC_only', 1, image=image, result=result)
    assert tracker.text_bytes == 32 * 1024 * 1024
    assert len(tracker.text_inputs) == 2
    assert tracker.text_dropped == 1
    monkeypatch.setenv('OCR_TEXT_DEBUG_ENABLE', '0')
    disabled = RecTracker()
    disabled.record(['1:name'], 'primary', 'v6', 'REC_only', 1, image=image, result=result)
    assert not disabled.text_inputs


def test_text_flags_are_heuristic_and_clean_name_can_pass():
    image = np.full((30, 200), 255, np.uint8)
    flags, _ = reasons({'text':'Товар', 'confidence':.99, 'blocks':[{}]}, image, [])
    assert flags == []
