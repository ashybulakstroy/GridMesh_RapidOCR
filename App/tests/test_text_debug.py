from types import SimpleNamespace

import numpy as np

from App.ocr.experiment_rec_trace import RecTracker
from App.ocr.text_debug import reasons, render_text_report


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


def test_report_explains_why_and_escapes_ocr_text(tmp_path):
    payload={'suspects':1, 'threshold':.8, 'records':[
        {'row_id':2, 'flags':['mixed_script_word','low_confidence'],
         'final':{'text':'Товар <ошибка>'}, 'attempts':[]},
        {'row_id':3, 'flags':[], 'final':{'text':'Товар'}, 'attempts':[]}]}
    render_text_report(tmp_path, payload)
    page=(tmp_path/'report.html').read_text(encoding='utf8')
    assert 'признаки возможной ошибки найдены в 1 из 2' in page
    assert 'смешаны кириллические и латинские буквы' in page
    assert 'Низкая уверенность' in page
    assert 'Признаков ошибки не обнаружено' in page
    assert '&lt;ошибка&gt;' in page


def test_filtered_report_uses_total_checked_count(tmp_path):
    render_text_report(tmp_path, {'suspects':1,'checked':7,'records':[
        {'row_id':2,'flags':['vat_name_uncertain'],'final':{'text':'?'},'attempts':[]}]})
    page=(tmp_path/'report.html').read_text(encoding='utf8')
    assert 'в 1 из 7 наименований' in page
    assert 'отличить наименование товара от строки НДС' in page
