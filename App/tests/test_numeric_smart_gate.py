from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from App.ocr import engine
from App.ocr.numeric_smart_gate import may_skip_detector


def make_cell(two=False, border=False):
    image=np.full((80,100),255,np.uint8)
    cv2.putText(image,'2',(30,25 if two else 44),cv2.FONT_HERSHEY_SIMPLEX,.75,0,2,cv2.LINE_AA)
    if two: cv2.putText(image,'5',(30,65),cv2.FONT_HERSHEY_SIMPLEX,.65,0,2,cv2.LINE_AA)
    if border: cv2.rectangle(image,(3,5),(96,10),0,-1)
    return image


def test_rule_accepts_single_line_with_border_but_not_two_lines():
    image=make_cell(border=True);original=image.copy()
    assert may_skip_detector(image,'2',.998)
    assert np.array_equal(image,original)
    assert not may_skip_detector(make_cell(two=True,border=True),'2',.999)
    assert not may_skip_detector(image,'2',.97)
    assert not may_skip_detector(image,'2т',.999)


@pytest.mark.parametrize('enabled,confidence,expected_calls', [('1',.999,0),('0',.999,1),('1',.97,1)])
def test_actual_engine_skips_detector_only_when_evidence_is_strong(monkeypatch,enabled,confidence,expected_calls):
    monkeypatch.setenv('OCR_NUMERIC_SMART_GATE',enabled)
    monkeypatch.setenv('OCR_NUMERIC_MULTILINE_FAST','0')
    monkeypatch.setattr(engine,'_has_separate_numeric_lines',lambda image:True)
    monkeypatch.setattr(engine,'_run_cyrillic_rec',lambda image:SimpleNamespace(txts=['2'],scores=[confidence]))
    calls=[]
    def detect(image):
        calls.append(image)
        return [{'text':'2','confidence':.9,'bbox':[[0,0],[5,0],[5,8],[0,8]]}]
    monkeypatch.setattr(engine,'read_full',detect)
    result=engine._read_cell(make_cell(border=True),2,'quantity',(0,0))
    assert result['text']=='2'
    assert len(calls)==expected_calls
