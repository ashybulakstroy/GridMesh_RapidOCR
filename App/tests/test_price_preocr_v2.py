from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

from App.ocr import p1_batch


@pytest.mark.parametrize('mode', ['cell','batch'])
@pytest.mark.parametrize('roi_enabled', ['0','1'])
def test_real_2080_currency_is_removed_before_primary_ocr(monkeypatch, mode, roi_enabled):
    image=cv2.imread(str(Path(__file__).parent/'fixtures/price_currency_2080.png'))
    received=[]
    monkeypatch.setenv('OCR_NUMERIC_ROI',roi_enabled)
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','1')
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    class Engine:
        def recognize_txt(self, images):
            received.extend(im.copy() for im in images)
            return SimpleNamespace(txts=['2080']*len(images),scores=[.99]*len(images))
    monkeypatch.setattr(p1_batch,'engine',lambda:Engine())
    def read_cell(img,rid,field,offset):
        received.append(img.copy())
        return {'text':'2080','confidence':.99,'bbox':[0,0,img.shape[1],img.shape[0]],'blocks':[]}
    monkeypatch.setattr(p1_batch,'read_cell',read_cell)
    result,_=p1_batch.read_batches([p1_batch.BatchCell(image,2,'price',(0,0),[])],mode=mode)
    assert len(received)==1
    assert received[0].shape[1] <= 85
    assert result[(2,'price')]['numeric_roi']['currency_removed_before_ocr'] is True
    assert result[(2,'price')]['text']=='2080'


def test_disabling_currency_and_roi_preserves_primary_image(monkeypatch):
    image=cv2.imread(str(Path(__file__).parent/'fixtures/price_currency_2080.png'))
    seen=[]
    monkeypatch.setenv('OCR_NUMERIC_ROI','0')
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','0')
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    class Engine:
        def recognize_txt(self,images):
            seen.extend(images)
            return SimpleNamespace(txts=['2080'],scores=[.99])
    monkeypatch.setattr(p1_batch,'engine',lambda:Engine())
    p1_batch.read_batches([p1_batch.BatchCell(image,2,'price',(0,0),[])])
    assert len(seen)==1 and np.array_equal(seen[0],image)


def test_currency_crop_still_runs_after_optional_budget_exhausted(monkeypatch):
    image=cv2.imread(str(Path(__file__).parent/'fixtures/price_currency_2080.png'))
    monkeypatch.setenv('OCR_NUMERIC_ROI','1')
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','1')
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    def slow_crop(img):
        time.sleep(.014)
        return img,{'changed':False}
    monkeypatch.setattr(p1_batch,'crop_numeric_ink',slow_crop)
    class Engine:
        def recognize_txt(self,images):
            return SimpleNamespace(txts=['2080']*len(images),scores=[.99]*len(images))
    monkeypatch.setattr(p1_batch,'engine',lambda:Engine())
    cells=[p1_batch.BatchCell(image,1,'price',(0,0),[]),p1_batch.BatchCell(image,2,'price',(0,0),[])]
    result,_=p1_batch.read_batches(cells)
    assert all(result[(rid,'price')]['numeric_roi']['currency_removed_before_ocr'] for rid in (1,2))
