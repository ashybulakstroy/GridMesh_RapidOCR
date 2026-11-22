from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from App.ocr.numeric_roi import prepare_numeric_input, split_price_currency
from App.ocr import p1_batch


@pytest.mark.parametrize('mode',['cell','batch'])
def test_actual_ocr_receives_price_pixels_without_currency(monkeypatch,mode):
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','1')
    image=cv2.imread(str(Path(__file__).parent/'fixtures/price_4810_currency.png'))
    # Visually verified T starts at x=111 in the original GridMesh cell.
    expected=image[0:57,5:111]
    calls=[]
    def check(received):
        calls.append(received)
        assert np.array_equal(received,expected)
    def read(image,rid,field,offset):
        check(image)
        return {'text':'4810','confidence':.99,'blocks':[]}
    def rec(images):
        assert len(images)==1
        check(images[0])
        return SimpleNamespace(txts=['4810'],scores=[.99])
    monkeypatch.setattr(p1_batch,'read_cell',read)
    monkeypatch.setattr(p1_batch,'engine',lambda:SimpleNamespace(recognize_txt=rec))
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    source=[[1,2],[3,4],[5,6],[7,8]]
    cell=p1_batch.BatchCell(image,1,'price',(10,20),source)
    result,stats=p1_batch.read_batches([cell],mode=mode,unlimited_roi=True)
    value=result[1,'price']
    assert len(calls)==1 and value['text']=='4810'
    assert value['bbox']==[10,20,10+image.shape[1],20+image.shape[0]]
    assert value['source_bbox']==source
    assert value['numeric_roi']['ocr_roi_bbox']==[5,0,111,57]
    assert value['numeric_roi']['currency']['removed_from_ocr_input']
    assert stats['numeric_currency_removed']==1


@pytest.mark.parametrize('number',['1','7','17','117','1907','1001','11.07'])
def test_final_real_digits_are_not_classified_as_currency(monkeypatch,number):
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','1')
    image=np.full((50,220,3),255,np.uint8)
    cv2.putText(image,number,(8,36),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2,cv2.LINE_AA)
    assert split_price_currency(image,clean_rule_fragments=True)[1] is None
    _,info=prepare_numeric_input(image,'price')
    assert not info['currency_removed_before_ocr']


def test_decimal_and_last_seven_pixels_survive_currency_crop(monkeypatch):
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','1')
    image=np.full((50,250,3),255,np.uint8)
    cv2.putText(image,'19.07',(8,36),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2,cv2.LINE_AA)
    digits=np.any(image<128,axis=2)
    cv2.putText(image,'T',(180,36),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2,cv2.LINE_AA)
    cropped,info=prepare_numeric_input(image,'price')
    assert info['currency_removed_before_ocr']
    x0,y0,x1,y1=info['ocr_roi_bbox']
    yy,xx=np.nonzero(digits)
    assert x0<=xx.min() and x1>xx.max() and y0<=yy.min() and y1>yy.max()
    assert np.array_equal(cropped,image[y0:y1,x0:x1])


def test_shared_v5_collector_uses_currency_free_pixels(monkeypatch):
    import App.pipeline as pipeline
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','1')
    image=cv2.imread(str(Path(__file__).parent/'fixtures/price_4810_currency.png'))
    def read(prepared):
        assert np.array_equal(prepared,image[0:57,5:111])
        return {'text':'4810','confidence':.99}
    monkeypatch.setattr(pipeline,'read_line',read)
    result=pipeline.collect_v5_candidates({(1,'price'):image})
    assert not result['errors'] and len(result['observations'])==1
    assert result['observations'][0]['numeric_roi']['currency_removed_before_ocr']


def test_disabling_flag_keeps_original_preparation(monkeypatch):
    from App.ocr.numeric_roi import crop_numeric_ink
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','0')
    image=cv2.imread(str(Path(__file__).parent/'fixtures/price_4810_currency.png'))
    actual,info=prepare_numeric_input(image,'price')
    expected,_=crop_numeric_ink(image)
    assert np.array_equal(actual,expected) and 'currency_removed_before_ocr' not in info
