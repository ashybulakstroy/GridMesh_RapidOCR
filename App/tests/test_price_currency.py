"""Currency correction requires visual evidence; never blindly strip trailing 7."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from App.ocr.numeric_roi import split_price_currency
from App.ocr import p1_batch
from App.pipeline import number

@pytest.mark.parametrize('text',['3807','1470','1700','117','111','101','7'])
def test_real_digits_are_not_currency(text):
    image=np.full((40,180,3),255,np.uint8)
    cv2.putText(image,text,(5,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
    assert split_price_currency(image)[1] is None

@pytest.mark.parametrize(('row','raw','expected'),[(14,'3807',380),(28,'17',1),(21,'1480T',1480)])
def test_user_price_currency_without_additional_ocr(monkeypatch,row,raw,expected):
    monkeypatch.setenv('OCR_PRICE_CURRENCY_CROP','0')  # legacy post-OCR recovery
    path=Path(__file__).parent/'fixtures'/f'price_currency_{expected}.png'
    image=cv2.imread(str(path))
    class Recognizer:
        def recognize_txt(self,images):
            assert len(images)==1
            return type('Response',(),{'txts':[raw],'scores':[.95]})()
    monkeypatch.setattr(p1_batch,'engine',lambda:Recognizer())
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    cell=p1_batch.BatchCell(image,row,'price',(0,0),[])
    result,stats=p1_batch.read_batches([cell])
    value=result[(row,'price')]
    assert number(value['text'])==expected
    assert stats['batch_calls']==1
    assert stats['numeric_roi_ms']<50  # generous guard against accidental inference
    if raw.endswith('7'):
        assert value['ocr_text']==raw
        assert value['currency_symbol']['ocr_confusion']=='7'
    else:
        assert value['text']==raw
