import cv2
import numpy as np
from App.ocr.price_candidate import prepare_price_candidate


def image(text):
    result=np.full((50,220,3),255,np.uint8)
    cv2.putText(result,text,(8,36),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2,cv2.LINE_AA)
    return result


def test_last_seven_remains_without_currency_evidence():
    assert prepare_price_candidate(image('1907'),'1907') is None


def test_decimal_prices_are_not_stripped():
    assert prepare_price_candidate(image('19.07     T'),'19.07T') is None


def test_only_isolated_t_is_accepted(monkeypatch):
    import App.ocr.numeric_roi as roi
    monkeypatch.setattr(roi,'crop_numeric_ink',lambda im:(im,{}))
    picture=image('190     T')
    cropped=prepare_price_candidate(picture,'190 T')
    assert cropped is not None and cropped.shape[1]<picture.shape[1]
    assert np.array_equal(cropped,picture[:,:cropped.shape[1]])
