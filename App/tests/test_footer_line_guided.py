import cv2
import numpy as np
from App.ocr.footer_line_guided_v1 import isolate_footer


def test_slanted_footer_borders_isolate_only_the_total():
    image=np.full((340,400,3),255,np.uint8)
    cv2.line(image,(200,200),(350,205),(0,0,0),1)
    cv2.line(image,(200,235),(350,240),(0,0,0),1)
    cv2.putText(image,'10324',(225,230),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    guide={'price_left':210,'price_right':345,'grid_bottom_y':203,'last_row_height':35}
    crop,bbox,diag=isolate_footer(image,guide)
    assert crop is not None and diag['method']=='observed_slanted_footer_borders'
    assert 198 <= bbox[1] <= 206 and 235 <= bbox[3] <= 244


def test_one_slanted_border_cannot_create_footer():
    image=np.full((340,400,3),255,np.uint8)
    cv2.line(image,(200,200),(350,205),(0,0,0),1)
    cv2.putText(image,'12505',(230,226),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    guide={'price_left':210,'price_right':345,'grid_bottom_y':203,'last_row_height':35}
    crop,_,_=isolate_footer(image,guide)
    assert crop is None


def test_tax_band_ending_at_grid_bottom_is_not_a_footer():
    image=np.full((340,400,3),255,np.uint8)
    cv2.line(image,(200,200),(350,200),(0,0,0),1)
    cv2.line(image,(200,240),(350,240),(0,0,0),1)
    cv2.putText(image,'12505',(230,226),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    guide={'price_left':220,'price_right':350,'grid_bottom_y':240,'last_row_height':40}
    crop,_,_=isolate_footer(image,guide)
    assert crop is None, 'The last tax row lies ABOVE the end of the table'


def test_recovered_tax_row_invalidates_old_footer_guide(monkeypatch):
    import App.pipeline as pipeline
    monkeypatch.setenv('OCR_FOOTER_FOCUS_ENABLE','1')
    monkeypatch.setattr(pipeline,'read_line',lambda _: {'text':'100','confidence':.99})
    image=np.full((340,400,3),255,np.uint8)
    guide={'price_left':220,'price_right':350,'grid_bottom_y':200,'last_row_height':30}
    result=pipeline.read_printed_footer(image,240,footer_guide=guide)
    assert result['total']=='100'
    assert not result['footer_focus']['accepted']
    assert [x['strategy'] for x in result['attempts']]==['single_line']


def test_footer_ocr_receives_image_without_confirmed_currency(monkeypatch):
    import App.pipeline as pipeline
    monkeypatch.setenv('OCR_FOOTER_FOCUS_ENABLE','0')
    image=np.full((300,400,3),255,np.uint8)
    cv2.putText(image,'1907',(210,214),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(image,'T',(320,214),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2,cv2.LINE_AA)
    received=[]
    def read(crop):
        received.append(crop.copy())
        return {'text':'1907','confidence':.99}
    monkeypatch.setattr(pipeline,'read_line',read)
    result=pipeline.read_printed_footer(image,190)
    line=result['attempts'][0]['lines'][0]
    assert result['total']=='1907'
    assert line['currency_crop']['removed_before_ocr']
    assert received[0].shape[1] < image.shape[1]//2


def test_v6_confirms_subthreshold_footer_without_detector(monkeypatch):
    import App.pipeline as pipeline
    monkeypatch.setenv('OCR_FOOTER_FOCUS_ENABLE','0')
    monkeypatch.setattr(pipeline,'read_line',lambda _: {'text':'10324.00','confidence':.4})
    class Pool:
        def recognize(self, model_id, image, stage):
            assert model_id=='v6_tiny_cyrillic_optional'
            return {'text':'10 324,00','confidence':.3,'wall_ms':5}
    monkeypatch.setattr(pipeline,'ExperimentModelPool',Pool)
    def forbidden(*args,**kwargs):
        raise AssertionError('detector must not run on agreement')
    monkeypatch.setattr(pipeline,'read_full',forbidden)
    result=pipeline.read_printed_footer(np.full((300,400,3),255,np.uint8),190)
    assert result['total']=='10324.00'
    assert [a['strategy'] for a in result['attempts']]==['single_line','v6_tiny_cyrillic_confirmation']
    assert result['attempts'][1]['matched']


def test_v6_disagreement_preserves_existing_detector_fallback(monkeypatch):
    import App.pipeline as pipeline
    monkeypatch.setenv('OCR_FOOTER_FOCUS_ENABLE','0')
    responses=iter([{'text':'10324.00','confidence':.91},None,None])
    monkeypatch.setattr(pipeline,'read_line',lambda _: next(responses))
    class Pool:
        def recognize(self, *args):return {'text':'10325','confidence':.99,'wall_ms':5}
    monkeypatch.setattr(pipeline,'ExperimentModelPool',Pool)
    monkeypatch.setattr(pipeline,'read_full',lambda *args,**kwargs:[{
        'text':'10324.00','confidence':.99,'bbox':[[0,20],[80,20],[80,40],[0,40]]}])
    result=pipeline.read_printed_footer(np.full((300,400,3),255,np.uint8),190)
    assert result['total']=='10324.00'
    assert not result['attempts'][1]['matched']
    assert result['attempts'][-1]['strategy']=='fast_detector'
