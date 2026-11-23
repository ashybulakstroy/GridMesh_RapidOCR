import cv2
import numpy as np
from App.ocr.footer_line_guided_v1 import isolate_footer


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
