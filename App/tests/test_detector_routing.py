import numpy as np

from App.geometry import detector_routing_v1 as routing


def test_only_long_incomplete_portrait_grid_uses_multiscale_first():
    state,token=routing.start('auto',10)
    try:
        image=np.zeros((200,100,3),np.uint8)
        assert routing.choose({'horizontal':[None]*10,'vertical':[None]*3},{},image)
        assert not routing.choose({'horizontal':[None]*5,'vertical':[None]*3},{},image)
        assert not routing.choose({'horizontal':[None]*10,'vertical':[None]*5},{},image)
        assert not routing.choose({'horizontal':[None]*10,'vertical':[None]*3},{},image.transpose(1,0,2))
        routing.attempt()
        assert not routing.choose({'horizontal':[None]*10,'vertical':[None]*3},{},image)
    finally:
        routing.end(token)


def test_retry_is_only_allowed_before_product_ocr():
    state,token=routing.start('auto')
    try:
        assert not routing.can_retry()
        routing.mark_candidate()
        assert routing.can_retry()
        routing.finish_geometry()
        assert not routing.can_retry()
    finally:
        routing.end(token)
