from types import SimpleNamespace
import numpy as np
import pytest
import App.ocr.engine as engine


def test_models_are_selected_by_field_and_context_restored(monkeypatch):
    seen=[]
    def read(image,row_id,column_id,offset,**kwargs):
        seen.append(engine._NAME_MODEL.get())
        if row_id==2:raise RuntimeError('test')
        return {'text':'x'}
    monkeypatch.setattr(engine,'_read_cell',read)
    image=np.zeros((10,10,3),dtype='uint8')
    for field in ('name','quantity','price','printed_total'):
        engine.read_cell(image,1,field,(0,0))
    with pytest.raises(RuntimeError):engine.read_cell(image,2,'name',(0,0))
    assert seen==['v6_tiny_cyrillic_optional','v6_tiny_official','v6_tiny_official','v5_cyrillic_mobile','v6_tiny_cyrillic_optional']
    assert engine._NAME_MODEL.get()=='v5_cyrillic_mobile'


def test_v5_fallback_when_tiny_fails(monkeypatch):
    class Broken:
        def __call__(self,*args,**kwargs):raise RuntimeError('missing model')
    class Good:
        def __call__(self,*args,**kwargs):return SimpleNamespace(txts=['123'],scores=[.99])
    monkeypatch.setattr(engine,'_models',{('table','v6_tiny_official'):Broken()})
    monkeypatch.setattr(engine,'engine',lambda:Good())
    token=engine._NAME_MODEL.set('v6_tiny_official')
    try:assert engine._run_cyrillic_rec(np.zeros((5,5,3))).txts==['123']
    finally:engine._NAME_MODEL.reset(token)
