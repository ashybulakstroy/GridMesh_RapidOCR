import numpy as np
import pytest
from App.ocr import p1_batch


class FakeEngine:
    def __init__(self):
        self.calls=[]
    def recognize_txt(self,imgs):
        self.calls.append(len(imgs))
        return type('Output',(),{'txts':[str(k+1) for k in range(len(imgs))],
                                 'scores':[.95]*len(imgs)})()


def test_batch_recognizes_multiple_images_in_one_inference(monkeypatch):
    fake=FakeEngine()
    monkeypatch.setattr(p1_batch,'engine',lambda:fake)
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    cells=[p1_batch.BatchCell(np.full((32,64,3),255,np.uint8),r,'quantity',(0,0),[])
           for r in range(1,10)]
    result,stats=p1_batch.read_batches(cells,batch_size=8)
    assert fake.calls==[8,1]
    assert stats['batch_fallback']==0
    assert result[(1,'quantity')]['text']=='1'
    assert result[(9,'quantity')]['text']=='1'
    assert result[(9,'quantity')]['strategy']=='p1_rec_batch'


def test_batch_fallback_is_visible(monkeypatch):
    class Broken:
        def recognize_txt(self,imgs):
            raise RuntimeError('Model is single-batch only')
    monkeypatch.setattr(p1_batch,'engine',lambda:Broken())
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    monkeypatch.setattr(p1_batch,'read_cell',lambda img,r,c,offset:{
        'text':'2','confidence':.9,'bbox':[0,0,60,30],'row_id':r,'column_id':c})
    cells=[p1_batch.BatchCell(np.full((30,60,3),255,np.uint8),1,'price',(0,0),[]),
           p1_batch.BatchCell(np.full((30,60,3),255,np.uint8),2,'price',(0,0),[])]
    result,stats=p1_batch.read_batches(cells,batch_size=8)
    assert stats['batch_fallback']==2
    assert stats['errors']
    assert result[(1,'price')]['strategy']=='legacy_batch_fallback'


def test_numeric_crop_preserves_cell_coordinates_and_obeys_budget(monkeypatch):
    fake=FakeEngine()
    monkeypatch.setattr(p1_batch,'engine',lambda:fake)
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    monkeypatch.setattr(p1_batch,'setting',lambda *_:'1')
    original=np.full((40,100,3),255,np.uint8)
    monkeypatch.setattr(p1_batch,'crop_numeric_ink',lambda image:(image[5:35,10:70], {'changed':True}))
    ticks=iter([0.0,.012])
    monkeypatch.setattr(p1_batch,'perf_counter',lambda:next(ticks))
    cells=[p1_batch.BatchCell(original,1,'price',(10,20),[[3,4]]),
           p1_batch.BatchCell(original,2,'quantity',(30,40),[[5,6]])]
    result,stats=p1_batch.read_batches(cells,batch_size=8)
    assert stats['numeric_roi_changed']==1
    assert stats['numeric_roi_skipped']==1
    assert stats['numeric_roi_crop_exceeded'] is False
    assert stats['numeric_roi_skipped_cells']==[{'row_id':2,'column_id':'quantity'}]
    assert result[(1,'price')]['bbox']==[10,20,110,60]
    assert result[(1,'price')]['source_bbox']==[[3,4]]
    assert result[(2,'quantity')]['bbox']==[30,40,130,80]


def test_crop_work_reserves_budget_for_currency_validation(monkeypatch):
    class Recognizer:
        def recognize_txt(self,images):
            return type('Response',(),{'txts':['3807','1'],'scores':[.9,.9]})()
    monkeypatch.setattr(p1_batch,'engine',lambda:Recognizer())
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    monkeypatch.setattr(p1_batch,'setting',lambda *_:'1')
    monkeypatch.setattr(p1_batch,'crop_numeric_ink',lambda image:(image,{'changed':False}))
    monkeypatch.setattr(p1_batch,'split_price_currency',lambda image:(image,{
        'symbol':'₸','numeric_glyph_count':3}))
    ticks=iter([0.,.0121,.02,.0203])
    monkeypatch.setattr(p1_batch,'perf_counter',lambda:next(ticks))
    image=np.full((40,80,3),255,np.uint8)
    cells=[p1_batch.BatchCell(image,1,'price',(0,0),[]),
           p1_batch.BatchCell(image,2,'quantity',(0,0),[])]
    result,stats=p1_batch.read_batches(cells)
    assert stats['numeric_roi_skipped']==1
    assert stats['numeric_roi_ms']==12.4
    assert stats['numeric_roi_crop_budget_ms']==12
    assert stats['numeric_roi_budget_ms']==16
    assert result[(1,'price')]['text']=='380 ₸'


def test_actual_overruns_and_doubled_budget_are_reported(monkeypatch):
    monkeypatch.setattr(p1_batch,'engine',lambda:FakeEngine())
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    monkeypatch.setattr(p1_batch,'setting',lambda *_:'1')
    monkeypatch.setattr(p1_batch,'crop_numeric_ink',lambda image:(image,{'changed':False}))
    ticks=iter([0., .025])
    monkeypatch.setattr(p1_batch,'perf_counter',lambda:next(ticks))
    image=np.full((40,80,3),255,np.uint8)
    cells=[p1_batch.BatchCell(image,1,'price',(0,0),[]),
           p1_batch.BatchCell(image,2,'quantity',(0,0),[])]
    _,stats=p1_batch.read_batches(cells,budget_multiplier=2)
    assert stats['numeric_roi_crop_budget_ms']==24
    assert stats['numeric_roi_budget_ms']==32
    assert stats['numeric_roi_crop_ms']==25
    assert stats['numeric_roi_crop_exceeded'] is True
    assert stats['numeric_roi_skipped_cells']==[{'row_id':2,'column_id':'quantity'}]


def test_doubled_total_overrun_is_reported_after_currency_check(monkeypatch):
    class Recognizer:
        def recognize_txt(self, images):
            return type('Response',(),{'txts':['17'],'scores':[.9]})()
    monkeypatch.setattr(p1_batch,'engine',lambda:Recognizer())
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda _:False)
    monkeypatch.setattr(p1_batch,'setting',lambda *_:'1')
    monkeypatch.setattr(p1_batch,'crop_numeric_ink',lambda image:(image,{'changed':False}))
    monkeypatch.setattr(p1_batch,'split_price_currency',lambda image:(image,None))
    ticks=iter([0.,.0239,.04,.05])
    monkeypatch.setattr(p1_batch,'perf_counter',lambda:next(ticks))
    image=np.full((40,80,3),255,np.uint8)
    _,stats=p1_batch.read_batches([p1_batch.BatchCell(image,1,'price',(0,0),[])],budget_multiplier=2)
    assert stats['numeric_roi_crop_exceeded'] is False
    assert stats['numeric_roi_budget_exceeded'] is True
    assert stats['numeric_roi_ms']==33.9
