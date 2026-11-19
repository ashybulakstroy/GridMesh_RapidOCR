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
    ticks=iter([0.0,.006])
    monkeypatch.setattr(p1_batch,'perf_counter',lambda:next(ticks))
    cells=[p1_batch.BatchCell(original,1,'price',(10,20),[[3,4]]),
           p1_batch.BatchCell(original,2,'quantity',(30,40),[[5,6]])]
    result,stats=p1_batch.read_batches(cells,batch_size=8)
    assert stats['numeric_roi_changed']==1
    assert stats['numeric_roi_skipped']==1
    assert result[(1,'price')]['bbox']==[10,20,110,60]
    assert result[(1,'price')]['source_bbox']==[[3,4]]
    assert result[(2,'quantity')]['bbox']==[30,40,130,80]
