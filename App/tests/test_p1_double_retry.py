"""The retry must use OCR evidence and preserve ambiguous alternatives."""
from decimal import Decimal
import numpy as np

from App.ocr import p1_double_retry
from App.ocr.p1_batch import BatchCell


def row(number, kind, quantity, price):
    return {'row_id':number, 'kind':kind, 'cells':{
        field:{'text':value, 'confidence':.6 if field == 'price' and kind == 'product' else .99,
               'row_id':number, 'column_id':field, 'bbox':[0,0,20,20]}
        for field,value in [('quantity',quantity),('price',price)]}}


def test_unique_ocr_alternative_is_confirmed_by_total_and_vat(monkeypatch):
    rows=[row(1,'product','1','120'),row(2,'tax','192','1')]
    image=np.full((20,20,3),255,np.uint8)
    cells=[BatchCell(image,1,'price',(0,0),[])]
    def fake_read(cells, *, batch_size, budget_multiplier):
        assert budget_multiplier == 2
        return {(1,'price'):{'text':'1200','confidence':.99,'strategy':'p1_rec_batch'}}, {'numeric_roi_budget_ms':16}
    monkeypatch.setattr(p1_double_retry,'read_batches',fake_read)
    stats={'numeric_roi_skipped_cells':[{'row_id':1,'column_id':'price'}]}
    result=p1_double_retry.retry_suspect_cells(rows,cells,stats,Decimal('1392'),Decimal('16'))
    assert rows[0]['cells']['price']['text']=='1200'
    assert result['details'][0]['original_text']=='120'
    assert result['details'][0]['confirmation']=='unique_total_and_vat'
    assert result['details'][0]['accepted']


def test_multiple_total_matching_candidates_are_not_silently_applied(monkeypatch):
    rows=[row(1,'product','1','100'),row(2,'product','1','100'),row(3,'tax','48','1')]
    image=np.full((20,20,3),255,np.uint8)
    cells=[BatchCell(image,i,'price',(0,0),[]) for i in (1,2)]
    def fake_read(cells, *, batch_size, budget_multiplier):
        return {(i,'price'):{'text':'200','confidence':.99} for i in (1,2)}, {'numeric_roi_budget_ms':16}
    monkeypatch.setattr(p1_double_retry,'read_batches',fake_read)
    result=p1_double_retry.retry_suspect_cells(rows,cells,{},Decimal('348'),Decimal('16'))
    assert [item['cells']['price']['text'] for item in rows[:2]]==['100','100']
    assert len(result['candidates'])==2
    assert not any(item['accepted'] for item in result['details'])


def test_incomplete_row_retries_its_other_numeric_cells(monkeypatch):
    item = row(1, 'product', '', '12959')
    item['cells']['price']['confidence'] = .99
    item['cells']['printed_total'] = {'text':'12950', 'confidence':.99}
    image = np.full((20,20,3),255,np.uint8)
    cells = [BatchCell(image,1,field,(0,0),[]) for field in ('quantity','price','printed_total')]
    def fake_read(cells, *, batch_size, budget_multiplier):
        assert {cell.column_id for cell in cells} == {'quantity','price','printed_total'}
        return {(cell.row_id,cell.column_id):{'text':item['cells'][cell.column_id]['text'],
                                             'confidence':.99} for cell in cells}, {}
    monkeypatch.setattr(p1_double_retry,'read_batches',fake_read)
    result = p1_double_retry.retry_suspect_cells([item],cells,{},Decimal('12950'),Decimal('16'))
    assert result['attempted'] == 3
