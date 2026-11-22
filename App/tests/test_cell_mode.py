from threading import Event
from types import SimpleNamespace
from contextvars import ContextVar
import numpy as np
import pytest
from App.ocr import cell_mode, p1_batch
from App.ocr.engine import inference_lane, _LANE, _locks
from App.ocr.experiment_rec_trace import RecTracker, activate, deactivate, record_rec


def test_two_lanes_overlap_and_preserve_trace():
    table_entered=Event();tracker=RecTracker();token=activate(tracker)
    def footer():
        assert _LANE.get()=='footer'
        assert table_entered.wait(2)
        record_rec(keys=['receipt_total'],ms=1)
        return {'total':'100'}
    def table(cells, **kwargs):
        assert _LANE.get()=='table'
        table_entered.set()
        record_rec(keys=['1:price'],ms=1)
        return {}, {}
    try:result,stats,total=cell_mode.read_parallel([],footer,table)
    finally:deactivate(token)
    assert total['total']=='100'
    assert stats['parallel_lanes']==2
    assert set(tracker.export()['per_cell'])=={'receipt_total','1:price'}
    assert _locks['table'] is not _locks['footer']


def test_cell_order_is_columns_then_rows(monkeypatch):
    seen=[]
    def read(cells,**kw):
        assert kw['mode']=='cell' and kw['unlimited_roi']
        seen.extend((c.row_id,c.column_id) for c in cells)
        return {},{}
    monkeypatch.setattr(p1_batch,'read_batches',read)
    cells=[SimpleNamespace(row_id=r,column_id=f) for r in (2,1) for f in ('printed_total','quantity','name','price')]
    p1_batch.read_cells(cells)
    order=seen
    assert order==[(r,f) for f in ('name','quantity','price','printed_total') for r in (1,2)]


def test_failed_cell_uses_batch_fallback(monkeypatch):
    monkeypatch.setattr(p1_batch,'setting',lambda name,default:'0' if name=='OCR_NUMERIC_ROI' else default)
    monkeypatch.setattr(p1_batch,'read_cell',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('cell failed')))
    monkeypatch.setattr(p1_batch,'_has_separate_numeric_lines',lambda image:False)
    monkeypatch.setattr(p1_batch,'engine',lambda:SimpleNamespace(recognize_txt=lambda images:SimpleNamespace(txts=['100'],scores=[.99])))
    cell=p1_batch.BatchCell(np.full((25,50,3),255,np.uint8),1,'price',(0,0),[])
    result,stats=p1_batch.read_cells([cell])
    assert result[(1,'price')]['text']=='100'
    assert stats['cell_fallback']==1


def test_footer_failure_does_not_discard_table():
    result,stats,total=cell_mode.read_parallel([],lambda:(_ for _ in ()).throw(ValueError('footer failed')),lambda *a,**k:({'table':'kept'},{}))
    assert result=={'table':'kept'} and total is None
    assert 'footer failed' in stats['footer_error']


def test_footer_models_are_cached_separately(monkeypatch):
    import App.ocr.engine as engine_module
    monkeypatch.setattr(engine_module,'_models',{})
    monkeypatch.setattr(engine_module,'RapidOCR',lambda **kw:object())
    table=engine_module.engine()
    with inference_lane('footer'):
        footer=engine_module.engine()
        assert footer is engine_module.engine()
    assert table is not footer
    assert table is engine_module.engine()


def test_real_cell_mode_preserves_lower_numeric_rows(monkeypatch):
    from pathlib import Path
    from App.pipeline import process
    from App.ocr.p1_numbers import parse_number
    monkeypatch.setenv('OCR_TABLE_MODE','cell')
    monkeypatch.setenv('OCR_E4_ENABLE','1')
    # Full receipt is the same known regression image supplied for the comparison.
    full=Path(__file__).resolve().parents[2]/'data/output/7a0d379e961547828dd608a2cbb74f4a/input.bin'
    if not full.exists():pytest.skip('Local performance receipt is unavailable')
    result=process(full.read_bytes())
    assert result['math_match'] and parse_number(result['total'])==252161
    assert result['geometry']['ocr_mode']=='cell'
    stats=result['geometry']['ocr_batch']
    assert stats['parallel_lanes']==2 and stats['batch_calls']==0
    assert stats['numeric_roi_skipped']==0
    for row_id,quantity,price in [(15,5,2000),(16,20,60)]:
        row=next(row for row in result['rows'] if row['row_id']==row_id)
        assert parse_number(row['cells']['quantity']['text'])==quantity
        assert parse_number(row['cells']['price']['text'])==price
    keys=result['rec_trace']['per_cell']
    assert 'receipt_total' in keys and '15:price' in keys


def test_all_roi_prepared_despite_budget_overrun(monkeypatch):
    monkeypatch.setattr(p1_batch,'setting',lambda name,default:'1' if name=='OCR_NUMERIC_ROI' else default)
    prepared=[]
    def crop(image):
        prepared.append(image)
        return image,{'changed':False}
    monkeypatch.setattr(p1_batch,'crop_numeric_ink',crop)
    clock=iter(range(100))
    monkeypatch.setattr(p1_batch,'perf_counter',lambda:next(clock))
    def read(image,row_id,column_id,offset):
        assert len(prepared)==3
        return {'text':'100','confidence':.9,'blocks':[]}
    monkeypatch.setattr(p1_batch,'read_cell',read)
    cells=[p1_batch.BatchCell(np.full((20,40,3),255,np.uint8),r,'price',(0,0),[]) for r in (1,2,3)]
    result,stats=p1_batch.read_cells(cells)
    assert len(result)==3 and stats['numeric_roi_skipped']==0
    assert stats['numeric_roi_unlimited']
    assert not stats['numeric_roi_crop_exceeded']
