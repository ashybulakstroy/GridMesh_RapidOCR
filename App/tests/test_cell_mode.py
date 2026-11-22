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
    assert result['numeric_matrix']['candidate_collection']['v5_calls'] >= 34
    book=result['numeric_candidate_book']['cells']
    for key,value in [('4:price','2480'),('5:price','230'),('16:price','60')]:
        assert any(o['number']==value and o['path']=='v5_numeric_candidate' for o in book[key]['observations'])
    assert result['numeric_matrix']['initial_status'] in ('unique_math_match','ambiguous')
    assert result['numeric_matrix']['fallback_used'] is False
    assert result['geometry']['ocr_mode']=='cell'
    stats=result['geometry']['ocr_batch']
    assert stats['parallel_lanes']==3 and stats['batch_calls']==0
    assert stats['columns_parallel']
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


def test_candidates_run_after_footer_while_table_is_active():
    from App.pipeline import collect_v5_candidates
    import App.pipeline as pipeline
    from unittest.mock import patch
    table_ready=Event();candidates_done=Event();order=[]
    cells={(1,'price'):np.zeros((20,40,3),dtype='uint8')}
    def footer():
        order.append('footer')
        assert table_ready.wait(2)
        result=collect_v5_candidates(cells)
        candidates_done.set()
        return {'numeric_candidates':result}
    def table(cells,**kwargs):
        table_ready.set()
        assert candidates_done.wait(2)
        return {},{}
    def read(image):
        order.append('candidate')
        assert _LANE.get()=='footer'
        return {'text':'2480T','confidence':.9}
    with patch.object(pipeline,'read_line',read), patch.object(pipeline,'crop_numeric_ink',lambda x:(x,{})):
        _,_,result=cell_mode.read_parallel([],footer,table)
    assert order==['footer','candidate']
    assert result['numeric_candidates']['observations'][0]['text']=='2480T'


def test_candidate_errors_are_preserved_and_other_cells_continue(monkeypatch):
    import App.pipeline as pipeline
    monkeypatch.setattr(pipeline,'crop_numeric_ink',lambda im:(im,{}))
    counter=[]
    def reader(image):
        counter.append(1)
        if len(counter)==1:raise RuntimeError('bad crop')
        return {'text':'60','confidence':.8}
    monkeypatch.setattr(pipeline,'read_line',reader)
    image=np.zeros((20,40,3),dtype='uint8')
    data=pipeline.collect_v5_candidates({(1,'price'):image,(2,'price'):image})
    assert len(data['attempted'])==2 and len(data['errors'])==1
    assert data['observations'][0]['text']=='60'


def test_names_numbers_and_footer_overlap_with_trace_and_separate_locks(monkeypatch):
    from App.ocr.engine import _lock
    from App.ocr.debug_performance import activate_telemetry, deactivate_telemetry, get_debug
    monkeypatch.setenv('OCR_TABLE_PARALLEL', '1')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    ready = {lane: Event() for lane in ('footer', 'table_names', 'table_numeric')}
    tracker = RecTracker()
    token = activate(tracker)
    telemetry = activate_telemetry('overlap')
    debug = get_debug()
    cells = [SimpleNamespace(row_id=1, column_id=field) for field in ('name', 'quantity', 'price')]

    def arrive():
        ready[_LANE.get()].set()
        assert all(event.wait(2) for event in ready.values())

    def footer():
        with _lock:
            arrive()
            record_rec(keys=['receipt_total'], ms=1)
        return {'total':'100'}

    def table(items, **kwargs):
        with _lock:
            arrive()
            result = {(c.row_id,c.column_id):{'text':c.column_id} for c in items}
            record_rec(keys=[f'{c.row_id}:{c.column_id}' for c in items], ms=1)
        return result, {'requested':len(items), 'cell_calls':len(items), 'errors':[],
                        'numeric_roi_ms':4 if _LANE.get()=='table_numeric' else 0,
                        'numeric_roi_skipped':0}

    try:
        result, stats, total = cell_mode.read_parallel(cells, footer, table)
        assert get_debug() is debug
    finally:
        deactivate_telemetry(telemetry)
        deactivate(token)
    assert total['total']=='100'
    assert set(result)=={(1,'name'),(1,'quantity'),(1,'price')}
    assert stats['requested']==stats['cell_calls']==3
    assert stats['numeric_roi_ms']==4 and stats['numeric_roi_skipped']==0
    assert stats['parallel_lanes']==3 and stats['columns_parallel']
    assert set(tracker.export()['per_cell'])=={'receipt_total','1:name','1:quantity','1:price'}
    assert {lane['lane'] for lane in debug.lanes}=={'table','footer','table_names','table_numeric'}
    assert _LANE.get()=='table'


def test_column_failure_retries_only_failed_group(monkeypatch):
    monkeypatch.setenv('OCR_TABLE_PARALLEL', '1')
    cells = [SimpleNamespace(row_id=1, column_id=f) for f in ('name','price')]
    attempts = []
    def reader(items, **kwargs):
        if items[0].column_id=='name':
            raise RuntimeError('name failure')
        return {(1,'price'):{'text':'100'}}, {'requested':1,'cell_calls':1,'errors':[]}
    def fallback(items, **kwargs):
        attempts.extend(c.column_id for c in items)
        assert _LANE.get()=='table_names'
        return {(1,'name'):{'text':'product'}}, {'requested':1,'batch_calls':1,'errors':[]}
    monkeypatch.setattr(cell_mode,'read_batches',fallback)
    result, stats = cell_mode.read_columns(cells,reader)
    assert attempts==['name']
    assert result[(1,'price')]['text']=='100'
    assert stats['cell_fallback']==1 and stats['batch_calls']==1
    assert any('name failure' in e for e in stats['errors'])


def test_sequential_switch_and_single_group_do_not_create_worker_lanes(monkeypatch):
    cells = [SimpleNamespace(row_id=1,column_id=f) for f in ('name','price')]
    def reader(items, **kwargs):
        assert _LANE.get()=='table'
        return {'retained':True}, {'requested':len(items)}
    monkeypatch.setenv('OCR_TABLE_PARALLEL','0')
    result, stats = cell_mode.read_columns(cells,reader)
    assert result=={'retained':True} and not stats['columns_parallel']
    monkeypatch.setenv('OCR_TABLE_PARALLEL','1')
    _, stats = cell_mode.read_columns(cells[:1],reader)
    assert not stats['columns_parallel']


def test_secondary_models_are_cached_per_lane(monkeypatch):
    import rapidocr
    from App.ocr import experiment_models
    monkeypatch.setattr(experiment_models,'_SESSIONS',{})
    monkeypatch.setattr(rapidocr,'RapidOCR',lambda **kwargs:object())
    pool=experiment_models.ExperimentModelPool(
        catalog={}, read=lambda name,default:'0' if name=='OCR_E2_ENABLE' else default)
    model={'enabled':True,'ocr_version':'PPOCRV6','model_type':'TINY','lang':'en'}
    with inference_lane('table_numeric'):
        numeric=pool._get_session('v6_tiny_official',model)
        assert numeric is pool._get_session('v6_tiny_official',model)
    with inference_lane('footer'):
        footer=pool._get_session('v6_tiny_official',model)
        assert footer is not numeric
        assert footer is pool._get_session('v6_tiny_official',model)


@pytest.mark.parametrize('budget', [2,4])
def test_table_thread_budget_does_not_change_footer_budget(monkeypatch,budget):
    from App.ocr.engine import onnx_threads
    monkeypatch.setenv('OCR_ONNX_THREADS','2')
    monkeypatch.setenv('OCR_TABLE_ONNX_THREADS',str(budget))
    for lane in ('table_names','table_numeric'):
        with inference_lane(lane):
            assert onnx_threads()==budget//2
    for lane in ('footer','table'):
        with inference_lane(lane):
            assert onnx_threads()==2


@pytest.mark.parametrize('threads', [1,2,3,4])
def test_parallel_columns_divide_configured_thread_budget(monkeypatch,threads):
    from App.ocr.engine import onnx_threads
    monkeypatch.setenv('OCR_ONNX_THREADS',str(threads))
    assert onnx_threads()==threads
    with inference_lane('table_names'):
        names=onnx_threads()
    with inference_lane('table_numeric'):
        numeric=onnx_threads()
    assert names+numeric==max(2,threads)
    with inference_lane('footer'):
        assert onnx_threads()==threads
