"""Acceptance tests for integrating OriginalROI Candidate Matrix V4."""
import copy
import json
from decimal import Decimal
import pytest

from App.ocr.experiment_matrix import CandidateBook
from App.ocr.experiment_vat import vat_source,seed_printed_vat
from App.ocr.experiment_auto import _support
from App.ocr.experiment_rec_trace import RecTracker,activate,deactivate,record_rec


def test_printed_vat_in_quantity_column_is_original_evidence():
    rows=[{'row_id':1,'kind':'tax','cells':{'quantity':{'text':'34781','confidence':.99},
                                          'price':{'text':'1 T','confidence':.98}}}]
    assert vat_source(rows)[0]==(1,'quantity')
    book=CandidateBook();seed_printed_vat(rows,book)
    assert book.vat[0]['number']=='34781'
    assert book.vat[0]['field']=='quantity'


def test_v5_aliases_cannot_be_two_independent_model_votes():
    observations=[dict(number='100',confidence=.99,model=name,authority='ocr_candidate')
                  for name in ['v5_cyrillic_mobile','PP-OCRv5 Cyrillic Mobile / RapidOCR 3.x','v5']]
    assert len(_support(observations,'100',.8))==1


def test_rec_context_restored_after_failure(monkeypatch):
    import App.pipeline as pipeline
    tracker=RecTracker();token=activate(tracker)
    def fail(*args,**kwargs):raise OSError('write failed')
    monkeypatch.setattr(pipeline,'_process_job',fail)
    try:
        with pytest.raises(OSError):pipeline.process(b'')
        record_rec(keys=['parent'],ms=1)
        assert tracker.export()['per_cell']['parent']['rec_only_exposures']==1
    finally:deactivate(token)


def test_manual_save_preserves_prior_user_truth_in_memory(tmp_path,monkeypatch):
    monkeypatch.setenv('OUTPUT_DIR',str(tmp_path/'data/output'))
    import App.web_server as server
    job_id='a'*32;directory=tmp_path/'data/output'/job_id;directory.mkdir(parents=True)
    original={'job_id':job_id,'job_directory':f'data/output/{job_id}',
              'rows':[{'row_id':1,'kind':'product','cells':{
                  'name':{'text':'Товар Brand'},'quantity':{'text':'1','confidence':.9},
                  'price':{'text':'100','confidence':.9}},'line_total':'100'}],
              'total':'100','vat':{'rate_percent':'16'},'geometry':{'warnings':[]}}
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr(server,'jobs',{job_id:{'status':'done','result':original}})
    submitted=copy.deepcopy(original)
    submitted['rows'][0]['cells']['price'].update(text='200',user_verified=True)
    submitted['total']='200'
    result=server.save_edited_receipt(job_id,submitted)
    assert result['receipt']['math_match']
    assert server.jobs[job_id]['result']['rows'][0]['cells']['price']['text']=='200'
    second=copy.deepcopy(server.jobs[job_id]['result'])
    second['rows'][0]['cells']['quantity'].update(text='2',user_verified=True)
    second['total']='400'
    result=server.save_edited_receipt(job_id,second)
    assert result['receipt']['computed_total']=='400'
    assert result['receipt']['rows'][0]['cells']['price']['user_verified']
    assert (directory/'numeric_user_truth.json').is_file()


def test_v4_enabled_replaces_legacy_retries_and_writes_trace(monkeypatch):
    from pathlib import Path
    from App.pipeline import process
    monkeypatch.setenv('OCR_E4_ENABLE','1')
    monkeypatch.setenv('OCR_E2_ENABLE','1')
    result=process((Path(__file__).parent/'receipt_5730.png').read_bytes())
    assert result['math_match']
    assert result['computed_total']=='5730'
    assert result['numeric_matrix']['status']!='experiment_error'
    assert result['geometry']['ocr_recovery']['initial']['delegated']=='e4_independent_original_roi'
    assert result['rec_trace']['rec_only_model_invocations']>0
    assert result['numeric_matrix']['initial_status']=='unique_math_match'
    assert result['numeric_matrix']['fallback_used'] is False
    assert result['original_roi_experiment']['rec_calls']==0
    assert result['geometry']['numeric_experiment']['model_ensemble']['model_calls']==0
    assert (Path(__file__).resolve().parents[2]/result['numeric_experiment_file']).exists()


def test_unique_matrix_resolves_footer_variants_without_model_quorum():
    from App.ocr.experiment_matrix import solve_matrix
    from App.ocr.experiment_auto import auto_apply_consensus
    rows=[{'row_id':1,'kind':'product','cells':{
        'quantity':{'text':'2','confidence':.9},
        'price':{'text':'8800','confidence':.99}}}]
    book=CandidateBook()
    book.add(1,'quantity','2',.9,'primary')
    book.add(1,'price','8800',.99,'primary')
    book.add(1,'price','6800',.4,'other',model='v6_small_official')
    book.add_footer('13600',.5,'footer')
    book.add_footer('13800',.99,'footer_other')
    solved=solve_matrix(rows,book,16)
    assert solved['status']=='unique_math_match'
    assert not solved['review_required']
    report=auto_apply_consensus(rows,book,solved,'13800')
    assert report['status']=='applied'
    assert rows[0]['cells']['price']['text']=='6800'
    assert report['printed_total']=='13600'


def test_numeric_scales_are_one_candidate():
    book=CandidateBook()
    for value in ['1','1.0','1.00']:
        book.add(1,'quantity',value,.9,value)
    assert len(book.possible_values(1,'quantity'))==1


def test_ambiguous_matrix_ranks_all_solutions_and_warns():
    from App.ocr.experiment_matrix import solve_matrix
    from App.ocr.experiment_auto import auto_apply_consensus
    rows=[{'row_id':1,'kind':'product','cells':{
        'quantity':{'text':'1'},'price':{'text':'100'}}}]
    book=CandidateBook();book.add(1,'quantity','1',.9,'primary')
    for value,score in [('100',.5),('200',.6),('300',.99)]:
        book.add(1,'price',value,score,value)
        book.add_footer(value,score,value)
    matrix=solve_matrix(rows,book,16)
    assert len(matrix['solutions'])==3
    assert matrix['solutions'][0]['footer']=='300'
    result=auto_apply_consensus(rows,book,matrix,'100')
    assert result['status']=='unresolved_preserved' and result['review_required']
    assert rows[0]['cells']['price']['text']=='100'
    bounded=solve_matrix(rows,book,16,max_nodes=1)
    assert auto_apply_consensus(rows,book,bounded,'300')['status']=='unresolved_preserved'


def test_original_roi_fallback_runs_when_primary_candidate_matrix_has_no_solution(monkeypatch):
    from pathlib import Path
    import App.pipeline as pipeline
    monkeypatch.setenv('OCR_E4_ENABLE','1')
    monkeypatch.setenv('OCR_E4_FAST_GATE','0')
    original=pipeline.solve_matrix
    calls=[]
    def no_solution_first(*args,**kwargs):
        value=original(*args,**kwargs)
        calls.append(value['status'])
        if len(calls)==1:
            value.update(status='no_solution',solutions=[],review_required=True)
        return value
    monkeypatch.setattr(pipeline,'solve_matrix',no_solution_first)
    result=pipeline.process((Path(__file__).parent/'receipt_5730.png').read_bytes())
    assert result['numeric_matrix']['initial_status']=='no_solution'
    assert result['numeric_matrix']['fallback_used'] is True
    assert result['original_roi_experiment']['cache_cells']>0
    assert result['math_match']
