"""E4 independent REC routing, audit and no mandatory manual review."""
from decimal import Decimal
from types import SimpleNamespace
import numpy as np

from App.ocr.experiment_rec_trace import RecTracker, activate, deactivate, rec_scope, record_rec
from App.ocr.experiment_matrix import CandidateBook, solve_matrix
from App.ocr.experiment_original import OriginalNumericCache


def sample_rows():
    return [{'row_id':1,'kind':'product','cells':{
        'quantity':{'text':'1','confidence':.98},
        'price':{'text':'1430','confidence':.79}},'line_total':'1430','printed_line_match':None},
        {'row_id':2,'kind':'tax','cells':{
        'quantity':{'text':'1','confidence':.98},
        'price':{'text':'150','confidence':.98}},'line_total':'150','printed_line_match':None}]


def create_book(rows):
    book=CandidateBook()
    for row in rows:
        for field,item in row['cells'].items():
            book.add(row['row_id'],field,item['text'],item['confidence'],'primary')
    book.add_footer('1630',.9,'printed_total')
    book.add_vat('150',.9,'printed_vat',row_id=2,field='price')
    return book


def cache_with_price():
    obj=object.__new__(OriginalNumericCache)
    obj.crops={(1,'price'):np.full((40,110,3),255,np.uint8)}
    obj.bytes=sum(im.nbytes for im in obj.crops.values())
    obj.prepare_ms=0.1
    obj.errors=[]
    return obj


def test_batch_model_invocation_counts_per_cell_exposures():
    tracer=RecTracker(); token=activate(tracer)
    try:
        record_rec(keys=['1:price','1:quantity'],model='v5_cyrillic_mobile',
                   stage='primary_rec_batch',ms=3.0)
        with rec_scope([(1,'price')],'independent_source'):
            record_rec(model='v6_small_official',ms=1.3)
        with rec_scope(['receipt_total'],'footer_detector'):
            record_rec(mode='DET+REC',ms=8,note='opaque internal model calls')
    finally:
        deactivate(token)
    result=tracer.export()
    assert result['rec_only_model_invocations']==2
    assert result['rec_only_cell_exposures']==3
    assert result['per_cell']['1:price']['rec_only_exposures']==2
    assert result['per_cell']['1:quantity']['rec_only_exposures']==1
    assert result['per_cell']['receipt_total']['detector_pipeline_calls']==1


def test_independent_model_replaces_redundant_v5_retry():
    rows=sample_rows();book=create_book(rows);cache=cache_with_price()
    class Pool:
        def models_for(self, task):return ['v6_small_official']
        def recognize(self,model_id,image,stage):
            return {'text':'1480','confidence':.97,'model_id':model_id}
    report=cache.run_suspects(rows,book,printed_total='1630',max_calls=2,
        read_line=lambda im: (_ for _ in ()).throw(AssertionError('v5 must not run')),
        model_pool=Pool())
    assert report['rec_calls']==1
    assert book.possible_values(1,'price')[0]['number']=='1480'
    assert book.possible_values(1,'price')[0]['observations'][0]['model']=='v6_small_official'
    solved=solve_matrix(rows,book,16)
    assert solved['status']=='unique_math_match'
    assert solved['solutions'][0]['vat_total']=='150'  # Operator VAT error is tolerated.


def test_fallback_v5_if_independent_recognizer_unavailable():
    rows=sample_rows();book=create_book(rows);cache=cache_with_price()
    class Pool:
        def models_for(self,task):return ['v6_small_official']
        def recognize(self,model_id,image,stage):return {'error':'onnx_missing'}
    used=[]
    report=cache.run_suspects(rows,book,printed_total='1630',max_calls=2,
        read_line=lambda im:(used.append(1) or {'text':'1480','confidence':.90}),
        model_pool=Pool())
    assert len(used)==1
    assert report['rec_results'][0]['model_priority']=='independent_first_fallback_v5'


def test_escalation_only_uses_second_model_and_keeps_original_data():
    rows=sample_rows(); book=create_book(rows); cache=cache_with_price()
    class Pool:
        def models_for(self,task):return ['v6_small_official','v6_tiny_official']
        def recognize(self,model_id,image,stage):
            assert model_id=='v6_tiny_official'
            return {'text':'1480','confidence':.91,'model_id':model_id}
    result=cache.escalate_suspects(book,{'rec_results':[{'row_id':1,'field':'price'}]},Pool(),max_calls=1)
    assert result['recognitions']==1
    assert rows[0]['cells']['price']['text']=='1430'
    assert any(o['model']=='v6_tiny_official' for o in book.cells['1:price']['observations'])


def test_conflict_is_not_called_correct_or_requiring_human():
    rows=sample_rows();book=create_book(rows)
    book.add(1,'price','1480',.92,'model_2',model='v6_small_official')
    result=solve_matrix(rows,book,16)
    assert result['status']=='unique_math_match'
    assert rows[0]['cells']['price']['text']=='1430'  # Candidate only; no silent overwrite.


def test_auto_consensus_uses_two_distinct_models_and_printed_vat():
    from App.ocr.experiment_auto import auto_apply_consensus
    rows=sample_rows(); book=create_book(rows)
    # Original price is 1430, actual photographic evidence is 1480.
    book.add(1,'price','1480',.94,'source_1',model='v6_small_official')
    book.add(1,'price','1480',.92,'source_2',model='v6_tiny_official')
    # The footer and VAT must each have evidence from TWO models as well.
    book.add_footer('1630',.95,'foot_1',model='v6_small_official')
    book.add_footer('1630',.93,'foot_2',model='v6_tiny_official')
    book.add_vat('150',.94,'vat_1',model='v6_small_official',row_id=2,field='price')
    book.add_vat('150',.91,'vat_2',model='v6_tiny_official',row_id=2,field='price')
    solved=solve_matrix(rows,book,16)
    assert solved['status']=='unique_math_match'
    report=auto_apply_consensus(rows,book,solved,'1630')
    assert report['status']=='applied'
    assert rows[0]['cells']['price']['text']=='1480'
    assert rows[0]['line_total']=='1480'
    assert report['changes'][0]['models']=={'v6_small_official':.94,'v6_tiny_official':.92}


def test_single_model_candidates_allow_unique_matrix_selection():
    from App.ocr.experiment_auto import auto_apply_consensus
    rows=sample_rows();book=create_book(rows)
    book.add(1,'price','1480',.97,'v5_inverted',model='v5')
    book.add(1,'price','1480',.98,'v5_bicubic',model='v5')
    book.add_footer('1630',.96,'footer_v6',model='v6')
    book.add_vat('150',.96,'vat_v6',model='v6',row_id=2,field='price')
    solved=solve_matrix(rows,book,16)
    result=auto_apply_consensus(rows,book,solved,'1630')
    assert result['status']=='applied'
    assert rows[0]['cells']['price']['text']=='1480'


def test_user_locked_cells_remain_immutable():
    from App.ocr.experiment_auto import auto_apply_consensus
    rows=sample_rows();book=create_book(rows)
    rows[0]['cells']['price']['user_verified']=True
    book.add(1,'price','1480',.97,'other_1',model='v6a')
    book.add(1,'price','1480',.98,'other_2',model='v6b')
    book.add_footer('1630',.96,'foot_a',model='v6a')
    book.add_footer('1630',.96,'foot_b',model='v6b')
    book.add_vat('150',.95,'vat_a',model='v6a',row_id=2,field='price')
    book.add_vat('150',.95,'vat_b',model='v6b',row_id=2,field='price')
    solved=solve_matrix(rows,book,16)
    result=auto_apply_consensus(rows,book,solved,'1630')
    assert result['status']=='not_enough_independent_evidence'
    assert rows[0]['cells']['price']['text']=='1430'


def test_ambiguous_receipt_keeps_original_without_user_prompt():
    from App.ocr.experiment_auto import auto_apply_consensus
    rows=sample_rows();book=create_book(rows)
    book.add(1,'price','1480',.94,'source_1',model='v6a')
    book.add(1,'price','1480',.93,'source_2',model='v6b')
    book.add_footer('1580',.94,'second_footer',model='v6a')
    result=solve_matrix(rows,book,16)
    assert result['status']=='ambiguous'
    corrected=auto_apply_consensus(rows,book,result,'1630')
    assert corrected['status']=='applied'
    assert corrected['review_required']
    assert rows[0]['cells']['price']['text']=='1480'
