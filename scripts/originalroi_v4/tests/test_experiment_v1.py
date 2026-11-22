"""Integration-friendly synthetic tests for the independent numeric OCR experiment.

Run from the host project root:
  python -m pytest -q App/tests/test_experiment_v1.py
No OCR model download required; source/geometry tests mock the recognizer.
"""
from decimal import Decimal
import numpy as np
import cv2
import pytest

from App.ocr.experiment_matrix import CandidateBook, solve_matrix
from App.ocr.experiment_original import OriginalNumericCache
from App.ocr.experiment_manual import apply_user_receipt
from App.ocr.experiment_footer import scan_footer


def make_row(number,kind,quantity,price,printed=None):
    items = {k:{'text':str(v),'confidence':.98,'row_id':number,'column_id':k}
             for k,v in (('quantity',quantity),('price',price))}
    if printed is not None:
        items['printed_total']={'text':str(printed),'confidence':.95,
                                'row_id':number,'column_id':'printed_total'}
    return {'row_id':number,'kind':kind,'cells':items,
            'line_total':str(Decimal(str(quantity))*Decimal(str(price)))
                 if str(quantity).isdigit() and str(price).isdigit() else None,
            'printed_line_match':None}


def seed(rows,footer):
    book=CandidateBook()
    for row in rows:
        for field,cell in row['cells'].items():
            book.add(row['row_id'],field,cell['text'],cell['confidence'],'gridmesh_initial')
    book.add_footer(footer,.96,'footer_detector')
    return book


def test_single_math_supported_alternative_with_vat():
    rows=[make_row(1,'product','1','120'),make_row(2,'tax','192','1')]
    book=seed(rows,'1392')
    book.add(1,'price','1200',.89,'original_pixels_numeric_roi_rec_only')
    result=solve_matrix(rows,book,16)
    assert result['status']=='unique_math_match'
    assert result['recommended_changes']==[{'row_id':1,'field':'price','from':'120',
                                             'to':'1200','status':'proposed_math_supported_not_ground_truth'}]
    assert any(x['assessment']=='math_conflicting' and x['path']=='gridmesh_initial'
               for x in result['model_assessments'])
    assert any(x['assessment']=='math_supported' and x['path']=='original_pixels_numeric_roi_rec_only'
               for x in result['model_assessments'])


def test_two_independent_numeric_corrections_can_combine():
    rows=[make_row(1,'product','2','125'),make_row(2,'product','1','80')]
    book=seed(rows,'530')
    book.add(1,'price','225',.88,'retry')
    book.add(2,'price','80',.93,'source')
    # More than one cell may vary; total=530 requires 2*225+1*80.
    result=solve_matrix(rows,book,16)
    assert result['status']=='unique_math_match'
    assert result['solutions'][0]['computed_total']=='530'


def test_two_assignments_same_total_are_not_automatically_confirmed():
    rows=[make_row(1,'product','1','100'),make_row(2,'product','1','100')]
    book=seed(rows,'300')
    book.add(1,'price','200',.85,'source_1')
    book.add(2,'price','200',.85,'source_2')
    result=solve_matrix(rows,book,16)
    assert result['status']=='ambiguous'
    assert len(result['solutions'])==2
    assert result['recommended_changes']


def test_unique_solution_resolves_footer_disagreement():
    rows=[make_row(1,'product','1','100')]
    book=seed(rows,'100')
    book.add_footer('101',.90,'footer_otsu')
    result=solve_matrix(rows,book,16)
    assert result['status']=='unique_math_match'
    assert result['footer_ambiguous'] is True
    assert result['review_required'] is False


def test_bounded_search_does_not_claim_unique():
    rows=[make_row(1,'product','1','100'),make_row(2,'product','1','100')]
    book=seed(rows,'300')
    book.add(1,'price','200',.9,'source')
    book.add(2,'price','200',.9,'source')
    result=solve_matrix(rows,book,16,max_nodes=1)
    assert result['status']=='bounded_search'
    assert result['search_truncated'] is True
    assert not result['recommended_changes']


def test_user_value_locks_candidate_dimension():
    rows=[make_row(1,'product','1','120'),make_row(2,'tax','192','1')]
    book=seed(rows,'1392')
    book.add(1,'price','1200',None,'web_user',user=True)
    assert [i['number'] for i in book.possible_values(1,'price')]==['1200']
    assert solve_matrix(rows,book,16)['status']=='unique_math_match'


def test_invalid_ocr_candidate_kept_for_audit_not_matrix():
    rows=[make_row(1,'product','1','100')]
    book=seed(rows,'100')
    book.add(1,'price','1О0',.92,'raw_image')  # Cyrillic О
    assert book.cells['1:price']['observations'][-1]['number'] is None
    assert solve_matrix(rows,book,16)['status']=='unique_math_match'


def test_server_manual_truth_overrides_ocr_and_recomputes_status():
    rows=[make_row(1,'product','1','120'),make_row(2,'tax','192','1')]
    book=seed(rows,'1392')
    original={'job_id':'abc','rows':rows,'total':'1392','computed_total':'312',
              'math_match':False,'status':'review',
              'vat':{'rate_percent':'16','status':'mismatch'},
              'numeric_candidate_book':book.export()}
    import copy
    sent=copy.deepcopy(original)
    sent['rows'][0]['cells']['price']['text']='1200'
    sent['rows'][0]['cells']['price']['user_verified']=True
    sent['math_match']=False  # intentionally stale/untrusted client value
    edited=apply_user_receipt(original,sent)
    assert edited['rows'][0]['cells']['price']['text']=='1200'
    assert edited['rows'][0]['cells']['price']['user_verified'] is True
    assert edited['math_match'] is True
    assert edited['numeric_user_truth'][0]['authority']=='user_confirmed'
    assert edited['numeric_matrix']['status']=='unique_math_match'
    assert any(x['assessment']=='user_confirmed_conflict' and x['path']=='gridmesh_initial'
               for x in edited['numeric_model_feedback'])
    assert original['rows'][0]['cells']['price']['text']=='120'


def test_invalid_manual_input_is_not_ground_truth():
    rows=[make_row(1,'product','1','100')]
    original={'rows':rows,'total':'100','vat':{'rate_percent':'16'},
              'numeric_candidate_book':seed(rows,'100').export()}
    import copy
    sent=copy.deepcopy(original)
    sent['rows'][0]['cells']['price']['text']='1O0'
    edited=apply_user_receipt(original,sent)
    assert edited['status']=='review'
    assert edited['numeric_user_truth'][0]['valid'] is False
    assert edited['rows'][0]['cells']['price']['user_verified'] is False


def test_original_source_cache_does_not_use_remap(monkeypatch):
    import App.ocr.experiment_original as source
    # Test local cell polygon; original raw pixels are preserved.
    monkeypatch.setattr(source,'cell_polygon',lambda rect,row,col:
        np.array([[5,5],[80,5],[80,32],[5,32]],dtype=float))
    monkeypatch.setattr(cv2,'remap',lambda *args,**kwargs:
        (_ for _ in ()).throw(AssertionError('remap must not be used')))
    img=np.full((70,110,3),255,dtype=np.uint8)
    cv2.putText(img,'100',(13,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
    class Rect:
        method='original_cells'
        info={'source_offset':(0,0)}
    cache=OriginalNumericCache(img,Rect())
    cache.store(1,'price',3)
    assert cache.bytes>0
    assert len(cache.crops)==1
    row=make_row(1,'product','1','80')
    row['cells']['price']['confidence']=.5
    book=seed([row],'100')
    result=cache.run_suspects([row],book,printed_total='100',read_line=lambda x:{'text':'100','confidence':.97})
    assert result['rec_calls']==1
    assert book.possible_values(1,'price')[0]['number']=='100'


def test_footer_scan_preserves_conflicting_observations():
    img=np.full((250,150,3),255,dtype=np.uint8)
    n=[0]
    def fake_line(crop):
        n[0]+=1
        return {'text':'100' if n[0]==1 else '108','confidence':.89}
    probes=scan_footer(img,160,'100',101,read_line=fake_line,read_full=lambda *args,**kwargs:[])
    vals={p['total'] for p in probes if p['total']}
    assert '100' in vals and '108' in vals
    assert len(probes)<=6


def test_more_than_four_distinct_cell_options_blocks_false_uniqueness():
    rows=[make_row(1,'product','1','100')]
    book=seed(rows,'100')
    for n in ('101','102','103','104'):
        book.add(1,'price',n,.8,'extra_'+n)
    result=solve_matrix(rows,book,16)
    assert result['candidate_pool_truncated'] is True
    assert result['status']=='bounded_search'


def test_detector_footer_combines_same_baseline_and_excludes_vat():
    img=np.full((400,1000,3),255,dtype=np.uint8)
    responses=[
        {'text':'192','confidence':.95,'bbox':[[120,0],[160,0],[160,8],[120,8]]},
        {'text':'5','confidence':.94,'bbox':[[80,48],[90,48],[90,61],[80,61]]},
        {'text':'730','confidence':.96,'bbox':[[95,48],[150,48],[150,61],[95,61]]},
    ]
    line_calls=[0]
    def fake_line(crop):
        line_calls[0]+=1
        return {'text':'', 'confidence':0}
    results=scan_footer(img,255,'5000','5730',read_line=fake_line,
                        read_full=lambda *args, **kwargs: responses)
    # For crop y0=y_bottom-27=228, the footer baseline y=276 is valid.
    detector=[v for v in results if v['strategy']=='footer_source_detector_baseline']
    assert len(detector)==1
    assert detector[0]['total']=='5730'


def test_manual_footer_remains_authoritative_when_arithmetic_disagrees():
    rows=[make_row(1,'product','1','100')]
    original={'rows':rows,'total':'100','vat':{'rate_percent':'16'},
              'numeric_candidate_book':seed(rows,'100').export()}
    import copy
    sent=copy.deepcopy(original)
    sent['total']='101'
    sent['user_total_verified']=True
    edited=apply_user_receipt(original,sent)
    assert edited['numeric_user_truth'][0]['field']=='receipt_total'
    assert edited['numeric_user_truth'][0]['authority']=='user_confirmed'
    assert edited['total']=='101'
    assert edited['numeric_matrix']['status']=='no_solution'
    assert edited['status']=='review'


def test_model_selection_is_opt_in_and_does_not_load_at_configuration_time():
    from App.ocr.experiment_models import ExperimentModelPool
    catalog={'v6_small_official':{'id':'v6_small_official','enabled':True,
                                 'name':'v6 small','ocr_version':'PPOCRV6',
                                 'model_type':'SMALL','lang':'en'}}
    off=lambda key,default: ('0' if key=='OCR_E2_ENABLE' else default)
    pool=ExperimentModelPool(read=off,catalog=catalog)
    assert pool.models_for('numeric')==[]
    on=lambda key,default: {'OCR_E2_ENABLE':'1',
                            'OCR_E2_NUMERIC_MODELS':'v6_small_official,v6_small_official',
                            'OCR_E2_FOOTER_MODELS':'v6_small_official',
                            'OCR_E2_MAX_MODEL_CALLS':'3'}.get(key,default)
    pool=ExperimentModelPool(read=on,catalog=catalog)
    assert pool.models_for('numeric')==['v6_small_official']
    assert pool.calls==0


def test_second_rapidocr_candidate_retains_model_id_and_exhausts_budget(monkeypatch):
    from App.ocr.experiment_models import ExperimentModelPool
    catalog={'v6_small_official':{'id':'v6_small_official','enabled':True,
                                 'name':'v6 small','ocr_version':'PPOCRV6',
                                 'model_type':'SMALL','lang':'en'}}
    on=lambda key,default: {'OCR_E2_ENABLE':'1',
                            'OCR_E2_NUMERIC_MODELS':'v6_small_official',
                            'OCR_E2_FOOTER_MODELS':'v6_small_official',
                            'OCR_E2_MAX_MODEL_CALLS':'1'}.get(key,default)
    pool=ExperimentModelPool(read=on,catalog=catalog)
    class FakeResponse:
        txts=['1250']
        scores=[.92]
    class FakeSession:
        def recognize_txt(self, batch):
            assert len(batch)==1
            return FakeResponse()
    monkeypatch.setattr(pool,'_get_session',lambda *args:FakeSession())
    import sys, types, threading
    monkeypatch.setitem(sys.modules,'App.ocr.engine',types.SimpleNamespace(_lock=threading.RLock()))
    img=np.full((20,30,3),255,dtype=np.uint8)
    first=pool.recognize('v6_small_official',img,'original_numeric_roi')
    assert first['model_id']=='v6_small_official'
    assert first['text']=='1250'
    assert pool.recognize('v6_small_official',img,'footer')['error']=='model_call_budget_exhausted'
    assert pool.stats()['model_calls']==1


def test_footer_uses_registered_rapidocr_models_without_tesseract():
    class FakePool:
        def models_for(self,task):
            return ['v6_small_official'] if task=='footer' else []
        def recognize(self,model_id,crop,stage):
            return {'text':'1299','confidence':.88,'model':'v6 small',
                    'model_id':model_id,'wall_ms':5.0}
    img=np.full((280,190,3),255,dtype=np.uint8)
    def primary(crop):return {'text':'1200','confidence':.94}
    scans=scan_footer(img,170,'1200',1201,read_line=primary,
                      read_full=lambda *args,**kwargs:[],model_pool=FakePool())
    assert any(item.get('model_id')=='v6_small_official' and item['total']=='1299'
               for item in scans)
