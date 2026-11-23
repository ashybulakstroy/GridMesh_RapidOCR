import json
import cv2
import numpy as np
from App.ocr.problem_digits import save_problem_digits


def test_debug_evidence_keeps_photographic_pixels_and_ocr_alternatives(tmp_path):
    image=np.arange(20*30,dtype=np.uint8).reshape(20,30)
    observations=[{'text':'17','number':'17','model':'v6'}, {'text':'1','number':'1','model':'v5'}]
    rows=[{'row_id':2,'cells':{'price':{'text':'1','source_bbox':[[1,2],[3,4]]}}}]
    book={'cells':{'2:price':{'row_id':2,'field':'price','observations':observations}}}
    changes={'changes':[{'row_id':2,'field':'price','before':'17','after':'1'}]}
    result=save_problem_digits(tmp_path,rows,{(2,'price'):image},book,changes)
    manifest=json.loads((tmp_path/result['manifest']).read_text(encoding='utf-8'))
    cell=manifest['cells'][0]
    assert result['count']==1 and cell['before']=='17' and cell['after']=='1'
    assert cell['observations']==observations
    restored=cv2.imread(str(tmp_path/'problem_digits'/cell['images']['gridmesh_cell']),cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(restored,image)
    assert (tmp_path/result['report']).is_file()


def test_missing_roi_and_footer_are_reported_without_fake_crop(tmp_path):
    changes={'changes':[{'row_id':1,'field':'quantity','before':'bad','after':'2'},
                        {'row_id':None,'field':'receipt_total','before':'100','after':'200'}]}
    save_problem_digits(tmp_path,[],{}, {},changes)
    cells=json.loads((tmp_path/'problem_digits/manifest.json').read_text(encoding='utf-8'))['cells']
    assert cells[0]['image_status']=='crop_unavailable'
    assert cells[1]['images']=={'full_receipt':'../original.png'}


def test_actual_primary_input_is_saved_separately_from_cell_metadata(tmp_path):
    image=np.full((30,80),200,np.uint8)
    actual=image[:,:50].copy()
    rows=[{'row_id':1,'cells':{'price':{'text':'9000','numeric_roi':{'ocr_roi_bbox':[0,0,70,30]}}}}]
    changes={'changes':[{'row_id':1,'field':'price','before':'000','after':'9000'}]}
    save_problem_digits(tmp_path,rows,{(1,'price'):image},{},changes,primary_inputs={(1,'price'):actual})
    folder=tmp_path/'problem_digits'
    record=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))['cells'][0]
    saved=cv2.imread(str(folder/record['images']['primary_ocr_input']),cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(saved,actual)
    assert 'recorded_roi' in record['images']


def test_footer_conflict_includes_source_photo_model_and_each_attempt_time(tmp_path):
    image=np.arange(50*80,dtype=np.uint8).reshape(50,80)
    book={'footer':[{'text':'90665','number':'90665','model':'v5'},
                    {'text':'66500','number':'66500','model':'v5'}]}
    trace={'events':[{'event_id':0,'cells':['receipt_total'],'model_id':'v5',
                     'stage':'printed_footer_primary','wall_ms':4.25,'mode':'REC_only','success':True}]}
    save_problem_digits(tmp_path,[],{},book,{},footer_context={'total':'90665','image':image,
        'attempts':[{'strategy':'single_line','total':'66500','lines':[{'text':'66500',
            'bbox':[[10,20],[50,20],[50,30],[10,30]]}]}]},rec_trace=trace,
        attempt_inputs={0:image[20:30,10:50]})
    directory=tmp_path/'problem_digits'
    record=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))['cells'][0]
    assert record['field']=='receipt_total' and record['after']=='90665'
    assert record['recognition_attempts'][0]['wall_ms']==4.25
    assert record['recognition_attempts'][0]['model_id']=='v5'
    assert record['footer_regions'][0]['source_bbox']==[10,20,50,30]
    assert np.array_equal(cv2.imread(str(directory/record['recognition_attempts'][0]['input_image']),0),image[20:30,10:50])


def test_low_confidence_failed_attempt_and_vat_conflict_are_included(tmp_path):
    rows=[{'row_id':1,'cells':{'quantity':{'text':'2','confidence':.5}}}]
    book={'vat':[{'text':'32','number':'32'},{'text':'320','number':'320'}]}
    trace={'events':[{'cells':['2:price','3:price'],'success':False,'stage':'batch',
                     'model_id':'v5','wall_ms':10,'mode':'REC_only'}]}
    save_problem_digits(tmp_path,rows,{},book,{},rec_trace=trace)
    cells=json.loads((tmp_path/'problem_digits/manifest.json').read_text(encoding='utf-8'))['cells']
    assert {(x['row_id'],x['field']) for x in cells}=={(1,'quantity'),(2,'price'),(3,'price'),(None,'printed_vat')}
    assert cells[1]['recognition_attempts'][0]['duration_scope']=='shared_batch'


def test_rec_tracker_preserves_result_model_time_and_debug_input(monkeypatch):
    from types import SimpleNamespace
    from App.ocr.experiment_rec_trace import RecTracker
    monkeypatch.setenv('LOG_LEVEL','DEBUG')
    tracker=RecTracker();image=np.ones((10,20),np.uint8)
    tracker.record(['1:price'],'retry','v6','REC_only',12.5,image=image,
                   result=SimpleNamespace(txts=['17'],scores=[.7]))
    event=tracker.export()['events'][0]
    assert event['texts']==['17'] and event['wall_ms']==12.5 and event['model_id']=='v6'
    image[:]=0
    assert tracker.inputs[event['event_id']].all()
    monkeypatch.setenv('LOG_LEVEL','INFO')
    quiet=RecTracker();quiet.record(['receipt_total'],'footer','v5','REC_only',2,image=image)
    assert quiet.inputs=={}
