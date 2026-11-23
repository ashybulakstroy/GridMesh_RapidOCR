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
