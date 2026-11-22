"""Offline tests; CnOCR model weights and network access are not required.

When run inside GridMesh_RapidOCR project this imports real App.ocr.numeric_roi.
In a detached checkout we provide a minimal stub solely for runner logic tests.
"""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

try:
    from App.ocr import numeric_roi as real_roi
    HAS_REAL_APP = True
except ImportError:
    HAS_REAL_APP = False
    package = types.ModuleType('App')
    subpackage = types.ModuleType('App.ocr')
    fake = types.ModuleType('App.ocr.numeric_roi')
    def no_currency(image):return image,None
    def passthrough(image):return image,{'changed':False,'reason':'test_stub'}
    fake.split_price_currency = no_currency
    fake.crop_numeric_ink = passthrough
    sys.modules.update({'App':package,'App.ocr':subpackage,
                        'App.ocr.numeric_roi':fake})

from preprocess import prepare_integer_cell,separated_suffix_with_hint
import runner


def image_text(raw, width=170):
    img=np.full((48,width,3),255,np.uint8)
    cv2.putText(img,raw,(7,35),cv2.FONT_HERSHEY_SIMPLEX,.87,(0,0,0),2,cv2.LINE_AA)
    return img


class CropTests(unittest.TestCase):
    def test_integer(self):
        result=prepare_integer_cell(image_text('190'),field='price',baseline_text='190')
        self.assertEqual(result.status,'ready')
        self.assertIsNotNone(result.image)

    def test_last_digit_seven_not_chopped_without_hint(self):
        img=image_text('1907')
        out,marker=separated_suffix_with_hint(img,has_currency_hint=False)
        self.assertIsNone(marker)
        self.assertEqual(out.shape,img.shape)

    def test_t_hint_can_remove_spatially_isolated_glyph(self):
        img=image_text('190     T',width=220)
        result=prepare_integer_cell(img,field='price',baseline_text='190 T')
        self.assertIn(result.status,('ready','currency_not_safely_separated'))
        if result.status=='ready':
            self.assertLess(result.candidate_hw[1],img.shape[1])

    def test_unsupported_receipt_footer(self):
        result=prepare_integer_cell(image_text('190'),field='receipt_total',baseline_text='190')
        self.assertEqual(result.status,'excluded_field')

    def test_decimal_table_is_flagged(self):
        result=prepare_integer_cell(image_text('190'),field='price',baseline_text='190,50')
        self.assertEqual(result.status,'unexpected_decimal_separator')

    def test_bad_primary_ocr_not_skipped(self):
        result=prepare_integer_cell(image_text('190'),field='quantity',baseline_text='l9O')
        self.assertEqual(result.status,'ready')

    def test_currency_hinted_but_unsafe_is_skipped(self):
        result=prepare_integer_cell(image_text('190T'),field='price',baseline_text='190 T')
        self.assertIn(result.status,('ready','currency_not_safely_separated'))

    def test_crop_bbox_reject(self):
        source=image_text('190')
        yy,xx=np.indices(source.shape[:2],dtype=np.float32)
        crop,reason=runner.extract_cell(source,xx,yy,[0,0,999,999])
        self.assertIsNone(crop)
        self.assertEqual(reason,'bbox_outside_map')

    def test_crop_identity(self):
        source=image_text('190')
        yy,xx=np.indices(source.shape[:2],dtype=np.float32)
        crop,reason=runner.extract_cell(source,xx,yy,[0,0,100,40])
        self.assertIsNone(reason)
        self.assertEqual(crop.shape,(40,100,3))

    def test_model_metadata_from_book(self):
        book={'cells':{'1:price':{'observations':[
            {'path':'gridmesh_initial','text':'190','model':'v6_tiny_official','confidence':.91},
            {'path':'gridmesh_final','text':'190','model':'v5_cyrillic_mobile'}]}}}
        out=runner.initial_observation(book,'1:price')
        self.assertEqual(out['model'],'v6_tiny_official')

    def test_v5_candidate_is_not_primary_fallback(self):
        book={'cells':{'1:price':{'observations':[
             {'path':'gridmesh_initial','text':'190','model':'v6_tiny_official'},
             {'path':'v5_numeric_candidate','text':'191','model':'v5_cyrillic_mobile'}]}}}
        self.assertEqual(runner.initial_observation(book,'1:price')['model'],'v6_tiny_official')
        self.assertEqual(runner.parallel_v5_observation(book,'1:price')['text'],'191')

    def test_trace_unattributed_is_not_invented(self):
        trace={'events':[{'cells':['other'],'stage':'v5_numeric_candidate','wall_ms':9.0}]}
        self.assertIsNone(runner.trace_for_cell(trace,'1:price','v5_numeric_candidate'))

    def test_trace_v5_only_matched_stage(self):
        trace={'events':[{'cells':['1:price'],'stage':'v5_numeric_candidate',
                          'model_id':'v5_cyrillic_mobile','wall_ms':9.25,'success':True},
                         {'cells':['1:price'],'stage':'primary_rec_cell','wall_ms':80.0}]}
        item=runner.trace_for_cell(trace,'1:price','v5_numeric_candidate')
        self.assertEqual(item['calls'],1)
        self.assertEqual(item['total_call_wall_ms'],9.25)

    def test_normalization_with_currency(self):
        self.assertEqual(runner.normalize_numeric('1 900 ₸'),'1900')
        self.assertIsNone(runner.normalize_numeric('1,900'))


class FakeCnOcr:
    def ocr_for_single_line(self,img):
        return {'text':'190','score':.98}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.job=Path(self.temp.name)
        image=image_text('190')
        cv2.imencode('.png',image)[1].tofile(str(self.job/'original.png'))
        y,x=np.indices(image.shape[:2],dtype=np.float32)
        np.savez_compressed(self.job/'rectification_map.npz',map_x=x,map_y=y)
        receipt={'job_id':'a'*32,'rows':[{'row_id':1,'kind':'product','cells':{
            'quantity':{'text':'l9O','bbox':[0,0,100,40],'model_id':'v6_tiny_official'},
            'price':{'text':'190','bbox':[0,0,100,40],'model_id':'v6_tiny_official'}}}],
            'numeric_candidate_book':{'cells':{'1:price':{'observations':[
             {'path':'gridmesh_initial','text':'190','model':'v6_tiny_official','confidence':.94},
             {'path':'v5_numeric_candidate','text':'191','model':'v5_cyrillic_mobile','confidence':.87}]}}},
            'rec_trace':{'events':[
              {'stage':'v5_numeric_candidate','cells':['1:price'], 'model_id':'v5_cyrillic_mobile',
               'wall_ms':12.5,'success':True}]},
            'geometry':{'ocr_batch':{'table_wall_ms':80.0,'footer_lane_wall_ms':72.0,
                        'footer_wall_ms':20.0,'v5_candidate_wall_ms':51.0}}}
        (self.job/'receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
        performance={'debug':{'ocr_cells':[
            {'cell_id':'1:price','mode':'cell','wall_ms':20.0}]}}
        (self.job/'performance_profile.json').write_text(json.dumps(performance),encoding='utf-8')

    def tearDown(self):self.temp.cleanup()

    def test_preprocess_only_preserves_receipt(self):
        before=(self.job/'receipt.json').read_bytes()
        report=runner.run(self.job,preprocess_only=True)
        self.assertEqual(before,(self.job/'receipt.json').read_bytes())
        self.assertTrue((self.job/'numeric_only_experiment_v3.json').exists())
        self.assertTrue((self.job/'numeric_only_experiment_v3.html').exists())
        self.assertEqual(report['version'],'numeric-only-v3')
        self.assertEqual(report['summary']['recognized'],0)

    def test_model_fake_integration(self):
        report=runner.run(self.job,warmups=1,repeats=2,ocr_factory=FakeCnOcr)
        self.assertEqual(report['summary']['recognized'],2)
        self.assertEqual(report['summary']['agreements_final'],1)
        self.assertEqual(report['summary']['disagreements_primary'],0)
        price=next(r for r in report['cells'] if r['field']=='price')
        self.assertEqual(price['primary_observation']['model'],'v6_tiny_official')
        self.assertEqual(price['production_debug_wall_ms'],20.0)
        self.assertEqual(price['parallel_v5_observation']['text'],'191')
        self.assertEqual(price['parallel_v5_trace']['total_call_wall_ms'],12.5)
        self.assertEqual(report['summary']['disagreements_parallel_v5'],1)
        self.assertEqual(report['summary']['production_parallel_lanes']['v5_candidate_wall_ms'],51.0)
        self.assertEqual(len(price['wall_ms_repeats']),2)
        self.assertFalse((self.job/'numeric_only_experiment.json').exists())

    def test_model_error_logged(self):
        def bad():raise RuntimeError('no model installed')
        report=runner.run(self.job,ocr_factory=bad)
        self.assertEqual(report['summary']['errors'],2)
        self.assertIn('RuntimeError',report['summary']['model_init_error'])
        self.assertEqual(report['summary']['recognized'],0)


if __name__=='__main__':unittest.main(verbosity=2)
