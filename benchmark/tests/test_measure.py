import unittest
from decimal import Decimal
import cv2
import numpy as np
from benchmark.core.measure import (decimal_ocr, levenshtein, verify_math, gt_fields, recognize_cells,
                                    summarize, evaluate, tighten_numeric_crop, split_name_lines, join_text_parts,
                                    extract_rois)
from benchmark.core.runner import read_ground_truth

class Reply:
    def __init__(self,txts):self.txts=txts;self.scores=[.9]*len(txts)
class FakeRecognizer:
    def __init__(self):self.calls=[]
    def recognize_txt(self, imgs):
        self.calls.append(len(imgs))
        return Reply(['42']*len(imgs))

class TestBenchmark(unittest.TestCase):
    def test_all_active_ground_truth_math(self):
        gts=read_ground_truth();self.assertEqual(3,len(gts))
        for g in gts.values():self.assertEqual([],verify_math(g),g['receipt_id'])
    def test_normalize(self):
        self.assertEqual(Decimal('73718.00'),decimal_ocr('73 718,00 ₸'))
        self.assertEqual(Decimal('1'),decimal_ocr('1 Т'))
        self.assertIsNone(decimal_ocr('12д 5'))
    def test_levenshtein(self):
        self.assertEqual(1,levenshtein('abc','abd'))
    def test_single_never_batches(self):
        engine=FakeRecognizer();cells=[('row:1:quantity',object()),('row:1:unit_price',object()),('printed_total',object())]
        res=recognize_cells(cells,engine,'single_cell',batch_size=16)
        self.assertEqual([1,1,1],engine.calls)
        self.assertEqual(3,res['ocr_calls'])
        self.assertAlmostEqual(sum(x['ocr_call_ms'] for x in res['calls']),res['sum_cell_ocr_ms'],delta=.001)
        self.assertIsNone(res['sum_batch_ocr_ms'])
    def test_batch_groups(self):
        engine=FakeRecognizer();cells=[(str(i),object()) for i in range(7)]
        res=recognize_cells(cells,engine,'batch',batch_size=3)
        self.assertEqual([3,3,1],engine.calls)
        self.assertEqual(3,res['ocr_calls'])
        self.assertIsNone(res['sum_cell_ocr_ms'])
    def test_percentile(self):
        self.assertEqual(5,summarize([1,2,3,4,5])['p95'])
    def test_v6_cyrillic_errors_are_scored(self):
        gt=read_ground_truth()['short_01_2142701072']
        metrics=evaluate(gt,{},script_supported=False)
        texts=[c for c in metrics['cell_metrics'] if c['type']=='text']
        self.assertTrue(any(not c['model_script_supported'] for c in texts))
        self.assertGreater(metrics['text_scored_fields'],0)
        self.assertGreater(metrics['text_cer'],0)
    def test_numeric_roi_is_tightened(self):
        img=np.full((120,80,3),255,np.uint8)
        cv2.rectangle(img,(1,1),(78,118),(0,0,0),1)
        cv2.putText(img,'42',(20,75),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),2,cv2.LINE_AA)
        out=tighten_numeric_crop(img)
        self.assertLess(out.shape[0],img.shape[0]*.8)
        self.assertLess(out.shape[1],img.shape[1]*.9)
    def test_multiline_name_is_split(self):
        img=np.full((150,260,3),255,np.uint8)
        cv2.putText(img,'First line',(5,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,'Second line',(5,110),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2,cv2.LINE_AA)
        self.assertGreaterEqual(len(split_name_lines(img)),2)
    def test_name_hyphenation_is_joined(self):
        self.assertEqual('Тачка садовая одноколесная',join_text_parts(['Тачка садовая одно-','колесная']))
        self.assertEqual('ПФ-115',join_text_parts(['ПФ-','115']))
    def test_extract_rois_splits_names_and_keeps_numeric_field(self):
        img=np.full((150,260,3),255,np.uint8)
        cv2.putText(img,'First line',(5,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,'Second line',(5,110),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2,cv2.LINE_AA)
        ann={'rois':[{'field_id':'row:1:name','bbox':[0,0,1,1]},{'field_id':'row:1:quantity','bbox':[0,0,1,1]}]}
        cells=extract_rois(img,ann)
        self.assertGreaterEqual(sum(f=='row:1:name' for f,_ in cells),2)
        self.assertEqual(1,sum(f=='row:1:quantity' for f,_ in cells))
    def test_agent_verified_names_are_scored(self):
        gt=read_ground_truth()['short_01_2142701072']
        expected=gt_fields(gt)['row:1:name']['value']
        metrics=evaluate(gt,{'row:1:name':expected},script_supported=True)
        statuses={c['status'] for c in metrics['cell_metrics'] if c['type']=='text'}
        self.assertIn('SCORED_AGENT_VERIFIED',statuses)
        self.assertEqual('AGENT_VERIFIED',metrics['text_quality_status'])

if __name__=='__main__':unittest.main()

class FullReceiptTests(unittest.TestCase):
    def test_all_product_names_scored_in_every_active_receipt(self):
        for gt in read_ground_truth().values():
            texts={f:m['value'] for f,m in gt_fields(gt).items()}
            quality=evaluate(gt,texts)
            self.assertEqual(len(gt['products']),quality['product_names_scored'])
            self.assertEqual(0,quality['product_name_cer'])
            self.assertTrue(quality['complete_receipt_correct'])
            self.assertTrue(quality['arithmetic_matches_printed_total'])
    def test_mixed_alphabets_and_total_errors_are_not_hidden(self):
        gt=read_ground_truth()['medium_08_2142830502']
        original={f:m['value'] for f,m in gt_fields(gt).items()}
        for field,text in [('row:1:name',original['row:1:name'].replace('SchnEl','SchnXX')),
                           ('row:1:name',original['row:1:name'].replace('Автомат','Автомот')),
                           ('printed_total','1')]:
            with self.subTest(text=text):
                values=dict(original);values[field]=text
                q=evaluate(gt,values)
                self.assertFalse(q['complete_receipt_correct'])
                if field.endswith(':name'):self.assertGreater(q['product_name_cer'],0)
                else:self.assertFalse(q['arithmetic_matches_printed_total'])
    def test_numeric_only_does_not_claim_full_receipt_success(self):
        gt=read_ground_truth()['medium_08_2142830502']
        values={f:m['value'] for f,m in gt_fields(gt).items() if m['type']=='numeric'}
        q=evaluate(gt,values,script_supported=False,field_scope='numeric_only')
        self.assertEqual(1,q['numeric_accuracy'])
        self.assertEqual(0,q['text_scored_fields'])
        self.assertIsNone(q['product_name_cer'])
        self.assertFalse(q['complete_receipt_correct'])
