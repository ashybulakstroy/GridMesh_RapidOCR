"""Run: .venv/Scripts/python scripts/e4_vat_reading_fix_v1/test_vat_fix.py

Tests direct VAT evidence, candidate recovery, and strict integration patches.
In a repository checkout this imports the actual project parser. Offline tests
use a strict stub parser only for numeric fixture values.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import importlib.util
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_witness():
    target = ROOT / 'App' / 'ocr' / 'vat_witness.py'
    try:
        if target.exists():
            sys.path.insert(0, str(ROOT))
            from App.ocr import vat_witness
            return vat_witness
    except ImportError:
        pass
    # The package can be tested without cloning the GitHub repository.
    pkg_app = types.ModuleType('App')
    pkg_app.__path__ = []
    pkg_ocr = types.ModuleType('App.ocr')
    pkg_ocr.__path__ = []
    pkg_numbers = types.ModuleType('App.ocr.p1_numbers')

    def parse_number(value):
        if value is None:return None
        raw=str(value).strip().replace(' ', '')
        if raw.endswith(('₸', 'Т', 'T', 'т')):
            raw = raw[:-1]
        if not raw or not raw.replace('.', '', 1).isdigit():return None
        try:return Decimal(raw)
        except Exception:return None

    pkg_numbers.parse_number=parse_number
    sys.modules.update({'App':pkg_app,'App.ocr':pkg_ocr,'App.ocr.p1_numbers':pkg_numbers})
    spec=importlib.util.spec_from_file_location('App.ocr.vat_witness',HERE/'vat_witness.py')
    obj=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=obj
    spec.loader.exec_module(obj)
    return obj

v = load_witness()
spec = importlib.util.spec_from_file_location('e4_vat_apply_patch', HERE / 'apply_patch.py')
patcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class FakeBook:
    def __init__(self):
        self.cells={}
        self.vat=[]
    @staticmethod
    def key(rid,field):return f'{rid}:{field}'
    def add(self,rid,field,text,conf=.9,path='candidate',model='v6_tiny_official',user=False):
        key=self.key(rid,field)
        bucket=self.cells.setdefault(key,{'observations':[]})
        value=v.parse_number(text)
        obs={'text':text,'number':str(value) if value is not None else None,
             'confidence':conf,'path':path,'model':model,
             'authority':'user_confirmed' if user else 'ocr_candidate'}
        bucket['observations'].append(obs)
        if user:bucket.update(user_locked=True,user_value=str(value))
    def add_vat(self,text,confidence,path,*,user=False,model=None,row_id=None,field=None):
        value=v.parse_number(text)
        item={'text':text,'number':str(value) if value is not None else None,
              'confidence':confidence,'path':path,'model':model,
              'row_id':row_id,'field':field,
              'authority':'user_confirmed' if user else 'ocr_candidate'}
        if not any((x['text'],x['path'],x['model'],x['authority'],x['row_id'],x['field']) ==
                   (item['text'],item['path'],item['model'],item['authority'],item['row_id'],item['field'])
                   for x in self.vat):
            self.vat.append(item)


def receipt(q='23200',p='1',printed=None,q_derived=False):
    cells={'quantity':{'text':q,'confidence':.95,'model_id':'v6_tiny_official', 'derived':q_derived},
           'price':{'text':p,'confidence':.98,'model_id':'v6_tiny_official'}}
    if printed is not None:
        cells['printed_total']={'text':printed,'confidence':.6,'model_id':'v6_tiny_official'}
    return [{'row_id':1,'kind':'product','cells':{'quantity':{'text':'5'},'price':{'text':'29000'}}},
            {'row_id':2,'kind':'tax','cells':cells}]


class TestVATWitness(unittest.TestCase):
    def test_photo_23200_times_one_direct(self):
        rows=receipt()
        self.assertEqual(v.vat_source(rows),((2,'quantity'),'single_unit_tax_quantity'))
        result=v.seed_printed_vat(rows,FakeBook())
        self.assertEqual((result['source']['field'],result['number']),('quantity','23200'))

    def test_blank_printed_total_does_not_hide_real_vat(self):
        rows=receipt(printed='')
        book=FakeBook()
        seed=v.seed_printed_vat(rows,book)
        self.assertEqual(seed['number'],'23200')
        self.assertEqual(seed['source']['field'],'quantity')
        self.assertEqual(v.observed_vat_text(book),'23200')

    def test_non_numeric_printed_total_does_not_hide_real_vat(self):
        seed=v.seed_printed_vat(receipt(printed='no text'),FakeBook())
        self.assertEqual(seed['number'],'23200')

    def test_valid_printed_total_takes_priority(self):
        seed=v.seed_printed_vat(receipt(printed='23200'),FakeBook())
        self.assertEqual(seed['source']['field'],'printed_total')

    def test_inverse_unit_column(self):
        seed=v.seed_printed_vat(receipt(q='1',p='23200'),FakeBook())
        self.assertEqual((seed['source']['field'],seed['number']),('price','23200'))

    def test_no_guess_from_nominal_vat_rate(self):
        result=v.seed_printed_vat(receipt(q='23200',p='2'),FakeBook())
        self.assertIsNone(result['source'])
        self.assertIsNone(result['number'])

    def test_more_than_one_tax_row_does_not_invent_vat(self):
        rows=receipt(); rows.append({'row_id':3,'kind':'tax','cells':rows[-1]['cells']})
        self.assertIsNone(v.seed_printed_vat(rows,FakeBook())['source'])

    def test_derived_amount_cannot_be_photo_witness(self):
        self.assertIsNone(v.seed_printed_vat(receipt(q_derived=True),FakeBook())['source'])

    def test_candidate_vat_after_weak_ocr(self):
        rows=receipt(q='bad',p='1',printed='unreadable')
        book=FakeBook()
        book.add(2,'quantity','bad')
        book.add(2,'price','1')
        self.assertIsNone(v.seed_printed_vat(rows,book)['source'])
        book.add(2,'quantity','23200',.89)
        result=v.seed_printed_vat(rows,book)
        self.assertEqual(result['number'],'23200')
        self.assertEqual(result['status'],'candidate_single_unit_tax_quantity')

    def test_price_unit_recognized_in_candidate_only(self):
        rows=receipt(q='23200',p='l')
        book=FakeBook()
        book.add(2,'quantity','23200')
        book.add(2,'price','1',.95)
        seed=v.seed_printed_vat(rows,book)
        self.assertEqual(seed['number'],'23200')

    def test_candidate_printed_total_even_when_primary_unreadable(self):
        rows=receipt(q='broken',p='broken',printed='')
        book=FakeBook();book.add(2,'printed_total','23200',.9)
        seed=v.seed_printed_vat(rows,book)
        self.assertEqual(seed['number'],'23200')
        self.assertEqual(seed['source']['field'],'printed_total')

    def test_primary_plus_alternative_preserve_both(self):
        rows=receipt()
        book=FakeBook();book.add(2,'quantity','2320',.4);book.add(2,'quantity','23200',.9)
        v.seed_printed_vat(rows,book)
        self.assertEqual({x['number'] for x in book.vat},{'23200','2320'})
        self.assertEqual(v.observed_vat_text(book,preferred='23200'),'23200')

    def test_user_lock_prevents_obsolete_vat_alternative(self):
        rows=receipt()
        book=FakeBook();book.add(2,'quantity','2320',.5)
        book.add(2,'quantity','23200',None,user=True)
        v.seed_printed_vat(rows,book)
        self.assertEqual({x['number'] for x in book.vat},{'23200'})

    def test_manual_evidence_supersedes_ocr_value(self):
        book=FakeBook();book.add_vat('2320',.99,'ocr',model='v5')
        book.add_vat('23200',None,'web',model='user',user=True)
        self.assertEqual(v.observed_vat_text(book,preferred='2320'),'23200')

    def test_invalid_first_vat_candidate_is_never_displayed(self):
        book=FakeBook();book.add_vat('T23OO',.99,'bad')
        book.add_vat('23200',.75,'good')
        self.assertEqual(v.observed_vat_text(book),'23200')

    def test_no_vat_witness_returns_none(self):
        self.assertIsNone(v.observed_vat_text(FakeBook()))

    def test_original_ocr_currency_suffix(self):
        result=v.seed_printed_vat(receipt(q='23200T',p='1'),FakeBook())
        self.assertEqual(result['number'],'23200')

    def test_patcher_rejects_unknown_source(self):
        with self.assertRaises(RuntimeError):patcher.patch_vat('hello')
        with self.assertRaises(RuntimeError):patcher.patch_pipeline('hello')
        with self.assertRaises(RuntimeError):patcher.patch_early('hello')

    def test_patcher_replaces_only_exact_known_anchors(self):
        original=''.join(['from .experiment_rec_trace import rec_scope\n',
                          'def vat_source(rows):\n    pass\n',
                          'def seed_printed_vat(rows, book):\n    pass\n',
                          'def _prep(image, mode):\n    pass\n'])
        new=patcher.patch_vat(original)
        self.assertIn('from .vat_witness import vat_source, seed_printed_vat, observed_vat_text',new)
        self.assertNotIn('def vat_source(rows):',new)
        new2=patcher.patch_early('                if _observed_numbers(book, key) - old_numbers:\n')
        self.assertIn('seed_printed_vat(rows, book)',new2)

    def test_blob_sha_recognizes_contents(self):
        self.assertEqual(patcher.git_blob(b'test'),'30d74d258442c7c65512eafab474568dd706c430')


    def test_e4_auto_correction_does_not_forge_primary_ocr_model(self):
        rows=receipt()
        rows[1]['cells']['quantity'].update(auto_corrected=True,model_id='v6_tiny_official')
        book=FakeBook()
        book.add(2,'quantity','23200',.91,path='v5_numeric_candidate',model='v5_cyrillic_mobile')
        result=v.seed_printed_vat(rows,book)
        self.assertEqual(result['number'],'23200')
        self.assertTrue(all(rec['model']=='v5_cyrillic_mobile' for rec in book.vat))
        self.assertTrue(all(rec['path'] != 'vat_row_primary_quantity' for rec in book.vat))

    def test_e4_correction_without_ocr_witness_is_not_invented(self):
        rows=receipt()
        rows[1]['cells']['quantity']['auto_corrected']=True
        book=FakeBook()
        result=v.seed_printed_vat(rows,book)
        self.assertIsNone(result['number'])
        self.assertEqual(book.vat, [])

    def test_pipeline_patch_anchors_match_and_compile(self):
        template="""from .ocr.experiment_vat import scan_printed_vat, seed_printed_vat
            if book.vat:
                printed_vat_observed=book.vat[0]['text']
                if book.vat:
                    printed_vat_observed=book.vat[0]['text']
            numeric_auto_corrections = auto_apply_consensus(
                rows,book,numeric_matrix,total)
                total=numeric_auto_corrections['printed_total']
                printed_vat_observed=(numeric_matrix['solutions'][0].get('vat_printed') or printed_vat_observed)
                numeric_matrix = solve_matrix(rows, book, vat_rate,
                geometry['warnings'].append('Нет независимого OCR-подтверждения суммы НДС; проверены только значения ячеек и общий итог.')
"""
        modified=patcher.patch_pipeline(template)
        self.assertIn('post_e4_vat_seed = seed_printed_vat(rows, book)',modified)
        self.assertIn('observed_vat_text',modified)
        self.assertNotIn('book.vat[0]',modified)

if __name__ == '__main__':
    unittest.main(verbosity=2)
