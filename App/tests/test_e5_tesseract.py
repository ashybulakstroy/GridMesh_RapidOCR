import unittest
from unittest.mock import patch

import numpy as np

from App.ocr import e5_tesseract
from App.ocr.experiment_matrix import CandidateBook


def row(value='1E50', printed=None):
    cells = {field: {'text':text, 'confidence':.8, 'row_id':1, 'column_id':field,
                     'bbox':[0, 0, 40, 20]}
             for field, text in [('quantity', '1'), ('price', value)]}
    if printed is not None:
        cells['printed_total'] = {'text':printed, 'confidence':.9, 'row_id':1,
                                  'column_id':'printed_total', 'bbox':[0, 0, 40, 20]}
    return {'row_id':1, 'cells':cells}


class TesseractFallbackTest(unittest.TestCase):
    def run_recovery(self, item, results, recovery=None, **kwargs):
        def read(*args, **kwargs):
            text = results.pop(0)
            return {'text':text, 'number':text, 'error':None,
                    'variant':kwargs['variant'], 'wall_ms':1}
        with patch.object(e5_tesseract, 'executable', return_value='tesseract'), \
             patch.object(e5_tesseract, 'read_tesseract', side_effect=read), \
             patch.object(e5_tesseract, 'prepare_numeric_input', side_effect=lambda img, field:(img, {})):
            return e5_tesseract.recover_with_tesseract(
                [item], {(1, 'price'):np.full((20,40,3),255,np.uint8)}, recovery or {}, **kwargs)

    def test_invalid_unanimous_retry_is_accepted_without_row_total(self):
        item = row()
        result = self.run_recovery(item, ['1550', '1550'])
        self.assertEqual(item['cells']['price']['text'], '1550')
        self.assertEqual(item['line_total'], '1550')
        self.assertTrue(result['details'][0]['accepted'])

    def test_math_confirmed_first_read_skips_second_presentation(self):
        item = row()
        item['kind'] = 'product'
        result = self.run_recovery(item, ['1550'], printed_total='1550',
                                   footer_attempts=[{'total':'1550', 'confidence':.95}])
        self.assertEqual(item['cells']['price']['text'], '1550')
        self.assertEqual(len(result['details'][0]['attempts']), 1)
        self.assertEqual(result['details'][0]['confirmation'], 'single_observation_and_receipt_math')

    def test_fractional_price_runs_both_tesseract_presentations(self):
        item = row(value='2.20')
        result = self.run_recovery(item, ['220', '220'], force_keys=[(1, 'price')],
                                   integer_price_keys=[(1, 'price')], accept_invalid=False)
        self.assertEqual([a['variant'] for a in result['details'][0]['attempts']],
                         ['raw_digits', '3x_plain'])
        self.assertEqual([a['text'] for a in result['candidates']], ['220', '220'])
        self.assertEqual(item['cells']['price']['text'], '2.20')

    def test_fractional_price_tries_binarized_image_when_no_integer_found(self):
        item = row(value='2.20')
        result = self.run_recovery(item, ['2.20', '2.20', '220'],
                                   force_keys=[(1, 'price')],
                                   integer_price_keys=[(1, 'price')], accept_invalid=False)
        self.assertEqual([a['variant'] for a in result['details'][0]['attempts']],
                         ['raw_digits', '3x_plain', '3x_otsu'])
        self.assertEqual(result['candidates'][-1]['text'], '220')

    def test_matrix_excludes_fractional_ocr_price_but_preserves_user_lock(self):
        book = CandidateBook()
        book.add(1, 'price', '2.20', .99, 'initial')
        book.add(1, 'price', '220', None, 'tesseract')
        self.assertEqual([x['number'] for x in book.possible_values(1, 'price')], ['220'])
        self.assertEqual(len(book.cells['1:price']['observations']), 2)
        book.add(1, 'price', '2.20', None, 'user', user=True)
        self.assertEqual([x['number'] for x in book.possible_values(1, 'price')], ['2.2'])

    def test_disagreeing_readings_are_not_accepted(self):
        item = row()
        result = self.run_recovery(item, ['1550', '15507'])
        self.assertEqual(item['cells']['price']['text'], '1E50')
        self.assertFalse(result['details'][0]['accepted'])

    def test_printed_row_total_must_agree(self):
        item = row(printed='1600')
        result = self.run_recovery(item, ['1550', '1550'])
        self.assertEqual(item['cells']['price']['text'], '1E50')
        self.assertFalse(result['details'][0]['accepted'])

    def test_numeric_conflict_is_candidate_only(self):
        item = row(value='4.10')
        recovery = {'footer': {'details':[{'row_id':1, 'column_id':'price',
                     'original_text':'4.10', 'alternative_text':'410'}]}}
        result = self.run_recovery(item, ['410', '410'], recovery)
        self.assertEqual(item['cells']['price']['text'], '4.10')
        self.assertEqual(result['candidates'][0]['text'], '410')
        self.assertFalse(result['details'][0]['accepted'])

    def test_missing_engine_keeps_original(self):
        item = row()
        with patch.object(e5_tesseract, 'executable', return_value=None):
            result = e5_tesseract.recover_with_tesseract(
                [item], {(1, 'price'):np.zeros((20, 40, 3), np.uint8)}, {})
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(item['cells']['price']['text'], '1E50')

    def test_e4_nonnumeric_observation_triggers_e5_for_valid_initial_value(self):
        item = row(value='4.10')
        book = CandidateBook()
        book.add(1, 'price', '4.10', .94, 'initial')
        book.add(1, 'price', '4:0', .8, 'source_pixel_recovery')
        result = self.run_recovery(item, ['40', '440'], book=book, accept_invalid=False)
        self.assertEqual([c['text'] for c in result['candidates']], ['40', '440'])
        self.assertEqual(item['cells']['price']['text'], '4.10')

    def test_e4_valid_observations_do_not_trigger_e5(self):
        item = row(value='4.10')
        book = CandidateBook()
        book.add(1, 'price', '4.10', .94, 'initial')
        result = self.run_recovery(item, [], book=book)
        self.assertEqual(result['status'], 'not_needed')


if __name__ == '__main__':
    unittest.main()
