import unittest

from App.ocr.e4_math_target import select_math_targets
from App.ocr.experiment_auto import auto_apply_consensus
from App.ocr.experiment_matrix import CandidateBook, solve_matrix


class TargetedE5Test(unittest.TestCase):
    def test_math_selects_location_without_inventing_candidate(self):
        rows = [
            {'row_id':1, 'kind':'product', 'cells':{'quantity':{'text':'2'}, 'price':{'text':'12500'}}},
            {'row_id':2, 'kind':'product', 'cells':{'quantity':{'text':'24'}, 'price':{'text':'4.10'}}},
            {'row_id':3, 'kind':'tax', 'cells':{'quantity':{'text':'5690'}, 'price':{'text':'1'}}},
        ]
        targets = select_math_targets(rows, '41250')
        self.assertEqual((targets[0]['row_id'], targets[0]['field']), (2, 'price'))
        self.assertNotIn('value', targets[0])
        self.assertEqual(rows[1]['cells']['price']['text'], '4.10')

    def test_ambiguous_solutions_do_not_change_price(self):
        row = {'row_id':1, 'kind':'product', 'cells':{'quantity':{'text':'1'}, 'price':{'text':'100'}}}
        book = CandidateBook()
        book.add(1, 'quantity', '1', .9, 'ocr')
        for value in ('100', '200'):
            book.add(1, 'price', value, .9, value)
            book.add_footer(value, .9, value)
        matrix = solve_matrix([row], book, 16)
        result = auto_apply_consensus([row], book, matrix, '100')
        self.assertEqual(result['status'], 'unresolved_preserved')
        self.assertEqual(row['cells']['price']['text'], '100')

    def test_unscored_tesseract_evidence_preserves_unknown_confidence(self):
        row = {'row_id':1, 'kind':'product', 'cells':{'quantity':{'text':'1'}, 'price':{'text':'4.10'}}}
        book = CandidateBook()
        book.add(1, 'quantity', '1', .9, 'ocr')
        book.add(1, 'price', '440', None, 'e5_3x_plain', model='tesseract_kaz')
        book.add_footer('440', .9, 'ocr')
        matrix = solve_matrix([row], book, 16)
        result = auto_apply_consensus([row], book, matrix, '440')
        self.assertEqual(result['status'], 'applied')
        self.assertIsNone(row['cells']['price']['confidence'])
        self.assertEqual(row['cells']['price']['auto_ocr_evidence'][0]['model'], 'tesseract_kaz')


if __name__ == '__main__':
    unittest.main()
