import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from App.ocr.e4_fast_gate import primary_consistency
from App.ocr.numeric_multiline_fast import read_numeric_multiline


class MultilineVatFastTest(unittest.TestCase):
    def test_join_only_observed_lines(self):
        image = np.full((60, 130, 3), 255, np.uint8)
        cv2.putText(image, '10627', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        cv2.putText(image, '5', (3, 53), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        values = iter(['10627', '5'])
        result = read_numeric_multiline(image, lambda _: SimpleNamespace(txts=[next(values)], scores=[.99]))
        self.assertEqual(result['text'], '106275')
        self.assertEqual([line['text'] for line in result['lines']], ['10627', '5'])

    def test_invalid_line_falls_back(self):
        image = np.full((60, 130, 3), 255, np.uint8)
        cv2.putText(image, '10627', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        cv2.putText(image, '5', (3, 53), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        values = iter(['10627', 'X'])
        result = read_numeric_multiline(image, lambda _: SimpleNamespace(txts=[next(values)], scores=[.99]))
        self.assertIsNone(result)

    def test_gate_requires_observed_footer_and_matching_sum(self):
        rows = [{'row_id':1, 'cells':{'quantity':{'text':'2', 'confidence':.9},
                                      'price':{'text':'10', 'confidence':.9}}}]
        self.assertFalse(primary_consistency(rows, '20', [])['eligible'])
        self.assertTrue(primary_consistency(rows, '20', [{'total':'20', 'confidence':.95}])['eligible'])
        self.assertFalse(primary_consistency(rows, '21', [{'total':'21', 'confidence':.95}])['eligible'])


if __name__ == '__main__':
    unittest.main()
