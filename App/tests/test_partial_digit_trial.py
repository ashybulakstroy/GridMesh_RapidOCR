from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np

from App.ocr.experiment_matrix import CandidateBook
from App.ocr.p1_numbers import parse_number
from App.ocr.partial_digit_lab.partial_digit_trial import segment, trial


def test_segment_ignores_disconnected_grid_fragments_above_text():
    image = np.full((60, 100), 255, np.uint8)
    cv2.putText(image, '390', (30, 38), cv2.FONT_HERSHEY_SIMPLEX, .7, 0, 2)
    image[0, 4:9] = 0
    image[0:7, 0:2] = 0
    assert any(len(slots) == 3 for slots, _, _ in segment(image, '39O'))


def test_shadow_marks_truncated_candidates_and_preserves_originals():
    rows = [{'row_id': 1, 'kind': 'product', 'cells': {'price': {'text': '39O'}}}]
    original = deepcopy(rows)
    book = CandidateBook()
    image = np.full((20, 40), 255, np.uint8)

    def solve(_rows, shadow, _rate, **_kwargs):
        assert shadow is not book
        assert shadow.cells
        return {'status': 'unique_math_match', 'review_required': False, 'solutions': [{}]}

    candidate_result = {'candidates': [{'text': '390', 'scope': 'free_beam'}],
                        'segments': [], 'budget_exhausted': False, 'pool_truncated': True}
    with patch('App.ocr.partial_digit_lab.partial_digit_trial.load_bank'), \
         patch('App.ocr.partial_digit_lab.partial_digit_trial.candidates', return_value=candidate_result):
        result = trial(rows, {(1, 'price'): image}, book, 16, solve, parse_number,
                       lambda crop, field: (crop, {'currency_removed_before_ocr': True}))

    assert rows == original
    assert not book.cells
    assert result['matrix']['review_required']
    assert result['matrix']['status'] == 'bounded_search'
    assert result['matrix']['candidate_pool_truncated']
