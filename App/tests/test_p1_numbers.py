import pytest
from decimal import Decimal

from App.ocr.p1_numbers import parse_number


@pytest.mark.parametrize(('text', 'expected'), [
    ('1234', '1234'), ('1234,50', '1234.50'), ('5730.00', '5730.00'),
    ('1 234,50', '1234.50'), ('1\u00a0234,50', '1234.50'),
    ('1.234,50', '1234.50'), ('1,234.50', '1234.50'),
    ('1.234.567', '1234567'), ('1 234 567', '1234567'),
    ('2 ₸', '2'), ('₸ 1 250', '1250'), ('6336Т', '6336'), ('1,5', '1.5'),
])
def test_valid_numbers(text, expected):
    assert parse_number(text) == Decimal(expected)


@pytest.mark.parametrize('text', [
    '', '12/34', 'S07', '1O0', '4а0', '1 23', '12 34,00', '1.234',
    '1,234', '1.2.3', '1,2,3', '-12', 'ten', '12abc', '1e3',
    '12..00', '12,345,67', None,
])
def test_invalid_numbers_never_silently_become_valid(text):
    assert parse_number(text) is None
