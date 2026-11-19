from pathlib import Path
import pytest
import cv2
import numpy as np

from App.pipeline import is_footer_label, number, process
from App.ocr.engine import read_full
import App.pipeline as pipeline

HERE = Path(__file__).resolve().parent

@pytest.mark.parametrize(('text','expected'), [
    ('2', '2'),
    ('4 000,000', '4000.000'),
    ('1.00', '1.00'),
    ('300 T', '300'),
    ('₸ 1 250', '1250'),
])
def test_number_accepts_valid_numeric_formats(text, expected):
    assert str(number(text)) == expected

@pytest.mark.parametrize('text', ['S07', '760тx', '12abc', '1,2,3', '*', ''])
def test_number_rejects_invalid_text_without_discarding_characters(text):
    assert number(text) is None

@pytest.mark.parametrize('text', ['Қорытынды:', 'Корытынды:', 'Жиыны:', 'Барлығы:', 'Итого:', 'Всего', 'Total:'])
def test_total_labels_are_recognized(text):
    assert is_footer_label(text)

@pytest.mark.parametrize('text', ['Итого раствор', 'Барлык цемент', 'НДС 16 %', 'subtotal'])
def test_product_and_tax_labels_are_not_total_markers(text):
    assert not is_footer_label(text)

def test_user_receipt_recovers_visible_row_boundary():
    photo = HERE.parent.parent / 'data' / 'output' / 'f0a67abf710b44beb2d33377f0e4d496' / 'input.bin'
    if not photo.is_file():
        pytest.skip('Локальное фото пользователя отсутствует')
    result = process(photo.read_bytes())
    assert [(r['kind'], r['cells']['quantity']['text'], number(r['cells']['price']['text']))
            for r in result['rows']] == [('product', '2', 20200), ('tax', '6464', 1)]
    assert result['computed_total'] == '46864'
    assert result['total'] == '46864.00'
    assert result['math_match']

def test_recovers_tax_row_below_detected_grid():
    photo = HERE.parent.parent / 'data' / 'output' / '5dbab730f56d4a959d6b0bd0e32d4ad4' / 'input.bin'
    if not photo.is_file():
        pytest.skip('Локальное фото пользователя отсутствует')
    result = process(photo.read_bytes())
    taxes = [row for row in result['rows'] if row['kind'] == 'tax']
    assert len(taxes) == 1
    assert number(taxes[0]['cells']['quantity']['text']) == 3005
    assert number(taxes[0]['cells']['price']['text']) == 1
    assert taxes[0]['cells']['name']['strategy'] == 'visible_tax_below_grid'
    assert result['total'] == '21785.00'
    assert number(result['rows'][4]['cells']['price']['text']) == 1800
    assert number(result['rows'][5]['cells']['price']['text']) == 5000
    assert result['computed_total'] == '21785'
    assert is_footer_label(result['geometry']['footer_marker']['text'])
    assert result['status'] == 'review'

@pytest.mark.parametrize('failure', ['empty', 'exception'])
def test_footer_detector_falls_back_when_fast_detector_fails(monkeypatch, failure):
    real_read_full = pipeline.read_full
    calls = []
    def read_with_failed_fast(image, *, footer_mode=False):
        if footer_mode:
            calls.append('fast')
            if failure == 'exception':
                raise RuntimeError('detector unavailable')
            return []
        calls.append('standard')
        return real_read_full(image)
    monkeypatch.setattr(pipeline, 'read_full', read_with_failed_fast)
    result = process((HERE/'receipt_2142830502.jpg').read_bytes())
    assert 'fast' in calls and 'standard' in calls
    assert result['total'] == '93612.00'
    assert any(a.get('strategy') == 'standard_detector' for a in result['footer_ocr_attempts'])

@pytest.mark.parametrize(('name','expected','products'), [
    ('receipt_5730.png', '5730.00', 2),
    ('receipt_575866.png', '45936.00', 3),
    ('receipt_597275.png', '23200.00', 2),
])
def test_real_receipt_geometry_and_math(name, expected, products):
    result = process((HERE/name).read_bytes())
    assert result['status'] in {'ok', 'review'}
    assert result['math_match']
    assert result['total'] == expected
    assert len([r for r in result['rows'] if r['kind'] == 'product']) == products
    assert len([r for r in result['rows'] if r['kind'] == 'tax']) == 1
    if name == 'receipt_597275.png':
        assert 'ботинки' in result['rows'][0]['cells']['name']['text'].casefold()
        assert 'bts' in result['rows'][0]['cells']['name']['text'].casefold()
        assert result['rows'][0]['cells']['quantity']['text'] == '1'
        assert result['rows'][1]['cells']['quantity']['text'] == '1'
        assert result['vat']['status'] == 'match'
    if name == 'receipt_575866.png':
        assert result['rows'][-1]['cells']['quantity']['text'] == '6336'
        assert 'tulex' in result['rows'][0]['cells']['name']['text'].casefold()
        assert 'ct15074' in result['rows'][1]['cells']['name']['text'].casefold()
    if name == 'receipt_5730.png':
        assert 'startul' in result['rows'][0]['cells']['name']['text'].casefold()
        assert 'st9084-50' in result['rows'][0]['cells']['name']['text'].casefold()
    assert all(r['cells']['name']['source_bbox'] for r in result['rows'])
    assert Path(result['geometry']['inverse_map']).is_file()
    directory = Path(result['job_directory'])
    for name in ('input.bin', 'original.png', 'rectified.png', 'receipt.json', 'performance_profile.json', 'rectification_map.npz'):
        assert (directory/name).is_file()

def test_problem_geometry_requires_review():
    result = process((HERE/'receipt_wheels.png').read_bytes())
    assert result['status'] == 'review'
    assert result['geometry']['warnings']
    assert result['geometry']['method'] == 'AutoGrid GridRefine V3'
    assert result['geometry']['safety']['status'] == 'MEASURED'
    assert result['geometry']['safety']['source_valid_fraction'] < .995
    assert result['total'] == '25346.00'
    assert number(result['rows'][2]['cells']['price']['text']) == 6350
    assert result['math_match']
    assert any(a.get('strategy') == 'compact_line' for a in result['footer_ocr_attempts'])

def test_six_column_receipt_uses_read_header_to_map_columns():
    result = process((HERE/'receipt_six_columns.jpg').read_bytes())
    assert result['geometry']['columns'] == {'number':0, 'name':1, 'price':3, 'quantity':4, 'printed_total':5}
    assert len(result['rows']) == 4
    # This low-quality image still has OCR errors, so it must not claim success.
    assert result['status'] == 'review'

def test_rotated_receipt_is_oriented_and_processed():
    source = cv2.imread(str(HERE/'receipt_5730.png'))
    ok, encoded = cv2.imencode('.png', cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE))
    assert ok
    result = process(encoded.tobytes())
    assert result['math_match']
    assert result['total'] == '5730.00'

def test_blank_image_does_not_fabricate_grid():
    ok, encoded = cv2.imencode('.png', np.full((500,400,3), 255, np.uint8))
    assert ok
    with pytest.raises(ValueError, match='сетк'):
        process(encoded.tobytes())

def test_cyrillic_latin_and_mixed_model():
    image = cv2.imread(str(HERE/'mixed_text.png'))
    text = ' '.join(r['text'] for r in read_full(image)).casefold()
    for word in ('клей', 'мдф', '400ml', 'ga065b', 'щетка', 'stayer', 'universal', 'труба', 'ppr', 'jakko'):
        assert word in text


def test_complete_grid_and_printed_total_on_receipt_2142830460():
    result = process((HERE/'receipt_2142830460.jpg').read_bytes())
    assert len(result['rows']) == 14
    assert len([r for r in result['rows'] if r['kind'] == 'tax']) == 1
    assert result['total'] == '38825.00'
    assert result['status'] == 'review'
    assert not result['math_match']
    assert result['geometry']['method'] == 'AutoGrid GridRefine V3 + Physical Grid Graph V3'


def test_fallback_when_detected_grid_has_duplicate_boundaries():
    result = process((HERE/'receipt_2142830502.jpg').read_bytes())
    assert result['geometry']['method'] == 'Physical Grid Graph V3'
    assert len(result['rows']) == 9
    assert len([r for r in result['rows'] if r['kind'] == 'tax']) == 1
    assert result['status'] == 'review'
    assert result['total'] == '93612.00'
