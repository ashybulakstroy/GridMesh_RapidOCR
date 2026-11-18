from pathlib import Path
import pytest
import cv2
import numpy as np

from App.pipeline import process
from App.ocr.engine import read_full

HERE = Path(__file__).resolve().parent

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
    assert not result['math_match']
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
