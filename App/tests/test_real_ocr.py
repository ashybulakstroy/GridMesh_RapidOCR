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
    monkeypatch.setenv('OCR_FOOTER_FOCUS_ENABLE','0')  # Explicitly exercise legacy detector fallback.
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
    assert number(result['total']) == 93612
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

def test_cyrillic_latin_and_mixed_model(monkeypatch):
    # English OCR is optional: keep this test as an explicit opt-in regression.
    monkeypatch.setenv('OCR_NAME_ENGLISH_FALLBACK','1')
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


def test_long_receipt_keeps_first_product_and_separates_sixth_seventh():
    result = process((HERE/'fixtures'/'receipt_long_grid.png').read_bytes())
    rows = result['rows']
    assert len(rows) == 30
    assert sum(row['kind'] == 'product' for row in rows) == 29
    assert rows[-1]['kind'] == 'tax'
    assert result['geometry']['ridge_refined_horizontal'] == 2
    for index, quantity, price in ((1,100,390),(6,10,150),(7,50,2250)):
        row = rows[index-1]
        assert number(row['cells']['quantity']['text']) == quantity
        assert number(row['cells']['price']['text']) == price
        assert row['cells']['number']['text'] == str(index)
    sixth = np.asarray(rows[5]['cells']['price']['source_bbox'])
    seventh = np.asarray(rows[6]['cells']['price']['source_bbox'])
    assert sixth[:,1].max() < seventh[:,1].min()
    assert number(rows[15]['cells']['price']['text']) == 380
    assert number(rows[-1]['cells']['price']['text']) == 1
    assert result['total'] == '538936.00'
    # Remaining numerical discrepancies must remain visible, never force balance.
    assert not result['math_match']


def test_price_1120_keeps_both_leading_ones():
    from App.ocr.numeric_roi import crop_numeric_ink
    from App.ocr.engine import read_line
    image = cv2.imread(str(HERE/'fixtures'/'price_1120_border.png'))
    crop, info = crop_numeric_ink(image)
    assert info['changed']
    assert crop.shape[0] < image.shape[0]*.5
    assert number(read_line(crop)['text']) == 1120


def test_receipt_29464_preserves_price_1120_and_matches_printed_total():
    result = process((HERE/'fixtures'/'receipt_29464.jpg').read_bytes())
    assert len(result['rows']) == 13
    price = result['rows'][8]['cells']['price']
    assert number(price['text']) == 1120
    assert price['confidence'] > .9
    assert not price.get('ocr_attempts')
    assert result['computed_total'] == '29464'
    assert result['total'] == '29464.00'
    assert result['math_match']


def test_budget_overrun_is_saved_and_mismatch_triggers_double_budget_retry(monkeypatch):
    import json

    original_read_batches = pipeline.read_batches
    def missed_price(cells, **kwargs):
        results, stats = original_read_batches(cells, **kwargs)
        results[(9, 'price')]['text'] = '120T'
        results[(9, 'price')]['confidence'] = .99
        stats.update(numeric_roi_crop_ms=12.5, numeric_roi_ms=12.5,
                     numeric_roi_crop_exceeded=True, numeric_roi_skipped=1,
                     numeric_roi_skipped_cells=[{'row_id':9,'column_id':'price'}])
        return results, stats

    monkeypatch.setattr(pipeline, 'read_batches', missed_price)
    result = process((HERE/'fixtures'/'receipt_29464.jpg').read_bytes())
    assert result['computed_total'] == '29464'
    assert result['math_match']
    assert number(result['rows'][8]['cells']['price']['text']) == 1120
    retry = result['geometry']['ocr_recovery']['double_budget']
    assert retry['batch']['numeric_roi_crop_budget_ms'] == 24
    assert retry['batch']['numeric_roi_budget_ms'] == 32
    assert any(d['row_id']==9 and d['column_id']=='price' and d['accepted'] for d in retry['details'])
    root = HERE.parent.parent
    saved_log = json.loads((root/result['processing_log_file']).read_text(encoding='utf-8'))
    saved_profile = json.loads((root/result['performance_file']).read_text(encoding='utf-8'))
    assert any(e['type']=='budget' and e['elapsed_ms']==12.5 for e in saved_log)
    assert any(e['type']=='check' and e['receipt_mismatch'] for e in saved_log if 'receipt_mismatch' in e)
    assert any(e['type']=='retry' and e.get('row_id')==9 and e['accepted'] for e in saved_log)
    assert saved_profile['budget_events']


def test_budget_overrun_is_preserved_when_processing_later_fails(monkeypatch):
    import json
    import uuid

    original_read_batches = pipeline.read_batches
    def overrun(cells, **kwargs):
        results, stats = original_read_batches(cells, **kwargs)
        stats.update(numeric_roi_crop_ms=12.5, numeric_roi_ms=12.5,
                     numeric_roi_crop_exceeded=True, numeric_roi_skipped=1,
                     numeric_roi_skipped_cells=[{'row_id':9,'column_id':'price'}])
        return results, stats
    def fail_later(*args, **kwargs):
        raise RuntimeError('test failure after budget event')
    monkeypatch.setattr(pipeline, 'read_batches', overrun)
    monkeypatch.setattr(pipeline, 'recover_invalid_and_rows', fail_later)
    job_id = uuid.uuid4().hex
    with pytest.raises(RuntimeError, match='test failure'):
        process((HERE/'fixtures'/'receipt_29464.jpg').read_bytes(), job_id=job_id)
    job_dir = HERE.parent.parent/'data'/'output'/job_id
    events = json.loads((job_dir/'processing_log.json').read_text(encoding='utf-8'))
    profile = json.loads((job_dir/'performance_profile.json').read_text(encoding='utf-8'))
    assert any(event['type']=='budget' and event['elapsed_ms']==12.5 for event in events)
    assert events[-1]['type']=='error'
    assert profile['budget_events']


def test_budget_limited_low_confidence_triggers_retry_even_if_total_matches(monkeypatch):
    original_read_batches = pipeline.read_batches
    def low_confidence(cells, **kwargs):
        results, stats = original_read_batches(cells, **kwargs)
        results[(9, 'price')]['confidence'] = .6
        stats.update(numeric_roi_crop_ms=12.5, numeric_roi_ms=12.5,
                     numeric_roi_crop_exceeded=True, numeric_roi_skipped=1,
                     numeric_roi_skipped_cells=[{'row_id':9,'column_id':'price'}])
        return results, stats
    monkeypatch.setattr(pipeline, 'read_batches', low_confidence)
    result = process((HERE/'fixtures'/'receipt_29464.jpg').read_bytes())
    assert result['math_match']
    check = next(event for event in result['processing_log'] if event['type']=='check')
    assert check['budget_limited'] and check['low_confidence']
    assert any(detail['row_id']==9 and detail['column_id']=='price'
               for detail in result['geometry']['ocr_recovery']['double_budget']['details'])


def test_folded_receipt_recovers_printed_grid_and_all_amounts():
    result = process((HERE / 'fixtures' / 'receipt_fold_36981.jpg').read_bytes())
    assert result['geometry']['fold_recovered_horizontal'] >= 3
    assert [(number(row['cells']['quantity']['text']), number(row['cells']['price']['text']))
            for row in result['rows']] == [
        (2, 2200), (5, 380), (5, 500), (1, 4550), (1, 3960),
        (2, 2650), (1, 770), (5, 400), (2, 3250), (5101, 1),
    ]
    assert result['math_match']
    assert number(result['total']) == 36981
    assert result['vat']['status'] == 'match'


def test_receipt_32596_keeps_price_300_and_matches_total():
    result = process((HERE / 'fixtures' / 'receipt_32596.jpg').read_bytes())
    row = result['rows'][2]
    assert number(row['cells']['quantity']['text']) == 30
    assert number(row['cells']['price']['text']) == 300
    assert number(result['total']) == 32596
    assert result['computed_total'] == '32596'
    assert result['math_match']


def test_six_column_problem_receipt_retains_prices_and_printed_totals():
    result = process((HERE / 'receipt_six_columns.jpg').read_bytes())
    for index, price, printed_total in ((0, 1700, 1700), (1, 1800, 3600), (3, 200, 400)):
        row = result['rows'][index]
        assert number(row['cells']['price']['text']) == price
        assert number(row['cells']['printed_total']['text']) == printed_total
    # Other fields on this receipt are still uncertain.
    assert result['status'] == 'review'


def test_short_receipt_uses_verified_grid_and_correct_numbers(monkeypatch):
    from App.geometry.lab_adapter_gridrefine_v3 import detect_for_lab
    from App.geometry.geometry_lab.physical_grid_graph_v3 import _complete_short_grid
    from App.geometry.geometry_lab import physical_grid_graph_v3 as detector
    image = cv2.imread(str(HERE/'fixtures'/'receipt_short_51040.jpg'))
    # The ordinary AutoGrid path must not accidentally mask the fallback test.
    grid, _ = detect_for_lab(image)
    assert grid is None or len(grid['vertical']) < 5
    data, info = detector.detect_physical_graph(image)
    assert info['selection'] == 'strong_short_grid'
    assert _complete_short_grid(data, info)
    result = process((HERE/'fixtures'/'receipt_short_51040.jpg').read_bytes())
    assert [(number(row['cells']['quantity']['text']), number(row['cells']['price']['text']))
            for row in result['rows']] == [(20,2200),(7040,1)]
    assert number(result['total']) == 51040
    assert result['math_match']
    assert result['vat']['status'] == 'match'
    assert result['geometry']['physical_detection_selection'] == 'strong_short_grid'
    fast, fast_overlay, _ = pipeline.build_grid(image)
    with monkeypatch.context() as patch:
        patch.setattr(detector, '_complete_short_grid', lambda data, report:False)
        full, full_overlay, _ = pipeline.build_grid(image)
    assert np.array_equal(fast_overlay, full_overlay)
    assert np.array_equal(fast.info['cell_mesh'].map_x, full.info['cell_mesh'].map_x)
    assert np.array_equal(fast.info['cell_mesh'].map_y, full.info['cell_mesh'].map_y)


def test_receipt252161_tall_numeric_rows_and_fast_orientation():
    photo=HERE.parent.parent/'data/output/2a0f0503794a42ada555c0bd465da692/input.bin'
    if not photo.exists():pytest.skip('Local user receipt absent')
    result=process(photo.read_bytes())
    expected=[(6,8120),(8,500),(10,700),(2,2480),(20,230),(4,2400),(10,500),(1,5100),
              (4,300),(1,7300),(3,9900),(20,3560),(20,190),(20,200),(5,2000),(20,60),(34781,1)]
    assert [(number(r['cells']['quantity']['text']),number(r['cells']['price']['text'])) for r in result['rows']]==expected
    assert result['total']=='252161.00'
    assert result['computed_total']=='252161'
    assert result['math_match']
    assert result['geometry']['orientation']['method']=='observed_header_cells'
    assert result['geometry']['ocr_batch']['numeric_roi_skipped']==0
