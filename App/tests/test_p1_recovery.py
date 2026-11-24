"""P1 safety: alternate OCR is retained and never invented from footer arithmetic."""
import numpy as np
from App.ocr import p1_recovery


def make_row(price='S07', printed='300', qty='1'):
    fields={'quantity':qty,'price':price,'printed_total':printed}
    cells={key:dict(text=value, confidence=.70, bbox=[0,0,35,30],
                    row_id=1, column_id=key, source_bbox=[]) for key,value in fields.items()}
    cells['printed_total']['confidence'] = .90
    return dict(row_id=1, kind='product', cells=cells)


def test_recovery_keeps_both_ocr_variants_and_accepts_when_printed_confirms(monkeypatch):
    row=make_row()
    monkeypatch.setattr(p1_recovery,'_retry', lambda original,image: dict(
        text='300',confidence=.94,bbox=original['bbox'],row_id=1,
        column_id=original['column_id']))
    source={(1,'price'):np.full((30,35,3),255,np.uint8)}
    result=p1_recovery.recover_invalid_and_rows([row],source,max_retries=1)
    assert row['cells']['price']['text']=='300'
    assert row['cells']['price']['ocr_attempts'][0]['original_text']=='S07'
    assert row['cells']['price']['ocr_attempts'][0]['alternative_text']=='300'
    assert result['changed']==[(1,'price','printed_row_match')]
    assert row['printed_line_match'] is True


def test_total_match_alone_does_not_authorize_correction(monkeypatch):
    row=make_row(price='200',printed='400')
    monkeypatch.setattr(p1_recovery,'_retry',lambda original,image: dict(
        text='300',confidence=.99,bbox=original['bbox'],row_id=1,
        column_id=original['column_id']))
    source={(1,'price'):np.full((30,35,3),255,np.uint8)}
    result=p1_recovery.recover_total_mismatch([row],source,300,max_retries=1)
    assert row['cells']['price']['text']=='200'
    assert result['changed']==[]
    assert result['candidates'][0]['text']=='300'
    assert row['cells']['price']['ocr_attempts'][0]['accepted'] is False


def test_invalid_number_accepts_valid_retry_without_printed_row_total(monkeypatch):
    row = make_row(qty='2.', price='1060')
    del row['cells']['printed_total']
    monkeypatch.setattr(p1_recovery, '_retry', lambda original, image: dict(
        text='2', confidence=.97, bbox=original['bbox'], row_id=1,
        column_id=original['column_id']))
    source = {(1, 'quantity'): np.full((30, 35, 3), 255, np.uint8)}
    result = p1_recovery.recover_invalid_and_rows([row], source, max_retries=1)
    assert row['cells']['quantity']['text'] == '2'
    assert row['line_total'] == '2120'
    assert result['changed'] == [(1, 'quantity', 'valid_retry_without_printed_row_total')]
    assert row['cells']['quantity']['ocr_attempts'][0]['original_text'] == '2.'
    assert result['details'][0]['accepted'] is True


def test_invalid_retry_does_not_replace_number_when_printed_row_total_disagrees(monkeypatch):
    row = make_row(qty='2.', price='1060', printed='3000')
    monkeypatch.setattr(p1_recovery, '_retry', lambda original, image: dict(
        text='2', confidence=.97, bbox=original['bbox'], row_id=1,
        column_id=original['column_id']))
    source = {(1, 'quantity'): np.full((30, 35, 3), 255, np.uint8)}
    result = p1_recovery.recover_invalid_and_rows([row], source, max_retries=1)
    assert row['cells']['quantity']['text'] == '2.'
    assert result['details'][0]['accepted'] is False
