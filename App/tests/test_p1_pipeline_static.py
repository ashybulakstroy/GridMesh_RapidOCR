"""P1 integration assertions using the existing repository's real photo dataset."""
from pathlib import Path
import pytest
from App.pipeline import process

HERE=Path(__file__).resolve().parent


@pytest.mark.parametrize('photo', ['receipt_5730.png','receipt_575866.png','receipt_597275.png'])
def test_p1_provides_batch_metrics_and_source_positions(photo):
    result=process((HERE/photo).read_bytes())
    batch=result['geometry']['ocr_batch']
    assert batch['requested']>=9
    assert batch['batch_calls']>=1
    assert result['geometry']['ocr_recovery']['initial']['attempted']>=0
    assert all('source_bbox' in r['cells']['name'] for r in result['rows'])
    assert all(r['cells']['number']['derived'] for r in result['rows'])
    assert result['math_match'] is True


def test_six_columns_validate_each_printed_line():
    result=process((HERE/'receipt_six_columns.jpg').read_bytes())
    assert 'printed_total' in result['geometry']['columns']
    assert len(result['rows'])==4
    assert all('printed_line_match' in r for r in result['rows'])
    assert result['status']=='review'  # Never turn poor OCR into false success.
