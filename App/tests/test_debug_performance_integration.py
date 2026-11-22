"""Real pipeline coverage for diagnostic persistence, isolation and both OCR modes."""
import json
from pathlib import Path
import pytest
import App.pipeline as pipeline
from App.ocr.debug_performance import get_debug, activate_telemetry, deactivate_telemetry


@pytest.mark.parametrize(('level','mode'), [('INFO','cell'),('DEBUG','cell'),('DEBUG','batch')])
def test_real_receipt_persists_diagnostics_without_changing_total(monkeypatch,level,mode):
    monkeypatch.setenv('LOG_LEVEL',level)
    monkeypatch.setenv('OCR_TABLE_MODE',mode)
    monkeypatch.setenv('OCR_E4_ENABLE','1')
    result=pipeline.process((Path(__file__).parent/'receipt_5730.png').read_bytes())
    assert result['math_match'] and result['computed_total']=='5730'
    saved=json.loads((pipeline.ROOT/result['performance_file']).read_text(encoding='utf-8'))
    assert isinstance(saved['issues'],list)
    assert get_debug() is None
    if level=='INFO':
        assert 'debug' not in saved
    else:
        debug=saved['debug']
        assert debug['job_id']==result['job_id']
        assert debug['configuration']['table_mode']==mode
        assert debug['geometry_substeps'] and debug['cell_crop_summary']
        assert debug['model_calls_from_rec_trace']
        assert debug['memory_mb']['max_rss_observed_at_checkpoints']>=debug['memory_mb']['rss_start']
        if mode=='cell':
            assert len(debug['ocr_cells'])>0 and not debug['ocr_batches']
            assert {r['lane'] for r in debug['ocr_lanes']}=={
                'table','footer','table_names','table_numeric'}
        else:
            assert debug['ocr_batches'] and not debug['ocr_cells']


@pytest.mark.parametrize('level',['INFO','DEBUG'])
def test_failure_persists_issue_and_restores_parent_context(tmp_path,monkeypatch,level):
    monkeypatch.setenv('LOG_LEVEL',level)
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    def fail(*args,**kwargs):raise RuntimeError('forced diagnostics failure')
    monkeypatch.setattr(pipeline,'_process',fail)
    parent=activate_telemetry('parent')
    original=get_debug()
    try:
        with pytest.raises(RuntimeError,match='forced diagnostics failure'):
            pipeline.process(b'photo',job_id='d'*32)
        assert get_debug() is original
    finally:deactivate_telemetry(parent)
    saved=json.loads((tmp_path/'data/output'/('d'*32)/'performance_profile.json').read_text(encoding='utf-8'))
    assert saved['issues'][0]['kind']=='uncaught_exception'
    assert saved['issues'][0]['exception_type']=='RuntimeError'
    if level=='DEBUG':assert 'RuntimeError' in saved['debug']['issues'][0]['traceback']
    else:assert 'debug' not in saved
