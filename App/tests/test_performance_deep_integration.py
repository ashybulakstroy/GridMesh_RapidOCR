import json

from App.ocr.performance_deep import live_capture, performance_report
from App.geometry.geometry_lab.physical_grid_graph_v3 import _complete_short_grid


def test_early_exit_rejection_records_actual_reason():
    token=live_capture.activate(True)
    try:
        assert not _complete_short_grid(None, {})
        assert not _complete_short_grid({'vertical':[], 'horizontal':[]}, {'vertical_table_local_refit':True})
        details=live_capture.current().export()['grid_counters']['details']
        assert [item['reason'] for item in details] == [
            'baseline_missing_or_vertical_fallback_or_no_local_refit', 'unexpected_grid_dimensions']
    finally:
        live_capture.deactivate(token)


def test_failed_job_report_keeps_error_and_job_id(tmp_path):
    profile={'wall_ms':2001, 'version':'ocr-performance-v1', 'samples':[],
             'debug':{'configuration':{'log_level':'DEBUG'}}}
    (tmp_path/'performance_profile.json').write_text(json.dumps(profile),encoding='utf8')
    report=performance_report.generate(tmp_path, receipt={'job_id':'abc', 'error':'test <error>'})
    assert report is not None
    assert 'test &lt;error&gt;' in report.read_text(encoding='utf8')
    manifest=json.loads((report.parent/'manifest.json').read_text(encoding='utf8'))
    assert manifest['job_id']=='abc' and manifest['error']=='test <error>'


def test_slow_grid_report_shows_internal_timing_and_missing_calls():
    profile={'wall_ms':2500,'version':'ocr-performance-v1',
             'samples':[{'name':'Поиск физической сетки','wall_ms':2100}],
             'debug':{'configuration':{'log_level':'DEBUG'}}}
    deep={'grid_counters':{'functions':{
        'physical_grid_graph.py:graph_verticals':{'calls':3,'sum_call_ms':125.5,
            'median_ms':4.0,'p90_ms':110.0,'min_ms':2.0,'max_ms':110.0}}}}
    report=performance_report.build(profile,{},deep)
    page=performance_report.render(report)
    assert 'Внутренние вызовы поиска сетки' in page
    assert 'physical_grid_graph.py:graph_verticals' in page and '125.5' in page
    assert 'trace_seed' in page and 'не зафиксировано' in page
