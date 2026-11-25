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


def test_export_preserves_raw_profile_only_for_slow_debug_job(tmp_path):
    profile={'wall_ms':2500,'version':'ocr-performance-v1',
             'samples':[{'name':'Поиск физической сетки','wall_ms':1500}],
             'debug':{'configuration':{'log_level':'DEBUG'}}}
    (tmp_path/'performance_profile.json').write_text(json.dumps(profile),encoding='utf8')
    measurement={'name':'grid_internal_timing','function':'trace_seed',
                 'breakdown_ms':{'поиск пути':17.5}}
    capture={'capture_status':'live','spans':[],
             'grid_counters':{'counts':{},'details':[measurement],'functions':{}},
             'ocr_decisions':[],'warnings':[]}
    report=performance_report.generate(tmp_path,capture,receipt={'job_id':'abc'})
    raw=json.loads((report.parent/'deep_profile.json').read_text(encoding='utf8'))
    assert raw['capture_status']=='live' and raw['job_id']=='abc'
    assert raw['grid_counters']['details']==[measurement]
    assert 'deep_profile.json' in report.read_text(encoding='utf8')
    quiet=tmp_path/'quiet';quiet.mkdir()
    profile['wall_ms']=2000
    (quiet/'performance_profile.json').write_text(json.dumps(profile),encoding='utf8')
    assert performance_report.generate(quiet,capture,receipt={'job_id':'quiet'}) is None
    assert not (quiet/'problem_performance').exists()


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


def test_internal_grid_timings_are_measured_without_changing_results():
    import numpy as np
    from App.geometry.geometry_lab.physical_grid_graph_local import trace_seed
    image=np.zeros((60,100),dtype=np.float32)
    image[30,:]=50
    xs=np.arange(5,95,dtype=float)
    baseline=trace_seed(image,(5,30,94,30),xs)
    token=live_capture.activate(True)
    try:
        actual=trace_seed(image,(5,30,94,30),xs)
        details=live_capture.current().export()['grid_counters']['details']
        measurement=next(x for x in details if x['name']=='grid_internal_timing' and x['function']=='trace_seed')
        assert {'подготовка и выборка','поиск пути','подгонка линии'} <= measurement['breakdown_ms'].keys()
        assert all(value>=0 for value in measurement['breakdown_ms'].values())
        assert all(np.array_equal(a,b) for a,b in zip(baseline,actual))
    finally:
        live_capture.deactivate(token)
