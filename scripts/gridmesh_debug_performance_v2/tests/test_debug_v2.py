"""Offline sanity tests; no OCR models or internet needed."""
import ast
import importlib.util
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import apply_experiment as installer


def load_collector(log_level='DEBUG'):
    app=types.ModuleType('_debug_v2_testapp'); app.__path__=[str(BASE/'App')]
    ocr=types.ModuleType('_debug_v2_testapp.ocr'); ocr.__path__=[str(BASE/'App/ocr')]
    config=types.ModuleType('_debug_v2_testapp.config')
    config.setting=lambda name,default: log_level if name=='LOG_LEVEL' else default
    config.positive_int=lambda name,default: default
    sys.modules.update({'_debug_v2_testapp':app,'_debug_v2_testapp.ocr':ocr,'_debug_v2_testapp.config':config})
    spec=importlib.util.spec_from_file_location('_debug_v2_testapp.ocr.debug_performance',BASE/'App/ocr/debug_performance.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


class Profile:
    def __init__(self):self.issues=[]
    def note_issue(self,**kw):self.issues.append(kw)


def test_debug_context_propagates_into_parallel_lane():
    m=load_collector('DEBUG')
    token=m.activate_telemetry('a'*32)
    try:
        p=Profile();m.bind_profile(p,'a'*32)
        def worker():
            d=m.get_debug()
            assert d is not None
            d.add_lane('footer',9.5)
            m.report_issue(source='RapidOCR',operation='recognize_txt',kind='empty_text',severity='warning')
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(copy_context().run,worker).result()
        d=m.get_debug(); d.capture_runtime((640,480,3))
        cell=SimpleNamespace(row_id=1,column_id='price',image=np.zeros((32,220,3),dtype='uint8'))
        d.add_cell(cell,1.25,strategy='primary_rec_cell')
        d.add_crop('price',0.15,cell.image.shape)
        d.add_batch('numeric',[cell],2.3,status='ok')
        d.add_geometry('grid_mesh_construction',22.3)
        tr={'events':[{'model_id':'v5_cyrillic_mobile','mode':'REC_only','wall_ms':4.0,
                       'success':True,'cells':['1:price']},
                      {'model_id':'v6_small_official','mode':'REC_only','wall_ms':2.0,
                       'success':False,'cells':['1:price']} ]}
        result=d.finish(tr)
        assert result['ocr_lanes'][0]['lane']=='footer'
        assert result['ocr_cells'][0]['field']=='price'
        assert result['ocr_batches'][0]['batch_size']==1
        assert result['geometry_substeps'][0]['step']=='grid_mesh_construction'
        assert result['model_calls_from_rec_trace']['v6_small_official|REC_only']['failed_calls']==1
        assert len(p.issues)==1
        assert result['configuration']['table_mode']=='cell'
        json.dumps(result,ensure_ascii=False)
    finally:
        m.deactivate_telemetry(token)
    assert m.get_debug() is None


def test_info_does_not_collect_debug_but_records_issues():
    m=load_collector('INFO')
    token=m.activate_telemetry('b'*32)
    try:
        p=Profile();m.bind_profile(p,'b'*32)
        assert m.get_debug() is None
        m.report_issue(source='ONNX',operation='rec',kind='exception',error=RuntimeError('bad'))
        assert p.issues[0]['source']=='ONNX'
        assert p.issues[0]['error'].args == ('bad',)
    finally:m.deactivate_telemetry(token)


def test_missing_duration_on_failed_footer_lane():
    m=load_collector()
    d=m.DebugPerformance('f'*32)
    d.add_lane('footer',None,status='error',error='retry')
    assert d.finish()['ocr_lanes'][0]['wall_ms'] is None


def test_padding_estimate():
    m=load_collector()
    assert m.DebugPerformance.estimate_padding([])['estimated_padding_fraction']==0
    data=m.DebugPerformance.estimate_padding([(35,80,3),(35,600,3)])
    assert 0<data['estimated_padding_fraction']<1


def test_refuse_unrecognized_project_without_write(tmp_path):
    (tmp_path/'.env.example').write_text('LOG_LEVEL=INFO\n')
    for file in installer.PATCHES:
        f=tmp_path/file;f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text('old version')
    with pytest.raises(ValueError,match='not the GitHub source'):
        installer.apply(tmp_path,dry_run=False)
    assert not (tmp_path/installer.NEW_FILE).exists()
    assert not (tmp_path/'data').exists()


def test_patch_performance_ast_and_issue_fields():
    src='''from threading import Event, Thread
class OCRPerformance:
    def __init__(self):
        self.samples = []
    def finish(self):
        return {'samples':self.samples,
                'version':'ocr-performance-v1'}
'''
    edited=installer.patch_performance(src)
    ast.parse(edited)
    assert 'issues' in edited
    assert 'issues_lock' in edited


def test_exact_anchor_rejects_drift():
    with pytest.raises(ValueError,match='anchor expected once'):
        installer.replace_once('changed source','expected source','patched',where='file.py')


def test_all_patchers_produce_syntactic_python():
    # Synthetic fixtures contain the actual structural anchors of this release.
    pipeline='''from .ocr.ocr_performance import OCRPerformance

def process(data):
    tracker=RecTracker()
    token=activate(tracker)
    try:
        return _process_job(data)
    finally:
        deactivate(token)

def _process_job(data):
    profile = OCRPerformance(publish)
    profile.processing_log = []
    try:
        result = _process(data, profile, job_dir, rectify)
    except Exception as exc:
        failure_profile = profile.finish()
        raise
    result['rec_trace']=rec_tracker.export()
    performance = profile.finish()
    result['performance']=performance
    return result

def _process(data,profile,job_dir,rectify):
    cv2.setNumThreads(positive_int('OCR_OPENCV_THREADS', 2))
    for idx in range(2):
        for field in field_order:
            crop, offset, source_bbox = cell_crop(rect, idx, columns[field])
            cell_requests.append(BatchCell(crop, idx, field, offset, source_bbox))
    geometry['ocr_mode'] = ocr_mode
    geometry['ocr_batch'] = batch_stats
    return geometry
'''
    assert ast.parse(installer.patch_pipeline(pipeline))

    batch='''from .experiment_rec_trace import rec_scope, record_rec

def read_batches(cells):
    if mode not in ('batch','cell'):raise ValueError('Unknown OCR mode: '+mode)
    for cell in working:
        if mode == 'cell':
            try:
                with rec_scope([key],'primary_rec_cell'):
                    item = read_cell(cell.image,cell.row_id,cell.column_id,cell.offset)
                stats['cell_calls'] += 1
            except Exception as exc:
                stats['cell_fallback'] += 1
            continue
        if _has_separate_numeric_lines(cell.image):
            with rec_scope([(cell.row_id,cell.column_id)],'primary_multiline_cell'):
                item = read_cell(cell.image,cell.row_id,cell.column_id,cell.offset)
            stats['multiline_fallback'] += 1
        else:
            pass
    for group in ('numeric','text'):
        for start in range(0,len(bucket),8):
            chunk=bucket[start:start+8]
            stats['batch_calls'] += 1
            try:
                response=engine().recognize_txt([c.image for c in chunk])
                for c,txt,score in zip(chunk,response.txts,response.scores):
                    out[(c.row_id,c.column_id)] = _convert(c, txt, score)
            except Exception as exc:
                stats['batch_fallback'] += len(chunk)
                for c in chunk:
                    with rec_scope([(c.row_id,c.column_id)],'primary_batch_exception_fallback'):
                        item = read_cell(c.image, c.row_id, c.column_id, c.offset)
    return out
'''
    assert ast.parse(installer.patch_batch(batch))

    geometry='''import cv2
import numpy as np

def build_grid(image, *, rectify=False, phase=None):
    data, report = detect_for_lab(image)
    if data is None:
        physical, physical_report = detect_physical_graph(image)
    if data is not None and method == 'Physical Grid Graph V3':
        data, recovered_horizontal = _recover_observed_horizontal(image, data, automatic_data)
    if mode:
        data, fold_recovered = recover_fold(image, data)
    data, left_recovered = recover_left_edges(image, data)
    if phase:
        phase('Создание сетки Grid Mesh')
    if not rectify:
        result = mesh(image,data)
    check = safety(result.info['cell_mesh'] if not rectify else result, image.shape)
    overlay = result.image.copy()
    return result, overlay, {'method': method, 'autogrid_vertical_count': auto_vertical_count,
                             'warnings': []}
'''
    assert ast.parse(installer.patch_geometry(geometry))

    cellmode='''from .engine import inference_lane

def read_parallel(cells, footer_reader, cell_reader, *, batch_size=8):
    started = Event()
    with ThreadPoolExecutor() as executor:
        try:
            readouts, stats=cell_reader(cells)
        except Exception as exc:
            readouts, stats = read_batches(cells, batch_size=batch_size)
        try:
            footer_result=footer_reader()
        except Exception as exc:
            # Retry independently in the legacy footer route after table assembly.
            footer_result=None
    stats['parallel_lanes'] = 2
    return readouts, stats, footer_result
'''
    assert ast.parse(installer.patch_cell_mode(cellmode))

    models='''from .experiment_rec_trace import record_rec

class ExperimentModelPool:
    def recognize(self,model_id,image,stage):
        try:
            return self.run(model_id,image)
        except Exception as exc:
            self.errors.append(item)
            return item
'''
    assert ast.parse(installer.patch_experiment_models(models))


def test_install_and_restore_atomic_with_byte_hashes(tmp_path,monkeypatch):
    from restore_experiment import restore
    (tmp_path/'.env.example').write_text('LOG_LEVEL=INFO\n')
    target=tmp_path/'App'/'pipeline.py';target.parent.mkdir(parents=True)
    original=b"print('old')\r\n"
    target.write_bytes(original)
    monkeypatch.setattr(installer,'EXPECTED_BLOBS',{'App/pipeline.py':installer.blob_id(original.replace(b'\r\n',b'\n'))})
    monkeypatch.setattr(installer,'PATCHES',{'App/pipeline.py':lambda text:text.replace("'old'","'new'")})
    assert installer.apply(tmp_path,dry_run=True)==['App/pipeline.py',installer.NEW_FILE]
    assert target.read_bytes()==original
    installer.apply(tmp_path)
    assert b"print('new')\r\n" == target.read_bytes()
    assert (tmp_path/installer.NEW_FILE).is_file()
    backup=tmp_path/'data'/'experiments'/installer.EXPERIMENT/'backup'/'App'/'pipeline.py'
    assert backup.read_bytes()==original
    restore(tmp_path,dry_run=True)
    assert b"'new'" in target.read_bytes()
    restore(tmp_path)
    assert target.read_bytes()==original
    assert not (tmp_path/installer.NEW_FILE).exists()


def test_restore_rejects_files_changed_after_install(tmp_path,monkeypatch):
    from restore_experiment import restore
    (tmp_path/'.env.example').write_text('LOG_LEVEL=INFO\n')
    target=tmp_path/'App'/'pipeline.py';target.parent.mkdir(parents=True)
    original=b"print('old')\n"
    target.write_bytes(original)
    monkeypatch.setattr(installer,'EXPECTED_BLOBS',{'App/pipeline.py':installer.blob_id(original)})
    monkeypatch.setattr(installer,'PATCHES',{'App/pipeline.py':lambda text:text.replace("'old'","'new'")})
    installer.apply(tmp_path)
    target.write_text("print('edited')\n")
    with pytest.raises(ValueError,match='changed since install'):
        restore(tmp_path)
    assert target.read_text()=="print('edited')\n"
