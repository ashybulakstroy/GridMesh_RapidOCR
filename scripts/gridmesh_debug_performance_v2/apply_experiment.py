#!/usr/bin/env python3
"""Strict, reversible DEBUG performance patch for GridMesh_RapidOCR (22 Nov 2026).

Dry-run validates git blob IDs and all patch anchors. No live JSON writes.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

EXPERIMENT = 'gridmesh_debug_performance_v2'
BASE_COMMIT = '1acd37d3cab4df1775f4e43db7d2fd81a33e07dc'
EXPECTED_BLOBS = {
    'App/pipeline.py': '69d12bb45c1655fd486ae93f21c7705acbd81c88',
    'App/ocr/p1_batch.py': '26ea6f7c488256f89b6fa3eec070761c7a724c06',
    'App/geometry/service.py': 'a6afebad1b5f7876316dc9a20055195b681aa126',
    'App/ocr/ocr_performance.py': 'aa33e2072d1eda254c1f5a7cec7b15ab983a389c',
    'App/ocr/cell_mode.py': '9992545546f11142bba4a0d066c29189cdd502ba',
    'App/ocr/experiment_models.py': '668b4cbc7c03e7d42577b8534301edb720286ef7',
}
NEW_FILE = 'App/ocr/debug_performance.py'


def blob_id(blob:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(blob)).encode('ascii')+b'\0'+blob).hexdigest()


def replace_once(source:str, old:str, new:str, *, where:str)->str:
    n=source.count(old)
    if n!=1:
        raise ValueError(f'{where}: anchor expected once, found {n}: {old[:110]!r}')
    return source.replace(old,new,1)


def patch_pipeline(source:str)->str:
    s=source
    def r(a,b):
        nonlocal s
        s=replace_once(s,a,b,where='App/pipeline.py')

    r('from .ocr.ocr_performance import OCRPerformance\n',
      'from .ocr.ocr_performance import OCRPerformance\n'
      'from .ocr.debug_performance import (activate_telemetry, deactivate_telemetry,\n'
      '    bind_profile, get_debug, report_issue)\n')
    r("    token=activate(tracker)\n    try:\n",
      "    token=activate(tracker)\n"
      "    telemetry_token=activate_telemetry(job_id)\n"
      "    try:\n")
    r('    finally:\n        deactivate(token)\n',
      '    finally:\n        deactivate_telemetry(telemetry_token)\n'
      '        deactivate(token)\n')
    r('    profile = OCRPerformance(publish)\n    profile.processing_log = []\n',
      '    profile = OCRPerformance(publish)\n'
      '    bind_profile(profile, job_id)\n'
      '    profile.processing_log = []\n')
    r('    except Exception as exc:\n        failure_profile = profile.finish()\n',
      "    except Exception as exc:\n"
      "        report_issue(source='pipeline',operation=profile.name or 'unknown',\n"
      "                     kind='uncaught_exception',error=exc,fallback='none')\n"
      "        failure_profile = profile.finish()\n"
      "        debug_perf=get_debug()\n"
      "        if debug_perf is not None:\n"
      "            failure_profile['debug']=debug_perf.finish(rec_tracker.export())\n")
    r("    result['rec_trace']=rec_tracker.export()\n    performance = profile.finish()\n",
      "    result['rec_trace']=rec_tracker.export()\n"
      "    for item in result.get('processing_log',[]):\n"
      "        if item.get('type') == 'error':\n"
      "            report_issue(source='processing_log',operation=str(item.get('stage','pipeline')),\n"
      "                         kind='caught_pipeline_error',context={'message':str(item.get('message',''))[:250]})\n"
      "    performance = profile.finish()\n"
      "    debug_perf=get_debug()\n"
      "    if debug_perf is not None:\n"
      "        performance['debug']=debug_perf.finish(result['rec_trace'])\n")
    r("    cv2.setNumThreads(positive_int('OCR_OPENCV_THREADS', 2))\n",
      "    cv2.setNumThreads(positive_int('OCR_OPENCV_THREADS', 2))\n"
      "    debug_perf=get_debug()\n"
      "    if debug_perf is not None:\n"
      "        debug_perf.capture_runtime(image.shape)\n")
    r('            crop, offset, source_bbox = cell_crop(rect, idx, columns[field])\n'
      '            cell_requests.append(BatchCell(crop, idx, field, offset, source_bbox))\n',
      '            _crop_t0=time.perf_counter() if debug_perf is not None else None\n'
      '            crop, offset, source_bbox = cell_crop(rect, idx, columns[field])\n'
      '            if debug_perf is not None:\n'
      '                debug_perf.add_crop(field,(time.perf_counter()-_crop_t0)*1000,crop.shape)\n'
      '            cell_requests.append(BatchCell(crop, idx, field, offset, source_bbox))\n')
    r("    geometry['ocr_mode'] = ocr_mode\n    geometry['ocr_batch'] = batch_stats\n",
      "    geometry['ocr_mode'] = ocr_mode\n"
      "    geometry['ocr_batch'] = batch_stats\n"
      "    if ocr_mode == 'batch' and debug_perf is not None:\n"
      "        debug_perf.add_lane('table',batch_stats.get('table_wall_ms'),status='batch')\n")
    return s


def patch_batch(source:str)->str:
    s=source
    def r(a,b):
        nonlocal s
        s=replace_once(s,a,b,where='App/ocr/p1_batch.py')
    r('from .experiment_rec_trace import rec_scope, record_rec\n',
      'from .experiment_rec_trace import rec_scope, record_rec\n'
      'from .debug_performance import get_debug, report_issue\n')
    r("    if mode not in ('batch','cell'):raise ValueError('Unknown OCR mode: '+mode)\n",
      "    if mode not in ('batch','cell'):raise ValueError('Unknown OCR mode: '+mode)\n"
      "    _debug=get_debug()\n")
    r("        if mode == 'cell':\n            try:\n",
      "        if mode == 'cell':\n"
      "            _cell_t0=rec_counter() if _debug is not None else None\n"
      "            try:\n")
    r("                stats['cell_calls'] += 1\n            except Exception as exc:\n",
      "                stats['cell_calls'] += 1\n"
      "                if _debug is not None:\n"
      "                    _debug.add_cell(cell,(rec_counter()-_cell_t0)*1000,\n"
      "                                    strategy=item.get('strategy','primary_rec_cell'))\n"
      "                if not str(item.get('text','')).strip():\n"
      "                    report_issue(source='RapidOCR',operation='read_cell',\n"
      "                                 kind='empty_text',severity='warning',\n"
      "                                 context={'cell_id':f'{cell.row_id}:{cell.column_id}',\n"
      "                                          'mode':'cell'})\n"
      "            except Exception as exc:\n"
      "                report_issue(source='RapidOCR',operation='read_cell',\n"
      "                             kind='cell_exception',error=exc,\n"
      "                             context={'cell_id':f'{cell.row_id}:{cell.column_id}'},\n"
      "                             fallback='batch')\n"
      "                if _debug is not None:\n"
      "                    _debug.add_cell(cell,(rec_counter()-_cell_t0)*1000,\n"
      "                                    status='error',error=f'{type(exc).__name__}: {exc}')\n")
    r("        if _has_separate_numeric_lines(cell.image):\n            with rec_scope([(cell.row_id,cell.column_id)],'primary_multiline_cell'):\n",
      "        if _has_separate_numeric_lines(cell.image):\n"
      "            _multi_t0=rec_counter() if _debug is not None else None\n"
      "            with rec_scope([(cell.row_id,cell.column_id)],'primary_multiline_cell'):\n")
    r("            stats['multiline_fallback'] += 1\n        else:\n",
      "            stats['multiline_fallback'] += 1\n"
      "            if _debug is not None:\n"
      "                _debug.add_fallback(cell,(rec_counter()-_multi_t0)*1000,\n"
      "                                    reason='multiline',strategy=item.get('strategy','legacy_multiline'))\n"
      "            if not str(item.get('text','')).strip():\n"
      "                report_issue(source='RapidOCR',operation='read_cell',kind='empty_text',\n"
      "                             severity='warning',context={'cell_id':f'{cell.row_id}:{cell.column_id}',\n"
      "                                                         'mode':'multiline'})\n"
      "        else:\n")
    r("            stats['batch_calls'] += 1\n            try:\n",
      "            stats['batch_calls'] += 1\n"
      "            _batch_t0=rec_counter() if _debug is not None else None\n"
      "            try:\n")
    r("                for c,txt,score in zip(chunk,response.txts,response.scores):\n"
      "                    out[(c.row_id,c.column_id)] = _convert(c, txt, score)\n",
      "                for c,txt,score in zip(chunk,response.txts,response.scores):\n"
      "                    out[(c.row_id,c.column_id)] = _convert(c, txt, score)\n"
      "                    if not str(txt).strip():\n"
      "                        report_issue(source='RapidOCR',operation='recognize_txt',\n"
      "                                     kind='empty_text',severity='warning',\n"
      "                                     context={'cell_id':f'{c.row_id}:{c.column_id}',\n"
      "                                              'group':group})\n"
      "                if _debug is not None:\n"
      "                    _debug.add_batch(group,chunk,(rec_counter()-_batch_t0)*1000,status='ok')\n")
    r("            except Exception as exc:\n                stats['batch_fallback'] += len(chunk)\n",
      "            except Exception as exc:\n"
      "                report_issue(source='RapidOCR',operation='recognize_txt',\n"
      "                             kind='batch_exception_or_invalid_response',error=exc,\n"
      "                             context={'group':group, 'size':len(chunk),\n"
      "                                      'cell_ids':[f'{c.row_id}:{c.column_id}' for c in chunk]},\n"
      "                             fallback='read_cell')\n"
      "                if _debug is not None:\n"
      "                    _debug.add_batch(group,chunk,(rec_counter()-_batch_t0)*1000,\n"
      "                                     status='error',error=f'{type(exc).__name__}: {exc}')\n"
      "                stats['batch_fallback'] += len(chunk)\n")
    r("                for c in chunk:\n                    with rec_scope([(c.row_id,c.column_id)],'primary_batch_exception_fallback'):\n"
      "                        item = read_cell(c.image, c.row_id, c.column_id, c.offset)\n",
      "                for c in chunk:\n"
      "                    _fallback_t0=rec_counter() if _debug is not None else None\n"
      "                    with rec_scope([(c.row_id,c.column_id)],'primary_batch_exception_fallback'):\n"
      "                        item = read_cell(c.image, c.row_id, c.column_id, c.offset)\n"
      "                    if _debug is not None:\n"
      "                        _debug.add_fallback(c,(rec_counter()-_fallback_t0)*1000,\n"
      "                                            reason='batch_error',strategy='read_cell')\n")
    return s


def patch_geometry(source:str)->str:
    s=source
    def r(a,b):
        nonlocal s
        s=replace_once(s,a,b,where='App/geometry/service.py')
    r('import cv2\nimport numpy as np\n',
      'import cv2\nimport numpy as np\nfrom time import perf_counter\n'
      'from ..ocr.debug_performance import get_debug\n')
    r('def build_grid(image, *, rectify=False, phase=None):\n    data, report = detect_for_lab(image)\n',
      'def build_grid(image, *, rectify=False, phase=None):\n'
      '    _debug=get_debug()\n'
      '    _grid_t0=perf_counter() if _debug is not None else None\n'
      '    data, report = detect_for_lab(image)\n'
      '    if _debug is not None:\n'
      "        _debug.add_geometry('autogrid_gridrefine_v3',(perf_counter()-_grid_t0)*1000,\n"
      "                            {'found':data is not None})\n")
    r('        physical, physical_report = detect_physical_graph(image)\n',
      '        _physical_t0=perf_counter() if _debug is not None else None\n'
      '        physical, physical_report = detect_physical_graph(image)\n'
      '        if _debug is not None:\n'
      "            _debug.add_geometry('physical_grid_graph_v3',(perf_counter()-_physical_t0)*1000,\n"
      "                                {'found':physical is not None})\n")
    r("    if data is not None and method == 'Physical Grid Graph V3':\n        data, recovered_horizontal = _recover_observed_horizontal(image, data, automatic_data)\n",
      "    if data is not None and method == 'Physical Grid Graph V3':\n"
      "        _ridge_t0=perf_counter() if _debug is not None else None\n"
      "        data, recovered_horizontal = _recover_observed_horizontal(image, data, automatic_data)\n"
      "        if _debug is not None:\n"
      "            _debug.add_geometry('recover_observed_horizontal',(perf_counter()-_ridge_t0)*1000,\n"
      "                                {'recovered':recovered_horizontal})\n")
    r("        data, fold_recovered = recover_fold(image, data)\n",
      "        _fold_t0=perf_counter() if _debug is not None else None\n"
      "        data, fold_recovered = recover_fold(image, data)\n"
      "        if _debug is not None:\n"
      "            _debug.add_geometry('recover_fold',(perf_counter()-_fold_t0)*1000,\n"
      "                                {'recovered':fold_recovered})\n")
    r('    data, left_recovered = recover_left_edges(image, data)\n',
      '    _left_t0=perf_counter() if _debug is not None else None\n'
      '    data, left_recovered = recover_left_edges(image, data)\n'
      '    if _debug is not None:\n'
      "        _debug.add_geometry('recover_left_edges',(perf_counter()-_left_t0)*1000,\n"
      "                            {'recovered':left_recovered})\n")
    r("    if phase:\n        phase('Создание сетки Grid Mesh')\n    if not rectify:\n",
      "    if phase:\n        phase('Создание сетки Grid Mesh')\n"
      "    _mesh_t0=perf_counter() if _debug is not None else None\n"
      "    if not rectify:\n")
    r("    check = safety(result.info['cell_mesh'] if not rectify else result, image.shape)\n",
      "    if _debug is not None:\n"
      "        _debug.add_geometry('grid_mesh_construction',(perf_counter()-_mesh_t0)*1000,\n"
      "                            {'rectify_full_grid':bool(rectify)})\n"
      "        _overlay_t0=perf_counter()\n"
      "    check = safety(result.info['cell_mesh'] if not rectify else result, image.shape)\n")
    r("    return result, overlay, {'method': method, 'autogrid_vertical_count': auto_vertical_count,\n",
      "    if _debug is not None:\n"
      "        _debug.add_geometry('safety_and_overlay',(perf_counter()-_overlay_t0)*1000)\n"
      "    return result, overlay, {'method': method, 'autogrid_vertical_count': auto_vertical_count,\n")
    return s


def patch_performance(s:str)->str:
    s=replace_once(s,'from threading import Event, Thread\n',
                   'from threading import Event, Thread, Lock\n',where='App/ocr/ocr_performance.py')
    s=replace_once(s,'        self.samples = []\n',
                   '        self.samples = []\n        self.issues = []\n        self.issues_lock = Lock()\n',
                   where='App/ocr/ocr_performance.py')
    s=replace_once(s,'    def finish(self):\n',
                   """    def note_issue(self, *, source, operation, kind, severity='error', error=None,
                   context=None, fallback=None):
        item={'source':str(source),'operation':str(operation),
              'kind':str(kind),'severity':str(severity)}
        if error is not None:
            item['exception_type']=type(error).__name__
            item['message']=str(error)[:300]
        if context:item['context']=context
        if fallback:item['fallback']=fallback
        with self.issues_lock:
            self.issues.append(item)

    def finish(self):
""",where='App/ocr/ocr_performance.py')
    s=replace_once(s,"'samples':self.samples,\n",
                   "'samples':self.samples, 'issues':list(self.issues),\n",
                   where='App/ocr/ocr_performance.py')
    return s


def patch_cell_mode(s:str)->str:
    s=replace_once(s,'from .engine import inference_lane\n',
                   'from .engine import inference_lane\n'
                   'from .debug_performance import get_debug, report_issue\n',
                   where='App/ocr/cell_mode.py')
    s=replace_once(s,"    started = Event()\n",
                   "    started = Event()\n    _debug=get_debug()\n",where='App/ocr/cell_mode.py')
    s=replace_once(s,"        except Exception as exc:\n            readouts, stats = read_batches(cells, batch_size=batch_size)\n",
                   "        except Exception as exc:\n"
                   "            report_issue(source='parallel_ocr',operation='table_lane',\n"
                   "                         kind='lane_exception',error=exc,\n"
                   "                         fallback='batch')\n"
                   "            readouts, stats = read_batches(cells, batch_size=batch_size)\n",
                   where='App/ocr/cell_mode.py')
    s=replace_once(s,"        except Exception as exc:\n            # Retry independently in the legacy footer route after table assembly.\n",
                   "        except Exception as exc:\n"
                   "            report_issue(source='parallel_ocr',operation='footer_lane',\n"
                   "                         kind='lane_exception',error=exc,\n"
                   "                         fallback='legacy_footer')\n"
                   "            # Retry independently in the legacy footer route after table assembly.\n",
                   where='App/ocr/cell_mode.py')
    s=replace_once(s,"    stats['parallel_lanes'] = 2\n    return readouts, stats, footer_result\n",
                   "    stats['parallel_lanes'] = 2\n"
                   "    if _debug is not None:\n"
                   "        _debug.add_lane('table',stats.get('table_wall_ms'),\n"
                   "                        status='batch_fallback' if stats.get('mode')=='batch_fallback' else 'ok')\n"
                   "        _debug.add_lane('footer',stats.get('footer_wall_ms'),\n"
                   "                        status='error' if footer_result is None else 'ok',\n"
                   "                        error=stats.get('footer_error'))\n"
                   "    return readouts, stats, footer_result\n",
                   where='App/ocr/cell_mode.py')
    return s


def patch_experiment_models(s:str)->str:
    s=replace_once(s,'from .experiment_rec_trace import record_rec\n',
                   'from .experiment_rec_trace import record_rec\n'
                   'from .debug_performance import report_issue\n',
                   where='App/ocr/experiment_models.py')
    s=replace_once(s,'            self.errors.append(item)\n            return item\n',
                   "            self.errors.append(item)\n"
                   "            report_issue(source='ExperimentModelPool',operation=stage,\n"
                   "                         kind='model_exception',error=exc,\n"
                   "                         context={'model_id':model_id},fallback='preserve_original')\n"
                   "            return item\n",
                   where='App/ocr/experiment_models.py')
    return s


PATCHES = {
    'App/pipeline.py':patch_pipeline,
    'App/ocr/p1_batch.py':patch_batch,
    'App/geometry/service.py':patch_geometry,
    'App/ocr/ocr_performance.py':patch_performance,
    'App/ocr/cell_mode.py':patch_cell_mode,
    'App/ocr/experiment_models.py':patch_experiment_models,
}


def apply(project:Path, *, dry_run=False):
    project=project.resolve()
    home=Path(__file__).resolve().parent
    if (project/NEW_FILE).exists():
        raise FileExistsError(NEW_FILE+' already exists. Inspect or restore previous installation.')
    if not (project/'.env.example').is_file():
        raise FileNotFoundError('Project .env.example missing')
    originals={}; patched={}; source_blobs={}
    for name,fn in PATCHES.items():
        path=project/name
        if not path.is_file():raise FileNotFoundError(name)
        original=path.read_bytes()
        actual=blob_id(original.replace(b'\r\n', b'\n'))
        if actual != EXPECTED_BLOBS[name]:
            raise ValueError(f'{name}: not the GitHub source reviewed at {BASE_COMMIT}. '
                             f'Expected blob {EXPECTED_BLOBS[name]}, found {actual}. '
                             'Update the experiment for this revision; do not force patch.')
        eol='\r\n' if b'\r\n' in original else '\n'
        text=original.decode('utf-8-sig').replace('\r\n','\n')
        edited=fn(text)
        ast.parse(edited,filename=name)
        originals[name]=original
        patched[name]=edited.replace('\n',eol).encode('utf-8')
    module=(home/NEW_FILE).read_bytes()
    ast.parse(module.decode('utf-8'),filename=NEW_FILE)
    if dry_run:return list(patched)+[NEW_FILE]
    target=project/'data'/'experiments'/EXPERIMENT
    if target.exists():raise FileExistsError('Experiment backup already exists: '+str(target))
    backup=target/'backup'
    backup.mkdir(parents=True,exist_ok=False)
    manifest={'experiment':EXPERIMENT,'base_commit':BASE_COMMIT,
              'original_sha256':{},'patched_sha256':{},
              'new_file':NEW_FILE,'new_file_sha256':hashlib.sha256(module).hexdigest()}
    try:
        for name,original in originals.items():
            b=backup/name
            b.parent.mkdir(parents=True,exist_ok=True)
            b.write_bytes(original)
            manifest['original_sha256'][name]=hashlib.sha256(original).hexdigest()
            manifest['patched_sha256'][name]=hashlib.sha256(patched[name]).hexdigest()
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        for name,new in patched.items():
            dest=project/name
            tmp=dest.with_suffix(dest.suffix+'.debugtmp')
            tmp.write_bytes(new)
            os.replace(tmp,dest)
        (project/NEW_FILE).parent.mkdir(parents=True,exist_ok=True)
        (project/NEW_FILE).write_bytes(module)
    except BaseException:
        for name,original in originals.items():
            (project/name).write_bytes(original)
        if (project/NEW_FILE).exists(): (project/NEW_FILE).unlink()
        shutil.rmtree(target,ignore_errors=True)
        raise
    return list(patched)+[NEW_FILE]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='.')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    changed=apply(Path(args.project),dry_run=args.dry_run)
    print('Dry-run OK' if args.dry_run else 'Experiment installed')
    for name in changed:print('  '+name)
    if not args.dry_run:print('Set LOG_LEVEL=DEBUG in .env and restart the service.')

if __name__=='__main__':main()
