"""Optional GridMesh performance diagnostics; no live file writes.

A ContextVar carries per-job instrumentation through copy_context() to both
OCR lanes. The normal INFO path keeps only a tiny structured error sink.
"""
from __future__ import annotations

from contextvars import ContextVar
from importlib import metadata
from threading import Lock
from time import perf_counter
import math
import os
import platform
import traceback

import cv2
import psutil

from ..config import setting, positive_int

_CONTEXT = ContextVar('gridmesh_performance_context_v2', default=None)


class TelemetryContext:
    def __init__(self, enabled: bool, job_id: str | None):
        self.profile = None
        self.debug = DebugPerformance(job_id) if enabled else None


def activate_telemetry(job_id=None):
    enabled = setting('LOG_LEVEL', 'INFO').strip().upper() == 'DEBUG'
    return _CONTEXT.set(TelemetryContext(enabled, job_id))


def deactivate_telemetry(token):
    _CONTEXT.reset(token)


def bind_profile(profile, job_id):
    context = _CONTEXT.get()
    if context is not None:
        context.profile = profile
        if context.debug is not None:
            context.debug.job_id = job_id


def get_debug():
    context = _CONTEXT.get()
    return None if context is None else context.debug


def report_issue(*, source, operation, kind, severity='error', error=None,
                 context=None, fallback=None, duration_ms=None):
    """Persist concise Python/library failures at INFO, detailed at DEBUG.

    Routine multicell fallback is not an issue; call only on anomalous results.
    """
    ctx = _CONTEXT.get()
    if ctx is None:
        return
    if ctx.profile is not None:
        ctx.profile.note_issue(source=source, operation=operation, kind=kind,
                               severity=severity, error=error,
                               context=context, fallback=fallback)
    if ctx.debug is not None:
        ctx.debug.add_issue(source=source, operation=operation, kind=kind,
                            severity=severity, error=error,
                            context=context, fallback=fallback,
                            duration_ms=duration_ms)


class DebugPerformance:
    def __init__(self, job_id=None):
        self.job_id = job_id
        self.started = perf_counter()
        self.proc = psutil.Process()
        self.lock = Lock()
        self.initial_rss = self._rss_mb()
        self.max_observed_rss = self.initial_rss
        self.configuration = {}
        self.cells = []
        self.batches = []
        self.fallbacks = []
        self.issues = []
        self.geometry = []
        self.crop_summary = {}
        self.lanes = []

    def _rss_mb(self):
        return round(self.proc.memory_info().rss / 1048576, 2)

    def _sample_rss(self):
        value = self._rss_mb()
        with self.lock:
            self.max_observed_rss = max(self.max_observed_rss, value)
        return value

    @staticmethod
    def _version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return 'not_installed'

    def capture_runtime(self, image_shape):
        config = {
            'log_level': 'DEBUG',
            'table_mode': setting('OCR_TABLE_MODE', 'cell'),
            'primary_recognizer': 'PP-OCRv5-cyrillic-mobile',
            'optional_numeric_models': setting('OCR_E2_NUMERIC_MODELS', ''),
            'e2_enabled': setting('OCR_E2_ENABLE','0') == '1',
            'e4_enabled': setting('OCR_E4_ENABLE','0') == '1',
            'onnx_intra_op_threads_configured': positive_int('OCR_ONNX_THREADS', 2),
            'onnx_inter_op_threads_configured': 1,
            'opencv_threads_configured': positive_int('OCR_OPENCV_THREADS', 2),
            'opencv_threads_reported': cv2.getNumThreads(),
            'rec_batch_num_configured': positive_int('OCR_REC_BATCH_NUM', 8),
            'numeric_roi': setting('OCR_NUMERIC_ROI', '1') == '1',
            'source_image_shape_hw': [int(image_shape[0]),int(image_shape[1])],
            'python_version': platform.python_version(),
            'rapidocr_version': self._version('rapidocr'),
            'onnxruntime_version': self._version('onnxruntime'),
            'opencv_version': cv2.__version__,
            'logical_cpus': os.cpu_count(),
        }
        with self.lock:
            self.configuration = config
        self._sample_rss()

    @staticmethod
    def estimate_padding(shapes):
        """Estimated padding at recognizer fixed H=48; NOT measured ONNX waste."""
        if not shapes:
            return {'estimated_input_width_px':0,'estimated_padding_fraction':0.0}
        ratios = [float(x[1])/max(1.,float(x[0])) for x in shapes]
        width=max(320,int(math.ceil(48 * max(ratios))))
        useful=sum(min(width,int(math.ceil(48*r))) for r in ratios)
        return {'estimated_input_width_px':width,
                'estimated_padding_fraction':round(max(0.,1.-useful/(len(ratios)*width)),4),
                'min_wh_ratio':round(min(ratios),3),'max_wh_ratio':round(max(ratios),3)}

    def add_crop(self, field, duration_ms, shape):
        h,w=int(shape[0]),int(shape[1])
        with self.lock:
            row=self.crop_summary.setdefault(str(field), {
                'count':0,'wall_ms':0.0,'total_pixels':0,'largest_image_hw':[0,0]})
            row['count']+=1
            row['wall_ms']=round(row['wall_ms']+duration_ms,3)
            row['total_pixels']+=h*w
            if h*w>math.prod(row['largest_image_hw']):
                row['largest_image_hw']=[h,w]

    def add_geometry(self, step, duration_ms, detail=None):
        row={'step':str(step),'wall_ms':round(duration_ms,3),
             'rss_after_mb':self._sample_rss()}
        if detail:row['detail']=detail
        with self.lock:self.geometry.append(row)

    def add_cell(self, cell, duration_ms, *, status='ok', strategy=None,
                 mode='cell', error=None):
        row={'cell_id':f'{cell.row_id}:{cell.column_id}',
             'field':str(cell.column_id),'mode':mode,
             'image_hw':[int(cell.image.shape[0]),int(cell.image.shape[1])],
             'wall_ms':round(duration_ms,3),'status':status,
             'rss_after_mb':self._sample_rss()}
        if strategy:row['strategy']=str(strategy)
        if error:row['error']=str(error)[:300]
        with self.lock:self.cells.append(row)

    def add_batch(self, group, cells, duration_ms, *, status, error=None):
        shapes=[c.image.shape for c in cells]
        row={'group':str(group),'batch_size':len(cells),
             'cell_ids':[f'{c.row_id}:{c.column_id}' for c in cells],
             'shapes_hw':[[int(s[0]),int(s[1])] for s in shapes],
             'recognize_txt_call_wall_ms':round(duration_ms,3),'status':status,
             'rss_after_mb':self._sample_rss(),**self.estimate_padding(shapes)}
        if error:row['error']=str(error)[:300]
        with self.lock:self.batches.append(row)

    def add_fallback(self, cell, duration_ms, reason, strategy):
        row={'cell_id':f'{cell.row_id}:{cell.column_id}',
             'reason':reason,'strategy':strategy,'wall_ms':round(duration_ms,3)}
        with self.lock:self.fallbacks.append(row)

    def add_lane(self, lane, wall_ms, status='ok', error=None):
        row={'lane':str(lane),'wall_ms':(round(wall_ms,3) if wall_ms is not None else None),'status':status}
        if error:row['error']=str(error)[:300]
        with self.lock:self.lanes.append(row)

    def add_issue(self, *, source, operation, kind, severity='error', error=None,
                  context=None, fallback=None, duration_ms=None):
        item={'source':str(source),'operation':str(operation),'kind':str(kind),
              'severity':str(severity),'at_ms':round((perf_counter()-self.started)*1000,3)}
        if error is not None:
            item['exception_type']=type(error).__name__
            item['message']=str(error)[:500]
            if error.__traceback__ is not None:
                item['traceback']=''.join(traceback.format_exception(
                    type(error),error,error.__traceback__))[-2000:]
        if context:item['context']=context
        if fallback:item['fallback']=fallback
        if duration_ms is not None:item['wall_ms']=round(duration_ms,3)
        with self.lock:self.issues.append(item)

    def finish(self, rec_trace=None):
        rss=self._sample_rss()
        with self.lock:
            records=list(self.cells)
            batches=list(self.batches)
            fallbacks=list(self.fallbacks)
            geometry=list(self.geometry)
            issues=list(self.issues)
            crops=dict(self.crop_summary)
            lanes=list(self.lanes)
            conf=dict(self.configuration)
            peak=self.max_observed_rss
        model_summary={}
        events=(rec_trace or {}).get('events',[])
        for event in events:
            key=(str(event.get('model_id','unknown')),str(event.get('mode','unknown')))
            row=model_summary.setdefault('|'.join(key),{
                'model_id':key[0],'mode':key[1],'calls':0,'total_call_wall_ms':0.0,
                'failed_calls':0,'cell_exposures':0})
            row['calls']+=1
            row['total_call_wall_ms']+=float(event.get('wall_ms',0))
            row['failed_calls']+=not bool(event.get('success',True))
            row['cell_exposures']+=len(event.get('cells',[]))
        for row in model_summary.values():
            row['total_call_wall_ms']=round(row['total_call_wall_ms'],3)
        return {
            'schema':'gridmesh-debug-performance-v2',
            'job_id':self.job_id,
            'configuration':conf,
            'ocr_cells':records,
            'ocr_batches':batches,
            'ocr_lanes':lanes,
            'ocr_fallbacks':fallbacks,
            'geometry_substeps':geometry,
            'cell_crop_summary':crops,
            'model_calls_from_rec_trace':model_summary,
            'issues':issues,
            'memory_mb':{'rss_start':self.initial_rss,'rss_last':rss,
                         'max_rss_observed_at_checkpoints':peak},
            'notes':[
                'Timing for each OCR cell/batch covers the entire Python call, not pure ONNX kernel time.',
                'Model call times can overlap between footer/table lanes; do NOT sum for total wall time.',
                'RecTrace excludes opaque internal detector-recognizer calls and may include shared contexts.',
                'Only checkpoint RSS measured; not an exact memory high-water mark.',
                'No periodic writes: profile is saved after success or caught failure.',
                'INFO records short Python/library issues. DEBUG additionally records call and input diagnostics.',
            ],
        }
