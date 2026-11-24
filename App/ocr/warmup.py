"""Background, lane-aware RapidOCR startup warmup.

Primary: exactly PP-OCRv6 Tiny Cyrillic in table_names and table_numeric.
Secondary: optional existing v5 Cyrillic sessions for footer / E4; NO English.
Does not import the pipeline or heavy OCR dependencies on module import.
"""
from __future__ import annotations

from threading import Event, Lock, Thread
from time import perf_counter, sleep
import logging
from ..config import setting

_LOGGER = logging.getLogger('uvicorn.error')
_START_LOCK = Lock()
_STATE_LOCK = Lock()
_STARTED = Event()
_PRIMARY_READY = Event()
_FULL_READY = Event()
_STATE = {
    'status': 'not_started',
    'primary_ready': False,
    'full_ready': False,
    'primary_ms': None,
    'full_ms': None,
    'steps': [],
    'errors': [],
}


def snapshot():
    with _STATE_LOCK:
        return {**_STATE, 'steps': list(_STATE['steps']), 'errors': list(_STATE['errors'])}


def _set(**fields):
    with _STATE_LOCK:
        _STATE.update(fields)


def _track(name, fn):
    began = perf_counter()
    error = None
    try:
        fn()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        with _STATE_LOCK:
            _STATE['errors'].append({'step': name, 'error': error})
        return False
    finally:
        elapsed = round((perf_counter() - began) * 1000, 2)
        with _STATE_LOCK:
            _STATE['steps'].append({'name': name, 'wall_ms': elapsed, 'success': error is None})
        if error is None:
            _LOGGER.info('Прогрев модели завершён: %s — %.2f мс', name, elapsed)
        else:
            _LOGGER.warning('Ошибка прогрева модели: %s — %.2f мс; %s', name, elapsed, error)
    return True


def _synthetic(text):
    import cv2
    import numpy as np
    width = max(110, len(text) * 18 + 24)
    sample = np.full((48, width, 3), 255, dtype=np.uint8)
    cv2.putText(sample, text, (7, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return sample


def _warm_tiny(lane, sample, model_id='v6_tiny_cyrillic_optional'):
    # Match engine._run_cyrillic_rec's exact session cache key and same lane.
    # Do not use read_cell here: empty synthetic outputs must not trigger v5 fallback.
    from . import engine as ocr_engine
    from .experiment_models import ExperimentModelPool, load_catalog

    with ocr_engine.inference_lane(lane):
        with ocr_engine._lock:
            key = (lane, model_id)
            if key not in ocr_engine._models:
                pool = ExperimentModelPool()
                catalog = load_catalog()
                if model_id not in catalog:
                    raise KeyError(f'Missing OCR catalog model: {model_id}')
                ocr_engine._models[key] = pool._get_session(model_id, catalog[model_id])
            ocr_engine._models[key](sample, use_det=False, use_cls=False)


def _warm_v5(lane, sample, *, detector=False):
    from . import engine as ocr_engine
    with ocr_engine.inference_lane(lane):
        with ocr_engine._lock:
            worker = ocr_engine.footer_engine() if detector else ocr_engine.engine()
            worker(sample, use_det=detector, use_cls=False)


def _worker(is_ocr_busy=None):
    began = perf_counter()
    primary_success = True
    try:
        delay = max(0, int(setting('OCR_WARMUP_DELAY_MS', '0')))
        if delay:
            sleep(delay / 1000)
        _set(status='warming_primary')
        name_sample = _synthetic('LUXRAY 150W SMD')
        num_sample = _synthetic('440')

        primary_success = _track(
            'PP-OCRv6 Tiny Cyrillic / названия',
            lambda: _warm_tiny('table_names', name_sample)) and primary_success
        numeric_model = ('v6_tiny_cyrillic_optional' if setting('OCR_UNIFIED_TINY_ENABLE', '1') == '1'
                         else 'v6_tiny_official')
        primary_success = _track(
            ('PP-OCRv6 Tiny Cyrillic' if numeric_model == 'v6_tiny_cyrillic_optional'
             else 'PP-OCRv6 Tiny Official') + ' / цифры',
            lambda: _warm_tiny('table_numeric', num_sample, numeric_model)) and primary_success
    except Exception as exc:
        primary_success = False
        _LOGGER.warning('Не удалось подготовить прогрев основных моделей: %s: %s', type(exc).__name__, exc)
        with _STATE_LOCK:
            _STATE['errors'].append({'step': 'primary_setup',
                                     'error': f'{type(exc).__name__}: {exc}'})
    finally:
        _set(primary_ready=primary_success,
             primary_ms=round((perf_counter()-began)*1000, 2),
             status='warming_secondary' if primary_success else 'primary_failed')
        # Unblock OCR even on failure: existing engine fallback remains available.
        _PRIMARY_READY.set()

    try:
        if setting('OCR_WARMUP_SECONDARY', '1') == '1':
            amount = _synthetic('41250')
            for label, lane, det in (
                    ('PP-OCRv5 Cyrillic Mobile / итог REC', 'footer', False),
                    ('PP-OCRv5 / итог DET+REC', 'footer', True),
                    ('PP-OCRv5 Cyrillic Mobile / E4 REC', 'table', False)):
                # The secondary phase must not compete with a running OCR job
                # when a busy checker was supplied by the HTTP server.
                if is_ocr_busy is not None:
                    while is_ocr_busy():
                        sleep(0.05)
                _track(label, lambda lane=lane, det=det: _warm_v5(
                    lane, amount, detector=det))
    except Exception as exc:
        _LOGGER.warning('Не удалось подготовить прогрев дополнительных моделей: %s: %s', type(exc).__name__, exc)
        with _STATE_LOCK:
            _STATE['errors'].append({'step': 'secondary_setup',
                                     'error': f'{type(exc).__name__}: {exc}'})
    finally:
        _set(full_ready=True, full_ms=round((perf_counter()-began)*1000, 2),
             status=('ready' if not snapshot()['errors']
                     else 'ready_with_errors'))
        _FULL_READY.set()


def start(is_ocr_busy=None):
    """Return immediately; safe to call more than once in one process."""
    with _START_LOCK:
        if _STARTED.is_set():
            return
        _STARTED.set()
        if setting('OCR_WARMUP_ENABLE', '1') != '1':
            _set(status='disabled', primary_ready=True, full_ready=True)
            _PRIMARY_READY.set()
            _FULL_READY.set()
            return
        thread = Thread(target=_worker, args=(is_ocr_busy,),
                        name='ocr-model-warmup', daemon=True)
        try:
            thread.start()
        except Exception as exc:
            _LOGGER.warning('Не удалось запустить фоновый прогрев моделей: %s: %s', type(exc).__name__, exc)
            with _STATE_LOCK:
                _STATE['errors'].append({'step': 'thread_start',
                                         'error': f'{type(exc).__name__}: {exc}'})
            _set(status='thread_start_failed', full_ready=True)
            # Always release waiting OCR jobs; original lazy fallback survives.
            _PRIMARY_READY.set()
            _FULL_READY.set()


def wait_primary():
    """Only OCR jobs wait. Never block FastAPI startup or read-only endpoints."""
    if not _STARTED.is_set():
        return True  # Lifespan not entered (e.g. direct calls from unit tests).
    _PRIMARY_READY.wait()
    return snapshot()['primary_ready']


def wait_full():
    if not _STARTED.is_set():
        return True
    _FULL_READY.wait()
    return snapshot()['full_ready']
