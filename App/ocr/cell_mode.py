"""Independent footer, product-name and numeric OCR workers."""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event
from time import perf_counter
from .engine import inference_lane, onnx_threads
from .debug_performance import get_debug, report_issue
from .p1_batch import read_batches
from ..config import setting


@_deep_timed("ocr", "App/ocr/cell_mode.py:read_columns")
def read_columns(cells, cell_reader, *, batch_size=8):
    """Overlap names with numbers without sharing mutable recognizer sessions."""
    names = [cell for cell in cells if cell.column_id == 'name']
    numeric = [cell for cell in cells if cell.column_id != 'name']
    if setting('OCR_TABLE_PARALLEL', '1') != '1' or not names or not numeric:
        result, stats = cell_reader(cells, batch_size=batch_size)
        stats['columns_parallel'] = False
        return result, stats

    origin = perf_counter()
    debug = get_debug()

    def recognize(lane, items):
        with inference_lane(lane):
            start = perf_counter()
            try:
                result, stats = cell_reader(items, batch_size=batch_size)
            except Exception as exc:
                report_issue(source='parallel_ocr', operation=lane,
                             kind='lane_exception', error=exc, fallback='batch')
                # Retry only the failed column group; keep the other result.
                result, stats = read_batches(items, batch_size=batch_size)
                stats.update(cell_fallback=len(items),
                             column_error=f'{type(exc).__name__}: {exc}')
            end = perf_counter()
            timing = {'start_ms': round((start-origin)*1000, 3),
                      'end_ms': round((end-origin)*1000, 3),
                      'wall_ms': round((end-start)*1000, 3),
                      'onnx_threads': onnx_threads()}
            if debug is not None:
                debug.add_lane(lane, timing['wall_ms'],
                               status='batch_fallback' if stats.get('column_error') else 'ok',
                               error=stats.get('column_error'))
            return result, stats, timing

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='table-ocr') as executor:
        names_future = executor.submit(copy_context().run, recognize, 'table_names', names)
        numeric_future = executor.submit(copy_context().run, recognize, 'table_numeric', numeric)
        name_results, name_stats, name_time = names_future.result()
        numeric_results, numeric_stats, numeric_time = numeric_future.result()

    # Keep ROI budgets and measurements exclusively from the numeric worker.
    # Its CPU preparation is no longer delayed by the product-name recognition.
    stats = dict(numeric_stats)
    for key in ('requested', 'cell_calls', 'cell_fallback', 'batch_calls',
                'multiline_fallback', 'batch_fallback'):
        stats[key] = numeric_stats.get(key, 0) + name_stats.get(key, 0)
    stats['errors'] = numeric_stats.get('errors', []) + name_stats.get('errors', [])
    for group, group_stats in (('names', name_stats), ('numeric', numeric_stats)):
        if group_stats.get('column_error'):
            stats['errors'].append(f"{group}: {group_stats['column_error']}")
    stats.update(columns_parallel=True, column_timings={'names':name_time, 'numeric':numeric_time},
                 names_wall_ms=name_time['wall_ms'], numeric_wall_ms=numeric_time['wall_ms'],
                 column_overlap_ms=round(max(0, min(name_time['end_ms'],numeric_time['end_ms']) -
                                                 max(name_time['start_ms'],numeric_time['start_ms'])), 3))
    # Preserve original request order so downstream tie-breaking is deterministic.
    combined = {**name_results, **numeric_results}
    return {(c.row_id,c.column_id):combined[(c.row_id,c.column_id)] for c in cells}, stats


@_deep_timed("ocr", "App/ocr/cell_mode.py:read_parallel")
def read_parallel(cells, footer_reader, cell_reader, *, batch_size=8):
    started = Event()
    _debug=get_debug()
    def footer():
        with inference_lane('footer'):
            started.set()
            begin = perf_counter()
            result = footer_reader()
            result['wall_ms'] = round((perf_counter()-begin)*1000, 3)
            return result
    def table():
        started.wait()
        begin = perf_counter()
        result, stats = read_columns(cells, cell_reader, batch_size=batch_size)
        stats['table_wall_ms'] = round((perf_counter()-begin)*1000, 3)
        return result, stats
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='receipt-ocr') as executor:
        footer_future = executor.submit(copy_context().run, footer)
        table_future = executor.submit(copy_context().run, table)
        try:
            readouts, stats = table_future.result()
        except Exception as exc:
            report_issue(source='parallel_ocr',operation='table_lane',
                         kind='lane_exception',error=exc,
                         fallback='batch')
            readouts, stats = read_batches(cells, batch_size=batch_size)
            stats.update(mode='batch_fallback', cell_fallback=len(cells),
                         cell_error=f'{type(exc).__name__}: {exc}')
        try:
            footer_result = footer_future.result()
            stats['footer_lane_wall_ms'] = footer_result['wall_ms']
            stats['footer_wall_ms'] = footer_result.get('footer_read_ms',footer_result['wall_ms'])
            if 'candidate_read_ms' in footer_result:
                stats['v5_candidate_wall_ms'] = footer_result['candidate_read_ms']
        except Exception as exc:
            report_issue(source='parallel_ocr',operation='footer_lane',
                         kind='lane_exception',error=exc,
                         fallback='legacy_footer')
            # Retry independently in the legacy footer route after table assembly.
            footer_result = None
            stats['footer_error'] = f'{type(exc).__name__}: {exc}'
    stats['parallel_lanes'] = 3 if stats.get('columns_parallel') else 2
    if _debug is not None:
        _debug.add_lane('table',stats.get('table_wall_ms'),
                        status='batch_fallback' if stats.get('mode')=='batch_fallback' else 'ok')
        _debug.add_lane('footer',stats.get('footer_wall_ms'),
                        status='error' if footer_result is None else 'ok',
                        error=stats.get('footer_error'))
    return readouts, stats, footer_result
