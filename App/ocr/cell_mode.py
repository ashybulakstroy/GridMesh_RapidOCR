"""Two independent OCR lanes: start footer first, then column-major table."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event
from time import perf_counter
from .engine import inference_lane
from .debug_performance import get_debug, report_issue
from .p1_batch import read_batches


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
        result, stats = cell_reader(cells, batch_size=batch_size)
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
            stats['footer_wall_ms'] = footer_result['wall_ms']
        except Exception as exc:
            report_issue(source='parallel_ocr',operation='footer_lane',
                         kind='lane_exception',error=exc,
                         fallback='legacy_footer')
            # Retry independently in the legacy footer route after table assembly.
            footer_result = None
            stats['footer_error'] = f'{type(exc).__name__}: {exc}'
    stats['parallel_lanes'] = 2
    if _debug is not None:
        _debug.add_lane('table',stats.get('table_wall_ms'),
                        status='batch_fallback' if stats.get('mode')=='batch_fallback' else 'ok')
        _debug.add_lane('footer',stats.get('footer_wall_ms'),
                        status='error' if footer_result is None else 'ok',
                        error=stats.get('footer_error'))
    return readouts, stats, footer_result
