from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from App.ocr import live_receipt


def test_parallel_cells_publish_ordered_immutable_snapshots():
    updates = []
    session, token = live_receipt.activate(updates.append)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, live_receipt.observe, key, {'text': text})
                       for key, text in [((2, 'price'), '220'), ((1, 'name'), 'Товар')]]
            for future in futures:
                future.result()
        live_receipt.flush(session)
    finally:
        live_receipt.deactivate(token)
    assert updates[-1]['live_receipt']['rows'] == [
        {'row_id': 1, 'cells': {'name': 'Товар'}},
        {'row_id': 2, 'cells': {'price': '220'}},
    ]
    first = updates[0]['live_receipt']['rows']
    assert sum(len(row['cells']) for row in first) == 1
    assert [update['live_receipt']['seq'] for update in updates] == list(range(1, len(updates) + 1))


def test_disabled_session_and_publisher_failure_do_not_affect_ocr():
    session, token = live_receipt.activate(lambda _: None, enabled=False)
    try:
        live_receipt.observe((1, 'name'), {'text': 'Товар'})
        assert session is None
    finally:
        live_receipt.deactivate(token)
    def fail(_):
        raise RuntimeError('unavailable')
    session, token = live_receipt.activate(fail)
    try:
        live_receipt.observe((1, 'price'), {'text': '100'})
        live_receipt.flush(session)
    finally:
        live_receipt.deactivate(token)
