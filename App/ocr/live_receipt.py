"""Publish read-only snapshots of primary OCR cells for the live receipt."""
from contextvars import ContextVar
from threading import Lock
from time import monotonic

_CURRENT = ContextVar('live_receipt', default=None)
_FIELDS = frozenset(('name', 'quantity', 'price', 'printed_total'))


class LiveReceipt:
    def __init__(self, publish, interval=.075):
        self.publish = publish
        self.interval = interval
        self.lock = Lock()
        self.rows = {}
        self.seq = 0
        self.last_sent = 0.0
        self.dirty = False

    def _emit(self, force=False):
        now = monotonic()
        if not self.dirty or (not force and now - self.last_sent < self.interval):
            return
        self.seq += 1
        snapshot = [{'row_id': row_id, 'cells': dict(cells)}
                    for row_id, cells in sorted(self.rows.items())]
        self.publish({'live_receipt': {'seq': self.seq, 'rows': snapshot}})
        self.last_sent = now
        self.dirty = False

    def observe_many(self, items):
        with self.lock:
            for (row_id, field), item in items:
                if field not in _FIELDS or not isinstance(item, dict):
                    continue
                text = str(item.get('text') or '')
                row = self.rows.setdefault(int(row_id), {})
                if row.get(field) != text:
                    row[field] = text
                    self.dirty = True
            self._emit()

    def flush(self):
        with self.lock:
            self._emit(force=True)


def activate(publish, enabled=True):
    session = LiveReceipt(publish) if enabled else None
    return session, _CURRENT.set(session)


def deactivate(token):
    _CURRENT.reset(token)


def observe(key, result):
    session = _CURRENT.get()
    if session is not None:
        try:
            session.observe_many([(key, result)])
        except Exception:
            pass  # Display updates must not interfere with OCR.


def observe_all(results):
    session = _CURRENT.get()
    if session is not None:
        try:
            session.observe_many(results.items())
        except Exception:
            pass


def flush(session):
    if session is not None:
        try:
            session.flush()
        except Exception:
            pass
