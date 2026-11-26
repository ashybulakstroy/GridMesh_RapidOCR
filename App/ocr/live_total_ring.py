"""Display-only running sum for OCR. No changes to recognition or validation."""
from contextvars import ContextVar
from decimal import Decimal
from threading import Lock
from .p1_numbers import parse_number

_CURRENT = ContextVar('live_total_ring', default=None)
FIELDS = ('quantity', 'price')

class LiveTotal:
    def __init__(self, publish, expected):
        self.publish = publish
        self.expected = expected
        self.cells = {}
        self.lines = {}
        self.lock = Lock()
        self.started = False
        self.seq = 0
        self.last = None

    def _emit(self, stage):
        total = sum(self.lines.values(), Decimal(0))
        key = (total, len(self.lines), self.expected, stage)
        if key == self.last:
            return
        self.last = key
        self.seq += 1
        # Immutable, compact snapshot; publish under the lock to preserve order.
        self.publish({'live_total_ring': {'version': 1, 'seq': self.seq,
            'stage': stage, 'active': True, 'amount': str(total),
            'ready_rows': len(self.lines), 'expected_rows': self.expected}})

    def start(self):
        with self.lock:
            if not self.started:
                self.started = True
                self._emit('scanning')

    def observe_many(self, items):
        with self.lock:
            touched = set()
            for (row, field), result in items:
                if field not in FIELDS:
                    continue
                value = parse_number(result.get('text'))
                if value is not None and not value.is_finite():
                    value = None
                self.cells[(row, field)] = value
                touched.add(row)
            if not touched:
                return
            for row in touched:
                q, p = (self.cells.get((row, field)) for field in FIELDS)
                if q is None or p is None:
                    self.lines.pop(row, None)
                else:
                    self.lines[row] = q*p
            if self.started:
                self._emit('scanning')

    def sync_rows(self, rows, stage='verification'):
        with self.lock:
            if not self.started:
                return
            lines = {}
            for row in rows:
                raw = row.get('line_total')
                if raw is not None:
                    # Internal Decimal strings are already canonical; do not
                    # apply OCR text ambiguity rules a second time.
                    value = Decimal(str(raw))
                    if value.is_finite():
                        lines[row['row_id']] = value
            self.lines = lines
            self.expected = len(rows)
            self._emit(stage)


def activate(publish, expected, enabled):
    session = LiveTotal(publish, expected) if enabled else None
    return session, _CURRENT.set(session)


def deactivate(token):
    _CURRENT.reset(token)


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception:
        # Optional display must never cause OCR failure or change OCR data.
        pass


def started(field):
    session = _CURRENT.get()
    if session is not None and field in FIELDS:
        _safe(session.start)


def observe(key, result):
    session = _CURRENT.get()
    if session is not None and key[1] in FIELDS:
        _safe(session.observe_many, [(key, result)])


def observe_all(results):
    session = _CURRENT.get()
    if session is not None:
        _safe(session.observe_many, results.items())


def sync(session, rows, stage='verification'):
    if session is not None:
        _safe(session.sync_rows, rows, stage)
