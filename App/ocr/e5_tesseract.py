"""Bounded Tesseract KAZ fallback for unresolved numeric cells."""
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from types import SimpleNamespace

import cv2

from ..config import ROOT, setting
from .experiment_rec_trace import record_rec
from .numeric_roi import prepare_numeric_input
from .p1_numbers import parse_number
from .p1_recovery import _recalculate


def executable():
    choices = [setting('TESSERACT_EXE', ''),
               str(ROOT / 'models/tesseract/tesseract.exe'), shutil.which('tesseract'),
               r'C:\Program Files\Tesseract-OCR\tesseract.exe',
               r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']
    return next((str(p) for p in choices if p and Path(p).is_file()), None)


def read_tesseract(image, exe, *, row_id, field, variant, whitelist):
    started = perf_counter()
    text, error = '', None
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        return {'text':'', 'error':'png_encode_failure', 'variant':variant}
    command = [exe, 'stdin', 'stdout', '--tessdata-dir',
               str(ROOT / 'models/tesseract_kaz/tessdata'), '-l', 'kaz',
               '--oem', '1', '--psm', '7']
    if whitelist:
        command += ['-c', 'tessedit_char_whitelist=0123456789.,']
    try:
        proc = subprocess.run(command, input=encoded.tobytes(), capture_output=True,
                              timeout=.8, env={**os.environ, 'OMP_THREAD_LIMIT':'1'},
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        text = proc.stdout.decode('utf-8', errors='replace').strip()
        if proc.returncode:
            error = 'tesseract_exit_' + str(proc.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        error = type(exc).__name__
    elapsed = round((perf_counter()-started)*1000, 3)
    record_rec(model='tesseract_kaz', mode='REC_only', ms=elapsed,
               success=error is None, note=error, keys=[f'{row_id}:{field}'],
               stage='e5_tesseract_'+variant, image=image,
               result=SimpleNamespace(txts=[text], scores=None))
    return {'text':text, 'number':str(parse_number(text)) if parse_number(text) is not None else None,
            'error':error, 'variant':variant, 'wall_ms':elapsed}


def recover_with_tesseract(rows, crops, recovery, *, max_cells=3, book=None,
                           exclude_keys=(), accept_invalid=True, force_keys=()):
    """Try invalid fields and conflicting OCR readings; keep conflicts for review."""
    if setting('OCR_E5_TESSERACT', '1') != '1':
        return {'status':'disabled', 'details':[], 'candidates':[]}
    targets = {}
    for row in rows:
        for field in ('quantity', 'price', 'printed_total'):
            cell = row['cells'].get(field)
            if cell and (row['row_id'], field) in crops and parse_number(cell.get('text')) is None:
                targets[(row['row_id'], field)] = (0, row, field)
    for block in recovery.values():
        for detail in block.get('details', []):
            key = (detail.get('row_id'), detail.get('column_id'))
            before, after = parse_number(detail.get('original_text')), parse_number(detail.get('alternative_text'))
            if key in crops and before is not None and after is not None and before != after:
                row = next((r for r in rows if r['row_id'] == key[0]), None)
                if row and key[1] in row['cells']:
                    targets.setdefault(key, (1, row, key[1]))
    # E4 can find malformed readings after the initial recovery has finished.
    # Inspect actual OCR observations, without treating a missing value as text.
    if book is not None:
        rows_by_id = {row['row_id']: row for row in rows}
        for bucket in book.cells.values():
            key = (bucket.get('row_id'), bucket.get('field'))
            row = rows_by_id.get(key[0])
            if not row or key not in crops or key[1] not in row['cells']:
                continue
            if any(str(obs.get('text') or '').strip() and
                   parse_number(obs.get('text')) is None and
                   obs.get('model') != 'tesseract_kaz'
                   for obs in bucket.get('observations', [])):
                targets.setdefault(key, (1, row, key[1]))
    # E4_MATH_TARGET_EXPERIMENT_V1: operator-selected OCR targets are not automatic edits.
    rows_by_id_forced = {r['row_id']:r for r in rows}
    for key in force_keys:
        key = tuple(key)
        row = rows_by_id_forced.get(key[0])
        if row and key in crops and key[1] in row.get('cells', {}):
            targets[key] = (-1, row, key[1])
    for key in exclude_keys:
        targets.pop(key, None)
    if max_cells <= 0:
        return {'status':'budget_exhausted' if targets else 'not_needed', 'details':[], 'candidates':[]}
    if not targets:
        return {'status':'not_needed', 'details':[], 'candidates':[]}
    exe = executable()
    if not exe or not (ROOT / 'models/tesseract_kaz/tessdata/kaz.traineddata').is_file():
        return {'status':'unavailable', 'details':[], 'candidates':[], 'error':'tesseract_or_kaz_model_missing'}
    details, candidates = [], []
    for key, (_, row, field) in sorted(targets.items(), key=lambda item:item[1][0])[:max_cells]:
        original = row['cells'][field]
        image, roi = prepare_numeric_input(crops[key], field)
        enlarged = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        attempts = [read_tesseract(image, exe, row_id=key[0], field=field, variant='raw_digits', whitelist=True),
                    read_tesseract(enlarged, exe, row_id=key[0], field=field, variant='3x_plain', whitelist=False)]
        values = {parse_number(a['text']) for a in attempts if not a.get('error') and parse_number(a['text']) is not None}
        item = {'row_id':key[0], 'column_id':field, 'original_text':original['text'],
                'attempts':attempts, 'numeric_roi':roi, 'accepted':False}
        for attempt in attempts:
            if not attempt.get('error') and attempt.get('number') is not None:
                candidates.append({'row_id':key[0], 'field':field, 'text':attempt['text'],
                                   'model':'tesseract_kaz', 'variant':attempt['variant']})
        # Both presentations must actually read the same valid number. A
        # whitelist alone can silently remove a confusing printed character.
        if (accept_invalid and parse_number(original['text']) is None and len(values) == 1 and
                all(not a.get('error') and parse_number(a['text']) in values for a in attempts)):
            alternative = dict(original, text=attempts[1]['text'], confidence=None,
                               model_id='tesseract_kaz', strategy='e5_tesseract_consensus')
            row['cells'][field] = alternative
            _recalculate(row)
            if 'printed_total' not in row['cells'] or row.get('printed_line_match') is True:
                item.update(accepted=True, confirmation='matching_tesseract_presentations')
                alternative['ocr_attempts'] = list(original.get('ocr_attempts', [])) + [item.copy()]
            else:
                row['cells'][field] = original
                _recalculate(row)
        details.append(item)
    return {'status':'completed', 'details':details, 'candidates':candidates}
