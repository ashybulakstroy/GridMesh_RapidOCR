"""Fast pixel-based numeric line segmentation; no DET, no arithmetic fabrication.

Only OCR-observed digits are joined. A geometrically uncertain cell returns None
so the application's established OCR fallback remains available.
"""
from __future__ import annotations
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import re
import math
import cv2
import numpy as np

_NUMERIC = re.compile(r"^[0-9][0-9\s.,]*$")


@_deep_timed("ocr", "App/ocr/numeric_multiline_fast.py:bands_from_pixels")
def bands_from_pixels(image):
    if image is None or not image.size or min(image.shape[:2]) < 9:
        return []
    h, w = image.shape[:2]
    if h < 28 or w < 7:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw[:2, :] = 0
    bw[-2:, :] = 0
    bw[:, :2] = 0
    bw[:, -2:] = 0
    ink = np.count_nonzero(bw, axis=1)
    occupied = ink >= max(2, round(w * 0.025))
    edges = np.diff(np.r_[False, occupied, False].astype(np.int8))
    raw = [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    bands = [(a, b) for a, b in merged if b - a >= 5]
    if not 2 <= len(bands) <= 3:
        return []
    # Distinct rows must really be separated by white pixels.
    if any(b[0] - a[1] < 4 for a, b in zip(bands, bands[1:])):
        return []
    if bands[-1][1] - bands[0][0] < 20:
        return []
    return bands


@_deep_timed("ocr", "App/ocr/numeric_multiline_fast.py:read_numeric_multiline")
def read_numeric_multiline(image, read_rec, *, min_score=0.50):
    """Return OCR-observed concatenated digits and individual line evidence.

    read_rec: callable(image) -> RapidOCR rec result with .txts, .scores.
    Full original column width is retained; no clipping of leftmost digits.
    """
    bands = bands_from_pixels(image)
    if not bands:
        return None
    readings = []
    for a, b in bands:
        top, bottom = max(0, a - 2), min(image.shape[0], b + 2)
        try:
            result = read_rec(image[top:bottom, :])
        except Exception:
            # Preserve the current full-cell recovery route if one line fails.
            return None
        texts = getattr(result, 'txts', None)
        scores = getattr(result, 'scores', None)
        if texts is None or scores is None or len(texts) != 1 or len(scores) != 1:
            return None
        raw = str(texts[0]).strip()
        confidence = float(scores[0])
        if not _NUMERIC.fullmatch(raw) or not math.isfinite(confidence) or confidence < min_score:
            return None
        cleaned = re.sub(r'\s+', '', raw)
        if not cleaned or not any(ch.isdigit() for ch in cleaned):
            return None
        readings.append({'text': raw, 'normalized': cleaned,
                         'confidence': confidence, 'top': a, 'bottom': b})
    joined = ''.join(r['normalized'] for r in readings)
    # Decimal separators can only occur in last visual line and only once.
    punctuation = sum(c in ',.' for c in joined)
    if punctuation > 1 or (punctuation == 1 and
                            any(any(c in ',.' for c in r['normalized']) for r in readings[:-1])):
        return None
    return {'text': joined, 'confidence': min(r['confidence'] for r in readings),
            'lines': readings, 'strategy': 'numeric_multiline_rec_only',
            'line_count': len(readings)}
