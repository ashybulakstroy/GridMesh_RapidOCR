"""Conservative extraction of numeric price candidates from isolated currency marks."""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import re
import cv2
import numpy as np

@_deep_timed("totals", "App/ocr/price_candidate.py:separated_suffix_with_hint")
def separated_suffix_with_hint(image: np.ndarray, *, has_currency_hint: bool):
    """Only an OCR-confirmed, isolated, T-shaped glyph may be removed.

    Do not strip the rightmost connected component just because it follows a gap:
    final 1 or 7 is a perfectly normal digit.
    """
    if not has_currency_hint or image is None or not image.size:
        return image, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    if min(h, w) < 12:
        return image, None
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask[np.count_nonzero(mask, axis=1) > w * .94, :] = 0
    mask[:, np.count_nonzero(mask, axis=0) > h * .94] = 0
    occupied = np.any(mask, axis=0)
    edges = np.diff(np.r_[False, occupied, False].astype(np.int8))
    groups = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if len(groups) < 2:
        return image, None
    left, right = map(int, groups[-1])
    gap = left - int(groups[-2][1])
    rows = np.flatnonzero(np.any(mask[:, left:right], axis=1))
    if not len(rows):
        return image, None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    glyph = mask[top:bottom, left:right] > 0
    gh, gw = glyph.shape
    if not (gh >= 6 and gw >= 4 and .30 <= gw / gh <= 1.25):
        return image, None
    if gap < max(2, round(gh * .15)) or left < max(8, round(w * .2)):
        return image, None
    # Typical T/tenge mark: a wide top crossbar and a relatively narrow
    # straight vertical stem in the bottom half. This rejects a slanted 7.
    upper = glyph[:max(2, round(gh * .4))]
    lower = glyph[round(gh * .45):]
    if lower.shape[0] < 3:
        return image, None
    top_rows = upper.sum(axis=1)
    if np.count_nonzero(top_rows >= gw * .7) < 1:
        return image, None
    widths = lower.sum(axis=1)
    if np.any(widths == 0) or np.max(widths) > gw * .6:
        return image, None
    centers = np.array([np.flatnonzero(row).mean() for row in lower])
    if not (np.std(centers) <= gw * .11 and abs(centers[0] - centers[-1]) <= gw * .15):
        return image, None
    if np.count_nonzero(mask[:, :left]) < 8:
        return image, None
    return image[:, :left].copy(), 'ocr_hint_and_t_shape'


@_deep_timed("totals", "App/ocr/price_candidate.py:prepare_price_candidate")
def prepare_price_candidate(image, observed_text):
    """Additional image only, preserving the original OCR candidate and decimals."""
    text=str(observed_text or '')
    if re.search(r'\d[,.]\d',text) or not re.search(r'[₸тТTt]\s*$',text):
        return None
    cropped,method=separated_suffix_with_hint(image,has_currency_hint=True)
    if method is None or cropped is None or not cropped.size:
        return None
    from .numeric_roi import crop_numeric_ink
    result,_=crop_numeric_ink(cropped)
    return result if result is not None and result.size else None
