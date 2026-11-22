"""Strict integer-cell ROI preparation for a read-only digits-only OCR experiment.

Currency is removed only when a visually isolated suffix has enough image
support. Never attempt to parse receipt footer decimals or alter original OCR.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re
import cv2
import numpy as np

FIELDS = frozenset(('quantity', 'price', 'printed_total'))
CURRENCY_SUFFIX = re.compile(r'(?:\s*[₸тТTt])\s*$')


@dataclass
class Prepared:
    image: Optional[np.ndarray]
    status: str
    currency: Optional[str] = None
    method: Optional[str] = None
    original_hw: Optional[tuple[int, int]] = None
    candidate_hw: Optional[tuple[int, int]] = None
    warning: Optional[str] = None
    crop_metadata: Optional[dict] = None


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


def prepare_integer_cell(image: np.ndarray, *, field: str,
                         baseline_text: str = '', baseline_ocr_text: str = '',
                         currency_hint: bool = False) -> Prepared:
    """Prepare an integer candidate. Unknown OCR text is *not* a reason to skip.

    It may be the exact reason why a second digits-only OCR model is useful.
    However, a decimal OCR result is flagged for review since only footer
    amounts are expected to contain genuine fractions.
    """
    if field not in FIELDS:
        return Prepared(None, 'excluded_field')
    if image is None or not hasattr(image, 'size') or not image.size:
        return Prepared(None, 'empty_image')
    h, w = map(int, image.shape[:2])
    original = h, w
    raw = str(baseline_text or '')
    from_ocr = str(baseline_ocr_text or '')
    has_currency_hint = (currency_hint or bool(CURRENCY_SUFFIX.search(raw)) or
                         bool(CURRENCY_SUFFIX.search(from_ocr)))
    decimal_warning = 'unexpected_decimal_in_table_cell' if re.search(r'\d[,.]\d', raw) else None
    if decimal_warning:
        return Prepared(None, 'unexpected_decimal_separator', original_hw=original,
                        warning=decimal_warning)
    candidate = image
    method, symbol = 'none', None
    if field in ('price', 'printed_total'):
        # Production GridMesh helper, unchanged since previous experiment.
        from App.ocr.numeric_roi import split_price_currency
        cropped, info = split_price_currency(candidate)
        if info is not None and cropped is not None and cropped.size:
            candidate = cropped
            method, symbol = 'project_t_shape', info.get('symbol', '₸')
        elif has_currency_hint:
            cropped, strategy = separated_suffix_with_hint(candidate,
                                                            has_currency_hint=True)
            if strategy is not None:
                candidate = cropped
                method, symbol = strategy, 'currency_suffix'
            else:
                return Prepared(None, 'currency_not_safely_separated',
                                original_hw=original, warning='currency_hint_unresolved')
    if candidate is None or not candidate.size or min(candidate.shape[:2]) < 5:
        return Prepared(None, 'crop_too_small', original_hw=original)
    from App.ocr.numeric_roi import crop_numeric_ink
    tightened, metadata = crop_numeric_ink(candidate)
    if tightened is not None and tightened.size:
        candidate = tightened
    return Prepared(candidate, 'ready', symbol, method, original,
                    tuple(map(int, candidate.shape[:2])), None, metadata)
