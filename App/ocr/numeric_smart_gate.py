"""Conservative single-line veto. Never change OCR pixels or fabricate digits."""
import math
import re
import cv2
import numpy as np


def single_line_evidence(image):
    """Return (confirmed, reason). Ambiguous geometry preserves old fallback."""
    if image is None or not image.size or image.ndim not in (2, 3):
        return False, 'invalid_image'
    h, w = image.shape[:2]
    if h < 28 or w < 12 or h*w > 160000:
        return False, 'size_guard'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bw[:2, :] = bw[-2:, :] = 0
    bw[:, :2] = bw[:, -2:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    # A long thin edge rule is not a digit. Estimate glyph height from
    # compact non-edge components; refuse to guess when none are available.
    candidates = [int(ch) for x,y,cw,ch,a in stats[1:]
                  if a >= 8 and ch >= 8 and .13 <= cw/ch <= 1.5
                  and ch < .75*h and x > 2 and x+cw < w-2]
    if not candidates:
        return False, 'no_reference_glyph'
    ref = float(np.median(candidates))
    # Ignore only isolated, elongated boundary rules, not short strokes that
    # could belong to a second digit. Keep all other ink, including decimals.
    for i,(x,y,cw,ch,a) in enumerate(stats[1:],1):
        horizontal = (y <= .25*h or y+ch >= .75*h) and cw >= max(.45*w, 2.3*ref) and ch <= max(6,.38*ref) and cw >= 6*ch
        vertical = (x <= 3 or x+cw >= w-3) and ch >= 2.1*ref and cw <= 2
        if horizontal or vertical:
            bw[labels==i] = 0
    # Fragmented thin horizontal rules can be composed of short disconnected
    # pieces. Require a wide aggregate envelope AND a very shallow edge band.
    rows = np.count_nonzero(bw, axis=1) > 0
    edges = np.diff(np.r_[False,rows,False].astype(np.int8))
    raw = list(zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)))
    for a,b in raw:
        if (a <= .25*h or b >= .75*h) and b-a <= max(2, .20*ref):
            xx = np.flatnonzero(np.any(bw[a:b],axis=0))
            if len(xx) and xx[-1]-xx[0]+1 >= .65*w and len(xx) >= .20*w:
                bw[a:b] = 0
    # A single compact vertical envelope is strong evidence. Isolated ink
    # away from it (even a small second digit) makes the decision ambiguous.
    ys,xs = np.nonzero(bw)
    if len(ys) < 8:
        return False, 'insufficient_ink'
    span = int(ys.max()-ys.min()+1)
    if not .70*ref <= span <= 1.30*ref:
        return False, 'multiple_or_ambiguous_bands'
    if ys.min() <= 2 or ys.max() >= h-3:
        return False, 'ink_touches_edge'
    return True, 'single_compact_digit_band'


def may_skip_detector(image, text, confidence, *, threshold=.98, evidence=None):
    """Use both strong physical evidence and high-confidence complete numeric REC."""
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(score) or score < threshold:
        return False
    if not re.fullmatch(r'[0-9]+(?:[,.][0-9]+)?', str(text).strip()):
        return False
    return (evidence if evidence is not None else single_line_evidence(image))[0]
