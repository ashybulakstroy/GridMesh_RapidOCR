"""Numeric ink ROI and evidence-based exclusion of a price currency glyph."""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import cv2
import numpy as np


@_deep_timed("ocr", "App/ocr/numeric_roi.py:split_price_currency")
def split_price_currency(image, *, clean_rule_fragments=False):
    """Remove only a separated, observed T-shaped currency glyph from OCR input."""
    if image is None or not image.size or min(image.shape[:2]) < 8:
        return image, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h,w = ink.shape
    ink[np.count_nonzero(ink,axis=1) > w*.94,:] = 0
    ink[:,np.count_nonzero(ink,axis=0) > h*.94] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    glyph_heights = [s[cv2.CC_STAT_HEIGHT] for s in stats[1:]
                     if s[cv2.CC_STAT_AREA] >= 8 and s[cv2.CC_STAT_HEIGHT] >= 6
                     and .15 <= s[cv2.CC_STAT_WIDTH]/s[cv2.CC_STAT_HEIGHT] <= 2
                     and s[cv2.CC_STAT_HEIGHT] < h*.9]
    small_height = max(3, float(np.median(glyph_heights))*.4) if glyph_heights else 3
    if clean_rule_fragments and glyph_heights:
        # A fragmented cell border can survive the full-column rule filter.
        # Require distributed support far taller than a glyph, so an isolated
        # narrow digit 1 is retained. Only the detection mask is changed.
        reference_height = float(np.median(glyph_heights))
        for edge in (stats[1:,0] <= 1, stats[1:,0]+stats[1:,2] >= w-1):
            fragments = np.flatnonzero(edge & (stats[1:,2] <= 2)) + 1
            if (len(fragments) and stats[fragments,3].sum() > h*.45 and
                    (stats[fragments,1]+stats[fragments,3]).max()-stats[fragments,1].min() > reference_height*2.5):
                ink[np.isin(labels,fragments)] = 0
    for component in range(1,count):
        bw = stats[component,cv2.CC_STAT_WIDTH]
        bh = stats[component,cv2.CC_STAT_HEIGHT]
        # Residual table-rule fragments can share the currency's x range and
        # make its measured height include the cell's upper border. Ignore
        # these shallow strokes only in the detection mask, never OCR pixels.
        shallow_rule = bh <= max(3, round(h*.08)) and bw >= max(4, 3*bh)
        if (stats[component,cv2.CC_STAT_AREA] < 3 or
                (clean_rule_fragments and (shallow_rule or bh < small_height))):
            ink[labels == component] = 0
    occupied = np.count_nonzero(ink,axis=0) > 0
    edges = np.diff(np.r_[False,occupied,False].astype(np.int8))
    groups = list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    if len(groups) < 2:
        return image, None
    left,right = groups[-1]
    ys = np.flatnonzero(np.count_nonzero(ink[:,left:right],axis=1))
    if not len(ys):
        return image, None
    top,bottom = int(ys[0]),int(ys[-1])+1
    glyph = ink[top:bottom,left:right] > 0
    gh,gw = glyph.shape
    if gh < 6 or gw < 4 or not .35 <= gw/gh <= 1.2 or left-groups[-2][1] < max(2,gh*.15):
        return image, None
    # A T/tenge sign has a wide upper bar and a narrow, centered vertical stem.
    # A final 7 normally has a sloping stem; 0/8 have wide lower strokes.
    upper = glyph[:max(2,round(gh*.4))]
    lower = glyph[round(gh*.45):]
    widths = lower.sum(axis=1)
    if np.count_nonzero(upper.sum(axis=1) >= gw*.75) < 2 or len(widths) < 3:
        return image, None
    if np.any(widths == 0) or np.max(widths) > gw*.55:
        return image, None
    centers = np.array([np.flatnonzero(row).mean() for row in lower])
    if centers.std() > gw*.10 or not gw*.25 <= centers.mean() <= gw*.70:
        return image, None
    if abs(centers[0]-centers[-1]) > gw*.15:
        return image, None
    cut = int(left)
    if cut < 5 or np.count_nonzero(ink[:,:cut]) < 8:
        return image, None
    digit_count = 0
    for a,b in groups[:-1]:
        glyph_rows = np.flatnonzero(np.count_nonzero(ink[:,a:b],axis=1))
        if not len(glyph_rows):
            continue
        digit_height = int(glyph_rows[-1]-glyph_rows[0]+1)
        if digit_height < gh*.5:  # decimal punctuation is not a digit
            continue
        if digit_height > gh*1.7:
            return image, None  # a ruling line contaminated this cell
        digit_count += max(1, round((b-a)/gw))
    return image[:,:cut].copy(), {'symbol':'₸', 'strategy':'separate_t_shape',
                                  'numeric_glyph_count':digit_count,
                                  'bbox':[int(left),top,int(right),bottom]}


@_deep_timed("ocr", "App/ocr/numeric_roi.py:exclude_price_currency")
def exclude_price_currency(image, roi):
    """V3's spatial split, applied BEFORE recognition, with cell-local bounds.

    Decimal punctuation stays in the image. An unconfirmed suffix stays intact;
    no text substitution or guessed digit is used to choose the rectangle.
    """
    if image is None or not image.size:
        return image, roi
    h,w = image.shape[:2]
    x,y = roi.get('offset', [0,0])
    cropped,currency = split_price_currency(image, clean_rule_fragments=True)
    info = {**roi, 'ocr_roi_bbox':[x,y,x+w,y+h],
            'coordinate_space':'gridmesh_cell_pixels',
            'currency_removed_before_ocr':False}
    if currency is None:
        info['currency_crop_status'] = 'not_confirmed'
        return image,info
    a,b,c,d = currency['bbox']
    info.update(changed=True, size=[cropped.shape[1],cropped.shape[0]], offset=[x,y],
                original_size=roi.get('original_size',[w,h]),
                ocr_roi_bbox=[x,y,x+cropped.shape[1],y+cropped.shape[0]],
                currency_removed_before_ocr=True, currency_crop_status='excluded',
                currency={**currency,'bbox':[x+a,y+b,x+c,y+d],
                          'removed_from_ocr_input':True})
    return cropped,info


@_deep_timed("ocr", "App/ocr/numeric_roi.py:prepare_numeric_input")
def prepare_numeric_input(image, field):
    """Use the same currency-free image preparation in primary/recovery paths."""
    cropped,roi = crop_numeric_ink(image)
    if field == 'price':
        from ..config import setting
        if setting('OCR_PRICE_CURRENCY_CROP','1') == '1':
            return exclude_price_currency(cropped,roi)
    return cropped,roi


@_deep_timed("ocr", "App/ocr/numeric_roi.py:_discard_isolated_rim_strokes")
def _discard_isolated_rim_strokes(ink):
    """Ignore isolated table-edge fragments for locating ink; never alter OCR pixels."""
    h, w = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    rim = 4
    candidates, body = [], []
    for index in range(1, count):
        x, y, cw, ch, area = stats[index]
        at_rim = x+cw <= rim or x >= w-rim or y+ch <= rim or y >= h-rim
        (candidates if at_rim else body).append(index)
    body = [index for index in body if stats[index, cv2.CC_STAT_AREA] >= 3]
    if not body or not candidates:
        return ink
    boxes = stats[body]
    left, top = boxes[:,0].min(), boxes[:,1].min()
    right = (boxes[:,0]+boxes[:,2]).max()
    bottom = (boxes[:,1]+boxes[:,3]).max()
    glyph_height = bottom-top
    if glyph_height < 6 or h < glyph_height*1.4:
        return ink
    remove = []
    for index in candidates:
        x, y, cw, ch, _ = stats[index]
        gap_x = max(left-(x+cw), x-right, 0)
        gap_y = max(top-(y+ch), y-bottom, 0)
        # A full-height digit at the side must remain, including a narrow 1.
        digit_like = ch >= glyph_height*.65 and cw >= 2
        if not digit_like and max(gap_x,gap_y) > max(3,glyph_height*.5):
            remove.append(index)
    if not remove:
        return ink
    cleaned = ink.copy()
    cleaned[np.isin(labels,remove)] = 0
    return cleaned


@_deep_timed("ocr", "App/ocr/numeric_roi.py:_compact_number_bounds")
def _compact_number_bounds(gray):
    """Find an isolated band of glyphs in a tall numeric cell."""
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 12)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    x, y, widths, heights, area = stats[1:].T
    interior = (x >= 3) & (x + widths <= w - 3) & (heights >= 8) & (area >= 3)
    if not interior.any():
        return None
    band_top = y[interior].min()
    band_bottom = (y + heights)[interior].max()
    outside_band = (y + heights < band_top - 3) | (y > band_bottom + 3)
    # Ignore only thin fragments attached to the outer edge of the crop.
    side = ((x == 0) | (x + widths == w)) & (widths <= 2)
    rails = side & ((outside_band & (heights < 8)) |
                    (heights > (band_bottom - band_top) * 2.5))
    # A fragmented rail has distributed support along the same outer edge.
    # Remove its short segments too, but preserve an isolated narrow digit 1.
    for at_edge in ((x == 0), (x + widths == w)):
        fragments = side & at_edge
        if (fragments.any() and heights[fragments].sum() > h*.45 and
                (y[fragments]+heights[fragments]).max()-y[fragments].min() > (band_bottom-band_top)*2.5):
            rails |= fragments
    rails |= (((y == 0) | (y + heights == h)) & (heights <= 2))
    # Include tiny decimal marks near the band, even when they occupy 1–2 pixels.
    nearby = ((x + widths >= x[interior].min() - 6) &
              (x <= (x + widths)[interior].max() + 6))
    keep = ((area >= 3) | (~outside_band & nearby)) & ~rails
    if not keep.any():
        return None
    glyph_height = int(heights[keep].max())
    if not 8 <= glyph_height <= h * .42:
        return None
    left, top = x[keep].min(), y[keep].min()
    right = (x + widths)[keep].max()
    bottom = (y + heights)[keep].max()
    if bottom - top > glyph_height * 1.45:
        return None  # Separate lines or ambiguous ink.
    if left < 3 or top < 3 or right > w - 3 or bottom > h - 3:
        return None  # The band may touch the original cell boundary.
    pad = max(3, round(glyph_height * .22))
    return max(0, left - pad), max(0, top - pad), min(w, right + pad), min(h, bottom + pad)


@_deep_timed("ocr", "App/ocr/numeric_roi.py:crop_numeric_ink")
def crop_numeric_ink(image):
    if image is None or image.ndim not in (2, 3) or not image.size:
        return image, {'changed': False, 'reason': 'empty'}
    h, w = image.shape[:2]
    if h < 12 or w < 12 or h * w > 160_000:
        return image, {'changed': False, 'reason': 'size_guard'}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    rows = np.count_nonzero(ink, axis=1)
    cols = np.count_nonzero(ink, axis=0)
    ink[rows > w * .94, :] = 0
    ink[:, cols > h * .94] = 0
    ys, xs = np.nonzero(ink)
    if xs.size < 8:
        return image, {'changed': False, 'reason': 'no_ink'}
    if xs.max()-xs.min()+1 >= w*.92 and ys.max()-ys.min()+1 >= h*.90:
        ink = _discard_isolated_rim_strokes(ink)
        ys, xs = np.nonzero(ink)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1-x0, y1-y0
    if width < 2 or height < 6:
        return image, {'changed': False, 'reason': 'tiny_ink'}
    padx = max(3, min(9, round(height * .28)))
    pady = max(3, min(8, round(height * .22)))
    lx, rx = max(0, x0-padx), min(w, x1+padx)
    ty, by = max(0, y0-pady), min(h, y1+pady)
    too_narrow = rx-lx < max(12, w*.24) or by-ty < 12
    if h >= 65 and (too_narrow or by-ty >= h*.6):
        compact = _compact_number_bounds(gray)
        if compact is not None:
            left, top, right, bottom = map(int, compact)
            return image[top:bottom, left:right].copy(), {
                'changed': True, 'reason': 'compact_number_band', 'offset': [left, top],
                'size': [right-left, bottom-top], 'original_size': [w,h]}
    if too_narrow:
        return image, {'changed': False, 'reason': 'too_narrow'}
    if rx-lx >= w*.92 and by-ty >= h*.90:
        return image, {'changed': False, 'reason': 'full'}
    return image[ty:by, lx:rx].copy(), {
        'changed': True, 'reason': 'ink_roi', 'offset': [lx,ty],
        'size': [rx-lx,by-ty], 'original_size': [w,h]}
