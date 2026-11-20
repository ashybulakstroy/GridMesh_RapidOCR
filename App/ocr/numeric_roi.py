"""Conservative numeric ink crop; currency and all visible digits are kept."""
import cv2
import numpy as np


def split_price_currency(image):
    """Remove only a separated, observed T-shaped currency glyph from OCR input."""
    if image is None or not image.size or min(image.shape[:2]) < 8:
        return image, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h,w = ink.shape
    ink[np.count_nonzero(ink,axis=1) > w*.94,:] = 0
    ink[:,np.count_nonzero(ink,axis=0) > h*.94] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    for component in range(1,count):
        if stats[component,cv2.CC_STAT_AREA] < 3:
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
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1-x0, y1-y0
    if width < 2 or height < 6:
        return image, {'changed': False, 'reason': 'tiny_ink'}
    padx = max(3, min(9, round(height * .28)))
    pady = max(3, min(8, round(height * .22)))
    lx, rx = max(0, x0-padx), min(w, x1+padx)
    ty, by = max(0, y0-pady), min(h, y1+pady)
    if rx-lx < max(12, w*.24) or by-ty < 12:
        return image, {'changed': False, 'reason': 'too_narrow'}
    if rx-lx >= w*.92 and by-ty >= h*.90:
        return image, {'changed': False, 'reason': 'full'}
    return image[ty:by, lx:rx].copy(), {
        'changed': True, 'reason': 'ink_roi', 'offset': [lx,ty],
        'size': [rx-lx,by-ty], 'original_size': [w,h]}
