"""Conservative numeric ink crop; currency and all visible digits are kept."""
import cv2
import numpy as np


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
