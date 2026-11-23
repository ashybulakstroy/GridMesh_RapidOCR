"""Experimental geometrically isolated footer OCR from observed horizontal rules.

No inferred amounts: OCR is performed on an actual source-image strip between
FOOTER horizontal rules. Unconfirmed geometry leaves the old route untouched.
"""
from __future__ import annotations

from time import perf_counter
import re
import cv2
import numpy as np

_NUMERIC_LINE = re.compile(r"\s*[\d\s.,]+\s*(?:[ТT₸]|тг\.?)?\s*\Z", re.I)


def guide_from_grid(rect, image_shape):
    """Original-photo price column and lowest detected grid line, not a guess.

    Return None if a required observed geometric feature is unavailable.
    Vertical/horizontal curves are sampled in absolute source coordinates.
    """
    try:
        grid = rect.info['source_grid']
        h, w = image_shape[:2]
        vertical, horizontal = grid['vertical'], grid['horizontal']
        if len(vertical) < 4 or len(horizontal) < 3:
            return None
        bottom_hint = float(horizontal[-1].position)
        normal_y = (bottom_hint - h / 2.0) / h
        left = float(vertical[-2].eval(normal_y))
        right = float(vertical[-1].eval(normal_y))
        if not (np.isfinite(left) and np.isfinite(right) and 22 <= right - left <= w*.7):
            return None
        mid = (left + right)/2
        norm_x = (mid-w/2.0)/w
        boundary_y = float(horizontal[-1].eval(norm_x))
        preceding_y = float(horizontal[-2].eval(norm_x))
        row_height = boundary_y - preceding_y
        if not (np.isfinite(boundary_y) and 10 <= row_height <= 130):
            return None
        return {'price_left':round(left,2), 'price_right':round(right,2),
                'grid_bottom_y':round(boundary_y,2), 'last_row_height':round(row_height,2),
                'method':'observed_source_grid_curves'}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def isolate_footer(image, guide):
    """Return (source crop, original-photo bbox, diagnostics), else (None,...).

    Require two REAL, long near-horizontal rule fragments bounding a text band.
    This avoids confusing a price or tax amount with the printed receipt total.
    """
    fail = lambda why, **kwargs: (None, None, {'status':why, **kwargs})
    if image is None or not image.size or not guide:
        return fail('missing_image_or_geometry')
    h,w = image.shape[:2]
    try:
        left=float(guide['price_left']); right=float(guide['price_right'])
        anchor=float(guide['grid_bottom_y']); row_h=float(guide['last_row_height'])
    except (KeyError, TypeError, ValueError):
        return fail('invalid_geometry')
    if not all(np.isfinite(n) for n in (left,right,anchor,row_h)) or right-left<22 or row_h<10:
        return fail('invalid_geometry')
    price_w=right-left
    # Allow the printed sum to slightly cross the left price-column boundary.
    x0=max(0,int(round(left-max(12,min(26,price_w*.20)))))
    x1=min(w,int(round(right+max(4,min(12,price_w*.09)))))
    reach=max(42,min(92,row_h*1.8))
    yy0=max(0,int(round(anchor-reach)))
    yy1=min(h,int(round(anchor+reach)))
    if x1-x0<28 or yy1-yy0<26:
        return fail('insufficient_window',search_bbox=[x0,yy0,x1,yy1])
    search=image[yy0:yy1,x0:x1]
    gray=cv2.cvtColor(search,cv2.COLOR_BGR2GRAY) if search.ndim==3 else search
    # Adaptive to photo illumination while retaining faint printed table rules.
    bg=float(np.percentile(gray,70))
    threshold=int(np.clip(bg-18,110,195))
    ink=np.uint8(gray<threshold)*255
    width=x1-x0
    min_run=max(22,int(round(width*.33)))
    opened=cv2.morphologyEx(ink,cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(min_run,1)))
    line_rows=np.flatnonzero((opened>0).sum(axis=1)>=max(22,int(round(width*.29))))
    groups=[]
    for r in line_rows.tolist():
        if groups and r<=groups[-1][-1]+2:
            groups[-1].append(r)
        else:
            groups.append([r])
    separators=[yy0+int(round(float(np.median(group)))) for group in groups]
    max_gap=max(44,min(72,int(round(row_h*1.9))))
    pairs=[]
    for a,b in zip(separators,separators[1:]):
        gap=b-a
        if not 11<=gap<=max_gap:
            continue
        # One of the two separators must straddle, or be near, the grid end.
        if not (a<=anchor+max(8,row_h*.32) and
                b>=anchor-max(8,row_h*.32) and
                b<=anchor+max(48,row_h*1.9)):
            continue
        # The footer starts at the observed end of the item grid. A band
        # ENDING there is the final product/VAT row, not the printed total.
        if abs(a-anchor)>max(8,row_h*.32) or b<=anchor+max(6,row_h*.15):
            continue
        pairs.append((a,b))
    if not pairs:
        return fail('no_isolated_footer_band',search_bbox=[x0,yy0,x1,yy1],
                    observed_separator_y=separators, threshold=threshold)
    a,b=max(pairs,key=lambda p:p[1])  # most downstream bounded printed line
    y0=max(0,a+2)
    y1=min(h,b-1)
    if y1-y0<9:
        return fail('footer_band_too_narrow',observed_separator_y=separators)
    image_crop=image[y0:y1,x0:x1].copy()
    if image_crop.size == 0:
        return fail('empty_footer_crop')
    # A completely blank strip is not OCR evidence.
    area=cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY) if image_crop.ndim==3 else image_crop
    if int(np.count_nonzero(area<max(115,threshold-25)))<15:
        return fail('footer_band_has_no_ink',source_roi_bbox=[x0,y0,x1,y1])
    return image_crop,[x0,y0,x1,y1],{
        'status':'isolated', 'source_roi_bbox':[x0,y0,x1,y1],
        'search_bbox':[x0,yy0,x1,yy1], 'observed_separator_y':separators,
        'selected_separators_y':[a,b], 'threshold':threshold,
        'crop_shape_hw':list(image_crop.shape[:2]),
        'method':'last_bounded_horizontal_rule_band'}


def attempt_focused_footer(image, guide, read_one, parse_number, *, min_confidence=.86):
    """Exactly one recognizer call if observed geometry permits; no arithmetic guess."""
    crop,bbox,diag=isolate_footer(image,guide)
    if crop is None:
        return {'accepted':False, 'total':None, 'lines':[],
                'attempt':None, 'diagnostics':diag}
    started=perf_counter()
    try:
        response=read_one(crop) or {}
        raw=str(response.get('text') or '').strip()
        conf=float(response.get('confidence') or 0)
        numeric=parse_number(raw) if _NUMERIC_LINE.fullmatch(raw) else None
        accepted=(numeric is not None and numeric>0 and conf>=min_confidence)
        amount=str(numeric) if numeric is not None and numeric>0 else None
        if accepted:
            line={**response,'bbox':[[bbox[0],bbox[1]],[bbox[2],bbox[1]],
                                     [bbox[2],bbox[3]],[bbox[0],bbox[3]]]}
        else:
            line=None
        attempt={'strategy':'focused_line_guided_v1',
                 'total':amount if accepted else None, 'unaccepted_numeric_candidate':amount if not accepted else None,
                 'text':raw,'confidence':conf,
                 'lines':[line] if line else [],'source_roi_bbox':bbox,
                 'wall_ms':round((perf_counter()-started)*1000,3),
                 'status':'accepted' if accepted else 'low_confidence_or_invalid',
                 'ocr_invocations':1,
                 'geometry':diag}
        return {'accepted':accepted, 'total':amount if accepted else None,
                'lines':[line] if line else [],'attempt':attempt,
                'diagnostics':diag}
    except Exception as exc:
        attempt={'strategy':'focused_line_guided_v1', 'total':None,
                 'source_roi_bbox':bbox,'lines':[],
                 'wall_ms':round((perf_counter()-started)*1000,3),
                 'status':'ocr_exception','error':f'{type(exc).__name__}: {exc}',
                 'ocr_invocations':1,'geometry':diag}
        return {'accepted':False,'total':None,'lines':[],
                'attempt':attempt, 'diagnostics':diag}
