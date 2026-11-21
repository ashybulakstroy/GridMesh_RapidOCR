"""Recover printed horizontal rules crossing the central fold of a receipt."""
import cv2
import numpy as np

from .geometry_lab.physical_grid_graph_local import ridge_response
from .geometry_lab.structure_diagnostics import PolylineRule


def recover_fold(image, data):
    vertical = data['vertical']
    horizontal = data['horizontal']
    if len(vertical) != 5 or len(horizontal) < 5:
        return data, 0
    height, width = image.shape[:2]
    fold = vertical[2]
    response, _ = ridge_response(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    slopes = np.linspace(-.26, -.055, 72)
    offsets = np.arange(-5, 6)
    replacements = {}
    for index, rule in enumerate(horizontal):
        y = float(rule.eval((fold.position - width/2) / width))
        x = float(fold.eval((y - height/2) / height))
        y = float(rule.eval((x - width/2) / width))
        end = float(vertical[-1].eval((y - height/2) / height))
        if not (0 <= x < end < width and end - x >= 65):
            continue
        samples = np.linspace(x + 8, end - 6, 130)
        ix = np.rint(samples).astype(int)
        original = rule.eval((samples - width/2) / width)
        old_y = np.rint(original).astype(int).clip(0, height - 1)
        old_coverage = (response[old_y, ix] >= 7).mean()
        if old_coverage >= .78:
            continue
        candidates = y + slopes[:, None, None] * (samples - x) + offsets[None, :, None]
        iy = np.rint(candidates).astype(int)
        strength = response[iy.clip(0, height - 1), ix]
        strength[(iy < 0) | (iy >= height)] = 0
        supported = strength >= 7
        coverage = supported.mean(axis=2)
        valid = ((coverage >= .70) & (coverage > old_coverage + .22)
                 & (supported[:, :, :65].mean(axis=2) >= .60)
                 & (supported[:, :, 65:].mean(axis=2) >= .60))
        if not valid.any():
            continue
        scores = np.where(valid, np.minimum(strength, 28).mean(axis=2), -1)
        slope_index, offset_index = np.unravel_index(scores.argmax(), scores.shape)
        offset = offsets[offset_index]
        left = np.linspace(vertical[0].position, x, 45)
        left_y = rule.eval((left - width/2) / width)
        left_y += offset * np.clip((left - x + 24) / 24, 0, 1)
        right_y = y + offset + slopes[slope_index] * (end - x)
        replacements[index] = PolylineRule(
            np.column_stack((np.r_[left, end], np.r_[left_y, right_y])),
            'h', (height, width))

    # A fold affects several neighbors, whereas a text underline need not.
    if not any(all(i + step in replacements for step in range(3))
               for i in replacements):
        return data, 0
    updated = [replacements.get(i, rule) for i, rule in enumerate(horizontal)]
    probe = (np.linspace(vertical[0].position, vertical[-1].position, 100)
             - width/2) / width
    if np.any(np.diff([rule.eval(probe) for rule in updated], axis=0) < 5):
        return data, 0
    return {**data, 'horizontal': updated}, len(replacements)


def recover_left_edges(image, data):
    """Replace unsupported left extrapolations only with repeated observed ridges."""
    vertical, horizontal = data['vertical'], data['horizontal']
    if len(vertical) != 5 or len(horizontal) < 5:
        return data, 0
    height, width = image.shape[:2]
    left = float(vertical[0].position)
    anchor = float(vertical[1].position + min(75, (vertical[2].position-vertical[1].position)*.3))
    if not (0 <= left+8 < anchor-45 < width):
        return data, 0
    response, _ = ridge_response(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    samples = np.linspace(left+8, anchor, 100)
    ix = np.rint(samples).astype(int)
    slopes = np.linspace(-.30, .30, 121)
    offsets = np.arange(-3, 4)
    replacements = {}
    for index, rule in enumerate(horizontal):
        original = rule.eval((samples-width/2)/width)
        old_y = np.rint(original).astype(int).clip(0,height-1)
        old_coverage = (response[old_y,ix] >= 7).mean()
        if old_coverage >= .78:
            continue
        y_anchor = float(rule.eval((anchor-width/2)/width))
        trial = y_anchor + slopes[:,None,None]*(samples-anchor) + offsets[None,:,None]
        iy = np.rint(trial).astype(int)
        strength = response[iy.clip(0,height-1),ix]
        strength[(iy<0)|(iy>=height)] = 0
        supported = strength >= 7
        coverage = supported.mean(axis=2)
        valid = ((coverage >= .82) & (coverage > old_coverage+.25)
                 & (supported[:,:,:50].mean(axis=2) >= .75)
                 & (supported[:,:,50:].mean(axis=2) >= .75))
        if not valid.any():
            continue
        score = np.where(valid,np.minimum(strength,28).mean(axis=2),-1)
        si, oi = np.unravel_index(score.argmax(),score.shape)
        slope, offset = slopes[si], offsets[oi]
        if abs((y_anchor+slope*(left-anchor)+offset)-float(rule.eval((left-width/2)/width))) < 5:
            continue
        right = np.linspace(anchor+1, float(vertical[-1].position), 80)
        if right[-1] <= right[0]:
            continue
        # Blend the small anchor offset back to the original curve.
        right_y = rule.eval((right-width/2)/width) + offset*np.clip(1-(right-anchor)/35,0,1)
        points = np.column_stack((np.r_[left,anchor,right], np.r_[y_anchor+slope*(left-anchor)+offset,y_anchor+offset,right_y]))
        replacements[index] = PolylineRule(points,'h',(height,width))
    if not any(all(i+step in replacements for step in range(3)) for i in replacements):
        return data, 0
    updated = [replacements.get(i,rule) for i,rule in enumerate(horizontal)]
    probe=(np.linspace(left,vertical[-1].position,120)-width/2)/width
    if np.any(np.diff([rule.eval(probe) for rule in updated],axis=0)<5):
        return data, 0
    return {**data,'horizontal':updated}, len(replacements)
