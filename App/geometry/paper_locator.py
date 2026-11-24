"""Conservative paper-contour proposal. No fixed receipt coordinates or OCR.

A bounding rectangle is a *candidate paper envelope*, NOT a perspective
rectification and NOT proof that every row is inside the envelope.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np


@dataclass
class PaperLocation:
    bbox: tuple[int, int, int, int] | None
    contour: list[list[int]] | None
    elapsed_ms: float
    reason: str
    metrics: dict


def locate_paper(image, *, target_width=450):
    start = perf_counter()
    H, W = image.shape[:2]
    if H < 160 or W < 160:
        return PaperLocation(None, None, (perf_counter()-start)*1000, 'too_small', {})
    scale = min(1.0, target_width/W)
    small = cv2.resize(image, (max(1,round(W*scale)), max(1,round(H*scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    h, w = gray.shape
    otsu, _ = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # A dark textured/photo-reflective background can join the white paper to
    # the image border at medium brightness. Above 200 the paper often becomes
    # its own connected component. These are *proposals*, not trusted crops.
    thresholds = sorted(set([int(np.clip(otsu+26, 120, 205)),
                             int(np.clip(otsu+46, 140, 215)),
                             165, 200, 210, 218]))
    candidates=[]
    kernel=np.ones((5,5),np.uint8)
    for threshold in thresholds:
        binary=cv2.inRange(gray,threshold,255)
        mask=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=2)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=float(cv2.contourArea(contour))
            x,y,cw,ch=cv2.boundingRect(contour)
            fraction=area/(h*w)
            bbox_ratio=cw/max(ch,1)
            rectangularity=area/max(cw*ch,1)
            # Ignore walls, truncated sheets and narrow disconnected text fragments.
            if not (.055 <= fraction <= .72 and .25 <= bbox_ratio <= 1.35):continue
            if not (.48 <= rectangularity <= 1.0):continue
            if cw < w*.20 or ch < h*.18:continue
            if (x<=2 or y<=2 or x+cw>=w-2 or y+ch>=h-2):continue
            region=gray[y:y+ch,x:x+cw]
            bright_fraction=float(np.mean(region>=threshold))
            if bright_fraction<.57:continue
            score=(fraction * min(rectangularity, .9)*bright_fraction +
                   .03*min(ch/h,1.))
            candidates.append((score,contour,(x,y,cw,ch),threshold,
                               rectangularity,bright_fraction,fraction))
    if not candidates:
        return PaperLocation(None,None,(perf_counter()-start)*1000,'paper_not_confident',
                             {'thresholds':thresholds,'original_hw':[H,W]})
    best=max(candidates,key=lambda item:item[0])
    _,contour,(x,y,cw,ch),threshold,rectangularity,bright_fraction,fraction=best
    # The paper boundary itself must be retained. Padding protects cut/fold edges.
    pad_x=max(8,int(W*.018));pad_y=max(8,int(H*.014))
    x0=max(0,int(np.floor(x/scale))-pad_x)
    y0=max(0,int(np.floor(y/scale))-pad_y)
    x1=min(W,int(np.ceil((x+cw)/scale))+pad_x)
    y1=min(H,int(np.ceil((y+ch)/scale))+pad_y)
    eps=max(1.,.009*cv2.arcLength(contour,True))
    polygon=cv2.approxPolyDP(contour,eps,True).reshape(-1,2).astype(float)/scale
    return PaperLocation((x0,y0,x1,y1),np.rint(polygon).astype(int).tolist(),
                         (perf_counter()-start)*1000,'paper_candidate',
                         {'threshold':threshold,
                          'threshold_strategy':'high_brightness_disconnection' if threshold>=190 else 'standard',
                          'mask_fraction':round(fraction,4),
                          'rectangularity':round(rectangularity,4),
                          'bright_fraction':round(bright_fraction,4),
                          'bbox_fraction':round((x1-x0)*(y1-y0)/(W*H),4),
                          'contour_vertices':len(polygon), 'count_candidates':len(candidates)})
