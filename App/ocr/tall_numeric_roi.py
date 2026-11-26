"""Conservative fallback for tall cells with isolated printed grid remnants.
Only changes the crop rectangle; original pixels and decimal marks are retained.
"""
import cv2
import numpy as np


def compact_tall(image):
    if image is None or not image.size or image.ndim not in (2,3):
        return None
    h,w=image.shape[:2]
    if h<65 or w<12 or h*w>160000:
        return None
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,12)
    _,_,stats,_=cv2.connectedComponentsWithStats(ink)
    s=stats[1:]
    if not len(s):return None
    x,y,bw,bh,area=s.T
    # Estimate a glyph band only from substantial interior components.
    core=(x>=4)&(x+bw<=w-4)&(y>=4)&(y+bh<=h-4)&(bh>=8)&(bh<=h*.40)&(bw>=2)&(bw<=bh*1.6)&(area>=12)
    if not core.any():return None
    gh=float(np.median(bh[core])); top=int(y[core].min()); bottom=int((y+bh)[core].max())
    if bottom-top>gh*1.5 or gh<8:return None
    # Only components tightly attached to physical edges can be excluded.
    # An edge-touching plausible digit is never removed.
    horizontal=((y==0)|(y+bh==h))&(bh<=2)
    vertical=((x==0)|(x+bw==w))&(bw<=2)
    corner=((y==0)|(y+bh==h))&((x<=2)|(x+bw>=w-2))&(bw<=5)&((bh>gh*1.8)|((bh<=7)&(area<=12)))
    far=(y+bh<top-3)|(y>bottom+3)
    rail=horizontal|corner|(vertical & (far|(bh>gh*1.8)))
    for side in ((x==0),(x+bw==w)):
        distributed=vertical&side
        if distributed.any():
            span=(y[distributed]+bh[distributed]).max()-y[distributed].min()
            if span>gh*2.5 and bh[distributed].sum()>h*.25:
                # Don't discard a plausible digit at the same height as the band.
                plausible=(bh>=gh*.65)&(bh<=gh*1.4)&(~far)
                rail|=distributed&~plausible
    # Preserve even a single pixel: it may be a decimal separator.
    keep=~rail
    if not keep.any():return None
    left=int(x[keep].min()); right=int((x+bw)[keep].max())
    t=int(y[keep].min()); b=int((y+bh)[keep].max())
    if b-t>gh*1.5 or left<3 or right>w-3 or t<3 or b>h-3:return None
    pad=max(3,round(gh*.25)); l=max(0,left-pad); t=max(0,t-pad); r=min(w,right+pad); b=min(h,b+pad)
    if b-t>=h*.65 or r-l<8:return None
    return image[t:b,l:r].copy(), {'changed':True,'reason':'tall_numeric_band_v1','offset':[l,t],'size':[r-l,b-t],'original_size':[w,h]}


def improve(image, cropped, roi):
    """Use only on failed or vertically oversized normal crops."""
    if image is None or not image.size:return cropped,roi
    h=image.shape[0]
    if h<65 or (roi.get('changed') and cropped.shape[0]<h*.55):return cropped,roi
    result=compact_tall(image)
    return result if result is not None else (cropped,roi)
