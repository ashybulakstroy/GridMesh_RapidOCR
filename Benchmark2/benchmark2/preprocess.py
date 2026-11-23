"""Identical segmentation of physical name cells for every recognizer.

Port of the basic split_name_lines / join_text_parts logic in project's
benchmark/core/measure.py. Physical 17 original ROIs are never resized.
"""
import re
import numpy as np
import cv2
from PIL import Image


def ink_mask(crop):
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY) if crop.ndim==3 else crop
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    h,w=binary.shape
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,int(w*.45)),1)))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(20,int(h*.45)))))
    clean=cv2.subtract(binary,cv2.bitwise_or(horizontal,vertical))
    n,labels,stats,_=cv2.connectedComponentsWithStats(clean,8)
    keep=np.zeros_like(clean)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area<4:continue
        if bw<=max(4,.04*w) and bh>=max(10,.25*h) and bh>=4*bw:continue
        if bh<=max(4,.04*h) and bw>=max(10,.25*w) and bw>=4*bh:continue
        if area/(bw*bh)<.18 and max(bw,bh)>=max(16,.25*max(w,h)):continue
        keep[labels==i]=255
    return keep


def split_name_lines(image):
    crop=np.asarray(image.convert('RGB'))
    mask=ink_mask(crop);h,w=mask.shape
    counts=np.count_nonzero(mask,axis=1)
    if counts.size==0 or counts.max()==0:return [image]
    active=(counts>=max(2,int(w*.01))).astype(np.uint8).reshape(-1,1)
    active=cv2.morphologyEx(active,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)).ravel().astype(bool)
    edges=np.diff(np.r_[False,active,False].astype(np.int8))
    bands=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=4]
    if len(bands)<=1:return [image]
    med=int(np.median([b-a for a,b in bands]))
    bands=[(a,b) for a,b in bands if b-a>=med*.5]
    if len(bands)<=1:return [image]
    return [image.crop((0,max(0,y0-2),w,min(h,y1+2))) for y0,y1 in bands]


def join_parts(parts):
    out=[]
    for x in parts:
        s=str(x).strip()
        if not s:continue
        if out and re.search(r'[-\u2010-\u2015]$',out[-1]) and re.match(r'^[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]',s):
            out[-1]=re.sub(r'[-\u2010-\u2015]$','',out[-1])+s
        else:out.append(s)
    return ' '.join(' '.join(out).split())


def precompute(manifest, root):
    results={}
    for s in manifest:
        with Image.open(root/s['file']) as im:
            cell=im.convert('RGB')
            pieces=split_name_lines(cell) if s['type']=='name' else [cell]
            results[s['id']] = pieces
    return results
