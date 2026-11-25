"""CPU partial-glyph ranking. Scores are distances, not calibrated probabilities.
Inputs are complete character slots (including missing areas), not tight ink boxes.
"""
from dataclasses import dataclass
from itertools import product
import cv2
import numpy as np

W,H=24,36
DIGITS='0123456789'

def ink(image):
    if image is None or image.size==0: raise ValueError('empty image')
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    if g.dtype!=np.uint8: raise ValueError('expected uint8')
    g=cv2.resize(g,(W,H),interpolation=cv2.INTER_LINEAR)
    _,b=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    return b>0

def shift(a,dx,dy):
    return cv2.warpAffine(a.astype('uint8'),np.float32([[1,0,dx],[0,1,dy]]),(W,H),flags=cv2.INTER_NEAREST,borderValue=0).astype(bool)

def dt(a):
    return cv2.distanceTransform((~a).astype('uint8'),cv2.DIST_L2,3).clip(0,5)/5

@dataclass
class Bank:
    labels: np.ndarray
    masks: np.ndarray
    def __init__(self, samples):
        labels=[]; masks=[]
        for label,img in samples:
            if label not in DIGITS or len(label)!=1: raise ValueError('digit label required')
            b=ink(img)
            if not 10<=b.sum()<=H*W*.8: continue
            for dx,dy in product((-1,0,1),repeat=2):
                labels.append(label);masks.append(shift(b,dx,dy))
        if not masks: raise ValueError('no usable templates')
        self.labels=np.array(labels);self.masks=np.stack(masks)
    @property
    def coverage(self):return ''.join(sorted(set(self.labels)))

def candidate_masks(b, known=None, discover=False):
    if known is not None:
        if known.ndim!=2: raise ValueError('mask must be grayscale; 255=visible')
        v=cv2.resize(known,(W,H),interpolation=cv2.INTER_NEAREST)>0
        if v.mean()<.55: return []
        return [(v,0.0,'provided_mask')]
    out=[(np.ones_like(b),0.0,'unmasked')]
    if not discover:return out
    # Candidate loss strips are derived from the OBSERVED image, never chosen
    # separately for each digit. Empty columns alone do not prove print damage.
    density=b.sum(0); cols=np.flatnonzero(density<=1)
    if not len(cols):return out
    for run in np.split(cols,np.where(np.diff(cols)>1)[0]+1):
        if not len(run):continue
        a,z=int(run[0]),int(run[-1])+1
        if not(1<=z-a<=7 and a>=3 and z<=W-3):continue
        if b[:,:a].sum()<12 or b[:,z:].sum()<12:continue
        v=np.ones_like(b);v[:,a:z]=False
        out.append((v,.04+.18*(z-a)/W,f'hypothetical_missing_strip_{a}_{z}'))
    return out

def rank(image,bank,*,visible_mask=None,discover=False,top_k=3):
    if not 1<=top_k<=10:raise ValueError('top_k must be 1..10')
    b=ink(image);best={}
    for v,penalty,kind in candidate_masks(b,visible_mask,discover):
        obs=b&v
        if obs.sum()<12 or v.mean()<.55:continue
        # Both directions matter: a template must explain present ink AND
        # cannot invent visible ink in verified blank areas. Unknown pixels ignored.
        d_obs=dt(obs)
        for label,t in zip(bank.labels,bank.masks):
            ref=t&v
            if ref.sum()<10 or ref.sum()/max(1,t.sum())<.40:continue
            d_ref=dt(ref)
            chamfer=.5*(d_ref[obs].mean()+d_obs[ref].mean())
            xor=np.count_nonzero(obs^ref)/max(1,np.count_nonzero(obs|ref))
            cost=float(.60*chamfer+.40*xor+penalty)
            old=best.get(label)
            if old is None or cost<old['distance']:
                best[label]={'digit':str(label),'distance':cost,'mask':kind,
                             'visible_fraction':float(v.mean()),'observed_ink':int(obs.sum())}
    result=sorted(best.values(),key=lambda x:(x['distance'],x['digit']))
    # Abstain if absolutely poor or too little evidence. All outputs hypotheses.
    if not result or result[0]['distance']>.55:return []
    return result[:top_k]

def cell_candidates(image,bank,slots,*,top_k=3,beam=12):
    """slots: ordered dicts {bbox:[x0,y0,x1,y1], mask?: ndarray, literal?: '.' or ','}.
    Coordinates must be independently established from glyph pitch or manual labels.
    No arithmetic-derived length, no silent character deletion.
    """
    if not 1<=beam<=32 or not 1<=len(slots)<=16:raise ValueError('bounds')
    h,w=image.shape[:2]; states=[('',0.,[])];last=0
    for slot in slots:
        x0,y0,x1,y1=map(int,slot['bbox'])
        if not(0<=last<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('invalid slots')
        last=x1
        if 'literal' in slot:
            if slot['literal'] not in ('.',','):raise ValueError('unsupported literal')
            options=[{'digit':slot['literal'],'distance':0.,'mask':'externally_observed_separator'}]
        else:options=rank(image[y0:y1,x0:x1],bank,visible_mask=slot.get('mask'),discover=slot.get('discover',False),top_k=top_k)
        if not options:return []
        states=sorted([(s+o['digit'],c+o['distance'],a+[o]) for s,c,a in states for o in options],key=lambda x:(x[1],x[0]))[:beam]
    return [{'text':s,'distance':c/len(slots),'glyphs':a,'source':'partial_template_hypothesis','requires_review':True} for s,c,a in states]
