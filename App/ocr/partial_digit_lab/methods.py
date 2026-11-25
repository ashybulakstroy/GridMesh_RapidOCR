"""CPU comparisons on the same slots, bank, registration and mask hypotheses.
Distances are not probabilities; smaller is better, only compare within method.
"""
import numpy as np
import cv2
from .previous_hybrid import ink,dt,candidate_masks,rank as old_rank

from .expanded import NEW, rank_new
BASE=('masked_template','partial_chamfer','adaptive_parts','hybrid_v1')
METHODS=BASE+NEW+('hybrid_rank_v2',)
# Fixed selection before measuring, not selected on the test answers.
FUSION=('masked_template','partial_chamfer','masked_hog','projection_profiles','local_ssim','partial_hausdorff')

def rank(method,image,bank,*,discover=False,visible_mask=None,top_k=5):
    if method not in METHODS or not 1<=top_k<=10: raise ValueError('method/top_k')
    if method in NEW:
        return rank_new(method,image,bank,discover=discover,visible_mask=visible_mask,top_k=top_k)
    if method=='hybrid_rank_v2':
        votes={}
        for m in FUSION:
            options=rank(m,image,bank,discover=discover,visible_mask=visible_mask,top_k=10)
            if not options:continue
            for i,o in enumerate(options):votes.setdefault(o['digit'],[]).append(i)
        if not votes:return []
        return sorted([{'digit':d,'distance':(sum(v)+10*(len(FUSION)-len(v)))/(10*len(FUSION)), 'mask':'rank_fusion','support':len(v)} for d,v in votes.items()],key=lambda x:(x['distance'],x['digit']))[:top_k]
    if method=='hybrid_v1':
        return old_rank(image,bank,discover=discover,visible_mask=visible_mask,top_k=top_k)
    b=ink(image); best={}
    for v,penalty,kind in candidate_masks(b,visible_mask,discover):
        obs=b&v
        if obs.sum()<12 or v.mean()<.55: continue
        ref=bank.masks&v
        n=ref.sum(axis=(1,2)); good=(n>=10)&(n/np.maximum(1,bank.masks.sum(axis=(1,2)))>=.4)
        union=(ref|obs).sum(axis=(1,2)); xor=(ref^obs).sum(axis=(1,2))/np.maximum(1,union)
        if method=='masked_template':
            # Masked binary pixel agreement, foreground-union normalization.
            cost=xor
        elif method=='partial_chamfer':
            # Distance transform from template stroke to observed stroke and
            # vice versa; visible pixels only. Directional weights favor observed evidence.
            obs_dist=dt(obs); forward=np.array([dt(t)[obs].mean() for t in ref]); reverse=(ref*obs_dist).sum(axis=(1,2))/np.maximum(1,n)
            cost=.8*forward+.2*reverse
        else:
            # Independent visible regions, robust aggregation; inspired by
            # partial-character recognition, NOT a reproduction of the patent.
            parts=[]; weights=[]
            for y in (0,12,24):
                for x in (0,12):
                    sl=np.s_[y:y+12,x:x+12]; vv=v[sl]; o=obs[sl]; rr=ref[:,y:y+12,x:x+12]
                    if vv.mean()<.55 or o.sum()<2: continue
                    den=(rr|o).sum(axis=(1,2)); parts.append((rr^o).sum(axis=(1,2))/np.maximum(1,den)); weights.append(float(o.sum())**.5)
            if len(parts)<2: continue
            cost=np.average(np.stack(parts),axis=0,weights=weights)*.85+xor*.15
        for i in np.flatnonzero(good):
            label=str(bank.labels[i]); d=float(cost[i]+penalty)
            if label not in best or d<best[label]['distance']:
                best[label]={'digit':label,'distance':d,'mask':kind,'visible_fraction':float(v.mean())}
    return sorted(best.values(),key=lambda x:(x['distance'],x['digit']))[:top_k]

def combine(rankings,limit=12):
    states=[('',0.)]
    for options in rankings:
        if not options:return []
        states=sorted([(s+o['digit'],d+o['distance']) for s,d in states for o in options],key=lambda x:(x[1],x[0]))[:limit]
    return [{'text':s,'distance':d/len(rankings)} for s,d in states]
