"""Exact kernels; opt-in, no mandatory extra dependencies."""
import numpy as np
import os
import logging
from functools import lru_cache
from ..config import setting

def _dp_numpy(rewards):
    N,K=rewards.shape;score=rewards[0].copy();parents=np.zeros((N,K),np.int16)
    padded=np.full(K+4,-1e6);neighbors=np.lib.stride_tricks.sliding_window_view(padded,5)[:,::-1]
    penalties=.13*np.abs(np.arange(-2,3));positions=np.arange(K)
    candidates=np.empty((K,5),dtype=np.float64)
    for i in range(1,N):
        padded[2:-2]=score
        np.subtract(neighbors,penalties,out=candidates)
        choice=np.argmax(candidates,axis=1)
        parents[i]=positions-(choice-2);score=candidates[positions,choice]+rewards[i]
    return score,parents

def _dp_loop(rewards):
    N,K=rewards.shape;score=rewards[0].copy();parents=np.zeros((N,K),np.int16)
    next_score=np.empty(K,np.float64)
    for i in range(1,N):
        for k in range(K):
            best=-np.inf;parent=k
            for choice in range(5):
                j=k+2-choice
                val=(score[j] if 0<=j<K else -1e6)-.13*abs(choice-2)
                if val>best:best=val;parent=j
            parents[i,k]=parent;next_score[k]=best+rewards[i,k]
        score,next_score=next_score,score
    return score,parents

@lru_cache(maxsize=8)
def _file_mode(name, default):
    return setting(name,default)

def mode(name,default):
    # Environment override for A/B; .env values are read once per process.
    return os.environ.get(name) if name in os.environ else _file_mode(name,default)

@lru_cache(maxsize=1)
def _jit():
    try:
        from numba import njit
        fn=njit(cache=True,fastmath=False)(_dp_loop)
        sample=np.zeros((2,21),np.float64)
        got=fn(sample);expected=_dp_numpy(sample)
        if not all(np.array_equal(a,b) for a,b in zip(got,expected)):
            raise RuntimeError('trace DP warmup equivalence failed')
        return fn
    except Exception as exc:
        logging.getLogger(__name__).warning('Optional Numba DP unavailable; NumPy fallback: %s',exc)
        return None

def trace_dp(rewards):
    # Optional acceleration only; do not import/compile Numba unless selected.
    if mode('OCR_GRID_TRACE_DP_V3','off')=='numba':
        fn=_jit()
        if fn is not None:
            try:return fn(np.ascontiguousarray(rewards,dtype=np.float64))
            except Exception:pass
    return _dp_numpy(rewards)

def connected_fast(parts,width,tolerance=3.0):
    # Exactly the baseline ordering, eligibility, pair tests and tie handling.
    records=[];groups=[];bounds=[];weights=[];gap=width*.18
    for p in sorted(parts,key=lambda p:-p[2]):
        _,(ax,ay,bx,by),_=p
        if bx<ax:ax,ay,bx,by=bx,by,ax,ay
        ma=(by-ay)/max(bx-ax,1);lo,hi=ax,bx;low,high=min(ay,by),max(ay,by);slope=abs(ma)
        def compatible(q):
            _,cx,cy,dx,dy,mb=q
            L=max(ax,cx);R=min(bx,dx)
            if L-R>gap or abs(ma-mb)>.12:return False
            start,end=min(L,R),max(L,R)
            first=abs(ay+ma*(start-ax)-cy-mb*(start-cx));last=abs(ay+ma*(end-ax)-cy-mb*(end-cx))
            return max(first,last)<=tolerance
        best=None
        for i,(gx0,gx1,gy0,gy1,gslope) in enumerate(bounds):
            if max(lo-gx1,gx0-hi,0)>gap or max(low-gy1,gy0-high,0)>tolerance+max(slope,gslope)*gap:continue
            any_match=False;all_overlap=True
            for q in records[i]:
                good=compatible(q);any_match|=good
                if not good and min(hi,q[3])>=max(lo,q[1]):all_overlap=False;break
            if any_match and all_overlap and (best is None or weights[i]>weights[best]):best=i
        rec=(p,ax,ay,bx,by,ma)
        if best is None:
            groups.append([p]);records.append([rec]);bounds.append((lo,hi,low,high,slope));weights.append(p[2])
        else:
            i=best;gx0,gx1,gy0,gy1,gslope=bounds[i];bounds[i]=(min(gx0,lo),max(gx1,hi),min(gy0,low),max(gy1,high),max(gslope,slope))
            groups[i].append(p);records[i].append(rec);weights[i]+=p[2]
    return groups
