"""Conservative exterior-rail pruning; no assumed count in ambiguous tables."""
import re, unicodedata
import cv2
import numpy as np
from .geometry_lab.algorithms import intersection
from ..config import setting


def enabled():
    return setting('OCR_GRID_ANOMALY_V53','0')=='1' and setting('OCR_GRID_ANOMALY_V3','1')=='1'


def header_role(text):
    s=unicodedata.normalize('NFKC',str(text or '')).casefold().strip()
    s=s.translate(str.maketrans({'r':'г','b':'ы','ғ':'г','ң':'н','қ':'к','ұ':'у'}))
    if re.fullmatch(r'(?:№|n[oеe]\.?|n°|#)',s):return 'number'
    if re.search(r'атау|наимен|назван',s):return 'name'
    if re.search(r'саны|кол.?во|колич',s):return 'quantity'
    if re.search(r'бага|цена|стоим',s):return 'price'
    if re.search(r'сома|сумм',s):return 'printed_total'
    return None


def strict_header(texts):
    return [header_role(t) for t in texts]==['number','name','quantity','price']


def _coverage(rule,a,b,axis,mask,margin=5):
    H,W=mask.shape; extent=H if axis=='v' else W; across=W if axis=='v' else H
    t=np.arange(int(np.ceil(a+margin)),int(np.floor(b-margin))+1)
    if len(t)<8 or t.min()<0 or t.max()>=extent:return None
    val=np.asarray(rule.eval((t-extent/2)/extent),float)
    if not np.isfinite(val).all():return None
    u=np.rint(val).astype(int); k=u[:,None]+np.arange(-2,3)
    if k.min()<0 or k.max()>=across:return None
    m=mask[t[:,None],k] if axis=='v' else mask[k,t[:,None]]
    return float(np.any(m>0,axis=1).mean())


def prune_exterior(image,graph):
    """Exactly one of two exterior rails may be removed. Keep original objects.
    Full real fifth-column tables, weak retained rails and ambiguous evidence fail closed.
    """
    d={'accepted':False,'reason':'not_six_rail','version':'v53'}
    if graph is None:return graph,d
    vs,hs=graph.get('vertical',[]),graph.get('horizontal',[])
    if len(vs)!=6 or len(hs)<3:return graph,d
    try:
        H,W=image.shape[:2]
        if tuple(graph['shape'][:2])!=(H,W):raise ValueError('shape_mismatch')
        n=np.array([[intersection(v,h,(H,W)) for v in vs] for h in hs],float)
        if (not np.isfinite(n).all() or n[:,:,0].min()<3 or n[:,:,0].max()>=W-3 or n[:,:,1].min()<3 or n[:,:,1].max()>=H-3 or np.min(np.diff(n[:,:,0],axis=1))<18 or np.min(np.diff(n[:,:,1],axis=0))<18):raise ValueError('invalid_nodes_or_narrow_strip')
        # Reject crossings between nodes, too, not just at nodes.
        yy=np.linspace(n[:,:,1].min(),n[:,:,1].max(),128);xx=np.linspace(n[:,:,0].min(),n[:,:,0].max(),128)
        if np.min(np.diff(np.stack([v.eval((yy-H/2)/H) for v in vs]),axis=0))<5 or np.min(np.diff(np.stack([h.eval((xx-W/2)/W) for h in hs]),axis=0))<5:raise ValueError('crossing_curves')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,6)
        vm=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((13,1),np.uint8));hm=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,13),np.uint8))
        vc=[[_coverage(v,n[i,j,1],n[i+1,j,1],'v',vm,6) for i in range(len(hs)-1)] for j,v in enumerate(vs)]
        hc=[[_coverage(h,n[i,j,0],n[i,j+1,0],'h',hm) for j in range(5)] for i,h in enumerate(hs)]
        if any(c is None for r in vc+hc for c in r):raise ValueError('insufficient_samples')
        vc=np.asarray(vc);hc=np.asarray(hc);d.update(rail_coverage=vc.tolist(),horizontal_column_coverage=hc.tolist())
        eligible=[]
        for edge in (0,5):
            keep=[j for j in range(6) if j!=edge];strip=0 if edge==0 else 4
            retained=[j for j in range(5) if j!=strip]
            # Thin outside strip only: avoids removing a real broad column.
            width=np.diff(n[:,:,0],axis=1); ratio=width[:,strip]/(n[:,keep[-1],0]-n[:,keep[0],0])
            # Require all real rails and all real horizontal segments, including both junction arms.
            # A weak vertical alone is never sufficient for deletion.
            if (ratio.max()<=.12 and vc[edge].max()<=.18 and vc[edge].mean()<=.08 and
                hc[:,strip].max()<=.20 and vc[keep].min()>=.80 and hc[:,retained].min()>=.78):
                # Refuse any exterior strip that contains text-like ink. Sample actual
                # curved cell interiors, not a bounding rectangle of a slanted grid.
                blanks=[]
                for i in range(len(hs)-1):
                    lo=max(0,int(np.floor(n[i,strip:strip+2,1].min())));hi=min(H,int(np.ceil(n[i+1,strip:strip+2,1].max()))+1)
                    sy=np.arange(lo,hi);x0=vs[strip].eval((sy-H/2)/H);x1=vs[strip+1].eval((sy-H/2)/H)
                    ax=max(0,int(np.floor(x0.min())));bx=min(W,int(np.ceil(x1.max()))+1);sx=np.arange(ax,bx)
                    low=hs[i].eval((sx-W/2)/W);high=hs[i+1].eval((sx-W/2)/W)
                    inner=(sx[None,:]>=x0[:,None]+6)&(sx[None,:]<=x1[:,None]-6)&(sy[:,None]>=low[None,:]+7)&(sy[:,None]<=high[None,:]-7)
                    blanks.append(float((bw[lo:hi,ax:bx][inner]>0).mean()) if inner.sum()>=80 else 1.)
                d.setdefault('strip_ink_by_edge',{})[str(edge)]=blanks
                if max(blanks)>.02:continue
                # Test inward horizontal arm and vertical stroke near each new outside junction.
                j=keep[0] if edge==0 else keep[-1]; sign=1 if edge==0 else -1
                arms=[]
                for i,h in enumerate(hs):
                    x,y=n[i,j]; a,b=sorted((x+sign*2,x+sign*17))
                    arm=_coverage(h,a,b,'h',hm,0)
                    v=vs[j]; a,b=sorted((y+2,y+18)) if i<len(hs)-1 else sorted((y-18,y-2))
                    rail=_coverage(v,a,b,'v',vm,0)
                    arms.append([arm,rail])
                if all(a is not None and b is not None and a>=.75 and b>=.75 for a,b in arms):eligible.append((edge,keep,arms))
        if len(eligible)!=1:d['reason']='no_unique_exterior_proof';return graph,d
        edge,keep,arms=eligible[0]
        d.update(accepted=True,reason='weak_exterior_and_empty_strip_with_five_supported_rails',removed_index=edge,junction_arms=arms)
        return {**graph,'vertical':[vs[j] for j in keep]},d
    except (ValueError,TypeError,IndexError,AttributeError,KeyError,cv2.error,OverflowError) as exc:
        d['reason']=str(exc);return graph,d


def multiscale_only(image):
    """Independent candidate generator; never marks adaptive state or runs V2."""
    from .geometry_lab.physical_grid_graph_v3 import _multiscale_core,ridge_response
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);response,dark=ridge_response(gray)
    return _multiscale_core(gray,response,dark,None,{'status':'NOT_RUN'})
