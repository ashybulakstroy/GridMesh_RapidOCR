"""CPU baselines, own implementations; no arithmetic or expected labels in ranking.
Each comparison uses the same visible mask on both images. Distances are not probabilities.
"""
import cv2
import numpy as np
from .previous_hybrid import ink, dt, candidate_masks
NEW=('masked_ncc','tversky','partial_hausdorff','distance_field','masked_hog','projection_profiles','zoning','hu_moments','local_binary_patterns','local_ssim','shape_context_lite','flood_topology')

def hog(x,v):
    a=x.astype(np.float32)
    gx=cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)
    mag,ang=cv2.cartToPolar(gx,gy,angleInDegrees=True)
    safe=cv2.erode(v.astype('uint8'),np.ones((3,3),np.uint8))>0
    mag*=safe; bins=np.floor((ang%180)/20).astype(int)%9
    out=[]
    for y in range(0,36,12):
        for z in range(0,24,8):
            sl=np.s_[y:y+12,z:z+8]; h=np.bincount(bins[sl].ravel(),weights=mag[sl].ravel(),minlength=9);out.extend(h/(h.sum()+1e-9))
    return np.array(out)

def lbp(x,v):
    a=x.astype('uint8'); center=a[1:-1,1:-1]; code=np.zeros_like(center)
    for i,(dy,dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        code|=((a[1+dy:35+dy,1+dx:23+dx]>center).astype('uint8')<<i)
    valid=cv2.erode(v.astype('uint8'),np.ones((3,3),np.uint8))[1:-1,1:-1]>0
    # retain stroke neighborhoods rather than overwhelming counts of flat white pixels
    near=cv2.dilate(a,np.ones((3,3),np.uint8))[1:-1,1:-1]>0
    h=np.bincount(code[valid&near],minlength=256).astype(float);return h/(h.sum()+1e-9)

def hu(x):
    a=cv2.HuMoments(cv2.moments(x.astype('uint8'),binaryImage=True)).ravel()
    return np.clip(-np.sign(a)*np.log10(np.maximum(abs(a),1e-12)),-12,12)/12

def shape_hist(x,v):
    # Log-polar pair distribution, compact approximation of shape context;
    # NOT full point correspondences/TPS from Belongie et al.
    edge=x & ~(cv2.erode(x.astype('uint8'),np.ones((3,3),np.uint8))>0)
    edge &= cv2.erode(v.astype('uint8'),np.ones((3,3),np.uint8))>0
    pts=np.argwhere(edge).astype(float)
    if len(pts)<2:return np.zeros(48)
    if len(pts)>64:pts=pts[np.linspace(0,len(pts)-1,64).astype(int)]
    d=pts[:,None,:]-pts[None,:,:];r=np.linalg.norm(d,axis=2); a=np.arctan2(d[:,:,0],d[:,:,1]); valid=r>0
    rb=np.digitize(r[valid],[3,7,14,25]); ab=np.floor((a[valid]+np.pi)/(2*np.pi)*8).astype(int)%8
    h=np.bincount(rb*8+ab,minlength=48).astype(float);return h/(h.sum()+1e-9)

def flood(x,v):
    # Directional accessibility of visible background, own simple flood descriptor.
    free=(~x)&v; n,labels,stats,_=cv2.connectedComponentsWithStats(free.astype('uint8'),connectivity=4)
    features=[]
    for side in (labels[0],labels[-1],labels[:,0],labels[:,-1]):
        ids=np.unique(side); ids=ids[ids>0]; seen=np.isin(labels,ids)&free
        for y in (0,12,24):features.append(seen[y:y+12].sum()/max(1,v[y:y+12].sum()))
    # True enclosed holes: unknown boundary may open a hole, so ignore those touching it.
    border=np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]
    unknown_adj=cv2.dilate((~v).astype('uint8'),np.ones((3,3),np.uint8))>0
    bad=set(border)|set(labels[unknown_adj]); holes=[stats[i,4] for i in range(1,n) if i not in bad]
    features.extend([min(len(holes),3)/3,sum(holes)/max(1,v.sum())]);return np.array(features)

def hist_dist(a,b):return float(np.abs(a-b).sum()/2)

def cost(method,o,r,v):
    if method=='masked_ncc':
        a=o[v].astype(float); b=r[v].astype(float); a-=a.mean(); b-=b.mean()
        return float((1-np.dot(a,b)/max(1e-9,np.linalg.norm(a)*np.linalg.norm(b)))/2)
    if method=='tversky':
        tp=(o&r).sum(); fp=(o&~r).sum(); fn=(r&~o).sum();return float(1-tp/max(1,tp+.7*fp+.3*fn))
    if method=='partial_hausdorff':
        a=dt(r)[o];b=dt(o)[r];return float(max(np.quantile(a,.85),np.quantile(b,.85)))
    if method=='distance_field':return float(np.abs(dt(o)[v]-dt(r)[v]).mean())
    if method=='masked_hog':return hist_dist(hog(o,v),hog(r,v))/9
    if method=='projection_profiles':
        cols=np.maximum(1,v.sum(0)); rows=np.maximum(1,v.sum(1));return float(.5*(np.abs(o.sum(0)-r.sum(0))/cols).mean()+.5*(np.abs(o.sum(1)-r.sum(1))/rows).mean())
    if method=='zoning':
        a=[]
        for y in range(0,36,6):
            for x in range(0,24,6):
                s=np.s_[y:y+6,x:x+6];n=v[s].sum()
                if n>=12:a.append(abs(int(o[s].sum())-int(r[s].sum()))/n)
        return float(np.mean(a)) if a else 1.
    if method=='hu_moments':return float(np.abs(hu(o)-hu(r)).mean())
    if method=='local_binary_patterns':return hist_dist(lbp(o,v),lbp(r,v))
    if method=='local_ssim':
        # Local SSIM with normalized masked window statistics (5x5 box).
        a=o.astype('float32');b=r.astype('float32');w=v.astype('float32');f=lambda x:cv2.boxFilter(x,-1,(5,5),normalize=False,borderType=cv2.BORDER_CONSTANT)
        n=f(w);safe=v&(n>=15);den=np.maximum(n,1);ma=f(a)/den;mb=f(b)/den;va=np.maximum(0,f(a*a)/den-ma*ma);vb=np.maximum(0,f(b*b)/den-mb*mb);cov=f(a*b)/den-ma*mb
        s=(2*ma*mb+.0001)*(2*cov+.0009)/((ma*ma+mb*mb+.0001)*(va+vb+.0009));focus=safe&((f(a)+f(b))>0)
        return float(np.clip((1-s[focus].mean())/2,0,1)) if focus.any() else 1.
    if method=='shape_context_lite':return hist_dist(shape_hist(o,v),shape_hist(r,v))
    if method=='flood_topology':return float(np.abs(flood(o,v)-flood(r,v)).mean())
    raise ValueError(method)

def rank_new(method,image,bank,*,discover=False,visible_mask=None,top_k=5):
    b=ink(image);best={}
    for v,penalty,kind in candidate_masks(b,visible_mask,discover):
        o=b&v
        if o.sum()<12 or v.mean()<.55:continue
        for label,t in zip(bank.labels,bank.masks):
            r=t&v
            if r.sum()<10 or r.sum()/max(1,t.sum())<.4:continue
            d=float(cost(method,o,r,v)+penalty)
            if not np.isfinite(d):continue
            label=str(label)
            if label not in best or d<best[label]['distance']:best[label]={'digit':label,'distance':d,'mask':kind,'visible_fraction':float(v.mean())}
    return sorted(best.values(),key=lambda x:(x['distance'],x['digit']))[:top_k]
