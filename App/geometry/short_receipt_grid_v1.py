"""Short table rescue. No fixed positions, VAT rate, or inferred boundaries.
Candidate only: caller MUST validate OCR header and Grid Mesh safety.
"""
from __future__ import annotations
from itertools import combinations
from time import perf_counter
import cv2
import numpy as np


def _runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))


def _levels(mask, left, right, top, bottom):
    span=right-left
    counts=np.count_nonzero(mask[top:bottom,left:right],axis=1)
    # Keep distinct peaks even if text creates a continuous low-support bridge.
    minimum=max(12,round(span*.065)); gap=max(7,round(span*.018))
    candidates=sorted((i for i in range(len(counts)) if counts[i]>=minimum and
                       counts[i]>=np.max(counts[max(0,i-2):i+3])),key=lambda i:-int(counts[i]))
    selected=[]
    for i in candidates:
        if all(abs(i-j)>gap for j in selected):selected.append(i)
    return sorted((top+i,int(counts[i])) for i in selected)


def _trace(mask, seed, l, r, radius, W):
    """Dynamic-programming observed stroke path, then robust polynomial fit."""
    H=mask.shape[0]; lo=max(0,seed-radius); hi=min(H,seed+radius+1)
    if hi-lo<3 or r-l<60:return None
    xs=np.unique(np.r_[np.arange(l,r,3),r-1]).astype(int)
    ys=np.arange(lo,hi); n=len(ys)
    # A 3x3 neighborhood absorbs double printed edges, but not distant text.
    hits=mask[lo:hi,:][:,xs].astype(np.float32)/255
    dp=hits[:,0]*3-.015*np.abs(ys-seed); back=np.zeros((len(xs),n),np.int16)
    for i in range(1,len(xs)):
        opts=np.stack([np.r_[np.full(max(0,k),-1e8),dp[:n-k]] if k>0 else
                       np.r_[dp[-k:],np.full(-k,-1e8)] if k<0 else dp
                       for k in range(-2,3)])
        opts-=np.abs(np.arange(-2,3))[:,None]*.4
        best=np.argmax(opts,axis=0); back[i]=np.arange(n)-(best-2)
        dp=opts[best,np.arange(n)]+3*hits[:,i]-.018*np.abs(ys-seed)
    path=np.empty(len(xs),int); p=int(np.argmax(dp))
    for i in range(len(xs)-1,-1,-1):path[i]=p;p=int(back[i,p]) if i else p
    observed=hits[path,np.arange(len(xs))]>0
    if observed.mean()<.48:return None
    xx=(xs[observed]-W/2)/W; yy=ys[path[observed]]
    if len(xx)<8:return None
    coeff=np.polyfit(xx,yy,2)
    err=np.abs(np.polyval(coeff,xx)-yy); keep=err<=4
    if keep.sum()<8 or keep.mean()<.7:return None
    coeff=np.polyfit(xx[keep],yy[keep],2)
    predicted=np.polyval(coeff,(xs-W/2)/W)
    if np.max(np.abs(predicted-seed))>radius+4:return None
    yp=np.rint(predicted).astype(int)
    valid=np.zeros(len(xs),bool)
    for d in range(-4,5): valid|=mask[np.clip(yp+d,0,H-1),xs]>0
    return coeff,xs,valid


def _coverage(mask, rule, a, b, H, W):
    yy=np.arange(max(0,int(np.ceil(a))),min(H,int(np.floor(b))+1));
    if len(yy)<8:return 0.
    xx=np.rint(rule.eval((yy-H/2)/H)).astype(int)
    hit=np.zeros(len(yy),bool)
    for d in range(-4,5):hit |= mask[yy,np.clip(xx+d,0,W-1)]>0
    return float(hit.mean())


def detect_short_grid(image, *, rule_factory=None):
    """Find a compact, complete 5-rail table (header + 1–2 body bands)."""
    started=perf_counter(); report={'status':'rejected','reason':'no_verified_short_grid'}
    if image is None or image.size==0:return None,report
    from .source_grid_v16 import _line_segments, _find_rails
    if rule_factory is None:
        from .geometry_lab.detection import Rule as rule_factory
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    H,W=gray.shape; raw=_line_segments(image)
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,6)
    hm=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,max(9,min(17,round(W*.012)))),np.uint8))
    vm=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((max(9,min(17,round(H*.008))),1),np.uint8))
    # Two independent candidate sources: original V16 rail tracks and local
    # LSD rail groups. The latter handles clean, unbroken one-item tables.
    proposals=[]; v16,_=_find_rails(raw,image.shape)
    if v16 is not None:
        rails,top,bottom=v16
        proposals.append((rails,max(0,int(top)-22),min(H,int(bottom)+22)))
    # Horizontal runs across image supply compact search windows; no assumed VAT.
    broad=_levels(hm,0,W,0,H)
    broad=sorted(sorted(broad,key=lambda p:-p[1])[:32]);
    for i in range(len(broad)):
        for count in (3,4):
            if i+count>len(broad):continue
            a,b=broad[i][0],broad[i+count-1][0]; height=b-a
            if height<max(28,H*.025) or height>H*.43:continue
            if min(np.diff([p[0] for p in broad[i:i+count]]))<10:continue
            mid=(a+b)/2; seg=[]
            for x0,y0,x1,y1 in raw:
                dy=y1-y0;dx=x1-x0
                if abs(dy)<max(12,height*.19) or abs(dx)>.16*abs(dy):continue
                overlap=min(max(y0,y1),b)-max(min(y0,y1),a)
                if overlap<max(8,height*.17):continue
                slope=dx/dy; x=x0+slope*(mid-y0)
                if 2<x<W-2:seg.append((float(x),float(slope),float(overlap),min(y0,y1),max(y0,y1)))
            groups=[]
            for s in sorted(seg):
                if not groups or s[0]-np.median([v[0] for v in groups[-1]])>max(5,W*.006):groups.append([s])
                else:groups[-1].append(s)
            rails=[]
            for g in groups:
                x=float(np.median([p[0] for p in g]));slope=float(np.median([p[1] for p in g]));
                rr=rule_factory('v',np.array([slope*H,x+slope*(H/2-mid)]),x+slope*(H/2-mid),1.,(a,b))
                cov=_coverage(vm,rr,a,b,H,W)
                if cov>=.70:rails.append({'x_ref':x+slope*(H*.45-mid),'slope':slope,'first':a,'last':b,'quality':cov})
            # Do not pick five arbitrary rails from a larger table.
            if len(rails)==5:proposals.append((rails,max(0,a-24),min(H,b+24)))
    best=[]; seen=set()
    for tracks,top,bottom in proposals[:40]:
        mid=(top+bottom)/2
        xmid=[r['x_ref']+r['slope']*(mid-H*.45) for r in tracks]
        if len(xmid)!=5 or min(np.diff(xmid))<12 or xmid[-1]-xmid[0]<W*.22:continue
        l=max(0,int(min(xmid)-10));r=min(W,int(max(xmid)+10));span=r-l
        # Re-evaluate using rail-limited horizontal evidence instead of full-page text.
        levels=_levels(hm,l,r,top,bottom)
        levels=sorted(sorted(levels,key=lambda p:-p[1])[:32])
        traced=[]; radius=max(9,min(24,round(span*.035)))
        for seed,_ in levels:
            lx=max(0,int(tracks[0]['x_ref']+tracks[0]['slope']*(seed-H*.45)));rx=min(W,int(tracks[-1]['x_ref']+tracks[-1]['slope']*(seed-H*.45))+1)
            trace=_trace(hm,seed,lx,rx,radius,W)
            if trace is None:continue
            coeff,xx,valid=trace; cross=[t['x_ref']+t['slope']*(seed-H*.45) for t in tracks]
            cols=[float(valid[(xx>=a)&(xx<=b)].mean()) if np.any((xx>=a)&(xx<=b)) else 0 for a,b in zip(cross,cross[1:])]
            if min(cols)<.34 or valid.mean()<.58:continue
            yy=float(np.polyval(coeff,0));
            if any(abs(t[0].position-yy)<max(7,span*.012) for t in traced):continue
            traced.append((rule_factory('h',coeff,yy,float(valid.mean()),(lx,rx)),cols))
        traced.sort(key=lambda p:p[0].position)
        # Reject long/partial grids; never choose an arbitrary subset of a long table.
        if len(traced) not in (3,4):continue
        hs=[p[0] for p in traced]; y0,y1=hs[0].position,hs[-1].position
        if y1-y0>H*.43 or min(np.diff([h.position for h in hs]))<10:continue
        # Refit each rail from observed segments within the verified table range.
        from .source_grid_v16 import _fit_rail
        vs=[]; covers=[]
        for t in tracks:
            cf=_fit_rail(raw,t,y0,y1,H); vr=rule_factory('v',cf,float(cf[-1]),1.,(y0,y1));vs.append(vr)
            covers.append([_coverage(vm,vr,a.position+4,b.position-4,H,W) for a,b in zip(hs,hs[1:])])
        if min(min(c) for c in covers)<.57:continue
        from .geometry_lab.algorithms import intersection
        nodes=np.array([[intersection(v,h,image.shape) for v in vs] for h in hs])
        if not np.isfinite(nodes).all() or nodes[:,:,0].min()<0 or nodes[:,:,0].max()>=W or nodes[:,:,1].min()<0 or nodes[:,:,1].max()>=H:continue
        if np.min(np.diff(nodes[:,:,0],axis=1))<10 or np.min(np.diff(nodes[:,:,1],axis=0))<9:continue
        # Both outer rails must end near the top/bottom. A partial long table
        # would still have all five rails continuing beyond the candidate.
        extension=max(14,min(35,round(np.median(np.diff(nodes[:,:,1],axis=0))*.45)))
        # Reject evidence of a further band on ALL rails (footer uses only right rails).
        continuations=[]
        for edge,sign in ((nodes[0],-1),(nodes[-1],1)):
            ext=[]
            for j,v in enumerate(vs):
                y=edge[j,1]; a,b=sorted((y+sign*6,y+sign*extension));ext.append(_coverage(vm,v,a,b,H,W))
            continuations.append(ext)
        if any(sum(c>.88 for c in ext)>=4 for ext in continuations):continue
        key=tuple(np.rint(nodes[:,:,1].mean(axis=1)/7).astype(int));
        if key in seen:continue
        seen.add(key);score=float(np.mean(covers)+np.mean([h.support for h in hs]))
        best.append((score,{'vertical':vs,'horizontal':hs,'shape':(H,W)}, {'status':'candidate_requires_header_and_mesh_safety','selection':'short_receipt_grid_v1','horizontal_boundaries':len(hs),'vertical_boundaries':5,'nodes':nodes.tolist(),'rail_coverage':covers,'horizontal_column_coverage':[p[1] for p in traced],'rail_extensions':continuations,'score':score}))
    if not best:
        report.update(proposals=len(proposals),wall_ms=round((perf_counter()-started)*1000,2));return None,report
    best.sort(key=lambda p:-p[0]);score,graph,report=best[0]
    # Refuse clearly different, similarly strong table candidates.
    if len(best)>1 and best[1][0]>=score-.025 and abs(best[1][1]['horizontal'][0].position-graph['horizontal'][0].position)>15:
        return None,{'status':'rejected','reason':'ambiguous_short_tables'}
    report['wall_ms']=round((perf_counter()-started)*1000,2)
    return graph,report


def validate_short_graph(image, graph):
    """Independent LSD row audit plus source-pixel rail and exterior checks."""
    from .geometry_lab.algorithms import intersection
    from .grid_edge_validator_v12 import audit_rowaware_printed_rows
    H,W=image.shape[:2]; vs=graph['vertical'];hs=graph['horizontal']
    if len(vs)!=5 or len(hs) not in (3,4):return False,{'reason':'not_short_five_rail'}
    nodes=np.asarray([[intersection(v,h,image.shape) for v in vs] for h in hs])
    if (not np.isfinite(nodes).all() or nodes[:,:,0].min()<0 or nodes[:,:,0].max()>=W or
        nodes[:,:,1].min()<0 or nodes[:,:,1].max()>=H or np.min(np.diff(nodes[:,:,0],axis=1))<10 or
        np.min(np.diff(nodes[:,:,1],axis=0))<9 or np.ptp(nodes[:,:,1])>H*.43):
        return False,{'reason':'invalid_nodes'}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,6)
    vm=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((max(9,min(17,round(H*.008))),1),np.uint8))
    covers=[[_coverage(vm,v,nodes[i,j,1]+4,nodes[i+1,j,1]-4,H,W) for i in range(len(hs)-1)] for j,v in enumerate(vs)]
    if min(min(c) for c in covers)<.57:return False,{'reason':'weak_rail','rail_coverage':covers}
    pad=max(14,min(35,round(np.median(np.diff(nodes[:,:,1],axis=0))*.45)))
    extensions=[]
    for edge,sign in ((nodes[0],-1),(nodes[-1],1)):
        ext=[]
        for j,v in enumerate(vs):
            a,b=sorted((edge[j,1]+sign*6,edge[j,1]+sign*pad));ext.append(_coverage(vm,v,a,b,H,W))
        extensions.append(ext)
    if any(sum(c>.88 for c in ext)>=4 for ext in extensions):return False,{'reason':'possible_partial_long_grid','extensions':extensions}
    # Audit the actual compact rail band, not distant page borders. A footer
    # with only right-hand rails is not a full product row.
    box=[0,max(0,int(nodes[:,:,1].min()-pad)),W,min(H,int(nodes[:,:,1].max()+pad))]
    ok,audit=audit_rowaware_printed_rows(image,nodes,paper_bbox=box)
    return bool(ok),{'audit':audit,'rail_coverage':covers,'rail_extensions':extensions,'audit_band':box,'nodes':nodes.tolist()}


def rescue(image, *, rectify, phase, header_validator, build_core, physical_detector):
    if header_validator is None:return None  # never accept without real header OCR
    graph,diagnostic=detect_short_grid(image)
    candidates=[]
    if graph is not None:candidates.append((graph,diagnostic))
    def attempt(graph,diagnostic):
        ok,audit=validate_short_graph(image,graph)
        if not ok:return None
        result,overlay,geometry=build_core(image,rectify=rectify,phase=phase,pre_detected=(graph,{
            'status':'OBSERVED_NOT_GT','selection':'source_verified_five_rail_v16','version':'short_receipt_grid_v1'}))
        safe=geometry.get('safety') or {}
        if (safe.get('status')!='MEASURED' or safe.get('folded_fraction',1)>0 or
                safe.get('source_valid_fraction',0)<.995 or geometry.get('unresolved_extreme_horizontal',0)>0 or
                len(result.xs)!=5 or len(result.ys) not in (3,4)):return None
        final_graph=result.info.get('source_grid')
        if final_graph is None:return None
        final_ok,final_audit=validate_short_graph(image,final_graph)
        if not final_ok or not header_validator(result):return None
        geometry['method']='Short Receipt Grid Recovery V1'
        from .paper_first_grid_v12 import json_native
        geometry['short_receipt_grid_v1']=json_native({'detector':diagnostic,'audit':audit,'final_audit':final_audit})
        return result,overlay,geometry
    for g,d in candidates:
        try:
            result=attempt(g,d)
            if result is not None:return result
        except (ValueError,cv2.error,TypeError,IndexError,KeyError,AttributeError):pass
    # Physical graph fallback, only after the compact detector fails and only
    # if the same short-grid evidence, safety and OCR-header guards pass.
    try:
        g,d=physical_detector(image)
        if g is not None:return attempt(g,{'selection':'physical_graph_short_fallback','detector':d})
    except (ValueError,cv2.error,TypeError,IndexError,KeyError,AttributeError):pass
    return None
