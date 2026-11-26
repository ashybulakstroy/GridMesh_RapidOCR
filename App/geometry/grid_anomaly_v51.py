"""V5.1: recover only optically confirmed missing interior boundaries; verify real curvature."""
import cv2
import numpy as np
from .grid_anomaly_v5 import _trace, _nodes, _rail_evidence, recover as recover_v5


def _gray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image


def _strip(gray, rule, rails, radius):
    H,W=gray.shape
    nodes=np.array([_nodes(v,rule,(H,W)) for v in rails]); left,right=nodes[0,0],nodes[-1,0]
    if not np.isfinite(nodes).all() or right-left<70 or left<0 or right>=W or np.min(np.diff(nodes[:,0]))<5:raise ValueError('v51_invalid_strip')
    x=np.linspace(left,right, max(100,min(600,round(right-left)+1)))
    y=rule.eval((x-W/2)/W); offsets=np.arange(-radius,radius+1)
    X=np.broadcast_to(x,(len(offsets),len(x)));Y=y[None,:]+offsets[:,None]
    patch=cv2.remap(gray,X.astype('f'),Y.astype('f'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    bh=cv2.morphologyEx(patch,cv2.MORPH_BLACKHAT,np.ones((9,1),np.uint8))
    ink=cv2.morphologyEx(bh,cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
    ink[(Y<1)|(Y>=H-1)]=0
    return x,y,ink,nodes


def _coverage(support,x,nodes):
    cols=[]
    for l,r in zip(nodes[:-1,0],nodes[1:,0]):
        mask=(x>=l+2)&(x<=r-2)
        cols.append(float(support[mask].mean()) if mask.any() else 0.)
    return float(support.mean()),cols


def _valid_graph(image,data):
    H,W=image.shape[:2];vs=data['vertical'];hs=data['horizontal']
    if len(vs)!=5 or len(hs)<3:return False
    n=np.array([[_nodes(v,h,(H,W)) for v in vs] for h in hs]);
    if not np.isfinite(n).all() or np.any(n<0) or np.any(n[:,:,0]>=W) or np.any(n[:,:,1]>=H) or np.min(np.diff(n[:,:,0],axis=1))<5 or np.min(np.diff(n[:,:,1],axis=0))<5:return False
    xx=(np.linspace(n[:,:,0].min(),n[:,:,0].max(),200)-W/2)/W; yy=(np.linspace(n[:,:,1].min(),n[:,:,1].max(),200)-H/2)/H
    return bool(np.min(np.diff(np.stack([r.eval(xx) for r in hs]),axis=0))>=4 and np.min(np.diff(np.stack([r.eval(yy) for r in vs]),axis=0))>=4)


def curve_proof(gray, rule, rails):
    x,y,ink,nodes=_strip(gray,rule,rails,7)
    support=ink[5:10].max(0)>=3
    cov,cols=_coverage(support,x,nodes)
    return {'coverage':cov,'columns':cols,'accepted':bool(cov>=.90 and min(cols)>=.70)}


def verified_curvature(image,data,index):
    """No exception to mesh safety. Only V5-repaired routes, and only optically supported curves."""
    if not data.get('v51_recovery_proven') or not _valid_graph(image,data):return False,{'reason':'no_verified_route_or_invalid_graph'}
    d=curve_proof(_gray(image),data['horizontal'][index],data['vertical'])
    return d['accepted'],d


def _trace_hint(gray,h,vs,hs,factory):
    H,W=gray.shape;x,y,ink,nodes=_strip(gray,h,vs,15); norm=(x-W/2)/W
    if any(np.percentile(abs(h.eval(norm)-old.eval(norm)),90)<8 for old in hs):return None,{'reason':'already_observed'}
    # Only internal candidates; outer rows must satisfy V5 termination checks.
    if np.min(y-hs[0].eval(norm))<6 or np.min(hs[-1].eval(norm)-y)<6:return None,{'reason':'not_internal'}
    # Never let the local path enter an existing row or switch across it.
    Y=y[None,:]+np.arange(-15,16)[:,None]
    for old in hs:ink[abs(Y-old.eval(norm)[None,:])<5]=0
    if (ink.max(0)>=3).mean()<.85:return None,{'reason':'weak_hint'}
    path=_trace(ink,15,15);j=np.arange(len(x));sup=ink[path,j]>=3;cov,cols=_coverage(sup,x,nodes)
    d={'coverage':cov,'columns':cols,'hint_position':float(h.position)}
    if cov<.90 or min(cols)<.70 or np.any((path==0)|(path==30)):return None,{**d,'reason':'weak_or_clipped_trace'}
    yy=y+path-15; coef=np.polyfit(norm[sup],yy[sup],3);res=np.abs(np.polyval(coef,norm)-yy);keep=sup&(res<=3)
    if keep.mean()<.85:return None,{**d,'reason':'unstable_fit'}
    coef=np.polyfit(norm[keep],yy[keep],3);new=factory('h',coef,float(np.polyval(coef,0)),cov,(float(x[0]),float(x[-1])))
    proof=curve_proof(gray,new,vs);d['curve_proof']=proof
    if not proof['accepted']:return None,{**d,'reason':'fitted_curve_not_confirmed'}
    if any(np.min(abs(new.eval(norm)-old.eval(norm)))<6 or np.min(new.eval(norm)-old.eval(norm))*np.max(new.eval(norm)-old.eval(norm))<=0 for old in hs):return None,{**d,'reason':'crossing_or_duplicate'}
    # A horizontal stroke must lie in the table, with continuing rails at its intersections.
    rail=_rail_evidence(gray,vs,new,'top');d['rails']=rail
    if sum(q['inside']>=.45 and q['outside']>=.45 for q in rail)<3:return None,{**d,'reason':'insufficient_continuing_rails'}
    return new,{**d,'reason':'confirmed','position':float(new.position)}


def recover_full(image,data,automatic,factory):
    """Preserve every old rule; AutoGrid hints are not ground truth."""
    if not _valid_graph(image,data):raise ValueError('v51_invalid_input_graph')
    gray=_gray(image);hs=list(data['horizontal']);vs=list(data['vertical']);details=[];added=0
    hints=[] if automatic is None else automatic.get('horizontal',[])
    if automatic is not None and tuple(automatic.get('shape',()))!=tuple(image.shape[:2]):raise ValueError('v51_hint_coordinate_mismatch')
    for h in hints:
        try:new,d=_trace_hint(gray,h,vs,hs,factory)
        except (ValueError,TypeError,IndexError,cv2.error) as exc:new,d=None,{'reason':str(exc)}
        details.append(d)
        if new is not None:
            trial=sorted(hs+[new],key=lambda r:r.position)
            if _valid_graph(image,{**data,'horizontal':trial}):hs=trial;added+=1
            else:d['reason']='invalid_full_graph'
    # A complete row count is not enough: the original MultiScale graph may
    # contain an inaccurate curve. Refit weak internal rules only with an
    # independently confirmed AutoGrid hint in the same physical band.
    corrected=[]
    for i in range(1,len(hs)-1):
        old=hs[i];old_proof=curve_proof(gray,old,vs)
        if old_proof['accepted']:continue
        near=sorted((h for h in hints if abs(h.position-old.position)<8),key=lambda h:abs(h.position-old.position))
        confirmed=False
        for hint in near:
            try:new,d=_trace_hint(gray,hint,vs,hs[:i]+hs[i+1:],factory)
            except (ValueError,TypeError,IndexError,cv2.error):continue
            if new is None or abs(new.position-old.position)>=8:continue
            x=(np.linspace(*new.domain,100)-gray.shape[1]/2)/gray.shape[1]
            if np.max(abs(new.eval(x)-old.eval(x)))>35:continue
            if d['curve_proof']['coverage']-old_proof['coverage']<.12:continue
            trial=hs[:i]+[new]+hs[i+1:]
            if not _valid_graph(image,{**data,'horizontal':trial}):continue
            hs=trial;confirmed=True;corrected.append({'index':i,'position':float(new.position),'before':old_proof,'after':d['curve_proof']});break
        if not confirmed:raise ValueError('v51_unconfirmed_existing_internal_curve:'+str(i))
    out={**data,'horizontal':hs};ext=None
    # Existing V5 remains sole authority for extending the table ends.
    try:out,ext=recover_v5(image,out,factory)
    except ValueError as exc:details.append({'extension':str(exc)})
    if not added and ext is None:raise ValueError('v51_no_confirmed_missing_edges')
    if not _valid_graph(image,out):raise ValueError('v51_invalid_recovered_graph')
    out['v51_recovery_proven']=True
    return out,{'version':'v5.1','added_internal':added,'added_rows':added+(ext or {}).get('added_rows',0),'extension':ext,'hints':details,'corrected_internal':corrected}
