"""Conservative source-pixel extension, preserving existing rules. No row count prior."""
import numpy as np
import cv2
from time import perf_counter


def _trace(hr,k,radius=13):
    lo=max(0,k-radius);hi=min(len(hr),k+radius+1)
    vals=hr[lo:hi].T.astype(float);N,K=vals.shape
    offsets=np.arange(lo,hi)-k
    reward=np.minimum(vals/5.,2.)-.015*abs(offsets)[None,:]
    dp=reward[0].copy();par=np.zeros((N,K),np.int16);pos=np.arange(K);shifts=np.arange(-2,3)
    idx=pos[:,None]+shifts;clip=idx.clip(0,K-1);valid=idx==clip
    for j in range(1,N):
        a=np.where(valid,dp[clip]-.35*abs(shifts),-1e6);q=a.argmax(1)
        par[j]=clip[pos,q];dp=a[pos,q]+reward[j]
    path=np.zeros(N,int);path[-1]=dp.argmax()
    for j in range(N-1,0,-1):path[j-1]=par[j,path[j]]
    return path+lo


def _peaks(score):
    # Highest peak first, minimum 9 px separation. No assumed row pitch.
    candidates=[i for i in range(1,len(score)-1) if score[i]>=.22 and score[i]>=score[i-1] and score[i]>=score[i+1]]
    chosen=[]
    for i in sorted(candidates,key=lambda i:(-score[i],i)):
        if all(abs(i-j)>=9 for j in chosen):chosen.append(i)
    return chosen[:24]


def _nodes(v,h,shape):
    H,W=shape[:2];x=float(v.position);y=float(h.position)
    for _ in range(10):x=float(v.eval((y-H/2)/H));y=float(h.eval((x-W/2)/W))
    return x,y


def _rail_evidence(gray, vertical, edge, side):
    H,W=gray.shape; out=[]; sign=1 if side=='top' else -1
    for v in vertical:
        x,y=_nodes(v,edge,(H,W)); offsets=np.arange(5,31)*sign
        def strength(yy):
            xx=v.eval((yy-H/2)/H); rad=4
            # vertical ridge in rail coordinates; accept only longitudinal ink.
            X=xx[:,None]+np.arange(-rad,rad+1)[None,:];Y=np.broadcast_to(yy[:,None],X.shape)
            patch=cv2.remap(gray,X.astype('f'),Y.astype('f'),cv2.INTER_LINEAR,borderValue=255)
            bh=cv2.morphologyEx(patch,cv2.MORPH_BLACKHAT,np.ones((1,9),np.uint8))
            vr=cv2.morphologyEx(bh,cv2.MORPH_OPEN,np.ones((7,1),np.uint8))
            return float((vr[:,2:-2].max(1)>=3).mean())
        out.append({'inside':strength(y+offsets),'outside':strength(y-offsets),'node':[x,y]})
    return out


def scan_side(image,data,side,rule_factory):
    t0=perf_counter(); H,W=image.shape[:2]; vs=data['vertical'];hs=data['horizontal'];r=hs[0 if side=='top' else -1]
    sign=-1 if side=='top' else 1
    # Limit scan to physical neighborhood, not the entire receipt or barcode.
    depth=min(260,max(100,round(H*.14))) if side=='top' else min(160,max(65,round(H*.075)))
    offsets=np.arange(-depth,3) if side=='top' else np.arange(-3,depth+1)
    levels=r.position+offsets; n=400;u=np.linspace(0,1,n)[None,:]
    left=vs[0].eval((levels-H/2)/H);right=vs[-1].eval((levels-H/2)/H)
    X=left[:,None]*(1-u)+right[:,None]*u;Y=r.eval((X-W/2)/W)+offsets[:,None]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    patch=cv2.remap(gray,X.astype('f'),Y.astype('f'),cv2.INTER_LINEAR,borderValue=255)
    bh=cv2.morphologyEx(patch,cv2.MORPH_BLACKHAT,np.ones((9,1),np.uint8))
    hr=cv2.morphologyEx(bh,cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
    score=(hr>=3).mean(1); info={'side':side,'depth':depth,'candidates':[],'status':'no_signal'}
    candidates=[]
    for k in _peaks(score):
        if offsets[k]*sign<8:continue
        path=_trace(hr,k); j=np.arange(n); sup=hr[path,j]>=3; octants=[float(a.mean()) for a in np.array_split(sup,8)]
        d={'seed_offset':int(offsets[k]),'coverage':float(sup.mean()),'octants':octants,'offset_median':float(np.median(offsets[path]))}
        info['candidates'].append(d)
        if d['coverage']<.90 or min(octants)<.70 or sign*d['offset_median']<8:
            d['reason']='weak_or_partial_or_existing';continue
        px=X[path,j][sup];py=Y[path,j][sup];norm=(px-W/2)/W
        # Smooth polynomial fits on observed pixels, never a straight line joining endpoints.
        coef=np.polyfit(norm,py,3);err=abs(np.polyval(coef,norm)-py);keep=err<=3
        if keep.mean()<.85:
            d['reason']='unstable_curve';continue
        coef=np.polyfit(norm[keep],py[keep],3); residual=abs(np.polyval(coef,norm)-py)
        if np.percentile(residual,90)>3:
            d['reason']='curve_residual';continue
        new=rule_factory('h',coef,float(np.polyval(coef,0)),d['coverage'],(float(px.min()),float(px.max())))
        # Across the entire width, reject merged paths and crossings with the old boundary.
        xx=np.linspace(float(px.min()),float(px.max()),100);q=(xx-W/2)/W
        if np.min(sign*(new.eval(q)-r.eval(q)))<5:
            d['reason']='crossing_or_existing';continue
        d['fit_p90']=float(np.percentile(residual,90)); d['position']=new.position
        candidates.append((new,d))
    # deduplicate by full curve, retain strongest evidence, not position alone
    unique=[]
    for new,d in sorted(candidates,key=lambda x:-x[1]['coverage']):
        xx=(np.linspace(*new.domain,100)-W/2)/W
        if any(np.percentile(abs(new.eval(xx)-v.eval(xx)),90)<5 for v,_ in unique):
            d['reason']='duplicate';continue
        unique.append((new,d))
    unique.sort(key=lambda z:z[0].position)
    if unique:
        outer=unique[0 if side=='top' else -1]
        rail=_rail_evidence(gray,vs,outer[0],side);info['outer_rail_evidence']=rail
        # Majority of rails inside; both external rails or majority show a termination.
        inside=sum(v['inside']>=.55 for v in rail); stops=sum(v['inside']>=.55 and v['outside']<=.40 for v in rail)
        info.update(inside_rails=inside,terminating_rails=stops)
        if inside>=3 and (stops>=2 or (side=='bottom' and inside==5 and rail[0]['outside']<=.20 and rail[1]['outside']<=.65 and outer[1]['coverage']>=.95)):
            for new,d in unique:d['reason']='confirmed'
            info['status']='confirmed';info['positions']=[float(n.position) for n,d in unique]
        else:
            info['status']='unconfirmed_anchor';
            for _,d in unique:d['reason']='unconfirmed_anchor'
            unique=[]
    info['elapsed_ms']=round((perf_counter()-t0)*1000,3)
    return [v for v,d in unique],info


def refine_rails(image, vs, hs, rule_factory):
    """Only change a rail if a coherent source track proves significant drift."""
    H,W=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    out=list(vs); details=[]
    for i,v in enumerate(vs):
        a,b=_nodes(v,hs[0],(H,W)),_nodes(v,hs[-1],(H,W));ys=np.arange(round(a[1])+4,round(b[1])-4)
        if len(ys)<80:continue
        xx=v.eval((ys-H/2)/H); radius=25
        X=xx[:,None]+np.arange(-radius,radius+1);Y=np.broadcast_to(ys[:,None],X.shape)
        patch=cv2.remap(gray,X.astype('f'),Y.astype('f'),cv2.INTER_LINEAR,borderValue=255)
        bh=cv2.morphologyEx(patch,cv2.MORPH_BLACKHAT,np.ones((1,9),np.uint8))
        vr=cv2.morphologyEx(bh,cv2.MORPH_OPEN,np.ones((15,1),np.uint8))
        # Search only if original rail has a substantial unsupported section.
        if min(float(a.mean()) for a in np.array_split(vr[:,radius-3:radius+4].max(1)>=5,8))>=.80:continue
        path=_trace(vr.T,radius,radius);sup=vr[np.arange(len(ys)),path]>=5;off=path-radius
        cov=float(sup.mean()); bins=[float(z.mean()) for z in np.array_split(sup,8)]
        if cov<.96 or min(bins)<.90 or np.max(abs(off))>=radius-1:continue
        if max(abs(float(np.median(z))) for z in np.array_split(off,8))<5:continue
        t=(ys-H/2)/H;coef=np.polyfit(t[sup],(xx+off)[sup],3);err=abs(np.polyval(coef,t)-(xx+off))
        if np.percentile(err[sup],90)>2:continue
        # Do not move regions where original curve already follows the observed track.
        delta=np.polyval(coef,t)-xx
        if np.any(abs(delta[abs(off)<=2])>4):continue
        out[i]=rule_factory('v',coef,float(np.polyval(coef,0)),cov,(float(ys[0]),float(ys[-1])))
        details.append({'rail':i,'coverage':cov,'max_correction':float(max(abs(delta))),'fit_p90':float(np.percentile(err[sup],90))})
    return out,details


def recover(image,data,rule_factory):
    if data is None or len(data.get('vertical',[]))!=5 or len(data.get('horizontal',[]))<3:
        raise ValueError('v5_requires_five_rails_and_three_lines')
    a,top=scan_side(image,data,'top',rule_factory); b,bottom=scan_side(image,data,'bottom',rule_factory)
    if not a and not b:raise ValueError('v5_no_confirmed_extension')
    vs=list(data['vertical']);old=list(data['horizontal']); hs=a+old+b;H,W=image.shape[:2]
    vs,rail_refinements=refine_rails(image,vs,hs,rule_factory)
    nodes=np.array([[_nodes(v,h,(H,W)) for v in vs] for h in hs])
    if not np.isfinite(nodes).all() or np.min(np.diff(nodes[:,:,0],axis=1))<5 or np.min(np.diff(nodes[:,:,1],axis=0))<5 or np.any(nodes<0) or np.any(nodes[:,:,0]>=W) or np.any(nodes[:,:,1]>=H):raise ValueError('v5_invalid_nodes')
    xx=(np.linspace(nodes[:,:,0].min(),nodes[:,:,0].max(),160)-W/2)/W
    yy=(np.linspace(nodes[:,:,1].min(),nodes[:,:,1].max(),160)-H/2)/H
    if np.min(np.diff(np.stack([h.eval(xx) for h in hs]),axis=0))<4 or np.min(np.diff(np.stack([v.eval(yy) for v in vs]),axis=0))<4:raise ValueError('v5_crossing_curves')
    return {**data,'horizontal':hs,'vertical':vs},{'version':'v5','added_top':len(a),'added_bottom':len(b),'added_rows':len(a)+len(b),'preserved_horizontal':len(old),'rail_refinements':rail_refinements,'top':top,'bottom':bottom,'nodes':nodes.tolist()}


def install_hooks(audit_v4,recover_v4):
    """Keep V4 as first choice for ordinary top-only cases; check lower extent too."""
    from functools import wraps
    from ..config import setting
    from .geometry_lab.detection import Rule
    @wraps(audit_v4)
    def audit(image,rect,geometry,**kw):
        started=perf_counter()
        base=audit_v4(image,rect,geometry,**kw)
        if setting('OCR_GRID_ANOMALY_V5','0')!='1':return base
        data=rect.info.get('source_grid')
        if not data or len(data.get('vertical',[]))!=5 or len(data.get('horizontal',[]))<3:return base
        try:
            _,bottom=scan_side(image,data,'bottom',Rule)
            base['v5_bottom']=bottom
            if bottom['status']=='confirmed':
                base['status']='suspect';base['reasons'].append('printed_grid_below_candidate');base['geometry_confidence']='defect';base['strong_geometry']=False
                base['source_extension']['v5_bottom_confirmed']=True
            # V4 may miss strongly curved top lines. Only run extra full top scan
            # for inconclusive geometry or contrary ordinal, not for strong V4 pass.
            if base['status']=='pass' and not base.get('strong_geometry') and base.get('number_check',{}).get('effect') in ('negative_strong','negative_weak'):
                _,top=scan_side(image,data,'top',Rule);base['v5_top']=top
                if top['status']=='confirmed':
                    base['status']='suspect';base['reasons'].append('printed_grid_above_candidate_v5');base['geometry_confidence']='defect';base['strong_geometry']=False
                    base['source_extension']['v5_top_confirmed']=True
        except (ValueError,TypeError,IndexError,KeyError,cv2.error,OverflowError) as exc:
            base['v5_scan_error']=str(exc)
        base['elapsed_ms']=round((perf_counter()-started)*1000,3)
        return base
    @wraps(recover_v4)
    def recovery(image,data,scan,*,rule_factory):
        if setting('OCR_GRID_ANOMALY_V5','0')!='1':return recover_v4(image,data,scan,rule_factory=rule_factory)
        # If the lower boundary is known missing, V4 cannot repair it.
        if scan.get('v5_top_confirmed') and scan.get('anchor',{}).get('status')!='candidate':
            return recover(image,data,rule_factory)
        try:
            old,diag=recover_v4(image,data,scan,rule_factory=rule_factory)
        except (ValueError,TypeError,IndexError,KeyError,cv2.error,OverflowError) as exc:
            try:return recover(image,data,rule_factory)
            except (ValueError,TypeError,IndexError,KeyError,cv2.error,OverflowError) as v5:
                raise ValueError(f'V4: {exc}; V5: {v5}') from v5
        # PreMesh entry point may not carry the bottom flag. Preserve the V4
        # recovery and independently check if an additional full-width edge exists.
        try:
            new,bottom=scan_side(image,old,'bottom',rule_factory)
            if new:
                fixed,extra=recover(image,old,rule_factory)
                return fixed,{**diag,'v5_extension':extra,'added_rows':diag.get('added_rows',0)+extra['added_rows']}
        except (ValueError,TypeError,IndexError,KeyError,cv2.error,OverflowError):
            if scan.get('v5_bottom_confirmed'):raise
        if scan.get('v5_bottom_confirmed'):
            raise ValueError('v5_bottom_not_recovered_after_v4')
        return old,diag
    return audit,recovery
