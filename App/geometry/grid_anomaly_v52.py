"""V5.2: locally supported piecewise geometry, independent of detector route."""
import cv2
import numpy as np
from time import perf_counter
from .grid_anomaly_v5 import _trace, _peaks, _nodes, _rail_evidence, refine_rails
from .grid_anomaly_v51 import _strip, _coverage, _valid_graph, curve_proof, _gray
from .geometry_lab.detection import Rule

class PiecewiseRule:
    """Continuous linear spline through robust local knots. No high-degree global fit."""
    def __init__(self, x, y, width, coverage):
        self.axis='h'; self.extent=width; self.t=np.asarray(x,float); self.values=np.asarray(y,float)
        if len(self.t)<2 or np.any(np.diff(self.t)<=0) or not np.isfinite(self.values).all():raise ValueError('invalid_piecewise')
        self.points=np.column_stack((self.t,self.values));self.domain=(float(self.t[0]),float(self.t[-1]));self.support=float(coverage);self.position=float(self.eval(0.))
    def eval(self, coord):
        q=np.asarray(coord)*self.extent+self.extent/2
        if q.ndim==0:
            v=float(q)
            if v<self.t[0]:return self.values[0]+(v-self.t[0])*(self.values[1]-self.values[0])/(self.t[1]-self.t[0])
            if v>self.t[-1]:return self.values[-1]+(v-self.t[-1])*(self.values[-1]-self.values[-2])/(self.t[-1]-self.t[-2])
            return np.interp(v,self.t,self.values)
        z=np.interp(q,self.t,self.values)
        z=np.where(q<self.t[0],self.values[0]+(q-self.t[0])*(self.values[1]-self.values[0])/(self.t[1]-self.t[0]),z)
        return np.where(q>self.t[-1],self.values[-1]+(q-self.t[-1])*(self.values[-1]-self.values[-2])/(self.t[-1]-self.t[-2]),z)

def piecewise(x,y,width,coverage):
    order=np.argsort(x);x=np.asarray(x)[order];y=np.asarray(y)[order]
    if len(x)<30 or np.max(np.diff(x))>18:raise ValueError('unsupported_trace_gap')
    # Fit local straight segments with robust median residual, not individual pixel noise.
    knots=np.linspace(x[0],x[-1],max(3,int(np.ceil((x[-1]-x[0])/8))+1))
    vals=[]
    for k in knots:
        m=abs(x-k)<=9
        if m.sum()<3:raise ValueError('unsupported_spline_knot')
        xx=x[m]-k; yy=y[m];a,b=np.polyfit(xx,yy,1);vals.append(float(np.median(yy-a*xx)))
    r=PiecewiseRule(knots,vals,width,coverage)
    err=abs(r.eval((x-width/2)/width)-y)
    if np.percentile(err,90)>3 or np.max(abs(np.diff(vals)/np.diff(knots)))>.85:raise ValueError('unstable_piecewise')
    return r,float(np.percentile(err,90))

def scan_side(image,data,side):
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
        try:new,residual=piecewise(px,py,W,d['coverage'])
        except ValueError as exc:d['reason']=str(exc);continue
        proof=curve_proof(gray,new,vs);d['proof']=proof
        if not proof['accepted']:d['reason']='piecewise_not_confirmed';continue
        # Across the entire width, reject merged paths and crossings with the old boundary.
        xx=np.linspace(float(px.min()),float(px.max()),100);q=(xx-W/2)/W
        if np.min(sign*(new.eval(q)-r.eval(q)))<5:
            d['reason']='crossing_or_existing';continue
        d['fit_p90']=residual; d['position']=new.position
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



def correct_rule(gray,old,vs,others):
    H,W=gray.shape; radius=25
    x,y,ink,nodes=_strip(gray,old,vs,radius); norm=(x-W/2)/W
    Y=y[None,:]+np.arange(-radius,radius+1)[:,None]
    for h in others:ink[abs(Y-h.eval(norm)[None,:])<6]=0
    path=_trace(ink,radius,radius); sup=ink[path,np.arange(len(x))]>=3
    cov,cols=_coverage(sup,x,nodes)
    if cov<.90 or min(cols)<.70 or np.any((path==0)|(path==2*radius)):raise ValueError('weak_local_trace')
    new,res=piecewise(x[sup],(y+path-radius)[sup],W,cov)
    proof=curve_proof(gray,new,vs)
    if not proof['accepted']:raise ValueError('weak_local_fit')
    if np.max(abs(new.eval(norm)-old.eval(norm)))>radius:raise ValueError('correction_too_far')
    return new,{'coverage':cov,'fit_p90':res,'proof':proof,'max_displacement':float(np.max(abs(new.eval(norm)-old.eval(norm))))}

def recover(image,data,*,automatic=None):
    if not _valid_graph(image,data):raise ValueError('v52_invalid_input_graph')
    gray=_gray(image);vs=list(data['vertical']);hs=list(data['horizontal']);start=perf_counter()
    a,top=scan_side(image,data,'top');b,bottom=scan_side(image,data,'bottom')
    # Outer boundaries use termination evidence; intermediate new rules must have continuing rails.
    for h in a[1:]+b[:-1]:
        evidence=_rail_evidence(gray,vs,h,'top')
        if sum(q['inside']>=.45 and q['outside']>=.45 for q in evidence)<3:
            raise ValueError('v52_unconfirmed_internal_extension')
    hs=a+hs+b; out={**data,'horizontal':hs};corrected=[];added_internal=[]
    # Missing interior edges are hypothesized by an independent detector only.
    if automatic is not None and tuple(automatic.get('shape',()))==tuple(image.shape[:2]):
        for h in automatic.get('horizontal',[]):
            try:
                x,y,_,_=_strip(gray,h,vs,3);u=(x-gray.shape[1]/2)/gray.shape[1]
                if any(np.percentile(abs(y-r.eval(u)),90)<10 for r in hs):continue
                if min(y-hs[0].eval(u))<8 or min(hs[-1].eval(u)-y)<8:continue
                new,d=correct_rule(gray,h,vs,hs)
                rail=_rail_evidence(gray,vs,new,'top')
                if sum(z['inside']>=.45 and z['outside']>=.45 for z in rail)<3:continue
                trial=sorted(hs+[new],key=lambda r:r.position)
                if _valid_graph(image,{**out,'horizontal':trial}):hs=trial;added_internal.append(d)
            except (ValueError,TypeError,IndexError,cv2.error):continue
    # Refine only failing curves, keeping all supported geometry untouched.
    for i,old in enumerate(hs):
        before=curve_proof(gray,old,vs)
        if before['accepted']:continue
        new,d=correct_rule(gray,old,vs,hs[:i]+hs[i+1:])
        if d['proof']['coverage']-before['coverage']<.04 and min(d['proof']['columns'])-min(before['columns'])<.10:raise ValueError('v52_no_measured_improvement:'+str(i))
        trial=hs[:i]+[new]+hs[i+1:]
        if not _valid_graph(image,{**out,'horizontal':trial}):raise ValueError('v52_repaired_crossing:'+str(i))
        hs=trial;corrected.append({'index':i,'before':before,**d})
    if not (a or b or corrected or added_internal):raise ValueError('v52_no_confirmed_change')
    out={**out,'horizontal':hs}
    vs,rails=refine_rails(image,vs,hs,Rule);out['vertical']=vs
    if not _valid_graph(image,out):raise ValueError('v52_invalid_recovered_graph')
    proofs=[curve_proof(gray,h,vs) for h in hs]
    if not all(p['accepted'] for p in proofs):raise ValueError('v52_unconfirmed_final_curves')
    out['v51_recovery_proven']=True;out['v52_recovery_proven']=True
    return out,{'version':'5.2','added_top':len(a),'added_bottom':len(b),'added_internal':len(added_internal),'added_rows':len(a)+len(b)+len(added_internal),'corrected':corrected,'interior':added_internal,'top':top,'bottom':bottom,'rails':rails,'proofs':proofs,'elapsed_ms':(perf_counter()-start)*1000}

def enabled():
    from ..config import setting
    return setting('OCR_GRID_ANOMALY_V52','0')=='1' and setting('OCR_GRID_ANOMALY_V3','1')=='1'


def audit_hook(original):
    """Escalate only when the existing audit has contrary or incomplete top evidence."""
    from functools import wraps
    @wraps(original)
    def audit(image,rect,geometry,**kw):
        result=original(image,rect,geometry,**kw)
        if not enabled():return result
        safety=geometry.get('safety') or {}
        result['v52_checks']={'mesh_safe':safety.get('status')=='MEASURED' and safety.get('folded_fraction',1)==0 and safety.get('source_valid_fraction',0)>=.995,
                             'unresolved_curves':geometry.get('unresolved_extreme_horizontal',0)}
        if result['status']!='pass':return result
        scan=result.get('source_extension',{});anchor=scan.get('anchor',{});starts=anchor.get('rail_start_y',[])
        y=anchor.get('candidate_top_y'); contrary=result.get('number_check',{}).get('effect') in ('negative_weak','negative_strong')
        earlier=y is not None and sum(z is not None and y-z>20 for z in starts)>=3
        if not (contrary or earlier):return result
        data=rect.info.get('source_grid')
        try:
            if data is None or len(data['vertical'])!=5:return result
            found,detail=scan_side(image,data,'top');result['v52_top']=detail
            if found:
                result['status']='suspect';result['reasons'].append('printed_grid_above_candidate_v52');result['strong_geometry']=False;result['geometry_confidence']='defect'
        except (ValueError,TypeError,IndexError,KeyError,cv2.error) as exc:result['v52_scan_error']=str(exc)
        return result
    return audit


def build_candidate(image,rect,geometry,scan,*,rectify=False,multiscale=False):
    """Uses the normal renderer; independent header/audit still mandatory at caller."""
    from .grid_anomaly_v3 import build_direct
    from .geometry_lab.physical_grid_graph_v3 import _multiscale_core,ridge_response
    from . import detector_routing_v1 as route
    if not enabled():return None
    data=rect.info.get('source_grid'); original=data
    if multiscale:
        if 'AutoGrid' not in geometry.get('method','') or 'Physical Grid Graph' in geometry.get('method',''):return None
        gray=_gray(image);response,dark=ridge_response(gray)
        data,report=_multiscale_core(gray,response,dark,None,{'status':'NOT_RUN'})
        route.record('v52_autogrid_multiscale',found=data is not None,selection=report.get('selection'),reason=report.get('reason'))
        if data is None:return None
    elif not (geometry.get('safety',{}).get('status')=='MEASURED' and geometry.get('safety',{}).get('folded_fraction',1)==0 and geometry.get('safety',{}).get('source_valid_fraction',0)>=.995):
        raise ValueError('v52_unsafe_initial_mesh_use_independent_detector')
    if data is None:return None
    try:fixed,diag=recover(image,data,automatic=original if multiscale else None)
    except ValueError as exc:
        # A completely supported independent graph need not be changed.
        if not multiscale or str(exc)!='v52_no_confirmed_change' or not _valid_graph(image,data):raise
        proofs=[curve_proof(_gray(image),h,data['vertical']) for h in data['horizontal']]
        if not all(p['accepted'] for p in proofs):raise
        fixed=data;diag={'version':'5.2','added_rows':0,'corrected':[],'proofs':proofs}
    # Independent candidate may not lose established horizontal boundaries.
    if multiscale and original is not None:
        if len(fixed['horizontal'])<len(original['horizontal']):raise ValueError('v52_multiscale_lost_rows')
        # Every independently supported original boundary must still be present.
        H,W=image.shape[:2]
        for old in original['horizontal']:
            if not curve_proof(_gray(image),old,original['vertical'])['accepted']:continue
            x=(np.linspace(*old.domain,80)-W/2)/W
            if not any(np.percentile(abs(old.eval(x)-h.eval(x)),90)<=10 for h in fixed['horizontal']):
                raise ValueError('v52_multiscale_lost_confirmed_boundary')
    clean={**geometry,'unresolved_extreme_horizontal':0,'v52_recovery':diag}
    clean['warnings']=[w for w in geometry.get('warnings',[]) if w!='Обнаружена сильно искривлённая граница строки; проверьте фото.']
    if multiscale:clean['method']='Physical Grid Graph V3 (AutoGrid rescue)';clean['physical_detection_selection']='v52_multiscale_rescue'
    result=build_direct(image,rect,clean,scan,rectify=rectify,prepared=(fixed,diag))
    route.record('v52_recovery',multiscale=multiscale,added=diag['added_rows'],corrected=len(diag['corrected']))
    return result
