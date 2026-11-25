"""Auto-grid fitting. Reimplements V2 local curve fitting; no ROI or GT file imports."""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _peaks(obs,length,sigma,distance,prominence):
    counts=np.zeros(length,float)
    for center,weight,_ in obs:
        idx=round(center)
        if 1<=idx<length-2: counts[idx-1:idx+2]+=weight
    smooth=gaussian_filter1d(counts,sigma)
    locations,_=find_peaks(smooth,distance=distance,prominence=prominence)
    return [(int(i),float(smooth[i])) for i in locations if smooth[i]>10]


def _h_support(y,obs,x0,x1,tol=10):
    width=x1-x0
    bits=np.zeros(width,bool)
    for center,weight,(xa,ya,xb,yb) in obs:
        if abs(center-y)>tol:continue
        start=max(x0,round(min(xa,xb)))
        stop=min(x1,round(max(xa,xb)))
        if stop>start:bits[start-x0:stop-x0]=True
    n=max(1,width//3)
    return {'left':float(bits[:n].mean()),'whole':float(bits.mean()),
            'center':float(bits[n:2*n].mean()),'right':float(bits[2*n:].mean())}


def _curve(axis,target,obs,ref,tol=10):
    samples=[]
    for intercept,weight,(xa,ya,xb,yb) in obs:
        if abs(intercept-target)>=tol:continue
        for t in np.linspace(0,1,max(3,int(weight/9))):
            samples.append((xa+(xb-xa)*t,ya+(yb-ya)*t))
    if len(samples)<4:return None
    arr=np.asarray(samples,float)
    along=arr[:,0] if axis=='h' else arr[:,1]
    across=arr[:,1] if axis=='h' else arr[:,0]
    along=(along-ref)/500
    degree=2 if np.ptp(along)>0.2 else 1
    coef=None
    for _ in range(5):
        try: coef=np.polyfit(along,across,degree)
        except (TypeError,ValueError,np.linalg.LinAlgError):return None
        err=np.abs(np.polyval(coef,along)-across)
        keep=err<max(3.5,np.median(err)*3)
        if all(keep) or sum(keep)<degree+1:break
        along=along[keep];across=across[keep]
    return coef


@_deep_timed("grid", "App/geometry/autogrid_gridrefine_v3/fitter.py:fit_grid")
def fit_grid(image,roi,relaxed_horizontal_support=False,anchor_right=False):
    height,width=image.shape[:2]
    x0,y0,x1,y1=[int(z) for z in roi]
    roi_im=image[y0:y1,x0:x1]
    if roi_im.size==0:
        return {'status':'NO_DETECTION','overlay':image.copy(),'horizontal_boundaries':0,'vertical_boundaries':0}
    xref=int(x0+.45*(x1-x0));yref=int(y0+.52*(y1-y0))
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(roi_im,cv2.COLOR_BGR2GRAY))[0]
    obs={'h':[],'v':[]}
    if raw is not None:
        for xa,ya,xb,yb in raw.reshape(-1,4):
            xa,xb,ya,yb=float(xa+x0),float(xb+x0),float(ya+y0),float(yb+y0)
            dx,dy=xb-xa,yb-ya
            if abs(dx)>2*abs(dy) and abs(dx)>max(10,(x1-x0)*.045):
                intercept=ya+(dy/dx)*(xref-xa)
                axis,weight='h',abs(dx)
                valid=y0-3<intercept<y1+3
            elif abs(dy)>2*abs(dx) and abs(dy)>max(10,(y1-y0)*.045):
                intercept=xa+(dx/dy)*(yref-ya)
                axis,weight='v',abs(dy)
                valid=x0-3<intercept<x1+3
            else:continue
            if valid:obs[axis].append((intercept,weight,(xa,ya,xb,yb)))
    hp=_peaks(obs['h'],height,5,19,10)
    vp=_peaks(obs['v'],width,5,20,6)
    goodh=[];reject=[]
    for y,score in hp:
        support=_h_support(y,obs['h'],x0,x1)
        # Standard mode preserves the previously validated 19-photo behavior.
        # The evidence-backed fallback tolerates weak/broken left-hand strokes,
        # but still requires support in all three table thirds. A short bottom
        # rule of the total has virtually no left support and is rejected.
        if relaxed_horizontal_support:
            physically_supported=(support['left']>=.095 and support['center']>=.22
                                  and support['right']>=.24 and support['whole']>=.245)
        else:
            physically_supported=support['left']>=.22 and support['whole']>=.33
        if physically_supported:goodh.append((y,score))
        else:reject.append({'y':int(y),'coverage':{k:round(v,3) for k,v in support.items()}})
    if vp:
        maximum=max(s for p,s in vp)
        goodv=[(x,s) for x,s in vp if s>=maximum*.45]
    else:goodv=[]
    hcurves=[_curve('h',y,obs['h'],xref) for y,s in goodh]
    vcurves=[_curve('v',x,obs['v'],yref) for x,s in goodv]
    right_anchor_diagnostics=[]
    if anchor_right and len(goodv)>=4:
        # Only for the independently triggered fallback: reconnect a partially
        # observed row at the right rail using *actual horizontal pixel ink*.
        # Fits from left-only LSD fragments otherwise extrapolate a bad parabola
        # across prices, VAT and totals. The anchor is determined without GT.
        bw=cv2.adaptiveThreshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),255,
             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
        thin_h=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,13),np.uint8))
        xr=goodv[-1][0]
        xa=max(x0,xr-72);xb=min(x1,xr-12)
        xanchor=(xa+xb)/2
        leftanchor=goodv[0][0]+min(50,(x1-x0)*.12)
        if xb-xa>=35 and xanchor>xref+40:
            evidence=gaussian_filter1d((thin_h[:,xa:xb]>0).mean(axis=1),1.5)
            for i,(y,s) in enumerate(goodh):
                coeff=hcurves[i]
                if coeff is None:continue
                lo=max(y0+2,int(y-32));hi=min(y1-2,int(y+33))
                peaks=[(float(evidence[j]),j) for j in range(lo+1,hi-1)
                       if evidence[j]>.12 and evidence[j]>evidence[j-1]
                       and evidence[j]>=evidence[j+1]]
                if not peaks:continue
                score,observed_y=max(peaks,key=lambda p:p[0]/(1+abs(p[1]-y)/38))
                fitted_at_anchor=float(np.polyval(coeff,(xanchor-xref)/500))
                delta=observed_y-fitted_at_anchor
                # Avoid large deviations that indicate another printed object.
                if abs(delta)>65:continue
                # A quadratic correction is zero at the left and centre, and
                # equals the measured change at the right: fully compatible with
                # V3 polynomial Rule representation and inverse mesh.
                normalized=((xanchor-leftanchor)*(xanchor-xref))/250000.
                if abs(normalized)<1e-6:continue
                # In u=(x-xref)/500: (x-leftanchor)*(x-xref)/250000
                correction=np.array([1.,(xref-leftanchor)/500.,0.])*(delta/normalized)
                hcurves[i]=np.polyadd(coeff,correction)
                right_anchor_diagnostics.append({'line_index':i,'measured_y':int(observed_y),
                          'x':round(xanchor,2),'source_coverage':round(score,3),
                          'pre_correction_y':round(fitted_at_anchor,2),
                          'correction_px':round(delta,2)})
    physical_refinements=[]
    if (anchor_right and len(goodv)>=4
            and any(abs(row['correction_px'])>20 for row in right_anchor_diagnostics)):
        from .refine import refine_uncertain_horizontals
        # The detected exterior verticals delimit *physical* table support.
        # Do not use the padded search ROI as actual table boundaries.
        region=(int(goodv[0][0]), y0, int(goodv[-1][0]), y1)
        hcurves,physical_refinements=refine_uncertain_horizontals(
            image,region,hcurves,xref)
    status='CURVES_FITTED' if len(hcurves)>=2 and len(vcurves)>=2 and all(c is not None for c in hcurves+vcurves) else 'INSUFFICIENT_SUPPORT'
    overlay=image.copy()
    def ev(coeff,x,ref):return float(np.polyval(coeff,(x-ref)/500))
    if status=='CURVES_FITTED':
        for coef in hcurves:
            pts=[]
            for x in range(x0,x1,2):
                y=ev(coef,x,xref)
                if ev(vcurves[0],y,yref)<=x<=ev(vcurves[-1],y,yref):pts.append((x,round(y)))
            if len(pts)>1:cv2.polylines(overlay,[np.asarray(pts,np.int32)],False,(255,120,0),2,cv2.LINE_AA)
        for coef in vcurves:
            pts=[]
            for y in range(y0,y1,2):
                x=ev(coef,y,yref)
                if ev(hcurves[0],x,xref)<=y<=ev(hcurves[-1],x,xref):pts.append((round(x),y))
            if len(pts)>1:cv2.polylines(overlay,[np.asarray(pts,np.int32)],False,(255,120,0),2,cv2.LINE_AA)
    return {'overlay':overlay,'status':status,'horizontal_boundaries':len(goodh),
            'vertical_boundaries':len(goodv),
            'h_positions':[y for y,s in goodh],'v_positions':[x for x,s in goodv],
            'horizontal_rejected':reject,
            'right_horizontal_ink_anchors':right_anchor_diagnostics,
            'physical_ink_refinements':physical_refinements,
            'h_curves':[c.tolist() if c is not None else None for c in hcurves],
            'v_curves':[c.tolist() if c is not None else None for c in vcurves]}
