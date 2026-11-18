"""Physical-ink based local correction for uncertain bent separator curves.

Never uses a receipt ID, GT annotation, row count or manual ROI. A proposal is
accepted only when direct printed-stroke support improves and topology stays
valid. The returned polynomial is the same curve used by both overlay and V3.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _evidence_map(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV,31,9)
    return cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,13),np.uint8)).astype(np.float32)/255.


def _support(mask,coeff,xref,xs):
    scores=[]
    for x in xs:
        y=int(round(np.polyval(coeff,(x-xref)/500)))
        if y<4 or y>=mask.shape[0]-4:continue
        a=max(0,int(x)-10);b=min(mask.shape[1],int(x)+10)
        scores.append(float(mask[y-2:y+3,a:b].max(axis=0).mean()))
    return float(np.mean(scores)) if scores else 0.


def _measure(mask,coeff,xref,x0,y0,x1,y1):
    points=[]
    for x in range(x0+35,x1-30,28):
        pred=float(np.polyval(coeff,(x-xref)/500))
        lo=max(y0+2,int(pred-24));hi=min(y1-2,int(pred+25))
        if hi-lo<9:continue
        ys=np.arange(lo,hi+1)
        sc=gaussian_filter1d(mask[ys,max(0,x-14):min(mask.shape[1],x+14)].mean(axis=1),1.2)
        inds,_=find_peaks(sc,height=.09,prominence=.06,distance=3)
        if not len(inds):continue
        ix=max(inds,key=lambda k:sc[k]-.007*abs(float(ys[k])-pred))
        points.append((float(x),float(ys[ix]),float(sc[ix])))
    if len(points)<9:return []
    # Reject dark text fragments/total-border spurs by the *local geometry* of
    # neighbouring observations, not by darkness alone.
    keep=np.ones(len(points),bool)
    for i in range(1,len(points)-1):
        xa,ya,_=points[i-1];x,y,_=points[i];xb,yb,_=points[i+1]
        expected=ya+(yb-ya)*(x-xa)/(xb-xa)
        # A solitary text segment makes a steep peak and immediate recovery;
        # a physical fold has neighbouring samples following the same slope.
        if abs(y-expected)>11 and abs(yb-ya)<14:
            keep[i]=False
    pts=[p for p,k in zip(points,keep) if k]
    while len(pts)>=3:
        xa,ya,_=pts[-3];xb,yb,_=pts[-2];x,y,_=pts[-1]
        expected=yb+(yb-ya)*(x-xb)/(xb-xa)
        if abs(y-expected)<=11:break
        pts.pop()
    return pts


def refine_uncertain_horizontals(image,roi,hcurves,xref):
    """Return copied coefficients, diagnostics; apply only to uncertain fallback.

    Only correct a line if enough observed printed-ink samples exist across
    most of the width and the new polynomial better follows those samples.
    """
    x0,y0,x1,y1=map(int,roi)
    mask=_evidence_map(image)
    original=[np.asarray(c,dtype=float) for c in hcurves]
    new=list(original)
    diagnostics=[]
    for idx,coeff in enumerate(original):
        pts=_measure(mask,coeff,xref,x0,y0,x1,y1)
        if len(pts)<10:
            diagnostics.append({'index':idx,'accepted':False,'reason':'too_few_anchors','anchors':len(pts)})
            continue
        xx=np.array([p[0] for p in pts]); yy=np.array([p[1] for p in pts]); ww=np.array([p[2] for p in pts]);
        spread=(xx[-1]-xx[0])/max(1,x1-x0)
        if spread<.70 or xx[0]-x0>80 or x1-xx[-1]>80:
            diagnostics.append({'index':idx,'accepted':False,'reason':'insufficient_span','span':round(spread,3)})
            continue
        # Endpoints are explicitly labelled as INFERRED short continuations, not
        # pixel observations. Clamp extrapolation to the measured adjacent slope,
        # avoiding the extreme end oscillation of high-order polynomials.
        slope_left=np.clip((yy[1]-yy[0])/(xx[1]-xx[0]),-.30,.30)
        slope_right=np.clip((yy[-1]-yy[-2])/(xx[-1]-xx[-2]),-.30,.30)
        endpoints=[(x0,yy[0]+slope_left*(x0-xx[0])),
                   (x1,yy[-1]+slope_right*(x1-xx[-1]))]
        xfit=np.r_[endpoints[0][0],xx,endpoints[1][0]]
        yfit=np.r_[endpoints[0][1],yy,endpoints[1][1]]
        wfit=np.r_[.65,ww,.65]
        # Degree 3..5 captures a central fold; bound both slope and displacement.
        cand=[]
        for degree in (3,4,5):
            fit=np.polyfit((xfit-xref)/500,yfit,degree,w=np.sqrt(wfit))
            probes=np.linspace(x0+5,x1-5,90)
            predicted=np.polyval(fit,(probes-xref)/500)
            old=np.polyval(coeff,(probes-xref)/500)
            disp=float(np.max(np.abs(predicted-old)))
            slope=float(np.max(np.abs(np.diff(predicted)/np.diff(probes))))
            rms=float(np.sqrt(np.average((np.polyval(fit,(xfit-xref)/500)-yfit)**2,weights=wfit)))
            if disp<=23 and slope<=.38 and rms<=4.0:
                cand.append((rms+.16*degree,fit,degree,disp,slope))
        if not cand:
            diagnostics.append({'index':idx,'accepted':False,'reason':'polynomial_guard_rejected','anchors':len(pts)})
            continue
        _,fit,degree,disp,slope=min(cand,key=lambda c:c[0])
        probes=np.linspace(x0+25,x1-25,52)
        oldscore=_support(mask,coeff,xref,probes)
        newscore=_support(mask,fit,xref,probes)
        if newscore < oldscore+.035 or newscore < oldscore*1.12:
            diagnostics.append({'index':idx,'accepted':False,'reason':'no_ink_support_gain',
                                'old_support':round(oldscore,4),'new_support':round(newscore,4),
                                'anchors':len(pts)})
            continue
        tentative=list(new);tentative[idx]=fit
        # Preserve the actual topology of printed rows at all x positions.
        test_x=np.linspace(x0+7,x1-7,64)
        stack=np.array([np.polyval(c,(test_x-xref)/500) for c in tentative])
        if np.min(np.diff(stack,axis=0))<8:
            diagnostics.append({'index':idx,'accepted':False,'reason':'crossing_rows'})
            continue
        new[idx]=fit
        diagnostics.append({'index':idx,'accepted':True,'polynomial_degree':degree,
                            'max_shift_px':round(disp,2),'max_slope':round(slope,3),
                            'old_support':round(oldscore,4),'new_support':round(newscore,4),
                            'anchors':len(pts),'inferred_endpoints':[(round(x,1),round(y,1)) for x,y in endpoints]})
    return new,diagnostics
