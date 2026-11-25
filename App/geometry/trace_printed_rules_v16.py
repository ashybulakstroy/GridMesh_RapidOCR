"""Trace actual horizontal printed ink, not a curve joining two side anchors.

The output horizontal rules are measured piecewise polylines. Missing ink is
permitted only for short local gaps, with independent support on all thirds.
The printed VAT row is not a prerequisite.
"""
from __future__ import annotations
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import numpy as np
import cv2
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


class MeasuredPolylineRule:
    """Minimal GridMesh-compatible rule: actual measured x/y ink samples."""
    def __init__(self, points, shape):
        arr=np.asarray(points,float)
        if arr.ndim!=2 or arr.shape[1]!=2 or len(arr)<3 or not np.isfinite(arr).all() or np.min(np.diff(arr[:,0]))<=0:
            raise ValueError('invalid measured polyline')
        self.axis='h'
        self.points=arr
        self.extent=int(shape[1])
        self.domain=(float(arr[0,0]),float(arr[-1,0]))
        self.support=(self.domain[1]-self.domain[0])/self.extent
        self.position=float(self.eval(0.))
    def eval(self, coord):
        x=np.asarray(coord,dtype=float)*self.extent+self.extent/2
        xx=self.points[:,0];yy=self.points[:,1]
        y=np.interp(x,xx,yy)
        left=yy[0]+(x-xx[0])*(yy[1]-yy[0])/(xx[1]-xx[0])
        right=yy[-1]+(x-xx[-1])*(yy[-1]-yy[-2])/(xx[-1]-xx[-2])
        return np.where(x<xx[0],left,np.where(x>xx[-1],right,y))


def _polyline(points, shape):
    return MeasuredPolylineRule(points,shape)

@_deep_timed("grid", "App/geometry/trace_printed_rules_v16.py:trace_printed_horizontals")
def trace_printed_horizontals(image, graph, *, points_per_row=33,
                              search_px=11, min_coverage=.70,
                              min_third_coverage=.55, max_gap=4):
    if image is None or image.size==0:
        return None, {'reason':'empty_image'}
    H,W=image.shape[:2]
    vertical=list(graph.get('vertical',[]))
    horizontal=list(graph.get('horizontal',[]))
    if len(vertical)!=5 or len(horizontal)<3:
        return None,{'reason':'not_a_five_rail_graph'}
    center_y=float(np.median([r.position for r in horizontal]))
    rail_x=np.array([v.eval((center_y-H/2)/H) for v in vertical],float)
    if np.any(np.diff(rail_x)<7):
        return None,{'reason':'invalid_rail_spacing'}
    x0,x1=float(rail_x[0]+6),float(rail_x[-1]-12)
    if x1-x0<120:
        return None,{'reason':'short_line_span'}
    x_samples=np.linspace(x0,x1,points_per_row)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV,31,6)
    # Suppress individual glyph strokes; retain printed, near-horizontal rules.
    strokes=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
    traces=[];records=[]
    for idx,h in enumerate(horizontal):
        predictions=np.array([float(h.eval((x-W/2)/W)) for x in x_samples])
        baseline=np.rint(predictions).astype(int)
        search=np.arange(-search_px,search_px+1)
        strengths=np.zeros((len(x_samples),len(search)),float)
        bins=[]
        for i,x in enumerate(x_samples):
            xi=int(round(x));xa=max(0,xi-6);xb=min(W,xi+7)
            bins.append((xa,xb))
            for k,off in enumerate(search):
                y=int(baseline[i]+off)
                if 0<=y<H:
                    strengths[i,k]=np.count_nonzero(strokes[y,xa:xb])
        # Dynamic search maximizes independent ink evidence, with limited
        # local bending. It does not derive a curve from endpoint anchors.
        local=.5*strengths-.08*np.abs(search)[None,:]
        n,m=local.shape
        dp=np.full((n,m),-1e10,float);back=np.zeros((n,m),np.int16)
        dp[0]=local[0]
        for i in range(1,n):
            earlier_y=baseline[i-1]+search
            for k,off in enumerate(search):
                current_y=baseline[i]+off
                delta=np.abs(earlier_y-current_y)
                costs=dp[i-1]-.35*delta-np.where(delta>4,10+4*delta,0)
                p=int(np.argmax(costs))
                dp[i,k]=costs[p]+local[i,k]
                back[i,k]=p
        indices=np.zeros(n,int);indices[-1]=int(np.argmax(dp[-1]))
        for i in range(n-2,-1,-1):indices[i]=back[i+1,indices[i+1]]
        raw_y=(baseline+search[indices]).astype(float)
        raw_ink=np.array([strengths[i,indices[i]]>=5 for i in range(n)])
        # V16: isolate short glyph-driven excursions. A real printed border
        # has a locally coherent tangent; a one/two-bin bulge does not.
        # Robust 5-bin median removes impulses; local polynomial smoothing
        # restores a continuous slope without turning the line into a chord.
        candidate=savgol_filter(
            median_filter(raw_y,size=5,mode='nearest'),
            window_length=7,polyorder=2,mode='interp')
        # Keep the proposal inside observed ink's original narrow corridor.
        # If smoothing would need to leave that corridor by >3.5 px,
        # refuse rather than pretending that a line runs across the text.
        change=np.abs(candidate-raw_y)
        if float(change.max())>3.5:
            return None,{'reason':'unverified_local_bulge',
                'failed_index':int(idx),'max_bulge_px':round(float(change.max()),3),
                'raw_points':raw_y.round(2).tolist(),
                'smoothed_points':candidate.round(2).tolist()}
        # Reject extreme, short-span bends independently of image coverage.
        # x points are regularly spaced; 1.25px second difference per bin
        # corresponds to a visibly sharp turn on this receipt resolution.
        curvature=np.abs(np.diff(candidate,n=2))
        if len(curvature) and float(curvature.max())>1.25:
            return None,{'reason':'excessive_local_curvature',
                'failed_index':int(idx),
                'max_second_difference_px':round(float(curvature.max()),3)}
        y_path=candidate.astype(float)
        # Score the SMOOTHED curve, not the discarded raw tracker.
        # Within 2px is allowed for anti-aliased / broken printed strokes;
        # a line that follows letters instead of a separator will fail the
        # independent row audit and/or continuous ink constraints below.
        on_ink=np.zeros(n,bool)
        for i,(x,y) in enumerate(zip(x_samples,y_path)):
            xa,xb=bins[i]
            low=max(0,int(np.floor(y))-2)
            high=min(H,int(np.ceil(y))+3)
            if low<high:
                on_ink[i]=max((np.count_nonzero(strokes[yy,xa:xb])
                                for yy in range(low,high)),default=0)>=5
        coverage=float(on_ink.mean())
        thirds=[float(z.mean()) for z in np.array_split(on_ink,3)]
        gaps=[];run=0
        for supported in on_ink:
            run=0 if supported else run+1
            gaps.append(run)
        longest=max(gaps)
        raw_run=0;raw_longest=0
        for supported in raw_ink:
            raw_run=0 if supported else raw_run+1
            raw_longest=max(raw_longest,raw_run)
        record={'index':int(idx),'ink_coverage':round(coverage,4),
                'thirds':[round(z,4) for z in thirds],
                'max_consecutive_unsupported':int(longest),
                'raw_max_consecutive_unsupported':int(raw_longest),
                'measured_points':int(on_ink.sum()),
                'raw_max_slope_px':round(float(np.abs(np.diff(raw_y)).max()),3),
                'smoothed_max_slope_px':round(float(np.abs(np.diff(y_path)).max()),3),
                'raw_max_curvature_px':round(float(np.abs(np.diff(raw_y,n=2)).max()),3),
                'smoothed_max_curvature_px':round(float(np.abs(np.diff(y_path,n=2)).max()),3),
                'max_displacement_px':round(float(change.max()),3),
                'total_points':int(n),
                'points':[[round(float(x),2),round(float(y),2)]
                          for x,y in zip(x_samples,y_path)]}
        if (coverage<min_coverage or min(thirds)<min_third_coverage or
                longest>max_gap or raw_longest>3 or
                not np.any(on_ink[:2]) or not np.any(on_ink[-2:])):
            return None,{'reason':'insufficient_continuous_printed_ink',
                         'failed_index':int(idx),'row':record,
                         'accepted_rows':records}
        # A short <4px endpoint extension is supported by the measured rail
        # crossing; everything else follows measured ink and short gaps only.
        points=np.column_stack((x_samples,y_path))
        try:
            traces.append(_polyline(points,(H,W)))
        except (ValueError,TypeError) as exc:
            return None,{'reason':'invalid_trace','index':int(idx),
                         'error':str(exc),'row':record}
        records.append(record)
    # Ensure local curves never cross between physical rails or inside columns.
    for x in np.linspace(x0,x1,150):
        levels=np.array([rule.eval((x-W/2)/W) for rule in traces],float)
        if not np.all(np.isfinite(levels)) or np.min(np.diff(levels))<5:
            return None,{'reason':'row_curves_cross','x':float(x)}
    return {**graph,'horizontal':traces}, {
        'status':'all_rules_traced',
        'method':'bounded_smooth_measured_polyline_on_printed_ink',
        'x_range':[round(x0,2),round(x1,2)],
        'rows':records}
