"""Grid anomaly gate. No generated row ordinal is used as OCR evidence."""
from __future__ import annotations
# GRID_ABORT_DIAGNOSTICS_V1
from App import grid_abort_diagnostics_v1 as _grid_diag
import re
from time import perf_counter
import cv2
import numpy as np


def _ordinal(result):
    result = result or {}
    raw = str(result.get('text', '')).strip()
    match = re.fullmatch(r'\d{1,4}[.)]?', raw)
    confidence = float(result.get('confidence') or 0)
    return {'raw': raw, 'value': int(raw.rstrip('.)')) if match else None,
            'confidence': confidence}


def classify_ordinals(observations, threshold=.85):
    """Soft independent evidence, never a standalone rejection condition."""
    first = _ordinal(observations.get(1))
    value, confidence = first['value'], first['confidence']
    reliable = value is not None and confidence >= threshold
    if value == 1:
        effect = 'positive_strong' if reliable else 'positive_weak'
    elif value is not None and value > 1:
        effect = 'negative_strong' if reliable else 'negative_weak'
    else:
        effect = 'neutral'
    return {'observations': {1:first} if 1 in observations else {},
            'effect':effect, 'reasons':[], 'first_verified':value==1,
            'low_confidence_one':bool(value==1 and not reliable),
            'read_performed':1 in observations}


def _merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]+3: out[-1]=(out[-1][0],max(out[-1][1],b))
        else: out.append((a,b))
    return sum(b-a for a,b in out)


@_grid_diag.observe('App/geometry/grid_anomaly_v3.py:scan_above')
def scan_above(image, map_x, map_y, xs, ys, *, detailed=False):
    """Search source ink ABOVE candidate grid, never the rendered overlay.
    Rectification coordinates map into the original oriented image.
    Conservative detector: two width-spanning lines and two extending rails.
    """
    t=perf_counter(); info={'status':'unknown','reason':'insufficient_geometry'}
    try:
        xs=np.rint(xs).astype(int); ys=np.rint(ys).astype(int)
        if len(xs)<4 or len(ys)<3 or np.any(np.diff(xs)<=0) or np.any(np.diff(ys)<=0): return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L51')
        if xs[0]<0 or ys[0]<0 or xs[-1]>=map_x.shape[1] or ys[1]>=map_x.shape[0]: return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L52')
        x0,x1=int(xs[0]),int(xs[-1]); y0,y1=int(ys[0]),int(ys[1]); w=x1-x0
        if w<60 or y1-y0<5: return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L54')
        src=np.float32([[map_x[y,x],map_y[y,x]] for y,x in [(y0,x0),(y0,x1),(y1,x0),(y1,x1)]])
        if not np.isfinite(src).all(): return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L56')
        # Bound scan to 6 ordinary row heights and at most 75% of table width.
        pitch=float(np.median(np.diff(ys)))
        depth=int(min(6*pitch,.75*w)); band=y1-y0
        if depth<20:return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L60')
        dst=np.float32([[0,depth],[w,depth],[0,depth+band],[w,depth+band]])
        matrix=cv2.getPerspectiveTransform(src,dst)
        # Crop source before warp and conversion (no full-photo grayscale allocation).
        inv=np.linalg.inv(matrix)
        ph=depth+band+max(6,round(.18*pitch))+1
        corners=cv2.perspectiveTransform(np.float32([[[0,0],[w,0],[0,ph],[w,ph]]]),inv)[0]
        ih,iw=image.shape[:2]
        l,t0=np.maximum(np.floor(corners.min(axis=0)).astype(int)-3,0)
        r,b0=np.minimum(np.ceil(corners.max(axis=0)).astype(int)+4,[iw,ih])
        if r<=l or b0<=t0: return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L70')
        roi=image[t0:b0,l:r]
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi
        local=matrix@np.array([[1,0,l],[0,1,t0],[0,0,1]],float)
        patch=cv2.warpPerspective(gray,local,(w+1,ph),borderValue=255)
        # Cheap independent optical screen. Only detailed candidates need LSD.
        bh=cv2.morphologyEx(patch,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(1,9)))
        ridge=cv2.morphologyEx(bh,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(15,round(.05*w)),1)))
        cov=(cv2.dilate(ridge,np.ones((9,1),np.uint8))>3).mean(axis=1)
        radius=max(6,round(.18*pitch))
        above=cov[5:max(5,depth-radius)]
        # At least one wide ridge is needed before costly segment extraction.
        # A non-one OCR observation forces detailed analysis independently.
        if not detailed and (not len(above) or float(above.max())<.38):
            top_support=float(cov[max(0,depth-radius):depth+radius+1].max())
            bottom_support=float(cov[max(0,depth+band-radius):depth+band+radius+1].max())
            info.update(status='no_signal',reason='fast_screen_no_wide_ridge',fast_screen=True,
                        top_support=round(top_support,3), bottom_support=round(bottom_support,3),
                        max_coverage=round(float(above.max()),3) if len(above) else 0.,
                        anchor={'status':'unconfirmed'})
            return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L90')
        lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(patch)[0]
        horizontal=[]; vertical=[]
        for x,y,X,Y in ([] if lsd is None else lsd.reshape(-1,4)):
            dx=abs(X-x);dy=abs(Y-y);ym=(y+Y)/2
            if dx>=max(15,.04*w) and dy<=.12*dx and 4<ym<depth-5:
                horizontal.append((float(ym),float(min(x,X)),float(max(x,X))))
            if dy>=max(14,.30*pitch) and dx<=.16*dy:
                vertical.append((float((x+X)/2),float(min(y,Y)),float(max(y,Y))))
        groups=[]
        for y,a,b in sorted(horizontal):
            if not groups or y-groups[-1][-1][0]>5:groups.append([])
            groups[-1].append((y,a,b))
        edges=[]
        for group in groups:
            coverage=_merge([(a,b) for _,a,b in group])/w
            if coverage>=.60:edges.append({'y':round(float(np.median([y for y,_,_ in group])),1),'coverage':round(coverage,3)})
        # Weak grey printed rules can be missed by LSD. Extract dark, locally
        # horizontal ridges independently, then require the same rail evidence.
        peaks=[]
        for y in range(5,depth-5):
            if cov[y]>=.60 and cov[y]>=max(cov[max(0,y-5):y+6]):
                if not peaks or y-peaks[-1]>max(9,.25*pitch): peaks.append(y)
                elif cov[y]>cov[peaks[-1]]:peaks[-1]=y
        for y in peaks:
            if not any(abs(y-e['y'])<=max(8,.20*pitch) for e in edges):
                edges.append({'y':int(y),'coverage':round(float(cov[y]),3),'method':'weak_ridge'})
        edges.sort(key=lambda e:e['y'])
        # A table rail must extend near the upper candidate boundary. Use
        # observed rail x positions, allowing small drift from extrapolation.
        rails=[]; rail_starts=[]; all_rail_starts=[]
        for rail in xs-x0:
            near=[(a,b) for x,a,b in vertical if abs(x-rail)<=max(5,.025*w)]
            if near and any(b>=depth+band*.45 and a<=depth+band*.45 for a,b in near):
                all_rail_starts.append(min(a for a,b in near))
            intervals=[(max(0,a),min(depth,b)) for x,a,b in vertical if abs(x-rail)<=max(5,.025*w) and a<depth and b>0]
            intervals=[(a,b) for a,b in intervals if b>a]
            support=_merge(intervals)
            if support>=max(20,.65*pitch) and any(b>=depth-.65*pitch for a,b in intervals):
                rails.append(int(rail))
                rail_starts.append(min(a for a,b in intervals))
        # A structural anchor is a candidate upper printed header edge near
        # the shared start of the table rails, followed by its lower edge.
        # Coordinates use the source image, not relative photo height.
        anchor={'status':'unconfirmed','candidate_top_y':depth,
                'candidate_first_product_y':depth+band,'rail_start_y':rail_starts}
        tol=max(12.,.45*pitch)
        # V4: use the edge nearest the shared rail starts, not the first
        # horizontal response above the table (e.g. seller text underline).
        aligned_edges=[]
        if len(rails)>=3:
            for edge in edges:
                y=float(edge['y'])
                count=sum(abs(start-y)<=tol for start in rail_starts)
                if count>=3:
                    error=float(np.median([abs(start-y) for start in rail_starts]))
                    aligned_edges.append((count,-error,edge['coverage'],y))
        # Preserve the previous multi-edge recovery on weak two-rail photos;
        # a SINGLE new upper edge requires the stricter three-rail proof.
        if not aligned_edges and len(edges)>=2 and len(rails)>=2:
            top=float(edges[0]['y'])
            aligned=sum(abs(y-top)<=tol for y in rail_starts)
            if aligned>=2:
                aligned_edges.append((aligned,0.,edges[0]['coverage'],top))
        if aligned_edges:
            aligned,_,_,top=max(aligned_edges)
            kept=[e for e in edges if e['y']>=top-2]
            # Existing top is a valid header-bottom hypothesis when only
            # one physical boundary above it remains.
            nxt=float(kept[1]['y']) if len(kept)>=2 else float(depth)
            if .35*pitch<=nxt-top<=3*pitch:
                coords=np.float32([[[w/2,top],[w/2,nxt],[w/2,depth+band]]])
                source=cv2.perspectiveTransform(coords,inv)[0]
                anchor.update(status='candidate',rail_start_matches=aligned,
                              header_top_patch_y=top,first_product_patch_y=nxt,
                              displacement_in_row_heights=round((depth+band-nxt)/pitch,3),
                              header_top_source_xy=source[0].round(1).tolist(),
                              first_product_top_source_xy=source[1].round(1).tolist(),
                              current_first_product_top_source_xy=source[2].round(1).tolist())
                info['discarded_above_anchor']=[e for e in edges if e['y']<top-2]
                edges=kept
        # V4: inspect a wide first existing band only on the repair route.
        # An extra separator must be optically present across all columns,
        # including both outer rails. Never insert by predicted pitch alone.
        internal=[]
        if anchor['status']=='candidate' and band>=1.5*pitch:
            lo=depth+max(10,round(.30*pitch)); hi=depth+band-max(10,round(.30*pitch))
            wide=cv2.dilate(ridge,np.ones((7,1),np.uint8))>3
            peaks=[]
            for y in range(lo,hi):
                if cov[y]<.88 or cov[y]<max(cov[max(0,y-4):y+5]):continue
                evidence=[float(wide[y,int(a):int(b)+1].mean()) for a,b in zip(xs[:-1]-x0,xs[1:]-x0)]
                if min(evidence)<.75:continue
                if not peaks or y-peaks[-1][0]>max(8,.22*pitch):peaks.append((y,evidence))
                elif cov[y]>cov[peaks[-1][0]]:peaks[-1]=(y,evidence)
            internal=[{'y':int(y),'coverage':round(float(cov[y]),3),'method':'internal_distributed_ridge','column_coverage':c} for y,c in peaks]
        info['internal_edges']=internal
        if anchor['status']=='unconfirmed':
            radius=max(3,round(.12*pitch))
            top_support=float(max(cov[max(0,depth-radius):depth+radius+1],default=0))
            bottom_support=float(max(cov[max(0,depth+band-radius):depth+band+radius+1],default=0))
            starts=sum(abs(y-depth)<=tol for y in all_rail_starts)
            anchor.update(current_top_support=round(top_support,3),
                          current_header_bottom_support=round(bottom_support,3),
                          current_rail_starts=all_rail_starts,current_rail_start_matches=starts)
            if starts>=2 and top_support>=.60 and bottom_support>=.60:
                anchor.update(status='aligned',displacement_in_row_heights=0.,
                              header_top_source_xy=src[:2].mean(axis=0).round(1).tolist(),
                              first_product_top_source_xy=src[2:].mean(axis=0).round(1).tolist())
        anomaly=len(edges)>=2 and len(rails)>=2
        # Single omitted boundary is suspicious only with several rails that
        # start at that independently visible edge (no inference from pitch alone).
        if len(edges)==1 and len(rails)>=3:
            anomaly=sum(abs(y-edges[0]['y'])<=tol for y in rail_starts)>=3
        info.update(status='suspect' if anomaly else 'no_signal',reason='printed_grid_above_candidate' if anomaly else 'no_confirmed_extension',edges=edges,rails=rails,depth=depth,patch_shape=list(patch.shape),anchor=anchor, source_to_patch=matrix.tolist(), width=w, pitch=pitch,
                    rail_positions=(xs-x0).tolist())
        return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L206')
    except (ValueError,TypeError,IndexError,cv2.error,OverflowError) as exc:
        _grid_diag.caught(locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L207')
        info['reason']=type(exc).__name__;return _grid_diag.ret((info), locals(), 'App/geometry/grid_anomaly_v3.py:scan_above:L208')
    finally: info['elapsed_ms']=round((perf_counter()-t)*1000,3)


@_grid_diag.observe('App/geometry/grid_anomaly_v3.py:audit_grid')
def audit_grid(image, rect, geometry, *, crop, read, number_col=0, force_number=False):
    """Geometry first; at most one optional ordinal read per candidate.

    Missing/ambiguous ordinal is neutral. Strong contrary OCR requests a
    source inspection, not automatic repair. Only confirmed geometry is hard.
    """
    t=perf_counter(); observations={}; errors=[]; reasons=[]
    mapping=rect.info.get('cell_mesh') if rect.method=='original_cells' else rect
    def scan_source(detailed=False):
        if mapping is None: return {'status':'unknown','reason':'missing_mapping'}
        return scan_above(image,mapping.map_x,mapping.map_y,mapping.xs,mapping.ys,detailed=detailed)
    scan=scan_source()
    safety=geometry.get('safety') or {}
    if safety.get('folded_fraction',0)>0 or safety.get('source_valid_fraction',1)<.995 or geometry.get('unresolved_extreme_horizontal',0)>0:
        reasons.append('invalid_mesh_geometry')
    if scan['status']=='suspect': reasons.append(scan['reason'])
    strong_geometry=(scan['status']=='no_signal' and
        (scan.get('anchor',{}).get('status')=='aligned' or
         (scan.get('fast_screen') and scan.get('top_support',0)>=.60 and
          scan.get('bottom_support',0)>=.60 and scan.get('max_coverage',1)<.25)) and
        safety.get('status')=='MEASURED' and not geometry.get('warnings') and not reasons)
    read_reason='geometry_sufficient' if strong_geometry else 'geometry_inconclusive'
    if reasons:
        read_reason='confirmed_geometric_defect'
    elif number_col is None:
        read_reason='number_column_unavailable'
    elif force_number or not strong_geometry:
        read_reason='post_recovery' if force_number else 'geometry_inconclusive'
        try:
            cell,_,_=crop(rect,1,number_col)
            observations[1]=None
            observations[1]=read(cell)
        except (ValueError,cv2.error,RuntimeError) as exc:
            _grid_diag.caught(locals(), 'App/geometry/grid_anomaly_v3.py:audit_grid:L244')
            errors.append({'row':1,'error':str(exc)})
    numbers=classify_ordinals(observations)
    numbers['read_reason']=read_reason
    if numbers['effect']=='negative_strong' and not reasons:
        if scan.get('fast_screen') or scan['status']=='unknown':
            scan=scan_source(detailed=True)
        if scan['status']=='suspect': reasons.append(scan['reason'])
    # This is a categorical evidence state, not a calibrated probability.
    # No OCR result cannot reduce the baseline geometric assessment.
    confidence='defect' if reasons else 'high' if strong_geometry or scan.get('anchor',{}).get('status')=='aligned' else 'limited'
    if not reasons and numbers['effect'].startswith('positive'):
        confidence='supported' if confidence=='limited' else 'high_supported'
    return _grid_diag.ret(({'status':'suspect' if reasons else 'pass',
            'reasons':reasons,'number_check':numbers,'source_extension':scan,'errors':errors,
            'geometry_confidence':confidence,'strong_geometry':bool(strong_geometry),
            'ordinal_evidence_delta':{'positive_strong':2,'positive_weak':1,'neutral':0,'negative_weak':-1,'negative_strong':-2}[numbers['effect']],
            'elapsed_ms':round((perf_counter()-t)*1000,3)}), locals(), 'App/geometry/grid_anomaly_v3.py:audit_grid:L257')


@_grid_diag.observe('App/geometry/grid_anomaly_v3.py:recover_graph')
def recover_graph(image, data, scan, *, rule_factory):
    """Prepend observed source rules, retaining all old rule objects.

    No row-pitch interpolation. Reject if existing rails do not physically
    continue through the new band. All coordinates refer to original image.
    """
    if scan.get('status')!='suspect' or scan.get('anchor',{}).get('status')!='candidate':
        raise ValueError('no_confirmed_top_anchor')
    edges=scan['edges']; anchor=scan['anchor']; depth=scan['depth']; w=scan['width']; pitch=scan['pitch']
    if not edges or anchor.get('rail_start_matches',0)<(3 if len(edges)==1 else 2):
        raise ValueError('insufficient_anchor_evidence')
    top=anchor['header_top_patch_y']
    edges=[e for e in edges if top-2<=e['y']<depth-max(5,.15*pitch)]
    if not edges:raise ValueError('insufficient_observed_boundaries')
    upper_edges=list(edges)
    internal_edges=list(scan.get('internal_edges',[]))
    edges=upper_edges+internal_edges
    M=np.asarray(scan['source_to_patch'],float); inv=np.linalg.inv(M)
    H,W=image.shape[:2]
    # Recovery path only: recompute source evidence in candidate coordinates.
    pad=max(8,round(.04*w))
    shifted=np.array([[1,0,pad],[0,1,0],[0,0,1]],float)@M
    patch=cv2.warpPerspective(image,shifted,(w+1+2*pad,int(max(depth+2*pitch,max([e['y'] for e in edges])+pitch))),borderValue=(255,255,255))
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY) if patch.ndim==3 else patch
    hb=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((9,1),np.uint8))
    hr=cv2.morphologyEx(hb,cv2.MORPH_OPEN,np.ones((1,max(15,round(.05*w))),np.uint8))[:,pad:pad+w+1]
    vb=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((1,11),np.uint8))
    vr=cv2.morphologyEx(vb,cv2.MORPH_OPEN,np.ones((max(13,round(.35*pitch)),1),np.uint8))
    def project(pts,m):return cv2.perspectiveTransform(np.asarray(pts,np.float32).reshape(1,-1,2),m)[0]
    # Existing vertical curves must remain close to optical rails through the
    # extension. Do not synthesize missing verticals based on regular spacing.
    vertical=list(data['vertical']); old=list(data['horizontal']); rail_support=[]
    for v in vertical:
        # Sample the existing curve in source coordinates, then project.
        start,end=project([[w/2,top+4],[w/2,depth-4]],inv)[:,1]
        yy=np.linspace(start,end,48); xx=v.eval((yy-H/2)/H)
        pp=project(np.column_stack((xx,yy)),M)
        good=0; valid=0; residuals=[]
        for x,y in pp:
            x+=pad
            iy=int(round(y)); ix=int(round(x)); rad=max(4,round(.02*w))
            if not 0<=iy<vr.shape[0] or not -rad<=ix<vr.shape[1]+rad:continue
            valid+=1; lo=max(0,ix-rad);hi=min(vr.shape[1],ix+rad+1)
            if hi<=lo:continue
            line=vr[iy,lo:hi]
            if int(line.max())>=3:good+=1;residuals.append(abs(lo+int(line.argmax())-x))
        rail_support.append({'fraction':good/max(1,valid),'median_offset':float(np.median(residuals)) if residuals else None})
    # Weak border ink is possible, but require independent majority rail evidence.
    if sum(r['fraction']>=.40 for r in rail_support)<max(3,len(vertical)-1):
        raise ValueError('existing_rails_not_supported_in_extension:'+str(rail_support))
    new=[]; fit_details=[]
    for edge in edges:
        y=float(edge['y']);rad=max(5,min(10,round(.19*pitch)))
        pts=[]
        for x in np.linspace(4,w-4,41):
            ix=int(round(x));lo=max(0,round(y)-rad);hi=min(hr.shape[0],round(y)+rad+1)
            response=hr[lo:hi,max(0,ix-3):min(w+1,ix+4)].mean(axis=1)
            if len(response) and response.max()>=3:
                k=int(response.argmax());pts.append((x,lo+k))
        if len(pts)<12 or np.ptp(np.asarray(pts)[:,0])<.65*w:
            raise ValueError('new_rule_not_supported')
        q=project(pts,inv); xx=(q[:,0]-W/2)/W; yy=q[:,1]
        coef=np.polyfit(xx,yy,1)
        err=np.abs(np.polyval(coef,xx)-yy); keep=err<=max(3.,float(np.median(err))*2.8)
        if keep.sum()<10:raise ValueError('unstable_new_rule')
        coef=np.polyfit(xx[keep],yy[keep],1)
        if np.max(np.abs(np.polyval(coef,xx[keep])-yy[keep]))>8:raise ValueError('unstable_new_rule')
        new.append(rule_factory('h',coef,float(np.polyval(coef,0)),float(edge['coverage']),(float(q[:,0].min()),float(q[:,0].max()))))
        fit_details.append({'patch_y':y,'samples':int(keep.sum()),'source_position':float(np.polyval(coef,0))})
    hs=sorted(new+old,key=lambda rule:rule.position)
    # Check all source nodes, including left/right edges where extrapolation
    # is most dangerous; require monotonic vertical and horizontal order.
    from itertools import product
    nodes=np.empty((len(hs),len(vertical),2),float)
    for i,j in product(range(len(hs)),range(len(vertical))):
        h,v=hs[i],vertical[j]; x=float(v.position); y=float(h.position)
        for _ in range(10):x=float(v.eval((y-H/2)/H));y=float(h.eval((x-W/2)/W))
        nodes[i,j]=(x,y)
    if not np.isfinite(nodes).all() or np.any(np.diff(nodes[:,:,0],axis=1)<5) or np.any(np.diff(nodes[:,:,1],axis=0)<5):
        raise ValueError('recovered_grid_crosses_or_overlaps')
    if np.any(nodes[:,:,0]<0) or np.any(nodes[:,:,0]>=W) or np.any(nodes[:,:,1]<0) or np.any(nodes[:,:,1]>=H):
        raise ValueError('recovered_grid_outside_photo')
    return _grid_diag.ret(({**data,'horizontal':hs,'vertical':vertical}, {'added_rows':len(new),'added_top':len(upper_edges),'added_internal':len(internal_edges),'preserved_horizontal':len(old),'rail_support':rail_support,'fits':fit_details,'nodes':nodes.tolist()}), locals(), 'App/geometry/grid_anomaly_v3.py:recover_graph:L346')


@_grid_diag.observe('App/geometry/grid_anomaly_v3.py:build_direct')
def build_direct(image, rect, geometry, scan, *, rectify=False, prepared=None):
    from .geometry_lab.detection import Rule
    from .geometry_lab.algorithms import mesh, Result, intersection
    from .geometry_lab.metrics import safety
    from .service import cell_polygon
    data=rect.info.get('source_grid')
    if data is None:raise ValueError('missing_source_graph')
    data,diag=recover_graph(image,data,scan,rule_factory=Rule) if prepared is None else prepared
    H,W=image.shape[:2]; vs,hs=data['vertical'],data['horizontal']
    # Same rendering contract as service.py; only the geometry input changes.
    if not rectify:
        nodes=np.array([[intersection(v,h,image.shape) for v in vs] for h in hs],float)
        left=max(0,int(np.floor(nodes[:,:,0].min()))-16);top=max(0,int(np.floor(nodes[:,:,1].min()))-16)
        right=min(W,int(np.ceil(nodes[:,:,0].max()))+17);bottom=min(H,int(np.ceil(nodes[:,:,1].max()))+17)
        vx=np.median(nodes[:,:,0],axis=0);hy=np.median(nodes[:,:,1],axis=1)
        gy,gx=np.indices((bottom-top,right-left),dtype=np.float32)
        result=Result('original_cells',image[top:bottom,left:right].copy(),gx+left,gy+top,
                      (vx-left).tolist(),(hy-top).tolist(),{'physical_grid':'observed_lines_source',
                      'source_grid':data,'source_shape':image.shape,'source_offset':(left,top),
                      'cell_mesh':mesh(image,data,sample_image=False),'source_image':image})
        check=safety(result.info['cell_mesh'],image.shape)
    else:
        result=mesh(image,data);result.info['source_grid']=data;check=safety(result,image.shape)
    if check.get('status')!='MEASURED' or check.get('folded_fraction',1)>0 or check.get('source_valid_fraction',0)<.995:
        raise ValueError('recovered_mesh_failed_safety')
    overlay=result.image.copy()
    if not rectify:
        from ..config import setting as _overlay_setting
        if _overlay_setting('OCR_GRID_OVERLAY_REUSE_V1','1')=='1':
            from .grid_overlay_reuse_v1 import polygon_reader
            _polygon=polygon_reader(result,nodes,cell_polygon)
        else:
            _polygon=lambda row,col:cell_polygon(result,row,col)
        for row in range(len(hs)):
            for col in range(len(vs)-1):
                poly=_polygon(max(0,min(row,len(hs)-2)),col)
                edge=poly[:17] if row<len(hs)-1 else poly[34:51]
                cv2.polylines(overlay,[np.rint(edge).astype(np.int32)],False,(255,100,0),2,cv2.LINE_AA)
        for col in range(len(vs)):
            for row in range(len(hs)-1):
                poly=_polygon(row,max(0,min(col,len(vs)-2)))
                edge=poly[17:34] if col==len(vs)-1 else poly[51:68]
                cv2.polylines(overlay,[np.rint(edge).astype(np.int32)],False,(255,100,0),2,cv2.LINE_AA)
    else:
        for x in result.xs:cv2.line(overlay,(round(x),round(result.ys[0])),(round(x),round(result.ys[-1])),(255,100,0),2)
        for y in result.ys:cv2.line(overlay,(round(result.xs[0]),round(y)),(round(result.xs[-1]),round(y)),(255,100,0),2)
    gg={**geometry,'warnings':list(geometry.get('warnings',[])), 'method':geometry.get('method','')+' + Direct Anchor V3',
        'safety':check,'rectification':result.method,'direct_anchor_v3':diag}
    return _grid_diag.ret((result,overlay,gg), locals(), 'App/geometry/grid_anomaly_v3.py:build_direct:L397')

# GRID_ANOMALY_V5_BIDIRECTIONAL
from .grid_anomaly_v5 import install_hooks as _install_v5
audit_grid, recover_graph = _install_v5(audit_grid, recover_graph)

# GRID_ANOMALY_V52_UNIVERSAL
from .grid_anomaly_v52 import audit_hook as _v52_audit_hook
audit_grid = _v52_audit_hook(audit_grid)
