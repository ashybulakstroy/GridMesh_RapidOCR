"""V3: multiscale physical segments and jointly scored five-column hypotheses.

Image-only experimental detector. Frozen V2 remains a conservative fallback.
No OCR, reference coordinates, receipt ID, or expected row count is accepted.
"""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import cv2,numpy as np
from itertools import combinations
from time import perf_counter
from ...ocr.debug_performance import get_debug
from .detection import Rule
from .physical_grid_graph import connected_fragments,graph_verticals
from .physical_grid_graph_local import (detect_physical_graph as v2,ridge_response,
                                       trace_seed,junction_outer_boundaries)
from .structure_diagnostics import PolylineRule
from .algorithms import intersection


@_deep_timed("grid", "App/geometry/geometry_lab/physical_grid_graph_v3.py:multiscale_segments")
def multiscale_segments(gray, *, scales=(1.,1.5)):
    H,W=gray.shape;parts={'h':[],'v':[]}; debug=get_debug()
    for scale in scales:
        started=perf_counter() if debug else None
        sized=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        if debug:
            debug.add_geometry('physical_v3_scale_resize',(perf_counter()-started)*1000,
                               {'scale':scale,'shape_hw':list(sized.shape)})
            started=perf_counter()
        sized=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(sized)
        if debug:
            debug.add_geometry('physical_v3_scale_clahe',(perf_counter()-started)*1000,{'scale':scale})
            started=perf_counter()
        found=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(sized)[0]
        if debug:
            debug.add_geometry('physical_v3_scale_line_detection',(perf_counter()-started)*1000,
                               {'scale':scale,'raw_segments':0 if found is None else len(found)})
            started=perf_counter()
            before_h,before_v=len(parts['h']),len(parts['v'])
        if found is None:
            if debug:debug.add_geometry('physical_v3_scale_classify',0,{'scale':scale,'horizontal':0,'vertical':0})
            continue
        for item in found.reshape(-1,4):
            x0,y0,x1,y1=(item+.5)/scale-.5;dx=x1-x0;dy=y1-y0
            if abs(dx)>=max(18,W*.025) and abs(dy)<=abs(dx)*.35:
                if x1<x0:x0,y0,x1,y1=x1,y1,x0,y0
                ref=y0+(W/2-x0)*(y1-y0)/max(x1-x0,1)
                parts['h'].append((float(ref),tuple(map(float,(x0,y0,x1,y1))),float(x1-x0)))
            elif abs(dy)>=max(30,H*.025) and abs(dx)<=abs(dy)*.35:
                # Transposed coordinates use the same fragment-connectivity test.
                if y1<y0:x0,y0,x1,y1=x1,y1,x0,y0
                ref=x0+(H/2-y0)*(x1-x0)/max(y1-y0,1)
                parts['v'].append((float(ref),tuple(map(float,(y0,x0,y1,x1))),float(y1-y0)))
        if debug:
            debug.add_geometry('physical_v3_scale_classify',(perf_counter()-started)*1000,
                               {'scale':scale,'horizontal':len(parts['h'])-before_h,
                                'vertical':len(parts['v'])-before_v})
    return parts


@_deep_timed("grid", "App/geometry/geometry_lab/physical_grid_graph_v3.py:fit_segments")
def fit_segments(parts,axis,shape):
    H,W=shape;extent=W if axis=='h' else H
    rules=[]
    for group in connected_fragments(parts,extent,tolerance=3.):
        p=[]
        for _,(a,b,c,d),length in group:
            t=np.linspace(0,1,max(3,int(length/8)));p.extend(zip(a+(c-a)*t,b+(d-b)*t))
        p=np.array(p);span=np.ptp(p[:,0])
        if span<max(35,extent*(.12 if axis=='h' else .035)):continue
        norm=(p[:,0]-extent/2)/extent;A=np.vander(norm,3);weight=np.ones(len(p))
        for _ in range(4):
            coeff=np.linalg.lstsq(A*weight[:,None],p[:,1]*weight,rcond=None)[0]
            weight=np.minimum(1.,2./np.maximum(abs(p[:,1]-A@coeff),.001))
        rules.append(Rule(axis,coeff,float(np.polyval(coeff,0)),float(span/extent),
                          (float(p[:,0].min()),float(p[:,0].max()))))
    return sorted(rules,key=lambda r:r.position)


def evidence(rule,vv,shape,response,dark):
    H,W=shape;nodes=[intersection(v,rule,shape) for v in vv]
    left,right=nodes[0][0],nodes[-1][0]
    if right-left<50:return None
    xs=np.linspace(left,right,max(70,int(right-left)));ys=rule.eval((xs-W/2)/W)
    valid=(xs>=0)&(xs<W)&(ys>=1)&(ys<H-1)
    ix=np.rint(xs).astype(int).clip(0,W-1);iy=np.rint(ys).astype(int).clip(0,H-1)
    pixel=np.maximum.reduce([dark[(iy+d).clip(0,H-1),ix] for d in [-1,0,1]])
    observed=valid&(response[iy,ix]>=7)&(pixel>=9)
    columns=[];junctions=[]
    for i in range(4):
        mask=(xs>=nodes[i][0]+3)&(xs<=nodes[i+1][0]-3)
        columns.append(float(observed[mask].mean()) if mask.any() else 0.)
    for i,(x,y) in enumerate(nodes):
        offsets=np.arange(3,11) if i==0 else -np.arange(3,11) if i==4 else np.r_[-np.arange(3,11),np.arange(3,11)]
        xx=x+offsets;yy=rule.eval((xx-W/2)/W);okay=(xx>=0)&(xx<W)&(yy>=1)&(yy<H-1)
        xx=np.rint(xx[okay]).astype(int).clip(0,W-1);yy=np.rint(yy[okay]).astype(int).clip(0,H-1)
        arms=np.maximum.reduce([dark[(yy+d).clip(0,H-1),xx] for d in [-1,0,1]]) if len(xx) else np.array([])
        junctions.append(bool(len(arms)>=5 and (arms>=9).mean()>=.65))
    cuts=np.r_[0,np.where(observed[1:]!=observed[:-1])[0]+1,len(xs)]
    intervals=[dict(x0=float(xs[a]),x1=float(xs[b-1]),observed=bool(observed[a])) for a,b in zip(cuts,cuts[1:])]
    gap=max([r['x1']-r['x0'] for r in intervals if not r['observed']]+[0])/(right-left)
    accepted=(min(columns)>=.55 and observed.mean()>=.64 and sum(junctions)>=3 and gap<=.18)
    return dict(position=rule.position,domain=list(rule.domain),column_coverage=columns,
        ink_coverage=float(observed.mean()),intersection_evidence=junctions,intervals=intervals,
        longest_unsupported_fraction=float(gap),accepted=bool(accepted),
        reason='physical_source_arms_and_four_columns' if accepted else 'insufficient_source_arms_or_column_ink')


@_deep_timed("grid", "App/geometry/geometry_lab/physical_grid_graph_v3.py:refine_nearby_ridge")
def refine_nearby_ridge(rule, vv, shape, response, dark, observed):
    """Refit a near-complete observed rule within 3 px; retain evidence thresholds."""
    if (observed['ink_coverage'] < .80 or min(observed['column_coverage']) < .35
            or sum(observed['intersection_evidence']) < 4):
        return rule, observed
    H, W = shape
    xs = np.arange(max(0,round(vv[0].position)+3),
                   min(W,round(vv[-1].position)-2), dtype=float)
    if len(xs) < 40:
        return rule, observed
    norm = (xs-W/2)/W
    base = rule.eval(norm)
    offsets = np.arange(-3,4)
    yy = np.rint(base[:,None]+offsets).astype(int).clip(0,H-1)
    values = response[yy,xs.astype(int)[:,None]] - .7*abs(offsets)[None,:]
    ys = yy[np.arange(len(xs)),values.argmax(axis=1)]
    for degree in (2,3):
        coeff = np.polyfit(norm,ys,degree)
        for _ in range(3):
            keep = abs(ys-np.polyval(coeff,norm)) < 1.6
            if keep.sum() < len(xs)*.75:
                break
            coeff = np.polyfit(norm[keep],ys[keep],degree)
        if np.max(abs(np.polyval(coeff,norm)-base)) > 3:
            continue
        refined = Rule('h',coeff,float(np.polyval(coeff,0)),rule.support,rule.domain)
        checked = evidence(refined,vv,shape,response,dark)
        if checked and checked['accepted']:
            checked['refinement'] = 'nearby_source_ridge_3px'
            return refined, checked
    return rule, observed


@_deep_timed("grid", "App/geometry/geometry_lab/physical_grid_graph_v3.py:_complete_short_grid")
def _complete_short_grid(data, report):
    """Accept an independently observed short grid only with strong ink evidence."""
    if data is None or report.get('vertical_fallback') or not report.get('vertical_table_local_refit'):
        from App.ocr.performance_deep.live_capture import count as _count_deep
        _count_deep('early_exit_rejected', reason='baseline_missing_or_vertical_fallback_or_no_local_refit', baseline_missing=data is None, vertical_fallback=bool(report.get('vertical_fallback')), local_refit=bool(report.get('vertical_table_local_refit')))
        return False
    horizontal = data['horizontal']
    edges = report.get('edges', [])
    if len(data['vertical']) != 5 or len(horizontal) != 4 or len(edges) != 4:
        from App.ocr.performance_deep.live_capture import count as _count_deep
        _count_deep('early_exit_rejected', reason='unexpected_grid_dimensions', verticals=len(data['vertical']), horizontals=len(horizontal), edges=len(edges))
        return False
    for edge in edges:
        if (edge['ink_coverage'] < .90 or min(edge['column_coverage']) < .85 or
                sum(edge['intersection_evidence']) < 4 or edge['longest_unsupported_fraction'] > .06):
            from App.ocr.performance_deep.live_capture import count as _count_deep
            _count_deep('early_exit_rejected', reason='insufficient_edge_evidence', edge=edge)
            return False
    first, last = horizontal[0].position, horizontal[-1].position
    for rejected in report.get('rejected', []):
        position = rejected['position']
        if (first + 8 < position < last - 8 and
                all(abs(position - rule.position) > 7 for rule in horizontal) and
                rejected['ink_coverage'] >= .55 and sum(rejected['intersection_evidence']) >= 3):
            from App.ocr.performance_deep.live_capture import count as _count_deep
            _count_deep('early_exit_rejected', reason='additional_supported_rejected_horizontal', position=float(position))
            return False
    return True


@_deep_timed("grid", "App/geometry/geometry_lab/physical_grid_graph_v3.py:detect_physical_graph")
def detect_physical_graph(image, *, prefer_multiscale=False, min_rows=0):
    debug=get_debug(); started=perf_counter() if debug else None
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);H,W=gray.shape
    if debug:
        debug.add_geometry('physical_v3_grayscale',(perf_counter()-started)*1000,{'shape_hw':[H,W]})
        started=perf_counter()
    response,dark=ridge_response(gray)
    if debug:
        debug.add_geometry('physical_v3_ridge_response',(perf_counter()-started)*1000)
        started=perf_counter()
    # ADAPTIVE_DETECTOR_ROUTING_V1
    from .. import detector_routing_v1 as _route
    _trial = None
    if prefer_multiscale:
        _route.attempt()
        _trial_t=perf_counter()
        _candidate,_trial_report=_multiscale_core(gray,response,dark,None,{'status':'NOT_RUN'})
        _trial=(_candidate,_trial_report)
        _accepted=_route.accept_candidate(_candidate,min_rows)
        _route.record('multiscale_trial',accepted=_accepted,wall_ms=round((perf_counter()-_trial_t)*1000,3),horizontal_count=len(_candidate['horizontal']) if _candidate is not None else 0)
        if debug:
            debug.add_geometry('grid_route_multiscale_trial',(perf_counter()-_trial_t)*1000,{'accepted':_accepted,'min_rows':min_rows})
        if _accepted:
            _route.mark_candidate()
            _trial_report['routing_v1']='multiscale_first'
            return _candidate,_trial_report
        # Preserve original selection and short-grid priority on failure.
        _route.record('fallback',reason='multiscale_candidate_rejected')
        started=perf_counter() if debug else None
    baseline,previous=v2(image, ridge=(response,dark))
    if debug:
        debug.add_geometry('physical_v3_baseline_v2',(perf_counter()-started)*1000,
                           {'found':baseline is not None,'status':previous.get('status'),
                            'reason':previous.get('reason'),
                            'horizontal_count':len(baseline['horizontal']) if baseline is not None else 0,
                            'vertical_count':len(baseline['vertical']) if baseline is not None else 0})
        started=perf_counter()
    if _complete_short_grid(baseline,previous):
        if debug:
            debug.add_geometry('physical_v3_short_grid_decision',(perf_counter()-started)*1000,
                               {'accepted':True,'selection':'strong_short_grid'})
        return baseline,dict(previous,version='physical_grid_graph_v3',
                             selection='strong_short_grid', v2_status=previous['status'])
    if debug:
        debug.add_geometry('physical_v3_short_grid_decision',(perf_counter()-started)*1000,
                           {'accepted':False,'selection':'continue_multiscale'})
    if _trial is not None:
        # The multiscale core does not depend on V2 except final selection.
        # Reuse the already computed graph instead of running LSD a second time.
        _candidate,_report=_trial
        _report=dict(_report,v2_status=previous['status'])
        if baseline is not None and (_candidate is None or len(_candidate['horizontal'])<=len(baseline['horizontal'])):
            _reason=_report.pop('reason',None) or 'baseline_has_at_least_as_many_boundaries'
            _report.update(status='OBSERVED_NOT_VERIFIED',selection='frozen_v2_fallback',edges=previous['edges'])
            # Do not retain nodes from a different candidate graph.
            _report.pop('nodes',None)
            _route.record('trial_reused',selection='frozen_v2_fallback',reason=_reason)
            if debug:debug.add_geometry('grid_route_trial_reused',0,{'selection':'frozen_v2_fallback','reason':_reason})
            return baseline,_report
        _route.record('trial_reused',selection=_report.get('selection','rejected'))
        return _candidate,_report
    return _multiscale_core(gray,response,dark,baseline,previous)


def _multiscale_core_impl(gray,response,dark,baseline,previous, *, supplied_parts=None):
    H,W=gray.shape; debug=get_debug()
    parts=multiscale_segments(gray) if supplied_parts is None else supplied_parts
    started=perf_counter() if debug else None
    vertical=fit_segments(parts['v'],'v',(H,W));horizontal=fit_segments(parts['h'],'h',(H,W))
    if debug:
        debug.add_geometry('physical_v3_fit_segments',(perf_counter()-started)*1000,
                           {'input_horizontal':len(parts['h']),'input_vertical':len(parts['v']),
                            'horizontal_candidates':len(horizontal),'vertical_candidates':len(vertical)})
        started=perf_counter()
    report=dict(version='physical_grid_graph_v3',status='REJECTED',vertical_candidates=len(vertical),
        horizontal_candidates=len(horizontal),edges=[],rejected=[],v2_status=previous['status'],
        evidence_policy='unchanged pixel thresholds; new multiscale observations, no row-count prior')
    # Local source windows prevent long poster/paper edges from consuming the
    # global top-20 candidate budget and hiding a short table's five borders.
    hypotheses=[]
    initial=graph_verticals(vertical,horizontal,W,H)
    if initial is not None:hypotheses.append(initial)
    seen=set()
    for h in horizontal:
        if h.domain[1]-h.domain[0]<W*.2:continue
        y=float(h.eval(((h.domain[0]+h.domain[1])/2-W/2)/W));near=[]
        for v in vertical:
            x=float(v.eval((y-H/2)/H))
            if v.domain[0]-10<=y<=v.domain[1]+10 and h.domain[0]-8<=x<=h.domain[1]+8:
                near.append((x,v))
        unique=[]
        for x,v in sorted(near,key=lambda p:-p[1].support):
            if all(abs(x-a)>4 for a,b in unique):unique.append((x,v))
        vv0=graph_verticals([v for x,v in unique],horizontal,W,H) if len(unique)>=5 else None
        if vv0 is None and 5<=len(unique)<=12:
            # A single common Y interval cannot describe staggered segment ends
            # at a slanted table or a broken VAT boundary. Bootstrap from five
            # observed vertical fragments and >=3 bounded horizontal proposals;
            # final source-arm/ink acceptance below remains unchanged.
            options=[]
            for combo in combinations(sorted(unique,key=lambda p:p[0]),5):
                xx=np.array([x for x,v in combo]);span=xx[-1]-xx[0]
                if not W*.18<span<W*.92:continue
                ratios=np.diff(xx)/span
                if not (.035<ratios[0]<.22 and .32<ratios[1]<.73 and .07<ratios[2]<.30 and .09<ratios[3]<.38):continue
                vv1=[v for x,v in combo];lo=max(v.domain[0] for v in vv1);hi=min(v.domain[1] for v in vv1)
                if hi-lo<40:continue
                bounded=sum(abs(r.domain[0]-xx[0])<span*.10 and abs(r.domain[1]-xx[-1])<span*.10 and
                    min(v.domain[0] for v in vv1)-12<=r.position<=max(v.domain[1] for v in vv1)+12 for r in horizontal)
                if bounded>=3:options.append((bounded-2*float(np.sum(abs(ratios-[.10,.52,.16,.22]))),vv1))
            if options:vv0=max(options,key=lambda p:p[0])[1]
        if vv0 is not None:
            sig=tuple(round(v.position/4) for v in vv0)
            if sig not in seen:hypotheses.append(vv0);seen.add(sig)
    if debug:
        debug.add_geometry('physical_v3_build_hypotheses',(perf_counter()-started)*1000,
                           {'hypotheses':len(hypotheses)})
        started=perf_counter()
    # GRID_SPEED_COMBINED_V1: identical object/rail evidence, no fuzzy matching.
    from ...config import setting as _combo_setting
    from ..grid_speed_combined_v1 import evidence_cache
    _ev=evidence_cache(evidence) if _combo_setting('OCR_GRID_COMBO_EVIDENCE','1')=='1' else evidence
    scored=[]
    for hypothesis in hypotheses:
        vv0=junction_outer_boundaries(hypothesis,vertical,horizontal,response)
        count=0;coverage=0.
        for h in horizontal:
            e=_ev(h,vv0,(H,W),response,dark)
            if e and e['accepted']:count+=1;coverage+=e['ink_coverage']
        scored.append((count,coverage,vv0))
    vv=max(scored,key=lambda p:(p[0],p[1]))[2] if scored else None
    report['local_hypotheses']=len(hypotheses)
    if debug:
        debug.add_geometry('physical_v3_score_hypotheses',(perf_counter()-started)*1000,
                           {'hypotheses':len(hypotheses),'horizontal_candidates':len(horizontal),
                            'best_accepted_edges':max((x[0] for x in scored),default=0),
                            'found':vv is not None})
        started=perf_counter()
    if vv is None:
        if debug:debug.add_geometry('physical_v3_decision',0,{'selection':'frozen_v2_fallback' if baseline is not None else 'rejected','reason':'no_evidence_backed_five_vertical_hypothesis'})
        if baseline is not None:return baseline,dict(report,status='OBSERVED_NOT_VERIFIED',selection='frozen_v2_fallback',edges=previous['edges'])
        report['reason']='no_evidence_backed_five_vertical_hypothesis';return None,report
    vv=junction_outer_boundaries(vv,vertical,horizontal,response)
    accepted=[]
    for rule in horizontal:
        ev=_ev(rule,vv,(H,W),response,dark)
        if ev is None:continue
        if not ev['accepted']:
            rule, ev = refine_nearby_ridge(rule,vv,(H,W),response,dark,ev)
        # Do not extrapolate an unrelated underline across the table.
        sample=np.linspace(rule.domain[0],rule.domain[1],30)
        y=rule.eval((sample-W/2)/W)
        if not any(v.domain[0]-10<=float(np.median(y))<=v.domain[1]+10 for v in vv):continue
        if ev['accepted']:accepted.append((rule,ev))
        else:report['rejected'].append(ev)
    if debug:
        debug.add_geometry('physical_v3_check_horizontal_evidence',(perf_counter()-started)*1000,
                           {'candidates':len(horizontal),'accepted':len(accepted),'rejected':len(report['rejected'])})
        started=perf_counter()
    picked=[]
    for r,e in sorted(accepted,key=lambda p:-p[1]['ink_coverage']):
        duplicate=False
        for old,_ in picked:
            lo=max(r.domain[0],old.domain[0]);hi=min(r.domain[1],old.domain[1])
            if hi>lo:
                t=(np.linspace(lo,hi,50)-W/2)/W
                if np.percentile(abs(r.eval(t)-old.eval(t)),90)<5:duplicate=True;break
        if not duplicate:picked.append((r,e))
    picked.sort(key=lambda p:p[0].position)
    sample=(np.linspace(vv[0].position,vv[-1].position,100)-W/2)/W
    clean=[]
    for r,e in picked:
        if clean and np.min(r.eval(sample)-clean[-1][0].eval(sample))<4:continue
        clean.append((r,e))
    if debug:
        debug.add_geometry('physical_v3_deduplicate_edges',(perf_counter()-started)*1000,
                           {'accepted':len(accepted),'unique':len(picked),'noncrossing':len(clean)})
        started=perf_counter()
    if baseline is not None and len(clean)<=len(baseline['horizontal']):
        if debug:debug.add_geometry('physical_v3_decision',(perf_counter()-started)*1000,{'selection':'frozen_v2_fallback','reason':'baseline_has_at_least_as_many_boundaries','multiscale_boundaries':len(clean),'baseline_boundaries':len(baseline['horizontal'])})
        return baseline,dict(report,status='OBSERVED_NOT_VERIFIED',selection='frozen_v2_fallback',edges=previous['edges'])
    report['edges']=[e for r,e in clean];report['nodes']=[dict(horizontal=j,vertical=i,point=intersection(v,r,(H,W))) for j,(r,e) in enumerate(clean) for i,v in enumerate(vv)]
    if len(clean)<3:
        report['reason']='insufficient_jointly_confirmed_multiscale_edges'
        if debug:debug.add_geometry('physical_v3_decision',(perf_counter()-started)*1000,{'selection':'rejected','reason':report['reason'],'horizontal_boundaries':len(clean)})
        return None,report
    report.update(status='OBSERVED_NOT_VERIFIED',selection='multiscale_joint_graph',horizontal=len(clean),vertical=5)
    if debug:debug.add_geometry('physical_v3_decision',(perf_counter()-started)*1000,{'selection':report['selection'],'horizontal_boundaries':len(clean),'vertical_boundaries':5})
    return dict(vertical=vv,horizontal=[r for r,e in clean],shape=(H,W)),report


def _multiscale_core(gray,response,dark,baseline,previous):
    from ...config import setting
    # Only speculative MultiScale-first branch. Legacy selection stays unchanged.
    if setting('OCR_GRID_COMBO_SCALE','0')!='1' or baseline is not None or previous.get('status')!='NOT_RUN':
        return _multiscale_core_impl(gray,response,dark,baseline,previous)
    from ..grid_speed_combined_v1 import scale_gate,combine_parts
    from ..grid_anomaly_v3 import scan_above
    from ..grid_edge_validator_v12 import audit_rowaware_printed_rows
    from .algorithms import _dest
    from .. import detector_routing_v1 as route
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    first=multiscale_segments(gray,scales=(1.,))
    candidate,report=_multiscale_core_impl(gray,response,dark,baseline,previous,supplied_parts=first)
    try:
        ok,reason=scale_gate(image,candidate,report,scan_above,_dest,intersection,audit_rowaware_printed_rows,int(setting('OCR_GRID_ROUTE_MIN_H','10')))
    except (ValueError,TypeError,IndexError,KeyError,AttributeError,cv2.error):
        ok,reason=False,'gate_exception'
    route.record('combo_scale_decision',accepted=ok,reason=reason)
    if ok:
        report['combo_scale']='1x_only_verified'
        return candidate,report
    # Identical segment ordering as the original: first 1x, then 1.5x.
    second=multiscale_segments(gray,scales=(1.5,))
    candidate,report=_multiscale_core_impl(gray,response,dark,baseline,previous,supplied_parts=combine_parts(first,second))
    report['combo_scale']='1x_plus_1_5x'
    return candidate,report
