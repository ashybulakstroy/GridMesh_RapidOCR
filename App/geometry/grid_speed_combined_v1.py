"""Experimental helpers; per-call state only. No changes to geometric thresholds."""
import re
import numpy as np

class SparseMap:
    def __init__(self, shape, points): self.shape=shape; self.points=points
    def __getitem__(self, key): return self.points[key]


def sparse_top_mapping(image, data, dest):
    """Exactly the mesh's eight iterations at the four queried rounded corners.
    No full output array. Missing indices raise, never return invented values.
    """
    xs,ys,shape=dest(data); H,W=image.shape[:2]; vv,hh=data['vertical'],data['horizontal']; mx={};my={}
    for y in (round(ys[0]),round(ys[1])):
        j=int(np.clip(np.searchsorted(ys,y,side='right')-1,0,len(ys)-2)); b=(y-ys[j])/(ys[j+1]-ys[j])
        for x in (round(xs[0]),round(xs[-1])):
            i=int(np.clip(np.searchsorted(xs,x,side='right')-1,0,len(xs)-2));a=(x-xs[i])/(xs[i+1]-xs[i]); yy=np.float64(y-ys[0]+hh[0].position)
            for _ in range(8):
                ny=(yy-H/2)/H; x0=vv[i].eval(ny);x1=vv[i+1].eval(ny);xx=x0+a*(x1-x0)
                nx=(xx-W/2)/W;y0=hh[j].eval(nx);y1=hh[j+1].eval(nx);yy=y0+b*(y1-y0)
            mx[y,x]=np.float32(xx);my[y,x]=np.float32(yy)
    return SparseMap(shape,mx),SparseMap(shape,my),xs,ys


def scan_graph(image,data,scan,dest):
    return scan(image,*sparse_top_mapping(image,data,dest))


def premesh(image,data,scan,recover,dest,rule_factory):
    try:
        s=scan_graph(image,data,scan,dest)
        if s.get('status')!='suspect':return data,None
        repaired,diag=recover(image,data,s,rule_factory=rule_factory)
        return repaired,diag
    except (ValueError,TypeError,IndexError,KeyError,AttributeError,OverflowError,RuntimeError):
        return data,None


def scale_gate(image,data,report,scan,dest,intersection,audit,min_h):
    """A single-scale candidate must independently prove completeness.
    Otherwise run original two-scale algorithm with cached first-scale fragments.
    """
    if data is None or len(data['vertical'])!=5 or len(data['horizontal'])<max(3,min_h):return False,'dimensions'
    edges=report.get('edges',[])
    if len(edges)!=len(data['horizontal']) or any(e.get('ink_coverage',0)<.85 or min(e.get('column_coverage',[0]))<.75 or sum(e.get('intersection_evidence',[]))<4 or e.get('longest_unsupported_fraction',1)>.09 for e in edges):return False,'edge_evidence'
    s=scan_graph(image,data,scan,dest)
    if s.get('status')!='no_signal':return False,'source_extension'
    anchor=s.get('anchor',{}).get('status')=='aligned'
    fast=s.get('fast_screen') and s.get('top_support',0)>=.60 and s.get('bottom_support',0)>=.60 and s.get('max_coverage',1)<.25
    if not (anchor or fast):return False,'header_anchor'
    nodes=np.array([[intersection(v,h,image.shape) for v in data['vertical']] for h in data['horizontal']],float)
    ok,detail=audit(image,nodes,paper_bbox=None)
    return bool(ok),detail.get('reason','independent_row_census')


def combine_parts(first,second):
    return {k:list(first[k])+list(second[k]) for k in ('h','v')}


def evidence_cache(fn):
    cache={}
    def evaluate(rule,rails,shape,response,dark):
        key=(id(rule),tuple(map(id,rails)))
        if key not in cache:
            cache[key]=(rule,tuple(rails),fn(rule,rails,shape,response,dark))
        return cache[key][2]
    return evaluate


def numeric_only(attempt,parse):
    text=str(attempt.get('text') or '').strip()
    # Allow decimal punctuation and conventional grouping spaces; no OCR letters.
    return not attempt.get('error') and bool(re.fullmatch(r'[0-9]+(?:[ ,\.\u00a0\u202f][0-9]+)*',text)) and parse(text) is not None
