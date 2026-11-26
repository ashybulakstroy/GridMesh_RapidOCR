"""V1 hook-compatible compact object store. No map/image encoding during detection."""
import hashlib,json
import numpy as np


def init(s):
    s.objects={};s.object_bytes=0;s.object_dropped=0;s.graph_index={};s.rule_index={}
    try:
        from App.config import ROOT
        s.project_path=str(ROOT)
    except ImportError:s.project_path=None


def intern(s,kind,value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    key=kind+'_'+hashlib.sha256(raw).hexdigest()[:24]
    if key not in s.objects:
        if s.object_bytes+len(raw)>12*1024*1024:
            s.object_dropped+=1;return {'omitted':'object_budget','kind':kind}
        s.objects[key]=value;s.object_bytes+=len(raw)
    return {'$ref':key}


def rule(s,r):
    if isinstance(r,dict):
        fields={k:r[k] for k in ('rule_class','axis','coeff','position','support','domain','points','extent','t','values') if k in r}
    else:
        fields={'rule_class':type(r).__name__,**{k:getattr(r,k) for k in ('axis','coeff','position','support','domain','points','extent','t','values') if hasattr(r,k)}}
    # Exact numerical fields, no eval() calls or repeated sampled curves.
    key=tuple((k,tuple(np.asarray(v).ravel().tolist()) if isinstance(v,(list,tuple,np.ndarray)) else v) for k,v in fields.items())
    if key not in s.rule_index:
        s.rule_index[key]=intern(s,'rule',{k:plain(s,v,1) for k,v in fields.items()})
    return s.rule_index[key]


def plain(s,x,depth=0):
    if depth>10:return {'omitted':'depth_limit'}
    if isinstance(x,str):
        if s.project_path:x=x.replace(s.project_path,'[PROJECT]').replace(s.project_path.replace('\\','/'),'[PROJECT]')
        return x[:16000]
    if x is None or isinstance(x,(bool,int)):return x
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.ndarray):
        if x.size>8192:return {'omitted':'large_array','shape':list(x.shape),'dtype':str(x.dtype)}
        return plain(s,x.tolist(),depth+1)
    if hasattr(x,'eval') and hasattr(x,'position'):return rule(s,x)
    if hasattr(x,'info') and hasattr(x,'xs') and hasattr(x,'ys'):
        return {'method':x.method,'xs':plain(s,x.xs,depth+1),'ys':plain(s,x.ys,depth+1),'info':plain(s,x.info,depth+1)}
    if isinstance(x,dict):
        if 'rule_class' in x and 'axis' in x and 'position' in x and 'domain' in x:return rule(s,x)
        if all(k in x for k in ('vertical','horizontal','shape')) and isinstance(x['vertical'],(tuple,list)) and isinstance(x['horizontal'],(tuple,list)):
            v=[rule(s,r) if hasattr(r,'eval') or isinstance(r,dict) and 'rule_class' in r else plain(s,r,depth+1) for r in x['vertical']]
            h=[rule(s,r) if hasattr(r,'eval') or isinstance(r,dict) and 'rule_class' in r else plain(s,r,depth+1) for r in x['horizontal']]
            val={'shape':plain(s,x['shape'],depth+1),'vertical':v,'horizontal':h}
            # Preserve other graph metadata, without duplicating image arrays.
            val.update({str(k):plain(s,y,depth+1) for k,y in x.items() if k not in ('vertical','horizontal','shape','image','source_image','map_x','map_y','cell_mesh')})
            return intern(s,'graph',val)
        return {str(k):plain(s,v,depth+1) for k,v in list(x.items())[:256] if str(k) not in ('source_image','image','map_x','map_y','cell_mesh','normalized_samples','evaluated_samples')}
    if isinstance(x,(tuple,list)):
        # Keep all normal row and rail arrays; large proposal sets explicitly truncated.
        out=[plain(s,v,depth+1) for v in x[:256]]
        if len(x)>256:out.append({'omitted_items':len(x)-256})
        return out
    if isinstance(x,BaseException):return {'type':type(x).__name__,'message':plain(s,str(x))[:4000]}
    if type(x).__name__=='RouteState':return plain(s,vars(x),depth+1)
    return {'type':type(x).__name__}


def snapshot(s,local):
    # Capture decisions and exact graph versions, not every transient variable.
    out={};names=('data','graph','physical','automatic_data','baseline','report','physical_report','previous','diagnostic','stats','audit','accepted','safe','safety','geometry','scan','rect','result','_grid_gate','_grid_trace','_grid_attempt','_grid_route','_ga','_gg','_state','failures','reason','retry','rectify','min_rows','prefer_multiscale','scales','number_col','force_number','detailed','rail_support','anchor','paper_bbox','bbox','roi','_combo_diag','_combo_scan','_premesh','_combo_recovery','_mode','_retry','_bands','_giant','_texts','_header','invalid_spacing','auto_vertical_count')
    for k in names:
        if k not in local:continue
        v=local[k]
        if callable(v):continue
        if hasattr(v,'map_x') and hasattr(v,'info'):
            if all(v is not x[1] for x in s.maps):s.maps.append((k,v))
        val=plain(s,v)
        out[k]=intern(s,'detail',val) if isinstance(val,(dict,list)) and '$ref' not in val else val
    for k in ('parts','supplied_parts','hypotheses','scored','clean','picked','rejected','edges','horizontal','vertical'):
        v=local.get(k)
        if isinstance(v,dict):out[k]={str(a):len(b) if hasattr(b,'__len__') else None for a,b in v.items()}
        elif isinstance(v,(tuple,list)):out[k]={'count':len(v)}
    im=local.get('image');gray=local.get('gray')
    if im is not None and hasattr(im,'shape'):
        out['input_shape']=list(im.shape)
        if getattr(im,'ndim',0)==3 and all(im is not i for i in s.source_images):s.source_images.append(im)
    elif gray is not None and hasattr(gray,'shape'):out['input_shape']=list(gray.shape)
    return out


def result(s,value):
    # Detector tuples normally contain a graph and report; omit image arrays.
    return intern(s,'result',plain(s,value))


def metadata(s):
    return {'schema':'grid-abort-diagnostics-v2','object_store':'objects.json','object_count':len(s.objects),'object_bytes':s.object_bytes,'objects_dropped':s.object_dropped}
