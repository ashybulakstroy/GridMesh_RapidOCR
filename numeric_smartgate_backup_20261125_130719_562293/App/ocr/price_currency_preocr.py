"""Second, conservative image-only pass for price suffixes; no OCR calls."""
from time import perf_counter
import cv2
import numpy as np


def split_rim_safe(image):
    """Return (original pixel crop, evidence or None, reason).

    Ignore isolated outer rules in the detection mask only. Require separated,
    centered T-like glyph with a broad bar and a straight narrow stem. Refuse
    ambiguous/multiline cells rather than sacrifice actual digits.
    """
    if image is None or not image.size or image.ndim not in (2,3):
        return image, None, 'empty'
    h,w=image.shape[:2]
    if min(h,w)<12 or h*w>160000:
        return image,None,'size_guard'
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ink[np.count_nonzero(ink,axis=1)>w*.94,:]=0
    ink[:,np.count_nonzero(ink,axis=0)>h*.94]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(ink)
    interior=[int(ch) for x,y,cw,ch,area in stats[1:] if x>=3 and x+cw<=w-3 and area>=8 and ch>=6 and .15<=cw/ch<=2 and ch<h*.9]
    if not interior:
        return image,None,'no_reference_glyphs'
    ref=float(np.median(interior))
    removed=[]
    for i,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        # Isolated thin outer vertical line much taller than a printed digit.
        rim=(x<=1 or x+cw>=w-1) and cw<=2 and ch>=2*ref
        # Outer horizontal fragment; do not discard decimal punctuation.
        rule=(y<=1 or y+ch>=h-1) and ch<=2 and cw>=max(8,ref)
        if rim or rule or area<3:
            ink[labels==i]=0
            if rim or rule:removed.append([int(x),int(y),int(cw),int(ch)])
    edges=np.diff(np.r_[False,np.any(ink,axis=0),False].astype(np.int8))
    groups=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    if len(groups)<2:return image,None,'no_separated_suffix'
    left,right=map(int,groups[-1]); gap=left-int(groups[-2][1]); ys=np.flatnonzero(np.any(ink[:,left:right],axis=1))
    if not len(ys):return image,None,'empty_suffix'
    top,bottom=int(ys[0]),int(ys[-1])+1; glyph=ink[top:bottom,left:right]>0; gh,gw=glyph.shape
    if not (gh>=6 and gw>=4 and .35<=gw/gh<=1.2 and .65*ref<=gh<=1.4*ref and gap>=max(2,gh*.15)):
        return image,None,'suffix_geometry'
    upper=glyph[:max(2,round(gh*.4))]; lower=glyph[round(gh*.45):]; widths=lower.sum(axis=1)
    if np.count_nonzero(upper.sum(axis=1)>=gw*.75)<2 or len(widths)<3 or np.any(widths==0) or np.max(widths)>gw*.55:
        return image,None,'not_t_bar_stem'
    centers=np.array([np.flatnonzero(row).mean() for row in lower])
    if centers.std()>gw*.10 or not gw*.25<=centers.mean()<=gw*.70 or abs(centers[0]-centers[-1])>gw*.15:
        return image,None,'stem_not_centered'
    # All preceding significant glyphs must occupy the same band; refuse
    # additional lines and cuts through any significant foreground.
    digit_count=0
    for a,b in groups[:-1]:
        yy=np.flatnonzero(np.any(ink[:,a:b],axis=1))
        if not len(yy):continue
        ht=int(yy[-1]-yy[0]+1)
        if ht<gh*.5:continue
        if ht>gh*1.7 or yy[0]<top-gh*.5 or yy[-1]>=bottom+gh*.5:
            return image,None,'ambiguous_digit_band'
        digit_count+=max(1,round((b-a)/gw))
    if not digit_count or np.count_nonzero(ink[:,:left])<8:return image,None,'no_digits'
    # Validate against original threshold mask: any non-noise component crossing
    # the cut or significant glyph farther right invalidates the proposal.
    # Outer rules already classified above are exempt, never modified in output.
    for i,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        if area<3 or [int(x),int(y),int(cw),int(ch)] in removed:continue
        if x<left<x+cw:return image,None,'cut_crosses_ink'
        if x>=left and (x+cw>right or y<top or y+ch>bottom):return image,None,'extra_suffix_ink'
    return image[:,:left].copy(), {'symbol':'₸','strategy':'separate_t_shape_rim_safe_v2','numeric_glyph_count':digit_count,'bbox':[left,top,right,bottom],'ignored_rim_rules':removed}, 'excluded'


def fallback_currency(image, cropped, currency):
    """Keep the established algorithm first; fallback only when it failed."""
    if currency is not None:return cropped,currency,'excluded_existing'
    return split_rim_safe(image)


def capture_price(debug, cell, prepared, info, elapsed_ms):
    """DEBUG only. Use existing job context, no writes in recognition loop."""
    if debug is None:return
    key=(cell.row_id,cell.column_id)
    with debug.lock:
        if not hasattr(debug,'price_currency_v2'):debug.price_currency_v2={}
        if key not in debug.price_currency_v2:
            debug.price_currency_v2[key]=(cell.image.copy(),prepared.copy(),dict(info),round(elapsed_ms,3))


def export_price_audit(job_dir, debug, tracker):
    """Record all primary prices and available exact REC event inputs."""
    import json
    if debug is None:return None
    from pathlib import Path
    root=Path(job_dir)/'price_currency'; data=getattr(debug,'price_currency_v2',{})
    if not data:return None
    root.mkdir(exist_ok=True); rows=[]
    def save(name,image):
        ok,buf=cv2.imencode('.png',image)
        if not ok:raise RuntimeError('PNG encoding failed')
        (root/name).write_bytes(buf.tobytes());return name
    for (rid,field),(source,prepared,info,ms) in sorted(data.items()):
        key=f'{rid}:{field}'; prefix=f'row_{int(rid)}_price'
        item={'cell_id':key,'prepare_ms':ms,'roi':info,'source':save(prefix+'_source.png',source),'prepared':save(prefix+'_prepared.png',prepared),'attempts':[]}
        for event in tracker.events:
            if key not in event.get('cells',[]):continue
            eid=int(event['event_id']); actual=tracker.inputs.get(eid)
            if isinstance(actual,dict):actual=actual.get(key)
            e={k:event[k] for k in ('event_id','stage','mode','model_id','wall_ms','texts','confidences') if k in event}
            e['actual_input']=save(prefix+f'_attempt_{eid}.png',actual) if actual is not None and actual.size else None
            item['attempts'].append(e)
        rows.append(item)
    (root/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'count':len(rows),'manifest':'price_currency/manifest.json'}
