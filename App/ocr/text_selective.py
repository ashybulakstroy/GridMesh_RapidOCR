"""Name-only experimental REC routing. No changes to numeric fields."""
from App.ocr.performance_deep.live_capture import timed as _deep_timed
import re
from time import perf_counter
import cv2
import numpy as np
from ..config import setting


@_deep_timed("ocr", "App/ocr/text_selective.py:ink")
def ink(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask[:2,:]=0; mask[-2:,:]=0; mask[:,:2]=0; mask[:,-2:]=0
    mask = cv2.subtract(mask, cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1,max(3,int(gray.shape[1]*.8))),np.uint8)))
    return mask


@_deep_timed("ocr", "App/ocr/text_selective.py:segments_v1")
def segments_v1(image):
    if image is None or not image.size: return [], 'empty'
    h,w=image.shape[:2]
    mask=ink(image)
    active=np.count_nonzero(mask,axis=1)>=max(2,w*.015)
    edges=np.diff(np.r_[False,active,False].astype(np.int8))
    raw=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    # Scale from observed ink runs, never absolute cell height.
    heights=[b-a for a,b in raw if b-a>=3]
    if not heights: return [], 'uncertain'
    glyph=float(np.median(heights)); merged=[]
    for a,b in raw:
        if merged and a-merged[-1][1]<=max(1,int(glyph*.20)):
            merged[-1]=(merged[-1][0],b)
        else: merged.append((a,b))
    bands=[(a,b) for a,b in merged if b-a>=max(3,glyph*.35)]
    if not bands or len(bands)>12: return [], 'uncertain'
    # Keep small marks in neighboring line, no overlap between line crops.
    result=[]
    for i,(a,b) in enumerate(bands):
        top=0 if i==0 else (bands[i-1][1]+a)//2
        bottom=h if i==len(bands)-1 else (b+bands[i+1][0])//2
        result.append((top,bottom))
    return result, 'observed' if len(result)>1 else 'single'


@_deep_timed("ocr", "App/ocr/text_selective.py:segments")
def segments(image):
    bands,state=segments_v1(image)
    if not bands:return bands,state
    density=np.count_nonzero(ink(image),axis=1).astype(float)
    # Detect two strong ink cores separated by a deep valley inside a merged band.
    peak=float(density.max())
    strong=density>=max(3,peak*.25)
    edges=np.diff(np.r_[False,strong,False].astype(np.int8))
    cores=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=4]
    if not cores:return bands,state
    glyph=float(np.median([b-a for a,b in cores])); out=[]
    for a,b in bands:
        inside=[(x,y) for x,y in cores if x>=a and y<=b]
        cuts=[]
        for (x,y),(u,v) in zip(inside,inside[1:]):
            if u-y<max(2,glyph*.20) or min(y-x,v-u)<glyph*.65:continue
            if u-x<glyph*1.15 or b-a<glyph*1.8:continue
            valley=density[y:u]
            if len(valley) and valley.min()<=.16*min(density[x:y].max(),density[u:v].max()):
                minima=np.flatnonzero(valley==valley.min()); cuts.append(y+int(minima[len(minima)//2]))
        bounds=[a]+cuts+[b];out.extend(zip(bounds,bounds[1:]))
    return out,'observed' if len(out)>1 else 'single'

@_deep_timed("ocr", "App/ocr/text_selective.py:trim_short")
def trim_short(crop):
    mask=ink(crop); h,w=mask.shape
    xs=np.flatnonzero(np.count_nonzero(mask,axis=0)>0)
    if not len(xs):return crop,0
    a,b=int(xs[0]),int(xs[-1])+1
    if b-a > w*.65:return crop,0
    pad=max(4,round(h*.35)); a=max(0,a-pad);b=min(w,b+pad)
    return np.ascontiguousarray(crop[:,a:b]),a


@_deep_timed("ocr", "App/ocr/text_selective.py:gamma")
def gamma(image):
    g=float(setting('OCR_TEXT_SELECTIVE_GAMMA','1.35'))
    if not .2<=g<=3: raise ValueError('gamma must be .2..3')
    lut=np.uint8(np.round(255*(np.arange(256)/255)**g))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    out=cv2.LUT(gray,lut)
    return cv2.cvtColor(out,cv2.COLOR_GRAY2BGR) if image.ndim==3 else out


@_deep_timed("ocr", "App/ocr/text_selective.py:suspect")
def suspect(text,score):
    return not text.strip() or score<float(setting('OCR_TEXT_SELECTIVE_MIN_CONF','0.80')) or not re.search(r'\w|[¼½¾⅛⅜⅝⅞]',text) or bool(re.search(r'[/–-]$',text.strip()))


@_deep_timed("ocr", "App/ocr/text_selective.py:join")
def join(parts):
    out=[]
    for value in parts:
        t=value.strip()
        if not t:continue
        # Only remove explicit wrap hyphen between letters; preserve dimension punctuation.
        if out and re.search(r'[^\W\d_]\-$',out[-1]) and re.match(r'[^\W\d_]',t):out[-1]=out[-1][:-1]+t
        else:out.append(t)
    return ' '.join(out)


@_deep_timed("ocr", "App/ocr/text_selective.py:read_name_v1")
def read_name_v1(image,row_id,column_id,offset,*,primary,secondary,last_model,lock,full):
    started=perf_counter(); bands,state=segments_v1(image)
    h,w=image.shape[:2]; ox,oy=offset
    bbox=[ox,oy,ox+w,oy+h]; blocks=[]; history=[]
    def rec(img, fn, route):
        try:
            with lock: r=fn(img)
            txt=str(r.txts[0]) if r.txts else ''
            score=float(r.scores[0]) if r.txts else 0.
            return dict(text=txt,confidence=score,model_id=last_model(),route=route)
        except Exception as exc:
            return dict(text='',confidence=0.,model_id=None,route=route,error=str(exc))
    if not bands:
        # Geometry uncertain only: full detector, not a whole-cell REC guess.
        lines=full(image,cell_mode=True)
        for l in lines:
            blocks.append(dict(text=l['text'],confidence=l['confidence'],model_id='v5_cyrillic_mobile',bbox=[[x+ox,y+oy] for x,y in l['bbox']],cell_id=f'{row_id}-{column_id}',row_id=row_id,column_id=column_id))
        blocks.sort(key=lambda b:(min(y for x,y in b['bbox']),min(x for x,y in b['bbox'])))
    else:
        limit=max(0,int(setting('OCR_TEXT_SELECTIVE_MAX_RETRIES','2')))
        budget=max(0.,float(setting('OCR_TEXT_SELECTIVE_BUDGET_MS','180')))
        retries=0
        for i,(a,b) in enumerate(bands):
            raw=image[a:b,:]; crop,x=trim_short(raw)
            first=rec(crop,primary,'primary_trimmed' if x or crop.shape[1]!=w else 'primary')
            candidates=[first]; chosen=first
            if suspect(first['text'],first['confidence']):
                for fn,variant,route in [(primary,lambda:gamma(crop),'gamma'),(secondary,lambda:crop,'v5_rec_only')]:
                    if retries>=limit or (perf_counter()-started)*1000>=budget:break
                    retries+=1; cand=rec(variant(),fn,route); candidates.append(cand)
                    # Heuristic, not a guarantee: prefer non-suspect and otherwise clear confidence gain.
                    if cand['text'].strip() and (not chosen['text'].strip() or (not suspect(cand['text'],cand['confidence']) and suspect(chosen['text'],chosen['confidence'])) or (suspect(cand['text'],cand['confidence'])==suspect(chosen['text'],chosen['confidence']) and cand['confidence']>chosen['confidence']+.03)):
                        chosen=cand
                    if not suspect(chosen['text'],chosen['confidence']):break
            history.append(dict(line=i+1,band=[a,b],trim_x=x,input_hw=list(crop.shape[:2]),candidates=candidates,selected=chosen['route']))
            blocks.append(dict(text=chosen['text'],confidence=chosen['confidence'],model_id=chosen['model_id'],bbox=[ox+x,oy+a,ox+x+crop.shape[1],oy+b],cell_id=f'{row_id}-{column_id}',row_id=row_id,column_id=column_id))
    models=list(dict.fromkeys(b['model_id'] for b in blocks if b.get('model_id')))
    return dict(text=join([b['text'] for b in blocks]),confidence=min((b['confidence'] for b in blocks),default=0.),bbox=bbox,cell_id=f'{row_id}-{column_id}',row_id=row_id,column_id=column_id,blocks=blocks,model_id=models[0] if len(models)==1 else 'mixed' if models else None,model_ids=models,strategy='text_selective_'+state,text_selective={'bands':bands,'state':state,'lines':history,'wall_ms':round((perf_counter()-started)*1000,3)})


@_deep_timed("ocr", "App/ocr/text_selective.py:read_name")
def read_name(image,row_id,column_id,offset,*,primary,secondary,last_model,lock,full):
    mode=setting('OCR_TEXT_FAST_MODE','E').upper()
    if mode=='G':return read_name_v1(image,row_id,column_id,offset,primary=primary,secondary=secondary,last_model=last_model,lock=lock,full=full)
    if mode not in ('B','C','D','E','F'):raise ValueError('Unknown text fast mode '+mode)
    start=perf_counter();bands,state=segments(image); prep=0.; blocks=[]; history=[];ox,oy=offset;h,w=image.shape[:2]
    if not bands:
        lines=full(image,cell_mode=True)
        for l in sorted(lines,key=lambda l:(min(y for x,y in l['bbox']),min(x for x,y in l['bbox']))):
            blocks.append(dict(text=l['text'],confidence=l['confidence'],model_id='v5_cyrillic_mobile',bbox=[[x+ox,y+oy] for x,y in l['bbox']],cell_id=f'{row_id}-{column_id}',row_id=row_id,column_id=column_id))
    for i,(a,b) in enumerate(bands):
        t=perf_counter();raw=image[a:b];crop,x=trim_short(raw) if mode in ('D','E','F') else (raw,0); prepared=gamma(crop) if mode in ('C','E','F') else crop;prep+=(perf_counter()-t)*1000
        def recognize(im,route):
            t=perf_counter()
            with lock:r=primary(im)
            return dict(text=str(r.txts[0]) if r.txts else '',confidence=float(r.scores[0]) if r.txts else 0.,model_id=last_model(),route=route,wall_ms=(perf_counter()-t)*1000)
        first=recognize(prepared,'primary'); chosen=first;candidates=[first]
        # F only: one alternative on a severe error, not on normal wrap hyphens.
        if mode=='F' and not any(len(v['candidates'])>1 for v in history) and (not first['text'].strip() or first['confidence']<.50):
            alt=recognize(crop,'original_retry');candidates.append(alt)
            if alt['text'].strip() and alt['confidence']>first['confidence']+.05:chosen=alt
        history.append(dict(line=i+1,band=[a,b],trim_x=x,input_hw=list(prepared.shape[:2]),candidates=candidates,selected=chosen['route']))
        blocks.append(dict(text=chosen['text'],confidence=chosen['confidence'],model_id=chosen['model_id'],bbox=[ox+x,oy+a,ox+x+crop.shape[1],oy+b],cell_id=f'{row_id}-{column_id}',row_id=row_id,column_id=column_id))
    models=list(dict.fromkeys(b['model_id'] for b in blocks if b.get('model_id')))
    return dict(text=join([b['text'] for b in blocks]),confidence=min((b['confidence'] for b in blocks),default=0.),bbox=[ox,oy,ox+w,oy+h],cell_id=f'{row_id}-{column_id}',row_id=row_id,column_id=column_id,blocks=blocks,model_id=models[0] if len(models)==1 else 'mixed' if models else None,model_ids=models,strategy='text_fast_'+state,text_selective={'bands':bands,'state':state,'lines':history,'mode':mode,'prep_ms':prep,'wall_ms':(perf_counter()-start)*1000})
