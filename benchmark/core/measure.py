"""Pure benchmark helpers; no import of the existing App OCR pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib, math, os, re, statistics, time

CYR = re.compile(r'[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]')


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def decimal_ocr(text):
    """Strict numeric parsing; never correct an OCR digit by guessing."""
    s=str(text).strip().replace('\u00a0',' ').replace('\u202f',' ')
    s=re.sub(r'(?i)(?:₸|тг|тенге)', '', s)
    # Printed currency 'Т' or Latin 'T' only at the end of an otherwise numeric candidate.
    s=re.sub(r'\s+[ТTтt]$', '', s).strip()
    s=re.sub(r'(?<=\d)[ТTтt]$', '', s).strip()
    s=s.replace(' ','')
    if not re.fullmatch(r'[+-]?[0-9]+(?:[.,][0-9]{1,3})?',s):
        return None
    try:return Decimal(s.replace(',','.'))
    except InvalidOperation:return None


def canonical_text(value):
    return ' '.join(str(value).split())


NUMERIC_SUFFIXES=(':quantity',':unit_price',':printed_quantity',':printed_unit_price')


def is_numeric_field(field):
    return field=='printed_total' or field.endswith(NUMERIC_SUFFIXES)


def is_name_field(field):
    return field.startswith('row:') and field.endswith(':name')


def _ink_mask(crop):
    """Dark ink without table rules; long-run pixels and line-like blobs are dropped."""
    import cv2
    import numpy as np
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    h,w=binary.shape
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,int(w*.45)),1)))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(20,int(h*.45)))))
    clean=cv2.subtract(binary,cv2.bitwise_or(horizontal,vertical))
    n,labels,stats,_=cv2.connectedComponentsWithStats(clean,8)
    keep=np.zeros_like(clean)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area<4:continue
        if bw<=max(4,.04*w) and bh>=max(10,.25*h) and bh>=4*bw:continue
        if bh<=max(4,.04*h) and bw>=max(10,.25*w) and bw>=4*bh:continue
        if area/(bw*bh)<.18 and max(bw,bh)>=max(16,.25*max(w,h)):continue
        keep[labels==i]=255
    return keep


def tighten_numeric_crop(crop):
    """Trim around the strongest printed-number ink band, with glyph padding."""
    import cv2
    import numpy as np
    if crop is None or crop.size==0:return crop
    mask=_ink_mask(crop);h,w=mask.shape
    counts=np.count_nonzero(mask,axis=1)
    if counts.size==0 or counts.max()==0:return crop
    active=(counts>=max(2,int(counts.max()*.12))).astype(np.uint8).reshape(-1,1)
    active=cv2.morphologyEx(active,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)).ravel().astype(bool)
    edges=np.diff(np.r_[False,active,False].astype(np.int8))
    bands=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=4]
    if not bands:return crop
    y0,y1=max(bands,key=lambda b:int(counts[b[0]:b[1]].sum()))
    pad_y=max(4,int((y1-y0)*.4))
    y0=max(0,y0-pad_y);y1=min(h,y1+pad_y)
    cols=np.count_nonzero(mask[y0:y1],axis=0)
    xs=np.flatnonzero(cols>0)
    if xs.size==0:return crop
    pad_x=max(3,int((y1-y0)*.15))
    x0=max(0,int(xs[0])-pad_x);x1=min(w,int(xs[-1])+1+pad_x)
    out=crop[y0:y1,x0:x1]
    if out.shape[0]<6 or out.shape[1]<6:return crop
    return np.ascontiguousarray(out)


def split_name_lines(crop):
    """Split only physically multi-line name crops; single lines stay whole."""
    import cv2
    import numpy as np
    if crop is None or crop.size==0:return [crop]
    mask=_ink_mask(crop);h,w=mask.shape
    counts=np.count_nonzero(mask,axis=1)
    if counts.size==0 or counts.max()==0:return [crop]
    active=(counts>=max(2,int(w*.01))).astype(np.uint8).reshape(-1,1)
    active=cv2.morphologyEx(active,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)).ravel().astype(bool)
    edges=np.diff(np.r_[False,active,False].astype(np.int8))
    bands=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=4]
    if len(bands)<=1:return [crop]
    med=int(np.median([b-a for a,b in bands]))
    bands=[(a,b) for a,b in bands if b-a>=med*.5]
    if len(bands)<=1:return [crop]
    out=[]
    for y0,y1 in bands:
        a=max(0,y0-2);b=min(h,y1+2)
        out.append(np.ascontiguousarray(crop[a:b]))
    return out or [crop]


def join_text_parts(parts):
    """Join OCR line results, undoing printed end-of-line hyphenation."""
    out=[]
    letter=re.compile(r'^[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]')
    hyphen=re.compile(r'[-\u2010-\u2015]$')
    for part in parts:
        text=str(part).strip()
        if not text:continue
        if out and hyphen.search(out[-1]):
            out[-1]=(hyphen.sub('',out[-1]) if letter.match(text) else out[-1])+text
        else:
            out.append(text)
    return canonical_text(' '.join(out))


def levenshtein(a,b):
    a,b=canonical_text(a),canonical_text(b)
    row=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        nxt=[i]+[0]*len(b)
        for j,cb in enumerate(b,1):
            nxt[j]=min(nxt[j-1]+1,row[j]+1,row[j-1]+(ca!=cb))
        row=nxt
    return row[-1]


def gt_fields(gt):
    result={}
    for p in gt['products']:
        k=f"row:{p['row_no']}"
        result[k+':name']={'value':p['name'],'type':'text','needs_review':p.get('name_needs_review',False)}
        for typ in ('quantity','unit_price'):
            result[k+':'+typ]={'value':p[typ],'type':'numeric','needs_review':False}
    vat=gt.get('vat_row')
    if vat:
        for field in ('printed_label','printed_quantity','printed_unit_price'):
            if field in vat:
                result['vat:'+field]={'value':vat[field],'type':'text' if field=='printed_label' else 'numeric','needs_review':False}
    result['printed_total']={'value':gt['printed_total'],'type':'numeric','needs_review':False}
    return result


def verify_math(gt):
    errors=[]; subtotal=Decimal('0')
    if len(gt.get('products',[]))!=gt.get('expected_product_count'):
        errors.append('expected_product_count differs from products length')
    row_ids=[]
    for p in gt.get('products',[]):
        row_ids.append(p['row_no'])
        try:
            value=Decimal(p['quantity'])*Decimal(p['unit_price'])
            if value!=Decimal(p['calculated_amount']):
                errors.append(f"row {p['row_no']}: calculated amount mismatch")
            subtotal+=value
        except (InvalidOperation,KeyError):errors.append(f"row {p.get('row_no')}: invalid numeric GT")
    if len(set(row_ids))!=len(row_ids):errors.append('duplicate row_no')
    try:
        if subtotal!=Decimal(gt['derived_products_subtotal']):errors.append('derived_products_subtotal mismatch')
        vat=gt.get('vat_row')
        add=Decimal(vat['amount']) if vat else Decimal('0')
        if vat and Decimal(vat['printed_quantity'])*Decimal(vat['printed_unit_price'])!=add:
            errors.append('VAT row multiplication mismatch')
        if subtotal+add!=Decimal(gt['printed_total']):errors.append('printed_total does not equal items + VAT')
    except (InvalidOperation,KeyError):errors.append('bad subtotal/VAT/total in GT')
    return errors


def percentile_nearest_rank(values,p):
    if not values:return None
    vals=sorted(values)
    return vals[min(len(vals)-1,max(0,math.ceil(p*len(vals))-1))]


def summarize(values):
    if not values:return None
    return {'median':round(statistics.median(values),3),'p95':round(percentile_nearest_rank(values,.95),3),'min':round(min(values),3),'max':round(max(values),3),'n':len(values)}


def evaluate(gt, transcripts, script_supported=True, field_scope="all"):
    if field_scope not in ("all","numeric_only"):raise ValueError("unknown field scope")
    expected=gt_fields(gt)
    if field_scope=='numeric_only':expected={f:m for f,m in expected.items() if m['type']=='numeric'}
    verification=gt.get('verification',{})
    agent_names=bool(verification.get('names_verified_by_agent'))
    nums=[]; text=[]; cell_metrics=[]
    for field,meta in expected.items():
        observed=transcripts.get(field,'')
        if meta['type']=='numeric':
            exp=decimal_ocr(meta['value']); obs=decimal_ocr(observed)
            good=obs is not None and obs==exp
            nums.append(good)
            cell_metrics.append({'field':field,'type':'numeric','expected':meta['value'],'observed':observed,'correct':good})
        else:
            contains_cyr=bool(CYR.search(meta['value']))
            if not (gt.get('approved_by_user',False) or agent_names) or meta['needs_review']:
                status='GT_DRAFT'
            elif gt.get('approved_by_user',False):
                status='SCORED'
            else:
                status='SCORED_AGENT_VERIFIED'
            d=levenshtein(meta['value'],observed) if status.startswith('SCORED') else None
            if d is not None:text.append((d,len(canonical_text(meta['value']))))
            cell_metrics.append({'field':field,'type':'text','expected':meta['value'],'observed':observed,'status':status,'char_edits':d,'model_script_supported':not contains_cyr or script_supported})
    product_names=[c for c in cell_metrics if c['field'].startswith('row:') and c['type']=='text']
    names_scored=[c for c in product_names if c['char_edits'] is not None]
    names_exact=sum(c['char_edits']==0 for c in names_scored)
    subtotal=Decimal('0'); amounts=[]; arithmetic_valid=True
    for product in gt['products']:
        prefix=f"row:{product['row_no']}"
        quantity=decimal_ocr(transcripts.get(prefix+':quantity',''))
        price=decimal_ocr(transcripts.get(prefix+':unit_price',''))
        value=quantity*price if quantity is not None and price is not None else None
        amounts.append({'row_no':product['row_no'],'calculated_amount':str(value) if value is not None else None,
                        'correct':value==Decimal(product['calculated_amount']) if value is not None else False})
        if value is None:arithmetic_valid=False
        else:subtotal+=value
    vat_amount=Decimal('0')
    if gt.get('vat_row'):
        vq=decimal_ocr(transcripts.get('vat:printed_quantity',''))
        vp=decimal_ocr(transcripts.get('vat:printed_unit_price',''))
        if vq is None or vp is None:arithmetic_valid=False
        else:vat_amount=vq*vp
    printed=decimal_ocr(transcripts.get('printed_total',''))
    arithmetic_matches=arithmetic_valid and printed is not None and subtotal+vat_amount==printed
    all_text_exact=bool(text) and all(c.get('char_edits')==0 for c in cell_metrics if c['type']=='text')
    return {'field_scope':field_scope,'product_names_total':len(product_names),'product_names_scored':len(names_scored),
            'product_names_exact':names_exact,
            'product_name_cer':round(sum(c['char_edits'] for c in names_scored)/max(1,sum(len(canonical_text(c['expected'])) for c in names_scored)),6) if names_scored else None,
            'derived_product_amounts':amounts,'derived_subtotal':str(subtotal) if arithmetic_valid else None,
            'arithmetic_matches_printed_total':arithmetic_matches,
            'complete_receipt_correct':all(nums) and all_text_exact and arithmetic_matches,
            'numeric_correct':sum(nums),'numeric_total':len(nums),'numeric_accuracy':round(sum(nums)/len(nums),6) if nums else None,
            'text_cer':round(sum(x[0] for x in text)/max(1,sum(x[1] for x in text)),6) if text else None,
            'text_scored_fields':len(text),'text_quality_status':'OWNER_APPROVED' if gt.get('approved_by_user',False) else ('AGENT_VERIFIED' if agent_names else 'GT_DRAFT'),
            'cell_metrics':cell_metrics,'complete_numeric_receipt':all(nums) if nums else False}


def extract_rois(image_bgr,annotation):
    """Return ordered cells; numeric crops tighten and multi-line names split."""
    import numpy as np
    h,w=image_bgr.shape[:2]; out=[]
    for entry in annotation['rois']:
        field=entry['field_id'];x0,y0,x1,y1=entry['bbox']
        # Product price digits can touch the drafted bbox; widen by blank cell margin.
        if field.endswith(':unit_price'):
            x0-=.022;x1+=.005
        a=max(0,min(w-1,int(round(x0*w))));c=max(1,min(w,int(round(x1*w))))
        d=max(0,min(h-1,int(round(y0*h))));e=max(1,min(h,int(round(y1*h))))
        if a>=c or d>=e:raise ValueError(f"invalid bbox for {field}")
        crop=np.ascontiguousarray(image_bgr[d:e,a:c])
        if is_numeric_field(field):out.append((field,tighten_numeric_crop(crop)))
        elif is_name_field(field):out.extend((field,part) for part in split_name_lines(crop))
        else:out.append((field,crop))
    return out


def recognize_cells(cells, recognizer, mode='single_cell', batch_size=8):
    """Recognize and time every actual engine call. Mock works only in unit tests."""
    if mode not in ('single_cell','batch'):raise ValueError('unknown mode')
    if batch_size<1:raise ValueError('invalid batch size')
    texts={};blocks=[]
    t0=time.perf_counter_ns()
    blocks_in=[[c] for c in cells] if mode=='single_cell' else [cells[i:i+batch_size] for i in range(0,len(cells),batch_size)]
    for group in blocks_in:
        images=[c[1] for c in group]
        st=time.perf_counter_ns()
        res=recognizer.recognize_txt(images)
        elapsed=(time.perf_counter_ns()-st)/1e6
        otexts=getattr(res,'txts',None)
        scores=getattr(res,'scores',None)
        if otexts is None or len(otexts)!=len(group):
            raise RuntimeError(f"OCR returned {0 if otexts is None else len(otexts)} results instead of {len(group)}")
        for j,(field,_) in enumerate(group):
            texts.setdefault(field,[]).append(str(otexts[j]))
        blocks.append({'fields':[x[0] for x in group],'batch_size':len(group),'ocr_call_ms':round(elapsed,4),'kind':'cell' if mode=='single_cell' else 'batch'})
    loop_ms=(time.perf_counter_ns()-t0)/1e6
    return {'texts':{field:join_text_parts(parts) for field,parts in texts.items()},
            'calls':blocks,'ocr_calls':len(blocks),'processed_rois':len(cells),'processed_fields':len({f for f,_ in cells}),
            'sum_cell_ocr_ms':round(sum(c['ocr_call_ms'] for c in blocks),4) if mode=='single_cell' else None,
            'sum_batch_ocr_ms':round(sum(c['ocr_call_ms'] for c in blocks),4) if mode=='batch' else None,
            'ocr_loop_wall_ms':round(loop_ms,4)}
