"""Experimental pixel hypotheses; shadow E4 only. No new OCR calls or files."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import re
import cv2
import numpy as np
from .previous_hybrid import Bank
from .methods import rank

METHODS=('masked_template','partial_chamfer')
EXTENDED=METHODS+('partial_hausdorff',)
FIELDS=('price','quantity','printed_total')

@lru_cache(maxsize=1)
def load_bank():
    base=Path(__file__).with_name('partial_digit_assets')/'templates'
    samples=[]
    for digit in '0123456789':
        image=cv2.imdecode(np.fromfile(base/f'template_{digit}.png',dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError('template missing: '+digit)
        samples.append((digit,image))
    bank=Bank(samples)
    if bank.coverage!='0123456789':raise ValueError('incomplete template bank')
    return bank


def segment(image, raw):
    """Pitch-guided experimental slots; reject grid residue / multiline / unknown length.
    Decimal-like OCR may yield a second integer hypothesis ONLY from visible ink slots.
    No automatic correction of separators; original OCR remains in the book.
    """
    if image is None or image.size==0:return []
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    if g.dtype!=np.uint8 or min(g.shape)<8:return []
    _,b=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    h,w=b.shape
    # suppress narrow edge-connected rules for analysis only
    n,labels,stats,_=cv2.connectedComponentsWithStats(b,8)
    for i in range(1,n):
        x,y,ww,hh,a=map(int,stats[i]); edge=x<=2 or x+ww>=w-2 or y<=2 or y+hh>=h-2
        if edge and ((hh>=.75*h and ww<=max(3,.08*w)) or (ww>=.8*w and hh<=max(3,.08*h))):b[labels==i]=0
    # Disconnected fragments of a grid rule can remain along the cell edge.
    # Ignore edge fragments only when they are entirely outside a separate
    # interior ink band; keep all components touching the band itself.
    interior=[tuple(map(int,stats[i])) for i in range(1,n)
              if stats[i,0]>2 and stats[i,0]+stats[i,2]<w-2
              and stats[i,1]>2 and stats[i,1]+stats[i,3]<h-2 and stats[i,4]>=10]
    if interior:
        top=min(y for x,y,ww,hh,a in interior)
        bottom=max(y+hh for x,y,ww,hh,a in interior)
        for i in range(1,n):
            x,y,ww,hh,a=map(int,stats[i])
            if (x<=2 or x+ww>=w-2 or y<=2 or y+hh>=h-2) and (y+hh<=top or y>=bottom):
                b[labels==i]=0
    ys,xs=np.where(b>0)
    if len(xs)<12:return []
    x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
    hh=y1-y0;ww=x1-x0
    if hh<7 or ww>hh*9:return []
    ink=b[y0:y1,x0:x1]>0
    row=ink.sum(1); good=row>=max(1,ww*.04)
    gaps=np.flatnonzero(~good)
    for a in np.split(gaps,np.where(np.diff(gaps)>1)[0]+1):
        if len(a)>=max(3,hh*.15) and a[0]>hh*.2 and a[-1]<hh*.8:return []
    text=re.sub(r'\s*(?:₸|[ТTтt])\s*$','',str(raw or '').strip())
    text=re.sub(r'\s+','',text)
    hints=re.sub(r'[.,]','',text)
    if not hints or not re.fullmatch(r'[0-9A-Za-zА-Яа-яЁё:|]+',hints):return []
    # Pitch is inferred from image height, not arithmetic. Try plausible count(s) near raw length.
    counts=[k for k in {len(hints),len(hints)-1,len(hints)+1} if 1<=k<=8 and .38<=ww/(hh*k)<=1.13]
    if not counts:return []
    den=ink.sum(0)/hh; results=[]
    for k in sorted(counts,key=lambda k:(abs(ww/(hh*k)-.76),abs(k-len(hints))))[:2]:
        pitch=ww/k; cuts=[0]; cost=0.
        for j in range(1,k):
            target=j*pitch; lo=max(cuts[-1]+max(2,int(.50*pitch)),int(target-.24*pitch)); hi=min(ww-2,int(target+.24*pitch)+1)
            if lo>hi:break
            cut=min(range(lo,hi+1),key=lambda x:float((den[x-1]+den[x])/2)+.17*abs(x-target)/pitch)
            cost+=float((den[cut-1]+den[cut])/2);cuts.append(cut)
        if len(cuts)!=k:continue
        cuts.append(ww)
        if cost>max(.28,.20*(k-1)):continue
        slots=[]
        for a,z in zip(cuts,cuts[1:]):
            sub=ink[:,a:z]
            if sub.sum()<max(4,hh*.35) or np.count_nonzero(sub.any(1))<.45*hh:break
            # Retain the entire slot (including missing pixels); tiny white rim via source crop.
            slots.append(g[max(0,y0-1):min(h,y1+1),x0+a:x0+z].copy())
        if len(slots)==k:results.append((slots,hints if len(hints)==k else '',cuts))
    return results


def candidates(image,raw,bank,*,deadline,limit=12,methods=METHODS):
    merged={}; details=[]; stopped=False
    for slots,hints,cuts in segment(image,raw):
        per=[]
        for slot in slots:
            ranks={}
            for method in methods:
                if perf_counter()>=deadline:stopped=True;break
                ranks[method]=rank(method,slot,bank,discover=True,top_k=10)
            if stopped:break
            votes={}
            for method,opts in ranks.items():
                for i,o in enumerate(opts,1): votes[o['digit']]=votes.get(o['digit'],0)+1/(10+i)
            if not votes:break
            per.append((sorted(votes,key=lambda d:(-votes[d],d)),ranks,votes))
        if len(per)!=len(slots):continue
        # Two modes: free beam + single-position substitutions on existing OCR/optical anchors.
        states=[('',0.)]
        for opts,_,votes in per:
            states=sorted([(s+d,v+votes[d]) for s,v in states for d in opts[:5]],key=lambda a:(-a[1],a[0]))[:limit]
        ranked={s:(v,'free_beam') for s,v in states}
        if hints:
            # Only one symbol can be treated as uncertain in a conditional hypothesis.
            unknown=[i for i,c in enumerate(hints) if c not in '0123456789']
            if len(unknown)<=1:
                indices=unknown or sorted(range(len(hints)),key=lambda i:per[i][2].get(hints[i],0))[:1]
                for i in indices:
                    for d in per[i][0]:
                        value=hints[:i]+d+hints[i+1:]
                        if value.isascii() and value.isdigit():
                            v=sum(p[2].get(c,0) for c,p in zip(value,per));ranked[value]=(v,'conditional_ocr_anchors')
        # reserve up to 10 slots for single-position hypothesis list; then free beam.
        conditional=sorted([(s,v,kind) for s,(v,kind) in ranked.items() if kind.startswith('conditional')],key=lambda t:(-t[1],t[0]))
        free=sorted([(s,v,kind) for s,(v,kind) in ranked.items() if kind=='free_beam'],key=lambda t:(-t[1],t[0]))
        for s,v,kind in conditional[:10]+free[:max(0,limit-len(conditional[:10]))]:
            old=merged.get(s)
            if old is None or v/len(slots)>old['rank_score']:merged[s]={'text':s,'rank_score':v/len(slots),'scope':kind}
        details.append({'cuts':cuts,'ocr_anchor':hints,'glyph_methods':[p[1] for p in per]})
    ordered=sorted(merged.values(),key=lambda o:(o['scope']!='conditional_ocr_anchors',-o['rank_score'],o['text']))
    return {'candidates':ordered[:limit],'segments':details,'budget_exhausted':stopped,'pool_truncated':len(ordered)>limit}


def trial(rows,crops,book,vat_rate,solve_matrix,parse_number,prepare_numeric_input,*,budget_ms=180,max_cells=3,limit=12,max_nodes=100000,methods=METHODS):
    start=perf_counter(); deadline=start+budget_ms/1000
    result={'status':'not_needed','review_only':True,'cells':[],'wall_ms':0,'candidate_scope':'bounded_pixel_hypotheses_not_optical_ground_truth'}
    def done():result['wall_ms']=round((perf_counter()-start)*1000,3);return result
    shadow=deepcopy(book);shadow.partial_trial_limit=16
    targets=[]
    for row in rows:
        if row.get('kind')=='tax':continue # v1 excludes tax cells / independent VAT witness
        for field in FIELDS:
            key=(row['row_id'],field);cell=row.get('cells',{}).get(field)
            if cell is None or key not in crops or cell.get('user_verified') or shadow.cells.get(shadow.key(*key),{}).get('user_locked'):continue
            val=parse_number(cell.get('text'));text=str(cell.get('text') or '')
            try:conf=float(cell.get('confidence') or 0)
            except (ValueError,TypeError):conf=0
            # Prefer malformed prices, then decimal-looking prices on mismatch; no change to clean routine.
            priority=0 if val is None else 1 if field=='price' and any(c in text for c in '.,') else 2 if conf<.8 or row.get('printed_line_match') is False else 3
            targets.append((priority,field!='price',conf,key,text))
    targets.sort(); added=0
    if not targets:return done()
    bank=load_bank()
    for _,_,_,key,text in targets[:max_cells]:
        if perf_counter()>=deadline:result['budget_exhausted']=True;break
        image,meta=prepare_numeric_input(crops[key],key[1])
        # Do not guess which glyph is currency when pre-OCR crop could not establish it.
        if key[1]=='price' and not meta.get('currency_removed_before_ocr'):
            result['cells'].append({'row_id':key[0],'field':key[1],'status':'currency_not_confirmed'});continue
        c=candidates(image,text,bank,deadline=deadline,limit=limit,methods=methods)
        c.update(row_id=key[0],field=key[1],original=text);result['cells'].append(c)
        for item in c['candidates']:
            if parse_number(item['text']) is None:continue
            path='partial_digit_trial_'+item['scope']
            shadow.add(*key,item['text'],None,path,model='partial_digit_image_hypothesis')
            bucket=shadow.cells[shadow.key(*key)]
            for obs in bucket['observations']:
                if obs.get('path')==path and obs.get('text')==item['text']:
                    obs['authority']='partial_image_hypothesis';obs['note']='review only; rank score is not confidence'
            added+=1
    result['added_hypotheses']=added
    if added:
        result['matrix']=solve_matrix(rows,shadow,vat_rate,max_nodes=max_nodes,per_row_cap=256)
        result['status']='review_candidates';result['matrix']['review_required']=True
        # The solver cannot see hypotheses discarded by the image generator.
        # A single surviving solution therefore does not prove uniqueness.
        incomplete=bool(result.get('budget_exhausted') or any(
            cell.get('pool_truncated') or cell.get('budget_exhausted') for cell in result['cells']))
        if incomplete:
            result['matrix']['candidate_pool_truncated']=True
            result['matrix']['search_truncated']=True
            result['matrix']['status']='bounded_search'
            result['matrix']['note']='Список гипотез неполный; решения требуют ручной проверки.'
    else:result['status']='no_candidates'
    return done()
