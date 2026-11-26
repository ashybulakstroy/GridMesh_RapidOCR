"""V3: review selected result only; VAT evidence controls verification, not classification."""
import re
from decimal import Decimal
from .vat_marker_strict_v1 import vat_marker as _vat_marker_strict


def enabled():
    from ..config import setting
    return setting('OCR_TEXT_REVIEW_V3','1') == '1'


def vat_marker(text):
    return bool(_vat_marker_strict(text))


def plan(cells, rows, parse, rate):
    name=cells['name']; t=str(name.get('text') or '').strip()
    if vat_marker(t): return 'already_read'
    q,p= (parse(cells[k].get('text','')) for k in ('quantity','price'))
    hint=bool(re.match(r'^\s*(?:[нh][дd]|[дd][сc]|v[аa]|[нh]\s+[дd])(?=\s|\d|%|$)',t,re.I))
    short_pct=bool(re.fullmatch(r'[\W\d_%]+',t)) and '%' in t
    weak=not t or float(name.get('confidence') or 0)<.8 or len(t)<=5
    base=Decimal(0); complete=bool(rows)
    for row in rows:
        if row.get('kind')=='tax': continue
        v=row.get('line_total')
        if v is None: complete=False
        else: base+=Decimal(str(v))
    expected=base*Decimal(str(rate))/100
    close=complete and expected>0 and q is not None and abs(q-expected)<=max(Decimal('1'),expected*Decimal('.02'))
    unit=p==1 and q is not None and q>0
    # Torgsoft template: a VAT label occupies exactly one physical text line.
    blocks=name.get('blocks') or []
    strong_multiline=(len(blocks)>=2 and not weak and
                      all(float(b.get('confidence') or 0)>=.8 and
                          len(re.findall(r'[^\W\d_]',str(b.get('text') or '')))>=2
                          for b in blocks) and
                      len(re.findall(r'[^\W\d_]',t))>=8)
    if strong_multiline:
        # Do not reread a confirmed multiline product even if numeric fields
        # happen to resemble VAT. A contradiction is flagged, not classified.
        if hint or short_pct or (unit and close):
            return 'uncertain_no_retry'
        return 'skip_multiline_product'
    if hint or short_pct or (unit and (close or weak)):
        return 'verify'
    if weak and (p is None or p==1): return 'verify'
    if p is None or (unit and not complete): return 'uncertain_no_retry'
    return 'skip_product'


def verify(candidate, old, read_cell, read_full):
    from .text_debug import bands
    img=candidate.image; rid=candidate.row_id; x,y=candidate.offset
    if len(bands(img))>=2 or len(old.get('blocks') or [])>=2:
        # Detector route only: never feed confirmed multiline cells into single-line REC.
        lines=read_full(img,cell_mode=True)
        lines.sort(key=lambda l:(min(p[1] for p in l['bbox']),min(p[0] for p in l['bbox'])))
        for l in lines:
            l['bbox']=[[a+x,b+y] for a,b in l['bbox']]
            l.update(row_id=rid,column_id='name',cell_id=f'{rid}-name')
        return dict(old,text=' '.join(l['text'] for l in lines),confidence=min((l['confidence'] for l in lines),default=0),blocks=lines,model_id='v5_cyrillic_mobile',strategy='vat_gate_det_rec')
    return read_cell(img,rid,'name',candidate.offset,force_single=True)


def selected_reasons(item, image, events, threshold=.8, *, bands_fn):
    t=str(item.get('text') or '').strip(); flags=[]
    if not t: flags.append('empty_name')
    if float(item.get('confidence') or 0)<threshold: flags.append('low_confidence')
    if re.search(r'[-/–]$',t): flags.append('unfinished_suffix')
    latin_to_cyr=str.maketrans('ABCEHKMOPTXYaceopxy','АВСЕНКМОРТХУасеорху')
    cyr_to_latin=str.maketrans('АВСЕНКМОРТХУасеорху','ABCEHKMOPTXYaceopxy')
    def mixed(w):
        return (bool(re.search('[A-Za-z]',w) and re.search('[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]',w))
                and bool(re.search('[A-Za-z]',w.translate(latin_to_cyr)))
                and bool(re.search('[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]',w.translate(cyr_to_latin))))
    if any(mixed(w) for w in re.findall(r'[^\W\d_]+',t)): flags.append('mixed_script_word')
    b=bands_fn(image)
    if len(b)>=2 and len(item.get('blocks') or [])<len(b): flags.append('possible_missing_line')
    if len(b)>=2 and len(t)<12: flags.append('short_multiline_name')
    if item.get('vat_name_review'): flags.append('vat_name_uncertain')
    # Failed or low-confidence rejected candidates are evidence only, not quality flags.
    return flags,b
