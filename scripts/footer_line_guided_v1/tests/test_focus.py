"""Local offline tests. OCR accuracy/timing must be measured on Windows with models."""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import sys
from decimal import Decimal
import numpy as np
import cv2
import pytest

HERE=Path(__file__).resolve().parents[1]
MODULE=HERE/'files'/'App'/'ocr'/'footer_line_guided_v1.py'
spec=importlib.util.spec_from_file_location('footer_focus_experiment',MODULE)
focus=importlib.util.module_from_spec(spec)
spec.loader.exec_module(focus)
PHOTO=HERE/'fixtures'/'receipt_90665.png'
GUIDE={'price_left':401,'price_right':485,'grid_bottom_y':715,'last_row_height':31}


def parse_number(text):
    from decimal import Decimal
    import re
    value=re.sub(r'\s*(?:[₸ТT])\s*$', '', str(text)).strip().replace(' ','').replace(',','.')
    try:return Decimal(value)
    except:return None


def test_real_receipt_roi_excludes_tax_row_and_contains_printed_total():
    image=cv2.imread(str(PHOTO))
    assert image is not None and image.shape[:2]==(1280,720)
    crop,bbox,diag=focus.isolate_footer(image,GUIDE)
    assert diag['status']=='isolated'
    assert 380<=bbox[0]<=410 and bbox[2]>480
    assert 714<=bbox[1]<=722 and 731<=bbox[3]<=739
    assert 9<=crop.shape[0]<=24  # actual photographed footer line, not tax
    assert len(diag['selected_separators_y'])==2
    assert diag['selected_separators_y'][0]>700


def test_observed_grid_anchor_both_ends_of_last_price_column():
    class Rule:
        def __init__(self,v):self.position=v
        def eval(self,x):return self.position
    class Rect:
        info={'source_grid':{'vertical':[Rule(v) for v in (120,163,348,401,485)],
                             'horizontal':[Rule(v) for v in (480,530,590,685,715)]}}
    g=focus.guide_from_grid(Rect(),(1280,720,3))
    assert g['price_left']==401 and g['price_right']==485
    assert g['grid_bottom_y']==715 and g['last_row_height']==30


def test_one_ocr_call_when_isolated_and_confident():
    image=cv2.imread(str(PHOTO))
    calls=[]
    def read(crop):
        calls.append(crop.copy());return {'text':'90 665,00 ₸','confidence':.934}
    r=focus.attempt_focused_footer(image,GUIDE,read,parse_number)
    assert len(calls)==1 and r['accepted']
    assert Decimal(r['total'])==Decimal(90665)
    assert r['attempt']['ocr_invocations']==1
    assert r['attempt']['strategy']=='focused_line_guided_v1'
    assert r['attempt']['source_roi_bbox'][1]>710


def test_unverified_ocr_falls_through():
    image=cv2.imread(str(PHOTO))
    calls=[]
    r=focus.attempt_focused_footer(image,GUIDE,lambda crop:(calls.append(1) or {'text':'66500','confidence':.31}),parse_number)
    assert not r['accepted'] and len(calls)==1
    assert r['attempt']['status']=='low_confidence_or_invalid'
    assert r['attempt']['total'] is None  # rejected result must NOT enter the E4 candidate pool
    assert r['attempt']['unaccepted_numeric_candidate']=='66500'


def test_wrong_pattern_refused_even_with_high_confidence():
    image=cv2.imread(str(PHOTO))
    r=focus.attempt_focused_footer(image,GUIDE,lambda crop:{'text':'12505 1 ₸ 90665','confidence':.999},parse_number)
    assert not r['accepted']


def test_no_observed_bounding_lines_means_zero_ocr_calls():
    image=np.full((280,320,3),255,np.uint8)
    guide={'price_left':185,'price_right':290,'grid_bottom_y':210,'last_row_height':40}
    called=[]
    r=focus.attempt_focused_footer(image,guide,lambda crop:called.append(1),parse_number)
    assert not r['accepted'] and not called and not r['attempt']
    assert r['diagnostics']['status']=='no_isolated_footer_band'


def test_recovered_source_not_guessed_and_invalid_fallback():
    class Rect:info={}
    assert focus.guide_from_grid(Rect(),(100,100,3)) is None
    assert focus.isolate_footer(np.ones((100,100),np.uint8),None)[0] is None


def test_crop_respects_image_safety():
    image=cv2.imread(str(PHOTO))
    for y in (714,715):
        guide={**GUIDE,'grid_bottom_y':y}
        crop,bbox,diag=focus.isolate_footer(image,guide)
        assert crop is not None and diag['status']=='isolated'
        assert bbox[1]>=715 and bbox[3]<=740
    # A boundary at the bottom of this band cannot prove it is BELOW the
    # item grid; decline the fast path instead of risking a VAT-row crop.
    assert focus.isolate_footer(image,{**GUIDE,'grid_bottom_y':736})[0] is None


def test_patch_matches_static_python_markers_and_is_idempotent():
    sys.path.insert(0,str(HERE))
    import patch
    sample='''from .ocr.experiment_footer import scan_footer

def part(rect,image):
    xs = [int(round(x)) for x in rect.xs]; ys = [int(round(y)) for y in rect.ys]
    footer_result=read_printed_footer(image, table_source_bottom(rect,xs,ys))
    y_bottom=3
    rows=[]
    footer_result=read_printed_footer(image, y_bottom, rows)
    return {'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts, 'timings': {}}

def read_printed_footer(image, y_bottom, rows=()):
    footer_top = max(0,y_bottom-20)
    footer_attempts = []
    total, footer_lines = None, []
    return {'total':total, 'lines':footer_lines, 'attempts':footer_attempts,
            'y_bottom':y_bottom}
'''
    patched=patch.patch_source(sample)
    ast.parse(patched)
    assert patch.MAGIC in patched
    assert 'footer_guide=footer_focus_guide' in patched
    assert 'footer_focus' in patched
    with pytest.raises(RuntimeError,match='уже установлен'):
        patch.patch_source(patched)
    with pytest.raises(RuntimeError,match='одно вхождение'):
        patch.patch_source(sample.replace('from .ocr.experiment_footer import scan_footer\n',''))


def test_installer_restores_exact_pipeline_bytes(tmp_path,monkeypatch):
    sys.path.insert(0,str(HERE))
    import patch
    # A minimal structurally valid stand-in for the precise GitHub code anchors.
    baseline='''from .ocr.experiment_footer import scan_footer

def run(rect,image):
    xs = [int(round(x)) for x in rect.xs]; ys = [int(round(y)) for y in rect.ys]
    footer_result=read_printed_footer(image, table_source_bottom(rect,xs,ys))
    y_bottom=10
    rows=[]
    footer_result=read_printed_footer(image, y_bottom, rows)
    return {'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts, 'timings': {}}

def read_printed_footer(image, y_bottom, rows=()):
    footer_top = max(0,y_bottom-20)
    footer_attempts = []
    total, footer_lines = None, []
    return {'total':total, 'lines':footer_lines, 'attempts':footer_attempts,
            'y_bottom':y_bottom}
'''.encode('utf-8')
    project=tmp_path/'project';(project/'App/ocr').mkdir(parents=True)
    pipeline=project/'App/pipeline.py';pipeline.write_bytes(baseline)
    module=project/'App/ocr/footer_line_guided_v1.py'
    monkeypatch.setattr(patch,'PIPELINE',pipeline)
    monkeypatch.setattr(patch,'TARGET',module)
    monkeypatch.setattr(patch,'BACKUPS',project/'backups')
    patch.install()
    assert patch.MAGIC in pipeline.read_text()
    assert module.exists() and list((project/'backups').glob('*/manifest.json'))
    patch.restore()
    assert pipeline.read_bytes()==baseline
    assert not module.exists()


def test_restore_refuses_to_overwrite_user_edits(tmp_path,monkeypatch):
    sys.path.insert(0,str(HERE))
    import patch
    root=tmp_path/'project';(root/'App/ocr').mkdir(parents=True)
    code='''from .ocr.experiment_footer import scan_footer

def example(rect,image):
    xs = [int(round(x)) for x in rect.xs]; ys = [int(round(y)) for y in rect.ys]
    read_printed_footer(image, table_source_bottom(rect,xs,ys))
    read_printed_footer(image, y_bottom, rows)
    return {'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts, 'timings': 0}

def read_printed_footer(image, y_bottom, rows=()):
    footer_top = max(0,y_bottom-20)
    footer_attempts = []
    total, footer_lines = None, []
    return {'total':total, 'lines':footer_lines, 'attempts':footer_attempts,
            'y_bottom':y_bottom}
'''
    pipeline=root/'App/pipeline.py';pipeline.write_text(code,encoding='utf-8')
    monkeypatch.setattr(patch,'PIPELINE',pipeline)
    monkeypatch.setattr(patch,'TARGET',root/'App/ocr/footer_line_guided_v1.py')
    monkeypatch.setattr(patch,'BACKUPS',root/'backups')
    patch.install()
    pipeline.write_text(pipeline.read_text()+'\n# manual developer modification\n',encoding='utf-8')
    with pytest.raises(RuntimeError,match='Код изменён после установки'):
        patch.restore()
    assert 'manual developer modification' in pipeline.read_text()
