from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import json

import pytest

from App.ocr.e4_adaptive_early import (adaptive_collect, eligible_first_result,
    make_early_snapshot, compare_early_final, rank_candidates)
from App.ocr.experiment_matrix import CandidateBook, solve_matrix
from App.ocr.experiment_vat import seed_printed_vat


def receipt():
    rows=[]
    for rid,kind,q,p in [(1,'product','2','100'),(2,'tax','32','1')]:
        rows.append({'row_id':rid,'kind':kind,'cells':{
            'name':{'text':'Товар' if kind=='product' else 'НДС 16 %'},
            'quantity':{'text':q,'confidence':.95,'model_id':'v6_tiny_official'},
            'price':{'text':p,'confidence':.96,'model_id':'v6_tiny_official'}},
            'line_total':str(Decimal(q)*Decimal(p)),'printed_line_match':None})
    return rows


def book_for(rows,total='232'):
    book=CandidateBook.from_receipt(rows,{}, {}, total)
    seed_printed_vat(rows,book)
    return book


def test_initial_real_matrix_needs_no_extra_ocr_and_snapshot_is_independent():
    rows=receipt();book=book_for(rows);published=[]
    def publish(matrix,b,scans):
        published.append(make_early_snapshot(rows,b,matrix,'232',16,100,scans,'a'*32))
    def forbidden(*args):raise AssertionError('Initial match must not scan another cell')
    out=adaptive_collect(rows,book,{(1,'price'):None},16,forbidden,publish)
    assert out['early_found'] and out['probes_used']==0 and out['matrix_checks']==1
    snapshot=published[0]
    assert snapshot['provisional'] and snapshot['read_only']
    snapshot['rows'][0]['cells']['price']['text']='999'
    assert rows[0]['cells']['price']['text']=='100'


@pytest.mark.parametrize('condition',['ambiguous','bounded','missing_total','missing_vat','incomplete'])
def test_real_matrix_rejects_unsafe_first_result(condition):
    rows=receipt();book=book_for(rows)
    if condition=='ambiguous':
        book.add(1,'quantity','4',.99,'other')
        book.add(1,'price','50',.99,'other')
    if condition=='missing_total':book.footer=[]
    if condition=='missing_vat':book.vat=[]
    matrix=solve_matrix(rows,book,16,max_nodes=1 if condition=='bounded' else 2500)
    if condition=='incomplete':matrix['solutions'][0]['assignments'][0].pop('price')
    assert not eligible_first_result(matrix,rows,book)


def test_current_price_candidate_is_preserved_and_stops_after_first_cell():
    rows=receipt();rows[0]['cells']['price']['text']='bad'
    book=book_for(rows);seen=[];published=[]
    def collect(crops):
        key=next(iter(crops));seen.append(key)
        return {'attempted':{key},'errors':[], 'observations':[
            {'row_id':1,'field':'price','text':'100T','confidence':.9},
            {'row_id':1,'field':'price','text':'100','confidence':.95,
             'path':'price_digits_candidate','model_id':'v5_cyrillic_mobile'}]}
    out=adaptive_collect(rows,book,{(2,'price'):None,(1,'price'):None},16,collect,
                         lambda *args:published.append(args))
    assert seen==[(1,'price')] and out['early_found'] and out['matrix_checks']==2
    assert len(out['collected']['observations'])==2
    assert any(o['path']=='price_digits_candidate' for o in book.cells['1:price']['observations'])


def test_equivalent_decimal_candidate_does_not_repeat_solver():
    rows=receipt();book=book_for(rows,'999')
    def collect(crops):
        return {'errors':[],'observations':[{'row_id':1,'field':'quantity','text':'2.0','confidence':.9}]}
    out=adaptive_collect(rows,book,{(1,'quantity'):None},16,collect,lambda *a:None)
    assert out['matrix_checks']==1 and out['probes_used']==1 and not out['early_found']


def test_new_unit_price_candidate_unlocks_observed_vat():
    rows=receipt();rows[1]['cells']['price']['text']='bad'
    book=book_for(rows);published=[]
    def collect(crops):
        return {'errors':[], 'observations':[{'row_id':2,'field':'price',
                'text':'1','confidence':.95,'model_id':'v5_cyrillic_mobile'}]}
    out=adaptive_collect(rows,book,{(2,'price'):None},16,collect,
                         lambda *args:published.append(args))
    assert out['early_found'] and len(published)==1
    assert [x['number'] for x in book.vat_values()]==['32']


def test_ocr_failure_continues_other_cells():
    rows=receipt();book=book_for(rows,'999');seen=[]
    def collect(crops):
        seen.extend(crops)
        if len(seen)==1:raise RuntimeError('OCR failed')
        return {'observations':[],'errors':[]}
    out=adaptive_collect(rows,book,{(1,'price'):None,(2,'price'):None},16,collect,lambda *a:None)
    assert len(seen)==2 and len(out['collected']['errors'])==1
    assert not out['early_found']


def test_publish_failure_is_recorded_without_claiming_early_ready():
    rows=receipt();book=book_for(rows)
    def fail(*a):raise OSError('disk full')
    out=adaptive_collect(rows,book,{},16,lambda *a:None,fail)
    assert not out['early_found'] and 'disk full' in out['collected']['errors'][0]['error']


def test_priority_invalid_before_row_mismatch_before_low_confidence():
    rows=receipt();rows[0]['cells']['price']['text']='bad'
    rows[1]['printed_line_match']=False
    rows[0]['cells']['quantity']['confidence']=.01
    assert rank_candidates(rows,{(1,'quantity'):None,(2,'quantity'):None,(1,'price'):None})==[
        (1,'price'),(2,'quantity'),(1,'quantity')]


def test_comparison_separates_formatting_changes_and_final_ambiguity():
    rows=receipt();book=book_for(rows);matrix=solve_matrix(rows,book,16)
    early=make_early_snapshot(rows,book,matrix,'232',16,100,0,'a'*32)
    final={'rows':deepcopy(rows),'total':'232.00','printed_vat':'32.00',
           'timings':{'total_s':.5},'numeric_matrix':{'status':'ambiguous','review_required':True}}
    final['rows'][0]['cells']['price']['text']='100T'
    diff=compare_early_final(early,final)
    assert diff['changed_count']==0 and len(diff['formatting_changes'])==3
    assert diff['final_review_required'] and diff['final_matrix_status']=='ambiguous'
    final['rows'][0]['cells']['quantity']['text']='3'
    diff=compare_early_final(early,final)
    assert diff['changed_fields'][0]['field']=='quantity' and diff['changed_count']==1


def test_download_early_requires_known_job_and_ready_file(tmp_path,monkeypatch):
    import App.web_server as server
    from fastapi import HTTPException
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr(server,'setting',lambda name,default:default)
    monkeypatch.setattr(server,'jobs',{})
    with pytest.raises(HTTPException) as error:server.download_early_result('a'*32)
    assert error.value.status_code==404
    server.jobs['a'*32]={'status':'running'}
    with pytest.raises(HTTPException):server.download_early_result('a'*32)
    path=tmp_path/'data/output'/('a'*32)/'e4_early_result.json'
    path.parent.mkdir(parents=True);path.write_text('{}',encoding='utf-8')
    assert server.download_early_result('a'*32).path==path


def test_real_adaptive_final_preserves_current_candidates_and_immutable_preview(monkeypatch):
    import App.pipeline as pipeline
    # This regression exercises the full E4 sweep, not the optional fast path.
    monkeypatch.setenv('OCR_E4_FAST_GATE','0')
    monkeypatch.setenv('OCR_E4_TARGETED_FIRST','0')
    monkeypatch.setenv('OCR_E4_ADAPTIVE','0')
    data=(Path(__file__).parent/'fixtures/receipt_253309.jpg').read_bytes()
    baseline=pipeline.process(data)
    monkeypatch.setenv('OCR_E4_ADAPTIVE','1')
    calls=[];collector=pipeline.collect_v5_candidates;published=[]
    def collect(crops):
        calls.extend(crops)
        return collector(crops)
    monkeypatch.setattr(pipeline,'collect_v5_candidates',collect)
    def publish(progress):
        if progress.get('e4_early_ready'):
            snapshot=progress['e4_first_result']
            path=pipeline.ROOT/'data/output'/snapshot['job_id']/'e4_early_result.json'
            published.append((path,path.read_bytes()))
    result=pipeline.process(data,publish)
    assert result['e4_adaptive']['first_ready'] and len(published)==1
    assert result['e4_early_comparison']['changed_count']==0
    assert published[0][0].read_bytes()==published[0][1]
    assert result['math_match'] and result['computed_total']==baseline['computed_total']
    assert result['numeric_matrix']['status']==baseline['numeric_matrix']['status']
    assert len(calls)==len(set(calls))
    def values(r):
        return [(x['row_id'],f,x['cells'][f]['text']) for x in r['rows'] for f in ('name','quantity','price')]
    assert values(result)==values(baseline)
    def evidence(r):
        return {k:{(o['text'],o['path'],o['model']) for o in b['observations']}
                for k,b in r['numeric_candidate_book']['cells'].items()}
    assert evidence(result)==evidence(baseline)
    assert result['e4_adaptive']['first_ready'] and len(published)==1
    assert published[0][0].read_bytes()==published[0][1]
    assert result['e4_early_comparison']['changed_count']==0
    perf=json.loads((pipeline.ROOT/result['performance_file']).read_text(encoding='utf-8'))
    assert perf['e4_adaptive']['first_elapsed_ms']<perf['wall_ms']
    assert result['geometry']['ocr_batch']['parallel_lanes']==3
