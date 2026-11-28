import pytest
from App.ocr.experiment_auto import auto_apply_consensus
from App.ocr.experiment_matrix import CandidateBook


def example():
    row={'row_id':1,'kind':'product','cells':{
        'quantity':{'text':'1','confidence':.99},
        'price':{'text':'1700 "','confidence':.93,'model_id':'v6_tiny_cyrillic_optional'},
        'printed_total':{'text':'1700','confidence':.99}}}
    book=CandidateBook()
    book.add(1,'price','1700',.99,'v5_numeric_candidate',model='v5_cyrillic_mobile')
    return row,book


def test_row_can_be_fixed_without_claiming_complete_receipt_solution():
    row,book=example()
    report=auto_apply_consensus([row],book,{'status':'insufficient_candidates','review_required':True},'18650')
    assert row['cells']['price']['text']=='1700'
    assert row['cells']['price']['model_id']=='v5_cyrillic_mobile'
    assert row['printed_line_match'] and row['line_total']=='1700'
    assert report['status']=='partial_row_repair' and report['review_required']
    assert report['printed_total']=='18650'
    assert report['changes'][0]['ocr_evidence'][0]['text']=='1700'
    assert book.cells['1:price']['observations'][0]['path']=='v5_numeric_candidate'


@pytest.mark.parametrize('reason',['valid_reading','wrong_total','weak_quantity','weak_total','weak_candidate','same_model','user_locked','book_lock','missing_total','no_candidate','ambiguous_candidates','disabled'])
def test_row_repair_requires_independent_and_unambiguous_evidence(reason):
    row,book=example();cell=row['cells']['price']
    if reason=='valid_reading':cell['text']='1900'
    elif reason=='wrong_total':row['cells']['printed_total']['text']='1800'
    elif reason=='weak_quantity':row['cells']['quantity']['confidence']=.5
    elif reason=='weak_total':row['cells']['printed_total']['confidence']=.5
    elif reason=='weak_candidate':book.cells['1:price']['observations'][0]['confidence']=.5
    elif reason=='same_model':cell['model_id']='v5_cyrillic_mobile'
    elif reason=='user_locked':cell['user_verified']=True
    elif reason=='book_lock':book.cells['1:price']['user_locked']=True
    elif reason=='missing_total':row['cells'].pop('printed_total')
    elif reason=='no_candidate':book.cells['1:price']['observations'].clear()
    elif reason=='ambiguous_candidates':book.add(1,'price','1700.005',.99,'other',model='v5_cyrillic_mobile')
    before=cell['text']
    report=auto_apply_consensus([row],book,{'status':'no_solution','review_required':True},'18650',enabled=reason!='disabled')
    assert cell['text']==before and not report['changes']
