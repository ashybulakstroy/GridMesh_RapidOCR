"""V3: the printed VAT is an observed OCR figure, not a number invented by tax maths."""
from decimal import Decimal
from types import SimpleNamespace
import copy
import numpy as np

from App.ocr.experiment_matrix import CandidateBook,solve_matrix
from App.ocr.experiment_vat import scan_printed_vat,seed_printed_vat,vat_source
from App.ocr.experiment_manual import apply_user_receipt


def row(rid,kind,quantity,price,printed=None):
    cells={field:{'text':str(value),'confidence':.95,'row_id':rid,'column_id':field}
           for field,value in (('quantity',quantity),('price',price))}
    if printed is not None:
        cells['printed_total']={'text':str(printed),'confidence':.95,'row_id':rid,'column_id':'printed_total'}
    return {'row_id':rid,'kind':kind,'cells':cells,
            'line_total':str(Decimal(str(quantity))*Decimal(str(price))),
            'printed_line_match':None}


def book(rows,footer):
    out=CandidateBook()
    for r in rows:
        for k,c in r['cells'].items():
            out.add(r['row_id'],k,c['text'],c['confidence'],'baseline')
    out.add_footer(footer,.95,'footer_baseline')
    seed_printed_vat(rows,out)
    return out


def test_independent_vat_reading_is_seeded_only_from_printed_amount():
    rows=[row(1,'product',1,1000),row(2,'tax',1,160)]
    b=book(rows,'1160')
    assert b.vat_values()[0]['number']=='160'
    assert b.vat[0]['path']=='vat_row_primary_price'
    result=solve_matrix(rows,b,16)
    assert result['status']=='unique_math_match'
    assert result['printed_vat_observed'] is True
    assert result['solutions'][0]['vat_crosscheck']=='direct_ocr_match'
    assert result['review_required'] is False


def test_no_vat_value_may_be_invented_from_multiplication():
    rows=[row(1,'product',1,1000),row(2,'tax',2,80)]
    b=book(rows,'1160')
    assert vat_source(rows)[0] is None
    assert b.vat==[]
    result=solve_matrix(rows,b,16)
    assert result['status']=='unique_math_match'
    assert result['printed_vat_observed'] is False
    assert result['review_required'] is False
    assert result['solutions'][0]['vat_crosscheck']=='calculated_only'


def test_vat_variant_filters_out_otherwise_math_consistent_rows():
    rows=[row(1,'product',1,1000),row(2,'tax',1,160)]
    b=book(rows,'1160')
    b.add(1,'quantity','2',.98,'extra_product_qty')
    b.add(2,'price','320',.95,'tax_extra')
    b.add_footer('2320',.97,'footer_extra')
    result=solve_matrix(rows,b,16)
    # The 2320 result is arithmetically valid at VAT 16%, but independent
    # OCR of printed VAT only saw 160, so the 320 scenario must be rejected.
    assert result['status']=='unique_math_match'
    assert result['solutions'][0]['footer']=='1160'
    assert result['vat_variants']==['160']


def test_multi_axis_ambiguity_is_critical_even_when_every_equation_passes():
    rows=[row(1,'product',1,1000),row(2,'tax',1,160)]
    b=book(rows,'1160')
    b.add(1,'quantity','2',.95,'model_qty')
    b.add(2,'price','320',.95,'model_tax')
    b.add_footer('2320',.95,'model_total')
    b.add_vat('320',.95,'model_vat')
    result=solve_matrix(rows,b,16)
    assert result['status']=='ambiguous'
    assert result['critical_ambiguity'] is True
    assert result['risk_level']=='review'
    assert len(result['solutions'])==2
    assert {'footer','vat_printed','1:quantity','2:price'} <= set(result['ambiguity_dimensions'])
    assert result['recommended_changes']==[]
    assert result['model_assessments']


def test_unique_solution_resolves_vat_readings():
    rows=[row(1,'product',1,1000),row(2,'tax',1,160)]
    b=book(rows,'1160')
    b.add_vat('180',.97,'second_model')
    res=solve_matrix(rows,b,16)
    assert res['status']=='unique_math_match'
    assert res['vat_ambiguous'] is True
    assert res['review_required'] is False


def test_original_vat_scanner_emits_raw_and_model_observations():
    rows=[row(1,'product',1,1000),row(2,'tax',1,160)]
    b=book(rows,'1160')
    cache=SimpleNamespace(crops={(2,'price'):np.full((24,110,3),255,dtype=np.uint8)})
    def read_line(_crop):return {'text':'160','confidence':.94}
    class FakePool:
        def models_for(self,task):
            assert task=='vat'
            return ['v6_small_official']
        def recognize(self,model_id,im,stage):
            assert stage=='printed_vat_original_roi'
            return {'text':'160','confidence':.93,'model_id':model_id,'model':'v6','wall_ms':3}
    res=scan_printed_vat(rows,cache,b,16,read_line=read_line,model_pool=FakePool())
    assert res['source_status']=='rescanned'
    assert res['probes']==2
    assert {'vat_original_raw_rec','vat_original_model_v6_small_official'} <= {o['strategy'] for o in res['observations']}
    assert len({o['model'] for o in b.vat})>=2


def test_user_printed_vat_overrides_ocr_but_conflict_stays_review():
    rows=[row(1,'product',1,1000),row(2,'tax',1,160)]
    b=book(rows,'1160')
    base={'rows':rows,'total':'1160','printed_vat':'160','vat':{'rate_percent':'16'},
          'vat_evidence':{'source':{'row_id':2,'field':'price'}},
          'numeric_candidate_book':b.export()}
    updated=copy.deepcopy(base)
    updated['printed_vat']='170'
    updated['user_vat_verified']=True
    result=apply_user_receipt(base,updated)
    assert result['printed_vat']=='170'
    assert result['numeric_matrix']['status']=='no_solution'
    assert result['status']=='review'
    assert [x['number'] for x in result['numeric_candidate_book']['vat'] if x['authority']=='user_confirmed']==['170']
    assert any(x['field']=='printed_vat' and x['authority']=='user_confirmed'
               for x in result['numeric_user_truth'])
    assert any(x['field']=='printed_vat' and x['assessment']=='user_confirmed_conflict'
               for x in result['numeric_model_feedback'])


def test_user_can_confirm_printed_vat_matching_original_without_conflict():
    rows=[row(1,'product',1,1000),row(2,'tax',1,160)]
    b=book(rows,'1160')
    base={'rows':rows,'total':'1160','printed_vat':'160','vat':{'rate_percent':'16'},
          'numeric_candidate_book':b.export()}
    edited=copy.deepcopy(base)
    edited['user_vat_verified']=True
    result=apply_user_receipt(base,edited)
    assert result['vat']['printed_match'] is True
    assert result['numeric_matrix']['status']=='unique_math_match'
    assert result['numeric_matrix']['review_required'] is False


def test_vat_external_models_are_not_created_until_called():
    from App.ocr.experiment_models import ExperimentModelPool
    catalog={'v6_small_official':{'id':'v6_small_official','enabled':True,
        'name':'v6 small','ocr_version':'PPOCRV6','model_type':'SMALL','lang':'en'}}
    conf=lambda k,d:{'OCR_E2_ENABLE':'1','OCR_E2_NUMERIC_MODELS':'',
                     'OCR_E2_FOOTER_MODELS':'','OCR_E2_VAT_MODELS':'v6_small_official'}.get(k,d)
    pool=ExperimentModelPool(read=conf,catalog=catalog)
    assert pool.models_for('vat')==['v6_small_official']
    assert pool.calls==0


def test_wrong_operator_vat_is_not_ocr_failure():
    # Paper shows a deliberately incorrect 15% tax, but every photographed
    # number agrees with the printed total. OCR must not fabricate 16%.
    rows=[row(1,'product',1,10000),row(2,'tax',1,1500)]
    b=book(rows,'11500')
    result=solve_matrix(rows,b,16)
    assert result['status']=='unique_math_match'
    assert result['review_required'] is False
    assert result['solutions'][0]['vat_total']=='1500'
    assert result['solutions'][0]['vat_rate_expected']=='1600'
    assert result['vat_rate_diagnostic']['status']=='operator_calculation_anomaly'
    assert result['vat_rate_diagnostic']['classification']=='financial_diagnostic_not_ocr_failure'

def test_operator_vat_error_does_not_trigger_extra_scans_by_itself():
    rows=[row(1,'product',1,10000),row(2,'tax',1,1500)]
    b=book(rows,'11500')
    cache=SimpleNamespace(crops={(2,'price'):np.full((24,110,3),255,dtype=np.uint8)})
    result=scan_printed_vat(rows,cache,b,16,read_line=lambda crop:
        (_ for _ in ()).throw(AssertionError('unexpected OCR')),always_check=False)
    assert result['probes']==0
    assert result['source_status']=='primary_consistent_not_independently_rescanned'

def test_multiple_ocr_consistent_combinations_are_critical_despite_rate():
    rows=[row(1,'product',1,10000),row(2,'tax',1,1500)]
    b=book(rows,'11500')
    b.add(1,'price','20000',.91,'alternate_product')
    b.add(2,'price','3000',.91,'alternate_tax')
    b.add_footer('23000',.94,'alternate_total')
    b.add_vat('3000',.96,'alternate_vat')
    result=solve_matrix(rows,b,16)
    assert result['status']=='ambiguous'
    assert result['critical_ambiguity'] is True
    assert len(result['solutions'])==2
    assert all(sol['vat_rate_consistent'] is False for sol in result['solutions'])

def test_manual_correction_keeps_operator_mismatch_separate():
    rows=[row(1,'product',1,10000),row(2,'tax',1,1500)]
    b=book(rows,'11500')
    base={'rows':rows,'total':'11500','printed_vat':'1500',
          'vat':{'rate_percent':'16'},
          'vat_evidence':{'source':{'row_id':2,'field':'price'}},
          'numeric_candidate_book':b.export(), 'geometry':{'warnings':[]}}
    edited=copy.deepcopy(base)
    edited['user_vat_verified']=True
    output=apply_user_receipt(base,edited)
    assert output['math_match'] is True
    assert output['status']=='ok'
    assert output['ocr_validation_status']=='consistent'
    assert output['vat']['status']=='mismatch'
    assert len(output['financial_warnings'])==1
    assert output['numeric_matrix']['vat_rate_diagnostic']['status']=='operator_calculation_anomaly'
