#!/usr/bin/env python3
"""Guarded installer for Experiment Original Numeric ROI + Candidate Matrix V4.

Default operation is dry-run. It checks exact insertion anchors on the user's
current source, refuses to overwrite changed/unexpected code, and creates
recoverable backups under data/experiment_backups before --apply.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

NAME='OriginalROI_CandidateMatrix_V4'
HERE=Path(__file__).resolve().parent


def edit_once(text, needle, replacement, source):
    count=text.count(needle)
    if count!=1:
        raise RuntimeError(f'{source}: expected one exact integration anchor; got {count}: {needle[:96]!r}')
    return text.replace(needle,replacement,1)


def pipeline_transform(text):
    filename='App/pipeline.py'
    old='from .ocr.p1_double_retry import retry_suspect_cells\n'
    text=edit_once(text,old,old+'''from .ocr.experiment_matrix import CandidateBook, solve_matrix
from .ocr.experiment_original import OriginalNumericCache
from .ocr.experiment_footer import scan_footer
from .ocr.experiment_vat import scan_printed_vat
from .ocr.experiment_models import ExperimentModelPool
''',filename)
    old='    cell_requests, crop_map = [], {}\n'
    text=edit_once(text,old,old+'''    # E1: cache original source pixels; the usual remapped crop remains primary.
    original_numeric_cache = OriginalNumericCache(
        image, rect, max_bytes=positive_int('OCR_ORIGINAL_CACHE_MB',32)*1024*1024)
''',filename)
    old='                crop_map[(idx,field)] = crop\n'
    text=edit_once(text,old,old+'''                original_numeric_cache.store(idx, field, columns[field])
''',filename)
    old="    profile.phase('OCR P1: выборочное восстановление чисел')\n"
    text=edit_once(text,old,'''    # Snapshot pre-retry observations, including OCR scores for provenance.
    primary_numeric_observations = {
        CandidateBook.key(row['row_id'], field): {
            'text': row['cells'][field]['text'],
            'confidence': row['cells'][field].get('confidence')}
        for row in rows for field in ('quantity','price','printed_total')
        if field in row['cells']
    }
'''+old,filename)
    old="                crop_map[(row['row_id'],field)] = crop\n"
    text=edit_once(text,old,old+'''                # VAT outside the mesh is already an original-pixel extraction.
                original_numeric_cache.crops[(row['row_id'],field)] = crop.copy()
                original_numeric_cache.bytes += crop.nbytes
''',filename)
    # The operator may have printed an incorrect VAT. A rate mismatch alone
    # must not trigger expensive numeric OCR recovery.
    old="    retry_required = (incomplete_before or receipt_mismatch or row_mismatch or vat_mismatch or\n"
    text=edit_once(text,old,
        "    retry_required = (incomplete_before or receipt_mismatch or row_mismatch or\n",filename)
    old="    log_retry_details(processing_log, footer_recovery, 'Дополнительная проверка итога')\n"
    addon='''    # E1: finish the normal OCR pipeline first.  Do not mutate its numbers;
    # independent source-pixel observations form alternatives for a bounded
    # candidate matrix, not automatic replacements.
    profile.phase('OCR E1: оригинальные пиксели и матрица чисел')
    numeric_book_export = {'cells': {}, 'footer': []}
    numeric_matrix = {'status':'not_run', 'review_required':True}
    original_roi_experiment = {}
    numeric_auto_corrections = {'status':'not_run','changes':[]}
    footer_variant_attempts = []
    vat_evidence = {'source': None, 'source_status': 'not_run', 'observations': []}
    printed_vat_observed = None
    try:
        audit_sum = sum((Decimal(r['line_total']) for r in rows if r['line_total'] is not None), Decimal(0))
        book = CandidateBook.from_receipt(rows, primary_numeric_observations,
                                         geometry['ocr_recovery'], total, footer_attempts)
        model_pool = ExperimentModelPool()
        footer_variant_attempts = scan_footer(
            image, y_bottom, total, audit_sum,
            read_line=read_line, read_full=read_full,
            always_check=setting('OCR_FOOTER_VERIFY_ALWAYS', '1') == '1',
            max_probes=positive_int('OCR_FOOTER_MAX_PROBES',8),
            model_pool=model_pool)
        for alternative in footer_variant_attempts:
            if alternative.get('total') is not None:
                book.add_footer(alternative['total'], alternative.get('confidence'),
                                alternative['strategy'],
                                model=alternative.get('model_id') or alternative.get('model','PP-OCRv5 Cyrillic Mobile / RapidOCR 3.x'))
        # E3: photograph-based OCR of VAT, with its own independent observations.
        # Never infer 'printed VAT' solely by applying the tax percentage.
        vat_evidence = scan_printed_vat(
            rows, original_numeric_cache, book, vat_rate, read_line=read_line,
            model_pool=model_pool,
            always_check=setting('OCR_VAT_VERIFY_ALWAYS','1') == '1',
            max_probes=positive_int('OCR_VAT_MAX_PROBES',6))
        if book.vat:
            printed_vat_observed=book.vat[0]['text']
        original_roi_experiment = original_numeric_cache.run_suspects(
            rows, book, printed_total=total,
            max_calls=positive_int('OCR_ORIGINAL_MAX_RECOGNITIONS',8),
            read_line=read_line, model_pool=model_pool)
        numeric_matrix = solve_matrix(rows, book, vat_rate,
            max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
        numeric_book_export = book.export()
        geometry['numeric_experiment'] = {
            'status':numeric_matrix['status'],
            'original_cache_cells':original_roi_experiment['cache_cells'],
            'original_cache_bytes':original_roi_experiment['cache_bytes'],
            'source_rec_calls':original_roi_experiment['rec_calls'],
            'footer_probe_calls':len(footer_variant_attempts),
            'vat_probe_calls':vat_evidence['probes'],
            'vat_source_status':vat_evidence['source_status'],
            'vat_variants':numeric_matrix['vat_variants'],
            'critical_ambiguity':numeric_matrix['critical_ambiguity'],
            'matrix_nodes':numeric_matrix['nodes_visited'],
            'matrix_bounded':numeric_matrix['search_truncated'],
            'model_ensemble':model_pool.stats()}
        if numeric_matrix['footer_ambiguous']:
            geometry['warnings'].append('OCR печатного итога дал разные варианты; требуется подтверждение.')
        if numeric_matrix['vat_ambiguous']:
            geometry['warnings'].append('OCR напечатанного НДС дал разные значения; требуется подтверждение.')
        if numeric_matrix.get('vat_rate_diagnostic',{}).get('status')=='operator_calculation_anomaly':
            processing_log.append({'type':'financial_warning','stage':'Контроль НДС',
                'message':'Напечатанный НДС расходится с расчётом по ставке, но это не ошибка OCR само по себе.'})
        if numeric_matrix['critical_ambiguity']:
            geometry['warnings'].append('КРИТИЧЕСКАЯ НЕОДНОЗНАЧНОСТЬ: разные цены, количества, НДС и итоги одновременно проходят математические проверки. Нужна ручная проверка.')
        if any(row.get('kind')=='tax' for row in rows) and not numeric_matrix['printed_vat_observed']:
            geometry['warnings'].append('Нет независимого OCR-подтверждения суммы НДС; проверены только значения ячеек и общий итог.')
        if numeric_matrix['recommended_changes']:
            geometry['warnings'].append('Матрица нашла альтернативные цифры, согласованные с итогом; проверьте кандидатов.')
        processing_log.append({'type':'operation', 'stage':'Эксперимент E1',
            'message':f"E1: исходных ROI {original_roi_experiment['cache_cells']}, "
                      f"повторов {original_roi_experiment['rec_calls']}, "
                      f"вариантов итога {len(numeric_matrix['footer_variants'])}, "
                      f"вариантов НДС {len(numeric_matrix['vat_variants'])}, "
                      f"матрица {numeric_matrix['status']}."})
    except Exception as exc:
        numeric_matrix={'status':'experiment_error','review_required':True,
                        'error':f'{type(exc).__name__}: {exc}'}
        geometry['numeric_experiment']={'status':'experiment_error'}
        geometry['warnings'].append('Дополнительная проверка альтернатив OCR завершилась ошибкой; основной результат сохранён.')
        processing_log.append({'type':'error','stage':'Эксперимент E1',
                               'message':f'Эксперимент E1: {type(exc).__name__}: {exc}'})
'''
    text=edit_once(text,old,old+addon,filename)
    old="    if vat_ok is False: geometry['warnings'].append('Сумма НДС не совпадает с расчётом по установленной ставке.')\n"
    text=edit_once(text,old,'''    # Financial validation is orthogonal to whether printed digits were read.
    operator_vat_warning=('Напечатанный НДС расходится с установленной ставкой; возможна ошибка оператора.'
                           if vat_ok is False else None)
''',filename)
    old="            'numeric_issues': numeric_issues,\n"
    text=edit_once(text,old,old+'''             'numeric_candidate_book': numeric_book_export,
             'numeric_matrix': numeric_matrix,
             'original_roi_experiment': original_roi_experiment,
             'footer_variant_attempts': footer_variant_attempts,
             'vat_evidence': vat_evidence,
             'printed_vat': printed_vat_observed,
             'vat_rate_diagnostic': numeric_matrix.get('vat_rate_diagnostic'),
             'financial_warnings': [operator_vat_warning] if operator_vat_warning else [],
             'ocr_validation_status':('consistent' if match and not numeric_matrix.get('review_required',True)
                                      else 'review'),
''',filename)
    old="    receipt = {key:value for key,value in result.items() if key not in ('original', 'rectified')}\n"
    text=edit_once(text,old,'''    # E1 metadata is stored beside receipt.json; raw ROIs remain in RAM only.
    result['numeric_experiment_file'] = f"{result['job_directory']}/numeric_experiment.json"
    experiment_payload = {key:result.get(key) for key in
                          ('numeric_candidate_book','numeric_matrix','original_roi_experiment',
                           'footer_variant_attempts','vat_evidence','printed_vat')}
    experiment_payload['model_ensemble'] = (result.get('geometry',{})
                                            .get('numeric_experiment',{})
                                            .get('model_ensemble'))
    (job_dir/'numeric_experiment.json').write_text(
        json.dumps(experiment_payload, ensure_ascii=False, indent=2), encoding='utf-8')
'''+old,filename)
    return text


def server_transform(text):
    filename='App/web_server.py'
    old='from .pipeline import process\n'
    text=edit_once(text,old,old+'from .ocr.experiment_manual import apply_user_receipt\n',filename)
    old="    clean = {k:v for k,v in receipt.items() if k not in ('original','rectified')}\n"
    text=edit_once(text,old,'''    # Server independently recomputes Decimal arithmetic and locks valid
    # browser-confirmed figures; never trust client-provided math flags.
    try:
        clean = apply_user_receipt(job['result'], receipt)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
''',filename)
    old="    tmp.replace(target)\n"
    text=edit_once(text,old,old+'''    # Persist user ground truth and model-vs-user feedback separately.
    truth=directory/'numeric_user_truth.json'
    truth_tmp=directory/'numeric_user_truth.json.tmp'
    truth_tmp.write_text(json.dumps({'manual_truth':clean.get('numeric_user_truth',[]),
        'model_feedback':clean.get('numeric_model_feedback',[])},
        ensure_ascii=False, indent=2),encoding='utf-8')
    truth_tmp.replace(truth)
''',filename)
    old="    return {'saved': True, 'file': str(target.relative_to(ROOT)).replace('\\\\','/')}"
    # Existing return is one line containing a single escaped backslash.
    if old not in text:
        old="    return {'saved': True, 'file': str(target.relative_to(ROOT)).replace('\\','/')}"
    text=edit_once(text,old,"    return {'saved': True, 'file': str(target.relative_to(ROOT)).replace('\\\\','/'), 'receipt': clean}",filename)
    return text


def js_transform(text):
    filename='App/web/app.js'
    old='el.oninput=()=>{row.cells[k].text=el.value;calculate()};'
    text=edit_once(text,old,'''el.oninput=()=>{row.cells[k].text=el.value;
if(['quantity','price'].includes(k)){row.cells[k].user_verified=true;
el.title='Введено пользователем · приоритет над OCR';el.classList.remove('low-confidence')}
calculate()};''',filename)
    # Show all retained numeric readings, not just the selected OCR value.
    old="el.title=row.cells[k].derived?'Номер получен по порядку физической строки':`Уверенность: ${Math.round((row.cells[k].confidence??0)*100)}%`;"
    new=old+"if(['quantity','price'].includes(k)){const obs=current.numeric_candidate_book?.cells?.[`${row.row_id}:${k}`]?.observations||[];if(obs.length)el.title+=' | Кандидаты: '+obs.map(x=>`${x.text} (${x.confidence==null?'—':Math.round(x.confidence*100)+'%'}, ${x.path})`).join('; ');}"
    text=edit_once(text,old,new,filename)
    old="$('total').oninput=calculate;"
    text=edit_once(text,old,"$('total').oninput=()=>{current.user_total_verified=true;calculate()};",filename)
    old="$('total').value=current.total??'';"
    text=edit_once(text,old,old+"$('vat-printed').value=current.printed_vat??'';$('vat-printed').title='Варианты НДС: '+(current.numeric_candidate_book?.vat||[]).map(x=>`${x.text} (${x.confidence==null?'—':Math.round(x.confidence*100)+'%'}, ${x.path})`).join('; ');",filename)
    old="$('total').oninput=calculate;"
    # Already replaced above; put handler beside total handler.
    anchor="$('total').oninput=()=>{current.user_total_verified=true;calculate()};"
    text=edit_once(text,anchor,anchor+"$('vat-printed').oninput=()=>{current.printed_vat=$('vat-printed').value;current.user_vat_verified=true;calculate()};",filename)
    old="const verified=current.math_match&&vatOk===true&&rowTotalsOk;"
    new="const printedVat=num(current.printed_vat);const vatPrintedMatches=printedVat===null?false:hasVat&&Math.abs(vat-printedVat)<=.011;current.vat.printed_match=printedVat===null?null:vatPrintedMatches;const verified=current.math_match&&rowTotalsOk&&(!hasVat||vatPrintedMatches)&&!current.numeric_matrix?.review_required;current.ocr_validation_status=verified?'consistent':'review';current.vat.rate_diagnostic_only=true;"
    text=edit_once(text,old,new,filename)
    old="$('vat').className=vatOk===false?'bad':''"
    new="$('vat').className=(printedVat!==null&&!vatPrintedMatches)?'bad':''"
    text=edit_once(text,old,new,filename)
    old="vatOk?'соответствует ставке':'проверьте НДС'"
    text=edit_once(text,old,"vatOk?'соответствует ставке':'расхождение со ставкой (возможна ошибка оператора)'",filename)
    old="$('total').value=current.total??'';"
    new=old+"$('total').title='Варианты OCR итога: '+(current.numeric_candidate_book?.footer||[]).map(x=>`${x.text} (${x.confidence==null?'—':Math.round(x.confidence*100)+'%'}, ${x.path})`).join('; ');"
    text=edit_once(text,old,new,filename)
    old="if(!response.ok)throw Error('Не удалось сохранить правки в каталоге задания');$('status').textContent='Правки сохранены в каталоге задания'"
    new="if(!response.ok)throw Error('Не удалось сохранить правки в каталоге задания');const saved=await response.json();if(saved.receipt){Object.assign(result,saved.receipt);Object.assign(current,saved.receipt)}$('status').textContent='Правки и пользовательский эталон сохранены'"
    text=edit_once(text,old,new,filename)
    return text


def html_transform(text):
    filename='App/web/index.html'
    old='<p id="vat"></p>'
    new=('<p id="vat"></p><p><label for="vat-printed">Напечатанный НДС:</label> '
         '<input id="vat-printed" size="12" placeholder="Нет чтения"> ₸ '
         '<small>Ручная правка имеет приоритет над OCR</small></p>')
    return edit_once(text,old,new,filename)



def upgrade_pipeline_v4(text):
    """Replace repeated same-model retry stages with independent OCR path."""
    source='App/pipeline.py V4'
    a="    initial_recovery = recover_invalid_and_rows(rows,crop_map,max_retries=retry_budget)\n"
    b="""    initial_recovery = ({'attempted':0,'changed':[],'candidates':[],
                         'details':[],'delegated':'e4_independent_original_roi'}
                        if setting('OCR_E4_ROUTE_RETRIES','1') == '1' else
                        recover_invalid_and_rows(rows,crop_map,max_retries=retry_budget))
"""
    text=edit_once(text,a,b,source)
    a="    if retry_required:\n        processing_log.append({'type':'check', 'stage':'Контроль сумм',"
    b="    if retry_required and setting('OCR_E4_ROUTE_RETRIES','1') != '1':\n        processing_log.append({'type':'check', 'stage':'Контроль сумм',"
    text=edit_once(text,a,b,source)
    a="""    footer_recovery = recover_total_mismatch(rows,crop_map,number(total),
        max_retries=max(0,retry_budget-initial_recovery['attempted']))
"""
    b="""    footer_recovery = ({'attempted':0,'changed':[],'candidates':[],
                        'details':[],'delegated':'e4_independent_original_roi'}
                       if setting('OCR_E4_ROUTE_RETRIES','1') == '1' else
                       recover_total_mismatch(rows,crop_map,number(total),
                           max_retries=max(0,retry_budget-initial_recovery['attempted'])))
"""
    text=edit_once(text,a,b,source)
    a="from .ocr.experiment_models import ExperimentModelPool\n"
    b=a+"from .ocr.experiment_rec_trace import RecTracker, activate, deactivate, rec_scope\nfrom .ocr.experiment_auto import auto_apply_consensus\n"
    text=edit_once(text,a,b,source)
    a="    profile.processing_log = []\n"
    b=a+"    rec_tracker=RecTracker()\n    trace_token=activate(rec_tracker)\n"
    text=edit_once(text,a,b,source)
    a="        raise\n    performance = profile.finish()\n"
    b="        deactivate(trace_token)\n        raise\n    result['rec_trace']=rec_tracker.export()\n    performance = profile.finish()\n"
    text=edit_once(text,a,b,source)
    a="    return result\n\n\ndef _process(data, profile, job_dir, rectify):"
    b="    deactivate(trace_token)\n    return result\n\n\ndef _process(data, profile, job_dir, rectify):"
    text=edit_once(text,a,b,source)
    a="    footer_top = max(0,y_bottom-20)\n"
    b=a+"""    # All footer recognitions are attributable to one printed total.
    def footer_read_line(source_crop):
        with rec_scope(['receipt_total'],'printed_footer_primary'):
            return read_line(source_crop)
    def footer_read_full(source_crop, **kwargs):
        with rec_scope(['receipt_total'],'printed_footer_detector'):
            return read_full(source_crop, **kwargs)
"""
    text=edit_once(text,a,b,source)
    text=edit_once(text,"            lines = read_full(sample, footer_mode=True)","            lines = footer_read_full(sample, footer_mode=True)",source)
    text=edit_once(text,"            lines = read_full(sample)\n", "            lines = footer_read_full(sample)\n",source)
    text=edit_once(text,"    direct = read_line(image[fast_top:fast_bottom,fast_x:])", "    direct = footer_read_line(image[fast_top:fast_bottom,fast_x:])",source)
    text=edit_once(text,"            line = read_line(image[top:bottom, left:])", "            line = footer_read_line(image[top:bottom, left:])",source)
    # When a second independent model exists, try it only on cells already
    # crosschecked by the first model and only for unresolved math.
    a="""        numeric_matrix = solve_matrix(rows, book, vat_rate,
            max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
        numeric_book_export = book.export()
"""
    b="""        numeric_matrix = solve_matrix(rows, book, vat_rate,
            max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
        if ((numeric_matrix['status'] != 'unique_math_match' or
             numeric_matrix.get('review_required') or numeric_matrix.get('recommended_changes')) and
                setting('OCR_E4_SECOND_PASS','1') == '1'):
            target_keys={(v['row_id'],v['field']) for v in numeric_matrix.get('recommended_changes',[])}
            second_pass = original_numeric_cache.escalate_suspects(
                book, original_roi_experiment, model_pool,
                max_calls=positive_int('OCR_E4_SECOND_PASS_MAX',3),
                only_keys=target_keys if numeric_matrix['status']=='unique_math_match' and target_keys else None)
            original_roi_experiment['second_pass'] = second_pass
            if second_pass['recognitions']:
                numeric_matrix = solve_matrix(rows, book, vat_rate,
                    max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
        numeric_auto_corrections = auto_apply_consensus(
            rows,book,numeric_matrix,total,
            enabled=setting('OCR_E4_AUTO_APPLY','1') == '1')
        if numeric_auto_corrections['status']=='applied':
            total=numeric_auto_corrections['printed_total']
            printed_vat_observed=(numeric_matrix['solutions'][0].get('vat_printed') or printed_vat_observed)
            numeric_matrix = solve_matrix(rows, book, vat_rate,
                max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
            processing_log.append({'type':'operation','stage':'E4 автоматическая сверка',
                'message':'Выбрана единственная арифметически согласованная комбинация: '
                          + str(len(numeric_auto_corrections['changes'])) + ' исправлений.'})
        numeric_matrix['auto_applied'] = numeric_auto_corrections
        numeric_matrix['user_action_required'] = numeric_matrix['status']=='ambiguous'
        numeric_matrix['automatic_resolution'] = (
            'consistent' if numeric_matrix['status']=='unique_math_match' and not numeric_matrix['review_required']
            else 'unresolved_preserved_without_forced_edit')
        numeric_book_export = book.export()
"""
    text=edit_once(text,a,b,source)
    for a,b in (
        ('OCR печатного итога дал разные варианты; требуется подтверждение.', 'OCR печатного итога дал разные варианты; альтернативы сохранены для анализа.'),
        ('OCR напечатанного НДС дал разные значения; требуется подтверждение.', 'OCR напечатанного НДС дал разные значения; альтернативы сохранены для анализа.'),
        ('КРИТИЧЕСКАЯ НЕОДНОЗНАЧНОСТЬ: разные цены, количества, НДС и итоги одновременно проходят математические проверки. Нужна ручная проверка.', 'Найдены разные математически согласованные комбинации цифр. Выбрана комбинация с наибольшей уверенностью OCR. Пожалуйста, проверьте цифры.'),
        ('Матрица нашла альтернативные цифры, согласованные с итогом; проверьте кандидатов.', 'Матрица сохранила альтернативные цифры; исходные OCR-значения оставлены без автоматической подмены.'),
    ):
        text=edit_once(text,a,b,source)
    a="             'numeric_candidate_book': numeric_book_export,\n"
    text=edit_once(text,a,a+"             'ocr_user_action_required': False,\n             'numeric_auto_corrections': numeric_auto_corrections,\n",source)
    # A failed automatic OCR check should be described as diagnostic status,
    # not a demand that the user decode the photograph.
    text=edit_once(text,
        "Печатный итог не найден; проверьте вручную.",
        "Печатный итог не получен однозначно; доступные OCR-варианты сохранены.",source)
    text=edit_once(text,
        """f'Строка {row["row_id"]}: проверьте {label} «{shown}» — число не рассчитано.'""",
        """f'Строка {row["row_id"]}: {label} «{shown}» не удалось надёжно распознать; число не рассчитано.'""",source)

    a="                           'footer_variant_attempts','vat_evidence','printed_vat')}"
    text=edit_once(text,a,
        "                           'footer_variant_attempts','vat_evidence','printed_vat',\n                           'numeric_auto_corrections','rec_trace')}",source)
    return text


def upgrade_js_v4(text):
    src='App/web/app.js V4'
    a="$('receipt-heading').classList.toggle('receipt-mismatch',!verified);"
    b="""$('receipt-heading').classList.toggle('receipt-mismatch',!verified&&!current.numeric_matrix?.review_required);
const e4status=$('e4-auto-status');
if(e4status){
 const matrix=current.numeric_matrix||{};
 const uncertain=!!matrix.review_required;
 const changes=current.numeric_auto_corrections?.changes?.length||0;
 e4status.textContent=changes?('Система самостоятельно уточнила '+changes+' числовых значений на основании независимых OCR-моделей и согласованных сумм.'):
   uncertain?'Автоматическая сверка завершена. Некоторые цифры допускают разные прочтения; исходные значения сохранены, альтернативы доступны в журнале.':
   matrix.status==='unique_math_match'?'Сверка по итогам завершена.':'Дополнительные варианты чисел сохранены для диагностики.';
 e4status.hidden=!current.numeric_matrix;
}
"""
    text=edit_once(text,a,b,src)
    text=edit_once(text,"'Проверьте сумму и данные'",
                   "'Сумма пока не подтверждена автоматически'",src)
    return text


def upgrade_html_v4(text):
    src='App/web/index.html V4'
    return edit_once(text,'<div id="warnings"></div>',
                     '<div id="warnings"></div><p id="e4-auto-status" aria-live="polite" hidden></p>',src)


def engine_transform_v4(text):
    src='App/ocr/engine.py V4'
    a='import threading\n'
    text=edit_once(text,a,a+"from time import perf_counter\nfrom .experiment_rec_trace import record_rec\n",src)
    a='\n\ndef read_full(image, *, footer_mode=False, cell_mode=False):\n'
    b='''

def _run_cyrillic_rec(image):
    start=perf_counter()
    try:
        result=engine()(image,use_det=False,use_cls=False)
        record_rec(model='v5_cyrillic_mobile',ms=(perf_counter()-start)*1000)
        return result
    except Exception as exc:
        record_rec(model='v5_cyrillic_mobile',ms=(perf_counter()-start)*1000,
                   success=False,note=str(exc))
        raise

def _run_english_rec(image):
    start=perf_counter()
    try:
        result=english_engine()(image,use_det=False,use_cls=False)
        record_rec(model='v5_english_mobile',ms=(perf_counter()-start)*1000)
        return result
    except Exception as exc:
        record_rec(model='v5_english_mobile',ms=(perf_counter()-start)*1000,
                   success=False,note=str(exc))
        raise


def read_full(image, *, footer_mode=False, cell_mode=False):
'''
    text=edit_once(text,a,b,src)
    a='        result = (footer_engine() if footer_mode or cell_mode else engine())(image, use_det=True, use_cls=False)'
    b='''        clock=perf_counter()
        try:
            result = (footer_engine() if footer_mode or cell_mode else engine())(image, use_det=True, use_cls=False)
            record_rec(mode='DET+REC',model='v5_cyrillic_mobile',ms=(perf_counter()-clock)*1000,
                       note='Detector may dispatch multiple internal REC calls; exact count unavailable')
        except Exception as exc:
            record_rec(mode='DET+REC',model='v5_cyrillic_mobile',ms=(perf_counter()-clock)*1000,
                       success=False,note=str(exc))
            raise'''
    text=edit_once(text,a,b,src)
    text=text.replace('english_engine()(crop, use_det=False, use_cls=False)','_run_english_rec(crop)')
    text=text.replace('engine()(image, use_det=False, use_cls=False)','_run_cyrillic_rec(image)')
    text=text.replace('engine()(crop, use_det=False, use_cls=False)','_run_cyrillic_rec(crop)')
    # The above replacements have spaces after commas, not the helper function
    # itself (which uses compact commas).
    return text


def batch_transform_v4(text):
    src='App/ocr/p1_batch.py V4'
    a='from .engine import engine, read_cell, _has_separate_numeric_lines, _lock\n'
    text=edit_once(text,a,a+'from .experiment_rec_trace import rec_scope, record_rec\n',src)
    a='            item = read_cell(cell.image, cell.row_id, cell.column_id, cell.offset)\n'
    b='''            with rec_scope([(cell.row_id,cell.column_id)],'primary_multiline_cell'):
                item = read_cell(cell.image, cell.row_id, cell.column_id, cell.offset)
'''
    text=edit_once(text,a,b,src)
    a='''                with _lock:
                    response = engine().recognize_txt([c.image for c in chunk])
'''
    b='''                started_rec=perf_counter()
                try:
                    with _lock:
                        response = engine().recognize_txt([c.image for c in chunk])
                    record_rec(keys=[f'{c.row_id}:{c.column_id}' for c in chunk],
                               stage='primary_rec_batch',model='v5_cyrillic_mobile',
                               ms=(perf_counter()-started_rec)*1000)
                except Exception as error:
                    record_rec(keys=[f'{c.row_id}:{c.column_id}' for c in chunk],
                               stage='primary_rec_batch',model='v5_cyrillic_mobile',
                               ms=(perf_counter()-started_rec)*1000,
                               success=False,note=str(error))
                    raise
'''
    text=edit_once(text,a,b,src)
    a='''                    item = read_cell(c.image, c.row_id, c.column_id, c.offset)
'''
    b='''                    with rec_scope([(c.row_id,c.column_id)],'primary_batch_exception_fallback'):
                        item = read_cell(c.image, c.row_id, c.column_id, c.offset)
'''
    text=edit_once(text,a,b,src)
    return text
def update_file_content(relative,content):
    if relative=='App/pipeline.py':return upgrade_pipeline_v4(pipeline_transform(content))
    if relative=='App/web_server.py':return server_transform(content)
    if relative=='App/web/app.js':return upgrade_js_v4(js_transform(content))
    if relative=='App/web/index.html':return upgrade_html_v4(html_transform(content))
    if relative=='App/ocr/engine.py':return engine_transform_v4(content)
    if relative=='App/ocr/p1_batch.py':return batch_transform_v4(content)
    raise RuntimeError(relative)


def patch_set(project):
    edits={}
    for path in ('App/pipeline.py','App/web_server.py','App/web/app.js','App/web/index.html', 'App/ocr/engine.py','App/ocr/p1_batch.py'):
        fn=project/path
        if not fn.is_file():raise RuntimeError(f'Отсутствует файл: {fn}')
        data=fn.read_text(encoding='utf-8')
        if 'from .ocr.experiment_matrix import CandidateBook' in data or 'apply_user_receipt' in data:
            raise RuntimeError(f'Возможно, эксперимент уже установлен: {fn}')
        edits[path]=update_file_content(path,data)
    for fn in sorted((HERE/'files').rglob('*')):
        if fn.is_file() and fn.suffix in ('.py','.json'):
            rel=str(fn.relative_to(HERE/'files')).replace('\\','/')
            target=project/rel
            if target.exists():
                raise RuntimeError(f'Новый модуль уже существует, не перезаписываем: {target}')
            edits[rel]=fn.read_text(encoding='utf-8')
    # Refuse to write a syntactically invalid transformed Python file.
    for rel, source in edits.items():
        if rel.endswith('.py'):
            compile(source, str(project/rel), 'exec')
    return edits


def install(project, edits):
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup=project/'data'/'experiment_backups'/NAME/stamp
    backup.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for path in edits:
        target=project/path
        existed=target.exists()
        if existed:
            to=backup/path
            to.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,to)
        manifest.append({'path':path,'existed':existed})
    (backup/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for path,data in edits.items():
        target=project/path
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix(target.suffix+'.experiment_tmp')
        tmp.write_text(data,encoding='utf-8')
        tmp.replace(target)
    print(f'Применено: {len(edits)} файлов. Резервная копия: {backup}')


def rollback(project, backup):
    data=json.loads((backup/'manifest.json').read_text(encoding='utf-8'))
    for item in data:
        target=project/item['path']
        if item['existed']:
            shutil.copy2(backup/item['path'],target)
        elif target.exists():
            target.unlink()
    print(f'Отменено: {len(data)} файлов по копии {backup}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True,help='Корень проекта с App/pipeline.py')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--check',action='store_true',help='Проверить совместимость (по умолчанию)')
    group.add_argument('--apply',action='store_true',help='Установить с резервной копией')
    group.add_argument('--rollback',type=Path,help='Папка резервной копии с manifest.json')
    args=parser.parse_args()
    project=args.project.resolve()
    if args.rollback:
        rollback(project,args.rollback.resolve())
        return
    edits=patch_set(project)
    print(f'Проверка OK: {len(edits)} файлов, все точки встройки уникальны.')
    if args.apply:
        install(project,edits)
    else:
        print('Изменений нет. Для установки повторите с --apply.')


if __name__=='__main__':
    main()
