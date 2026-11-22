"""V4 installer: exact anchors, rollback and independent retry routing."""
from pathlib import Path
import importlib.util
import pytest


def load_installer():
    location=Path(__file__).resolve().parents[1]/'install_experiment.py'
    spec=importlib.util.spec_from_file_location('e4_installer',location)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_installer_rejects_partial_or_unknown_source(tmp_path):
    inst=load_installer()
    with pytest.raises(RuntimeError, match='Отсутствует файл'):
        inst.patch_set(tmp_path)


def test_v4_pipeline_rewrites_legacy_retries_and_attribution():
    inst=load_installer()
    src='''from .ocr.experiment_models import ExperimentModelPool
    profile.processing_log = []
        raise
    performance = profile.finish()
    return result


def _process(data, profile, job_dir, rectify):
    initial_recovery = recover_invalid_and_rows(rows,crop_map,max_retries=retry_budget)
    if retry_required:
        processing_log.append({'type':'check', 'stage':'Контроль сумм',
    footer_recovery = recover_total_mismatch(rows,crop_map,number(total),
        max_retries=max(0,retry_budget-initial_recovery['attempted']))
    footer_top = max(0,y_bottom-20)
            lines = read_full(sample, footer_mode=True)
            lines = read_full(sample)
    direct = read_line(image[fast_top:fast_bottom,fast_x:])
            line = read_line(image[top:bottom, left:])
        numeric_matrix = solve_matrix(rows, book, vat_rate,
            max_nodes=positive_int('OCR_MATRIX_MAX_NODES',100000))
        numeric_book_export = book.export()
        if numeric_matrix['footer_ambiguous']:
            geometry['warnings'].append('OCR печатного итога дал разные варианты; требуется подтверждение.')
        if numeric_matrix['vat_ambiguous']:
            geometry['warnings'].append('OCR напечатанного НДС дал разные значения; требуется подтверждение.')
        if numeric_matrix['critical_ambiguity']:
            geometry['warnings'].append('КРИТИЧЕСКАЯ НЕОДНОЗНАЧНОСТЬ: разные цены, количества, НДС и итоги одновременно проходят математические проверки. Нужна ручная проверка.')
        if numeric_matrix['recommended_changes']:
            geometry['warnings'].append('Матрица нашла альтернативные цифры, согласованные с итогом; проверьте кандидатов.')
        if total is None: geometry['warnings'].append('Печатный итог не найден; проверьте вручную.')
        geometry['warnings'].append(f'Строка {row["row_id"]}: проверьте {label} «{shown}» — число не рассчитано.')
             'numeric_candidate_book': numeric_book_export,
                           'footer_variant_attempts','vat_evidence','printed_vat')}
'''
    out=inst.upgrade_pipeline_v4(src)
    assert "'delegated':'e4_independent_original_roi'" in out
    assert 'rec_tracker.export()' in out
    assert 'original_numeric_cache.escalate_suspects' in out
    assert "numeric_matrix['user_action_required'] = numeric_matrix['status']=='ambiguous'" in out
    assert 'footer_read_line' in out
    assert 'Нужна ручная проверка' not in out
    assert 'требуется подтверждение' not in out


def test_trace_engine_and_batch_transform_structure():
    inst=load_installer()
    engine='''import threading

def read_full(image, *, footer_mode=False, cell_mode=False):
        result = (footer_engine() if footer_mode or cell_mode else engine())(image, use_det=True, use_cls=False)
        something=english_engine()(crop, use_det=False, use_cls=False)
        d=engine()(image, use_det=False, use_cls=False)
        d=engine()(crop, use_det=False, use_cls=False)
'''
    out=inst.engine_transform_v4(engine)
    assert 'def _run_cyrillic_rec' in out
    assert "record_rec(mode='DET+REC'" in out
    assert 'something=_run_english_rec(crop)' in out
    batch='''from .engine import engine, read_cell, _has_separate_numeric_lines, _lock
            item = read_cell(cell.image, cell.row_id, cell.column_id, cell.offset)
                with _lock:
                    response = engine().recognize_txt([c.image for c in chunk])
                    item = read_cell(c.image, c.row_id, c.column_id, c.offset)
'''
    changed=inst.batch_transform_v4(batch)
    assert 'record_rec(keys=' in changed
    assert "rec_scope([(cell.row_id,cell.column_id)],'primary_multiline_cell')" in changed
    assert "rec_scope([(c.row_id,c.column_id)],'primary_batch_exception_fallback')" in changed


def test_installer_backup_round_trip(tmp_path):
    inst=load_installer()
    project=tmp_path/'project'
    target=project/'App/pipeline.py'
    target.parent.mkdir(parents=True)
    target.write_text('original',encoding='utf-8')
    edits={'App/pipeline.py':'changed','App/ocr/experiment_rec_trace.py':'new'}
    inst.install(project,edits)
    assert target.read_text()=='changed'
    backup=next((project/'data/experiment_backups'/inst.NAME).iterdir())
    inst.rollback(project,backup)
    assert target.read_text()=='original'
    assert not (project/'App/ocr/experiment_rec_trace.py').exists()
