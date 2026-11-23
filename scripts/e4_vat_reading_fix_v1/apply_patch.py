"""Strict, reversible VAT-reading patch for GridMesh_RapidOCR @ 04a13da.

No pip installs, no model changes and no .env writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'scripts' / 'e4_vat_reading_fix_v1'
BACKUP = ROOT / 'data' / 'experiments' / 'e4_vat_reading_fix_v1' / 'backup'
MARKER = BACKUP / 'manifest.json'
ORIGINAL_BLOBS = {
    'App/ocr/experiment_vat.py': 'bd30563da83fc1d4c32a6a948e6aefa9fc79ac86',
    'App/pipeline.py': '44ee822ecde084aea49f6360d695d5e2501c99ee',
    'App/ocr/e4_adaptive_early.py': '4216da884c915293ac3a20d985a6f7fe15e126a4',
}
NEW_FILE = 'App/ocr/vat_witness.py'


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'{name}: ожидается одно совпадение, найдено {count}; патч отменён')
    return source.replace(old, new, 1)


def patch_vat(src: str) -> str:
    start = 'def vat_source(rows):\n'
    end = 'def _prep(image, mode):\n'
    if src.count(start) != 1 or src.count(end) != 1:
        raise RuntimeError('experiment_vat: неизвестная структура исходника')
    i, j = src.index(start), src.index(end)
    if j <= i:
        raise RuntimeError('experiment_vat: неверные границы')
    src = src[:i] + src[j:]
    return replace_once(src,
        'from .experiment_rec_trace import rec_scope\n',
        'from .experiment_rec_trace import rec_scope\n'
        'from .vat_witness import vat_source, seed_printed_vat, observed_vat_text\n',
        'experiment_vat: подключение прямого свидетеля НДС')


def patch_early(src: str) -> str:
    old = '                if _observed_numbers(book, key) - old_numbers:\n'
    new = ('                # A newly read 1-unit tax column may unlock a real VAT witness.\n'
           '                # Rebuild the VAT evidence from actual OCR candidates before E4.\n'
           '                seed_printed_vat(rows, book)\n'
           '                if _observed_numbers(book, key) - old_numbers:\n')
    return replace_once(src, old, new, 'e4_adaptive_early: обновление НДС')


def patch_pipeline(src: str) -> str:
    src = replace_once(src,
        'from .ocr.experiment_vat import scan_printed_vat, seed_printed_vat\n',
        'from .ocr.experiment_vat import scan_printed_vat, seed_printed_vat, observed_vat_text\n',
        'pipeline: импорт функции выбора НДС')
    src = replace_once(src,
        "            if book.vat:\n                printed_vat_observed=book.vat[0]['text']\n",
        "            printed_vat_observed = (observed_vat_text(book, preferred=seed.get('text'))\n"
        "                                    or printed_vat_observed)\n",
        'pipeline: начальный прямой OCR НДС')
    src = replace_once(src,
        "                if book.vat:\n                    printed_vat_observed=book.vat[0]['text']\n",
        "                printed_vat_observed = (observed_vat_text(book, preferred=printed_vat_observed)\n"
        "                                        or printed_vat_observed)\n",
        'pipeline: повторный OCR НДС')
    src = replace_once(src,
        '            numeric_auto_corrections = auto_apply_consensus(\n',
        "            # Display only a VAT figure that was actually observed by OCR.\n"
        "            if numeric_matrix.get('status') == 'unique_math_match' and numeric_matrix.get('solutions'):\n"
        "                printed_vat_observed = (observed_vat_text(\n"
        "                    book, preferred=numeric_matrix['solutions'][0].get('vat_printed'))\n"
        "                    or printed_vat_observed)\n"
        '            numeric_auto_corrections = auto_apply_consensus(\n',
        'pipeline: значение подтверждённой матрицы НДС')
    src = replace_once(src,
        "                total=numeric_auto_corrections['printed_total']\n"
        "                printed_vat_observed=(numeric_matrix['solutions'][0].get('vat_printed') or printed_vat_observed)\n"
        "                numeric_matrix = solve_matrix(rows, book, vat_rate,\n",
        "                total=numeric_auto_corrections['printed_total']\n"
        "                # Corrections may turn an unreadable VAT row into a valid\n"
        "                # photographed value (e.g. quantity=23200, price=1).\n"
        "                post_e4_vat_seed = seed_printed_vat(rows, book)\n"
        "                if post_e4_vat_seed['source'] is not None:\n"
        "                    vat_evidence['source'] = post_e4_vat_seed['source']\n"
        "                    vat_evidence['post_e4_source_status'] = post_e4_vat_seed['status']\n"
        "                printed_vat_observed = (observed_vat_text(\n"
        "                    book, preferred=numeric_matrix['solutions'][0].get('vat_printed'))\n"
        "                    or printed_vat_observed)\n"
        "                numeric_matrix = solve_matrix(rows, book, vat_rate,\n",
        'pipeline: обновление после коррекции E4')
    return replace_once(src,
        "                geometry['warnings'].append('Нет независимого OCR-подтверждения суммы НДС; проверены только значения ячеек и общий итог.')\n",
        "                geometry['warnings'].append('Напечатанный НДС не удалось надёжно прочитать из числовых ячеек строки НДС; сумма, рассчитанная по ставке, не подставлялась.')\n",
        'pipeline: точная диагностика отсутствующего НДС')


PATCHERS = {
    'App/ocr/experiment_vat.py': patch_vat,
    'App/pipeline.py': patch_pipeline,
    'App/ocr/e4_adaptive_early.py': patch_early,
}


def ensure_project() -> None:
    if not (ROOT / 'App' / 'pipeline.py').is_file():
        raise RuntimeError(f'Не найден корень GridMesh_RapidOCR: {ROOT}; распакуйте ZIP в корень проекта')
    if not (EXPERIMENT / 'apply_patch.py').is_file():
        raise RuntimeError('Скрипт установки должен находиться в scripts/e4_vat_reading_fix_v1/')


def install() -> None:
    ensure_project()
    if MARKER.is_file():
        raise RuntimeError('Патч уже установлен. Для отката запустите restore_vat_fix.bat')
    if (ROOT / NEW_FILE).exists():
        raise RuntimeError(f'{NEW_FILE} уже существует: перезапись запрещена')
    patched = {}
    originals = {}
    for rel, expected in ORIGINAL_BLOBS.items():
        p = ROOT / rel
        data = p.read_bytes()
        # Windows git checkouts may use CRLF while Git stores an LF blob.
        normalized = data.replace(b'\r\n', b'\n')
        sha = git_blob(normalized) if git_blob(data) != expected else git_blob(data)
        if sha != expected:
            raise RuntimeError(f'{rel}: Git blob {sha} != {expected}. Исходник обновлён или локально изменён, установка отменена.')
        originals[rel] = data
        text = PATCHERS[rel](normalized.decode('utf-8-sig'))
        patched[rel] = (text.replace('\n', '\r\n') if b'\r\n' in data else text).encode('utf-8')
        compile(patched[rel].decode('utf-8'), str(p), 'exec')
    witness_path = ROOT / NEW_FILE
    if not witness_path.is_file():
        # The payload stays next to the installer until installation.
        witness = EXPERIMENT / 'vat_witness.py'
        if not witness.is_file():
            raise RuntimeError('В ZIP отсутствует scripts/e4_vat_reading_fix_v1/vat_witness.py')
        candidate = witness.read_bytes()
    else:
        raise RuntimeError('vat_witness.py уже установлен; перезапись запрещена')
    compile(candidate.decode('utf-8'), NEW_FILE, 'exec')
    BACKUP.mkdir(parents=True, exist_ok=False)
    touched = []
    try:
        manifest = {'project_commit':'04a13da2f35f46f74ff7886cbcfb8edf0199aaa5',
                    'original_blobs':ORIGINAL_BLOBS,'patched_blobs':{},
                    'new_file':NEW_FILE,'witness_sha256':hashlib.sha256(candidate).hexdigest()}
        for rel, data in originals.items():
            dst = BACKUP / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        for rel, data in patched.items():
            path = ROOT / rel
            path.write_bytes(data)
            touched.append(rel)
            manifest['patched_blobs'][rel] = git_blob(data)
        witness_path.write_bytes(candidate)
        touched.append(NEW_FILE)
        MARKER.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    except BaseException:
        for rel in reversed(touched):
            if rel in originals:
                (ROOT / rel).write_bytes(originals[rel])
            elif rel == NEW_FILE:
                (ROOT / rel).unlink(missing_ok=True)
        shutil.rmtree(BACKUP,ignore_errors=True)
        raise
    print('OK: исправлено чтение НДС из действительных OCR-ячеек и E4. Резервные копии:',BACKUP)


def restore() -> None:
    ensure_project()
    if not MARKER.is_file():
        raise RuntimeError('Нет manifest.json установленного эксперимента; откат отменён')
    info=json.loads(MARKER.read_text(encoding='utf-8'))
    for rel, expected in info['patched_blobs'].items():
        if git_blob((ROOT / rel).read_bytes()) != expected:
            raise RuntimeError(f'{rel} изменился после установки; откат не будет затирать новые правки')
    witness_path = ROOT / info['new_file']
    if not witness_path.is_file() or hashlib.sha256(witness_path.read_bytes()).hexdigest() != info['witness_sha256']:
        raise RuntimeError(f'{witness_path}: исходный экспериментальный файл изменился; откат отменён')
    for rel, expected in info['original_blobs'].items():
        data = (BACKUP / rel).read_bytes()
        if git_blob(data.replace(b'\r\n', b'\n')) != expected:
            raise RuntimeError(f'Резервная копия повреждена: {rel}')
    for rel in info['original_blobs']:
        (ROOT / rel).write_bytes((BACKUP / rel).read_bytes())
    witness_path.unlink()
    shutil.rmtree(BACKUP)
    print('OK: код возвращён побайтово к исходному GitHub-коммиту')


def main() -> int:
    arg=argparse.ArgumentParser(description='E4 VAT Reading Fix V1')
    arg.add_argument('action',choices=['install','restore'])
    args=arg.parse_args()
    try:
        if args.action=='install':install()
        else:restore()
    except Exception as exc:
        print(f'ОШИБКА: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
