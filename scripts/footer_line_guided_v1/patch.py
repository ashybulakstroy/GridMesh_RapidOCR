"""Reversible anchored patch; does not modify OCR dependencies or .env."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

EXPERIMENT='footer_line_guided_v1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIPELINE=ROOT/'App'/'pipeline.py'
TARGET=ROOT/'App'/'ocr'/'footer_line_guided_v1.py'
BUNDLED=HERE/'files'/'App'/'ocr'/'footer_line_guided_v1.py'
BACKUPS=HERE/'_backups'
MAGIC='# FOCUSED_FOOTER_EXPERIMENT_V1'


def replace_once(code, old, new, title):
    n=code.count(old)
    if n!=1:
        raise RuntimeError(f'{title}: ожидалось одно вхождение, найдено {n}. Исходный код не изменён.')
    return code.replace(old,new,1)


def patch_source(code):
    if MAGIC in code:
        raise RuntimeError('Патч уже установлен. Повторная установка запрещена.')
    code=replace_once(code,
        'from .ocr.experiment_footer import scan_footer\n',
        'from .ocr.experiment_footer import scan_footer\n'
        'from .ocr.footer_line_guided_v1 import guide_from_grid, attempt_focused_footer\n',
        'Импорт модуля')
    code=replace_once(code,
        '    xs = [int(round(x)) for x in rect.xs]; ys = [int(round(y)) for y in rect.ys]\n',
        '    xs = [int(round(x)) for x in rect.xs]; ys = [int(round(y)) for y in rect.ys]\n'
        '    footer_focus_guide = guide_from_grid(rect, image.shape)\n',
        'Получение геометрии')
    code=replace_once(code,
        'read_printed_footer(image, table_source_bottom(rect,xs,ys))',
        'read_printed_footer(image, table_source_bottom(rect,xs,ys), footer_guide=footer_focus_guide)',
        'Параллельный поток итога')
    code=replace_once(code,
        'read_printed_footer(image, y_bottom, rows)',
        'read_printed_footer(image, y_bottom, rows, footer_guide=footer_focus_guide)',
        'Основной маршрут итога')
    code=replace_once(code,
        'def read_printed_footer(image, y_bottom, rows=()):',
        'def read_printed_footer(image, y_bottom, rows=(), *, footer_guide=None):',
        'Сигнатура')
    entry='''    # FOCUSED_FOOTER_EXPERIMENT_V1: optical evidence strictly between source rules.
    focused_attempt=None
    focus_info={'status':'disabled'}
    if setting('OCR_FOOTER_FOCUS_ENABLE','0') == '1':
        def focused_reader(crop):
            with rec_scope(['receipt_total'], 'printed_footer_focused'):
                return read_line(crop)
        min_conf=float(setting('OCR_FOOTER_FOCUS_MIN_CONF','0.86'))
        focused=attempt_focused_footer(image,footer_guide,focused_reader,number,
                                       min_confidence=min_conf)
        focus_info={**focused['diagnostics'],
                    'accepted':focused['accepted'],
                    'result':focused['total']}
        focused_attempt=focused['attempt']
        if focused['accepted']:
            return {'total':focused['total'],'lines':focused['lines'],
                    'attempts':[focused_attempt], 'y_bottom':y_bottom,
                    'footer_focus':focus_info}
'''
    code=replace_once(code,
        '    footer_top = max(0,y_bottom-20)\n',
        entry+'    footer_top = max(0,y_bottom-20)\n',
        'Приоритет одного OCR')
    code=replace_once(code,
        '    footer_attempts = []\n    total, footer_lines = None, []\n',
        '    footer_attempts = [focused_attempt] if focused_attempt is not None else []\n'
        '    total, footer_lines = None, []\n',
        'Пути fallback')
    code=replace_once(code,
        "    return {'total':total, 'lines':footer_lines, 'attempts':footer_attempts,\n            'y_bottom':y_bottom}\n",
        "    return {'total':total, 'lines':footer_lines, 'attempts':footer_attempts,\n"
        "            'y_bottom':y_bottom, 'footer_focus':focus_info}\n",
        'Возврат результата')
    code=replace_once(code,
        "'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts, 'timings':",
        "'footer_ocr': footer_lines, 'footer_ocr_attempts': footer_attempts,\n"
        "            'footer_focus': footer_result.get('footer_focus'), 'timings':",
        'Диагностика результата')
    ast.parse(code)
    return code


def sha(data):return hashlib.sha256(data).hexdigest()


def latest_backup():
    options=sorted(p for p in BACKUPS.glob('*/manifest.json') if p.is_file())
    if not options:raise RuntimeError('Нет резервной копии для отката')
    return options[-1]


def install():
    original=PIPELINE.read_bytes()
    source=original.decode('utf-8').replace('\r\n','\n')
    patched_text=patch_source(source)
    if b'\r\n' in original:patched_text=patched_text.replace('\n','\r\n')
    patched=patched_text.encode('utf-8')
    if not BUNDLED.exists():raise RuntimeError('Не найден новый модуль OCR')
    ast.parse(BUNDLED.read_text(encoding='utf-8'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=BACKUPS/stamp
    backup.mkdir(parents=True,exist_ok=False)
    (backup/'pipeline.py').write_bytes(original)
    had_module=TARGET.exists()
    if had_module:(backup/'footer_line_guided_v1.py').write_bytes(TARGET.read_bytes())
    manifest={'experiment':EXPERIMENT,'before_sha256':sha(original),'after_sha256':sha(patched),
              'module_preexisted':had_module,
              'module_after_sha256':sha(BUNDLED.read_bytes()),
              'source_repo':'ashybulakstroy/GridMesh_RapidOCR',
              'source_commit_checked':'5c1e94980ec8eb32ff96b5267dd83f34c055aa30'}
    (backup/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    temp=PIPELINE.with_name('pipeline.py.footer_focus.tmp')
    try:
        TARGET.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BUNDLED,TARGET)
        temp.write_bytes(patched)
        temp.replace(PIPELINE)
    except Exception:
        if temp.exists():temp.unlink()
        PIPELINE.write_bytes(original)
        if had_module:TARGET.write_bytes((backup/'footer_line_guided_v1.py').read_bytes())
        else:TARGET.unlink(missing_ok=True)
        raise
    print('Установлено. Резервная копия:',backup)
    print('Эксперимент выключен по умолчанию; включите через start_focus.bat')


def restore():
    path=latest_backup();record=json.loads(path.read_text(encoding='utf-8'))
    original=(path.parent/'pipeline.py').read_bytes()
    if sha(original)!=record['before_sha256']:
        raise RuntimeError('Повреждена резервная копия')
    if sha(PIPELINE.read_bytes())!=record['after_sha256']:
        raise RuntimeError('Код изменён после установки. Автоматический откат отказался его перезаписывать.')
    if not TARGET.exists() or sha(TARGET.read_bytes())!=record['module_after_sha256']:
        raise RuntimeError('Модуль эксперимента изменился. Автоматический откат отменён.')
    PIPELINE.write_bytes(original)
    if record['module_preexisted']:
        TARGET.write_bytes((path.parent/'footer_line_guided_v1.py').read_bytes())
    else:
        TARGET.unlink()
    print('Исходный pipeline.py восстановлен побайтово. Использована:',path.parent)


def verify():
    if not PIPELINE.exists():raise RuntimeError(f'Не найден {PIPELINE}')
    if not BUNDLED.exists():raise RuntimeError('Не найден модуль эксперимента в архиве')
    code=PIPELINE.read_text(encoding='utf-8')
    if MAGIC in code:
        print('Патч уже установлен');return
    final=patch_source(code)
    print('Проверка якорей пройдена, Python AST корректен.',len(final),'символов')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['install','restore','verify'])
    action=parser.parse_args().action
    try:
        {'install':install,'restore':restore,'verify':verify}[action]()
    except Exception as exc:
        print('ОШИБКА:',exc,file=sys.stderr)
        sys.exit(1)
