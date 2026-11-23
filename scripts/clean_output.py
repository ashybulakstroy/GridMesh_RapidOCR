"""Safe cleanup for data/output.

Two independent operations:
  duplicates - delete older job folders whose input.bin is byte-identical
  clean      - delete job folders that finished without any known error

Dry-run is the default. Deletion requires --apply (and confirmation unless
--yes is passed). Manifests are written under data/diagnostics/cleanup_manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / 'data' / 'output'
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / 'data' / 'diagnostics' / 'cleanup_manifests'
JOB_RE = re.compile(r'^[0-9a-fA-F]{32}$')
CLEAN_DENY_EVENTS = {'error', 'retry', 'check', 'budget', 'financial_warning'}
ACTIVE_MINUTES = 15


def human_bytes(value):
    return f'{value / 1024 ** 3:.3f} GiB ({value / 1e9:.3f} GB)'


def human_time(value):
    return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')


def dir_size(path):
    total = 0
    for file in path.rglob('*'):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            pass
    return total


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt(path):
    receipt_path = path / 'receipt.json'
    if not receipt_path.is_file():
        return None
    try:
        return json.loads(receipt_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def job_time(path):
    receipt_path = path / 'receipt.json'
    if receipt_path.is_file():
        return receipt_path.stat().st_mtime
    latest = 0.0
    for file in path.rglob('*'):
        try:
            if file.is_file():
                latest = max(latest, file.stat().st_mtime)
        except OSError:
            pass
    return latest or path.stat().st_mtime


def event_types(receipt):
    return {event.get('type') for event in (receipt.get('processing_log') or [])
            if isinstance(event, dict)}


def clean_reason(path, receipt):
    """Return None for a strictly clean job, otherwise a human-readable reason."""
    if receipt is None:
        return 'нет receipt.json'
    if receipt.get('status') != 'ok':
        return f"status={receipt.get('status')}"
    if receipt.get('math_match') is not True:
        return 'math_match не true'
    if (receipt.get('geometry') or {}).get('warnings'):
        return 'есть предупреждения'
    if receipt.get('numeric_issues'):
        return 'есть numeric_issues'
    if receipt.get('financial_warnings'):
        return 'есть financial_warnings'
    if (path / 'receipt_edited.json').exists():
        return 'есть ручные правки'
    if (path / 'error.json').exists():
        return 'есть error.json'
    if event_types(receipt) & CLEAN_DENY_EVENTS:
        return 'есть retry/check/budget'
    if receipt.get('ocr_user_action_required') not in (None, False):
        return 'требуется действие пользователя'
    if receipt.get('ocr_validation_status') not in (None, 'consistent'):
        return 'validation=review'
    if (receipt.get('vat') or {}).get('status') not in (None, 'match', 'not_applicable'):
        return 'НДС не совпал'
    matrix = receipt.get('numeric_matrix') or {}
    if matrix.get('status') not in (None, 'disabled', 'unique_math_match'):
        return f"matrix={matrix.get('status')}"
    if matrix.get('review_required'):
        return 'matrix требует проверки'
    return None


def plan_item(path, **extra):
    item = {
        'dir': path,
        'name': path.name,
        'size': dir_size(path),
        'mtime': job_time(path),
        'receipt': load_receipt(path),
    }
    item.update(extra)
    return item


def scan_clean(root):
    deletions = []
    skipped = Counter()
    jobs = 0
    for path in root.iterdir():
        if not path.is_dir() or not JOB_RE.match(path.name):
            continue
        jobs += 1
        receipt = load_receipt(path)
        reason = clean_reason(path, receipt)
        if reason is None:
            deletions.append(plan_item(path, reason='clean'))
        else:
            skipped[reason] += 1
    deletions.sort(key=lambda item: (-item['mtime'], item['name']))
    return {
        'kind': 'clean',
        'root': root,
        'job_dirs': jobs,
        'delete': deletions,
        'delete_count': len(deletions),
        'delete_bytes': sum(item['size'] for item in deletions),
        'skipped': dict(skipped),
    }


def scan_duplicates(root, active_minutes=ACTIVE_MINUTES, exclude_names=frozenset()):
    items = []
    for path in root.iterdir():
        if not path.is_dir() or not JOB_RE.match(path.name) or path.name in exclude_names:
            continue
        input_path = path / 'input.bin'
        if not input_path.is_file():
            continue
        items.append({
            'dir': path,
            'name': path.name,
            'input': input_path,
            'size': dir_size(path),
            'mtime': job_time(path),
            'receipt': load_receipt(path),
        })

    print(f'Сканирую input.bin: {len(items)} папок...')
    groups = defaultdict(list)
    for index, item in enumerate(items, 1):
        item['hash'] = file_hash(item['input'])
        groups[item['hash']].append(item)
        if len(items) >= 200 and (index % 250 == 0 or index == len(items)):
            print(f'\r  проверено {index}/{len(items)}', end='', flush=True)
    if len(items) >= 200:
        print()

    now = time.time()
    deletions = []
    preserved = []
    keepers = []
    duplicate_groups = {digest: group for digest, group in groups.items() if len(group) > 1}
    for digest, group in duplicate_groups.items():
        group.sort(key=lambda item: (-item['mtime'], item['name']))
        keeper = group[0]
        keep_names = {keeper['name']}
        # If the newest job did not finish, keep the newest finished copy too.
        if keeper['receipt'] is None:
            finished = [item for item in group if item['receipt'] is not None]
            if finished:
                keep_names.add(finished[0]['name'])
        keepers.append({
            'job_id': keeper['name'],
            'hash': digest,
            'bytes': keeper['size'],
            'mtime': keeper['mtime'],
            'status': (keeper['receipt'] or {}).get('status'),
            'total': (keeper['receipt'] or {}).get('total'),
            'extra_kept': sorted(keep_names - {keeper['name']}),
        })
        for item in group:
            if item['name'] in keep_names:
                continue
            if (item['dir'] / 'receipt_edited.json').exists():
                preserved.append(plan_item(item['dir'], hash=digest,
                                           reason='есть ручные правки'))
                continue
            if item['receipt'] is None and now - item['mtime'] < active_minutes * 60:
                preserved.append(plan_item(
                    item['dir'], hash=digest,
                    reason=f'без receipt.json и изменена менее {active_minutes} мин назад'))
                continue
            deletions.append(plan_item(item['dir'], hash=digest,
                                       keeper=keeper['name']))

    deletions.sort(key=lambda item: (-item['mtime'], item['name']))
    return {
        'kind': 'duplicates',
        'root': root,
        'job_dirs_with_input': len(items),
        'unique_hashes': len(groups),
        'duplicate_groups': len(duplicate_groups),
        'keepers': keepers,
        'delete': deletions,
        'delete_count': len(deletions),
        'delete_bytes': sum(item['size'] for item in deletions),
        'preserved': preserved,
    }


def print_plan(plan, preview=10):
    print()
    if plan['kind'] == 'clean':
        print('--- Чистые чеки ---')
        print(f"Всего job-папок: {plan['job_dirs']}")
        print(f"К удалению: {plan['delete_count']} папок, {human_bytes(plan['delete_bytes'])}")
        if plan['skipped']:
            print('Пропущено:')
            for reason, count in sorted(plan['skipped'].items(), key=lambda item: (-item[1], item[0])):
                print(f'  {count:4}  {reason}')
        for item in plan['delete'][:preview]:
            receipt = item['receipt'] or {}
            print(f"  DEL {item['name']}  {item['size'] / 1024 ** 2:7.2f} MiB  "
                  f"{human_time(item['mtime'])}  total={receipt.get('total')}")
        if plan['delete_count'] > preview:
            print(f'  ... ещё {plan["delete_count"] - preview}')
    else:
        print('--- Дубликаты input.bin ---')
        print(f"Job-папок с input.bin: {plan['job_dirs_with_input']}")
        print(f"Уникальных файлов: {plan['unique_hashes']}, "
              f"групп дубликатов: {plan['duplicate_groups']}")
        print(f"К удалению: {plan['delete_count']} папок, {human_bytes(plan['delete_bytes'])}")
        print(f"Сохранено: {len(plan['preserved'])} дубликатов "
              f"(ручные правки/возможно активные)")
        for item in plan['preserved'][:preview]:
            print(f"  KEEP {item['name']}  {item['reason']}")
        if len(plan['preserved']) > preview:
            print(f'  ... ещё {len(plan["preserved"]) - preview}')
        for item in plan['delete'][:preview]:
            receipt = item['receipt'] or {}
            print(f"  DEL {item['name']}  {item['size'] / 1024 ** 2:7.2f} MiB  "
                  f"{human_time(item['mtime'])}  keep={item['keeper']}  "
                  f"status={receipt.get('status')}")
        if plan['delete_count'] > preview:
            print(f'  ... ещё {plan["delete_count"] - preview}')


def public_plan(plan):
    payload = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'kind': plan['kind'],
        'root': str(plan['root']),
        'delete_count': plan['delete_count'],
        'delete_bytes': plan['delete_bytes'],
    }
    if plan['kind'] == 'clean':
        payload['job_dirs'] = plan['job_dirs']
        payload['skipped'] = plan['skipped']
        payload['delete'] = [{
            'job_id': item['name'],
            'bytes': item['size'],
            'mtime': datetime.fromtimestamp(item['mtime'], timezone.utc).isoformat(),
            'status': (item['receipt'] or {}).get('status'),
            'total': (item['receipt'] or {}).get('total'),
        } for item in plan['delete']]
    else:
        payload.update({
            'job_dirs_with_input': plan['job_dirs_with_input'],
            'unique_hashes': plan['unique_hashes'],
            'duplicate_groups': plan['duplicate_groups'],
            'keepers': [{
                **keeper,
                'mtime': datetime.fromtimestamp(keeper['mtime'], timezone.utc).isoformat(),
            } for keeper in plan['keepers']],
            'delete': [{
                'job_id': item['name'],
                'keeper': item['keeper'],
                'hash': item['hash'],
                'bytes': item['size'],
                'mtime': datetime.fromtimestamp(item['mtime'], timezone.utc).isoformat(),
                'status': (item['receipt'] or {}).get('status'),
                'total': (item['receipt'] or {}).get('total'),
            } for item in plan['delete']],
            'preserved': [{
                'job_id': item['name'],
                'reason': item['reason'],
                'hash': item['hash'],
                'bytes': item['size'],
                'mtime': datetime.fromtimestamp(item['mtime'], timezone.utc).isoformat(),
                'status': (item['receipt'] or {}).get('status'),
                'total': (item['receipt'] or {}).get('total'),
            } for item in plan['preserved']],
        })
    return payload


def force_remove(function, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except Exception:
        raise


def apply_plan(plan, manifest_dir):
    if plan['delete_count'] == 0:
        print('\nНечего удалять.')
        return None
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manifest_path = manifest_dir / f"{plan['kind']}_{stamp}.json"
    payload = public_plan(plan)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    root = plan['root'].resolve()
    deleted = []
    failed = []
    freed = 0
    for item in plan['delete']:
        path = item['dir'].resolve()
        if path.parent != root or not JOB_RE.match(path.name):
            failed.append({'job_id': item['name'], 'error': 'path safety check failed'})
            continue
        if not path.is_dir():
            continue
        if (path / 'receipt_edited.json').exists():
            failed.append({'job_id': item['name'],
                           'error': 'appeared receipt_edited.json; folder preserved'})
            continue
        for attempt in range(2):
            try:
                shutil.rmtree(path, onerror=force_remove)
                deleted.append(item['name'])
                freed += item['size']
                break
            except Exception as exc:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                failed.append({'job_id': item['name'], 'error': f'{type(exc).__name__}: {exc}'})

    payload.update({
        'deleted_count': len(deleted),
        'freed_bytes': freed,
        'deleted_job_ids': deleted,
        'failed': failed,
    })
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nУдалено {len(deleted)} папок, освобождено {human_bytes(freed)}')
    if failed:
        print(f'Не удалось удалить: {len(failed)}')
        for failure in failed[:20]:
            print(f"  FAIL {failure['job_id']}: {failure['error']}")
    print(f'Манифест: {manifest_path}')
    return manifest_path


def ask_yes(prompt):
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {'y', 'yes', 'д', 'да'}


def run_mode(mode, apply, root, manifest_dir, active_minutes=ACTIVE_MINUTES, yes=False):
    if mode == 'clean':
        plan = scan_clean(root)
        print_plan(plan)
        if not apply:
            print('\nРежим проверки: ничего не удалено.')
            return
        if plan['delete_count'] == 0:
            return
        if not yes and not ask_yes(f"\nУдалить {plan['delete_count']} папок чистых чеков? [y/N]: "):
            print('Отменено.')
            return
        apply_plan(plan, manifest_dir)
        return

    if mode == 'duplicates':
        plan = scan_duplicates(root, active_minutes)
        print_plan(plan)
        if not apply:
            print('\nРежим проверки: ничего не удалено.')
            return
        if plan['delete_count'] == 0:
            return
        if not yes and not ask_yes(f"\nУдалить {plan['delete_count']} папок-дубликатов? [y/N]: "):
            print('Отменено.')
            return
        apply_plan(plan, manifest_dir)
        return

    if mode == 'both':
        clean_plan = scan_clean(root)
        print_plan(clean_plan)
        if not apply:
            exclude = {item['name'] for item in clean_plan['delete']}
            duplicate_plan = scan_duplicates(root, active_minutes, exclude_names=exclude)
            print_plan(duplicate_plan)
            print('\nРежим проверки: ничего не удалено.')
            return
        if clean_plan['delete_count']:
            if not yes and not ask_yes(
                    f"\nШаг 1/2. Удалить {clean_plan['delete_count']} папок чистых чеков? [y/N]: "):
                print('Отменено.')
                return
            apply_plan(clean_plan, manifest_dir)
        # Re-scan after the first step: keepers may have changed.
        duplicate_plan = scan_duplicates(root, active_minutes)
        print_plan(duplicate_plan)
        if duplicate_plan['delete_count']:
            if not yes and not ask_yes(
                    f"\nШаг 2/2. Удалить {duplicate_plan['delete_count']} папок-дубликатов? [y/N]: "):
                print('Отменено.')
                return
            apply_plan(duplicate_plan, manifest_dir)
        return

    raise ValueError(f'Unknown mode: {mode}')


def interactive(root, manifest_dir, active_minutes):
    actions = {
        '1': ('duplicates', False),
        '2': ('duplicates', True),
        '3': ('clean', False),
        '4': ('clean', True),
        '5': ('both', False),
        '6': ('both', True),
    }
    while True:
        print('\n=== Очистка data/output ===')
        print('1. Показать дубликаты input.bin')
        print('2. Удалить дубликаты input.bin')
        print('3. Показать чистые чеки')
        print('4. Удалить чистые чеки')
        print('5. Показать чистые чеки и дубликаты')
        print('6. Удалить чистые чеки, затем дубликаты')
        print('0. Выход')
        try:
            choice = input('Выбор: ').strip()
        except EOFError:
            return 0
        if choice == '0':
            return 0
        if choice not in actions:
            print('Неизвестный пункт меню.')
            continue
        mode, apply = actions[choice]
        try:
            run_mode(mode, apply, root, manifest_dir, active_minutes)
        except KeyboardInterrupt:
            print('\nОтменено.')
        except Exception as exc:
            print(f'Ошибка: {type(exc).__name__}: {exc}')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Очистка data/output: дубликаты input.bin и чистые чеки.')
    parser.add_argument('--mode', choices=['duplicates', 'clean', 'both'],
                        help='Операция; без параметра открывается меню.')
    parser.add_argument('--apply', action='store_true',
                        help='Выполнить удаление (по умолчанию только проверка).')
    parser.add_argument('--yes', action='store_true',
                        help='Не спрашивать подтверждение при --apply.')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help=f'Папка с job-папками (по умолчанию {DEFAULT_ROOT}).')
    parser.add_argument('--manifest-dir', type=Path, default=DEFAULT_MANIFEST_DIR,
                        help=f'Куда писать манифесты (по умолчанию {DEFAULT_MANIFEST_DIR}).')
    parser.add_argument('--active-minutes', type=int, default=ACTIVE_MINUTES,
                        help='Не удалять недавние папки без receipt.json (по умолчанию 15).')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f'Папка не найдена: {root}', file=sys.stderr)
        return 2
    manifest_dir = args.manifest_dir.resolve()
    if args.mode is None:
        return interactive(root, manifest_dir, args.active_minutes)
    try:
        run_mode(args.mode, args.apply, root, manifest_dir,
                 args.active_minutes, args.yes)
    except KeyboardInterrupt:
        print('\nОтменено.')
        return 130
    except Exception as exc:
        print(f'Ошибка: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
