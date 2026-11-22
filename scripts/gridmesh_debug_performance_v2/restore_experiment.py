#!/usr/bin/env python3
"""Undo GridMesh DEBUG performance V2 experiment, refusing overwrite of later edits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

EXPERIMENT = 'gridmesh_debug_performance_v2'

def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def restore(project: Path, *, dry_run: bool = False, force: bool = False) -> list[str]:
    project = project.resolve()
    exp = project / 'data' / 'experiments' / EXPERIMENT
    backup = exp / 'backup'
    manifest = json.loads((exp / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['experiment'] != EXPERIMENT:
        raise ValueError('Unexpected experiment manifest')
    old, installed = manifest['original_sha256'], manifest['patched_sha256']
    paths = list(old)
    for path in paths:
        target = project / path
        saved = backup / path
        if not saved.is_file() or digest(saved.read_bytes()) != old[path]:
            raise ValueError('Backup missing or modified: ' + path)
        if target.is_file() and digest(target.read_bytes()) != installed[path] and not force:
            raise ValueError('Project file changed since install; refusing to overwrite: ' + path)
    new = project / manifest['new_file']
    if new.is_file() and digest(new.read_bytes()) != manifest['new_file_sha256'] and not force:
        raise ValueError('DEBUG module has since changed; refusing to remove it')
    if not dry_run:
        for path in paths:
            target = project / path
            temp = target.with_suffix(target.suffix + '.restoretmp')
            temp.write_bytes((backup / path).read_bytes())
            temp.replace(target)
        if new.exists():
            new.unlink()
    return paths + [manifest['new_file']]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='Overwrite any later edits (dangerous)')
    args = ap.parse_args()
    print('Restored:' if not args.dry_run else 'Would restore:', *restore(Path(args.project), dry_run=args.dry_run, force=args.force), sep='\n  ')
