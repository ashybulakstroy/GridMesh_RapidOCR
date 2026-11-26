"""Local .env configuration; process environment has priority."""
# GRID_ABORT_DIAGNOSTICS_V1
from App import grid_abort_diagnostics_v1 as _grid_diag
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@_grid_diag.config_read
def setting(name, default):
    if name in os.environ:
        return os.environ[name]
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip() == name:
                return value.strip().strip('"\'')
    return default

def positive_int(name, default):
    value = int(setting(name, str(default)))
    if value < 1: raise ValueError(f'{name}: требуется положительное целое')
    return value

def output_directory():
    """Allow project output and explicitly configured system temporary directories."""
    path = (ROOT / setting('OUTPUT_DIR', 'data/output')).resolve()
    allowed = [ROOT.resolve()]
    allowed.extend(Path(value).resolve() for name in ('TEMP', 'TMP')
                   if (value := os.environ.get(name)))
    if not any(path.is_relative_to(base) for base in allowed):
        raise ValueError('OUTPUT_DIR должен находиться внутри проекта, TEMP или TMP')
    return path

def display_path(path):
    path = Path(path).resolve()
    return (path.relative_to(ROOT.resolve()) if path.is_relative_to(ROOT.resolve()) else path).as_posix()
