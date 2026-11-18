"""Local .env configuration; process environment has priority."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
