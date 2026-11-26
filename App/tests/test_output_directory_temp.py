from pathlib import Path

import pytest

from App import config


def test_output_allows_project_and_system_temp(monkeypatch, tmp_path):
    root = tmp_path / 'project'
    temp = tmp_path / 'system_temp'
    monkeypatch.setattr(config, 'ROOT', root)
    monkeypatch.setenv('TEMP', str(temp))
    monkeypatch.delenv('TMP', raising=False)
    monkeypatch.setenv('OUTPUT_DIR', str(temp / 'runs'))
    assert config.output_directory() == (temp / 'runs').resolve()
    assert config.display_path(temp / 'runs') == (temp / 'runs').resolve().as_posix()
    monkeypatch.setenv('OUTPUT_DIR', 'data/output')
    assert config.output_directory() == (root / 'data/output').resolve()
    assert config.display_path(root / 'data/output') == 'data/output'


def test_output_rejects_paths_outside_project_and_temp(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'ROOT', tmp_path / 'project')
    monkeypatch.setenv('TEMP', str(tmp_path / 'temp'))
    monkeypatch.setenv('TMP', str(tmp_path / 'tmp'))
    monkeypatch.setenv('OUTPUT_DIR', str(tmp_path / 'other'))
    with pytest.raises(ValueError):
        config.output_directory()
    monkeypatch.setenv('OUTPUT_DIR', str(tmp_path / 'tmp' / 'runs'))
    assert config.output_directory() == (tmp_path / 'tmp' / 'runs').resolve()
