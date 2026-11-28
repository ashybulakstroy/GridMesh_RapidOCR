"""Keep functional test artifacts out of the user's live receipt directory."""
import pytest


@pytest.fixture(autouse=True)
def isolated_receipt_output(tmp_path, monkeypatch):
    monkeypatch.setenv('OUTPUT_DIR', str(tmp_path / 'receipts'))
