"""Frozen grid-search regressions on distinct real receipts at LOG_LEVEL=INFO.

The pinned boundaries are observed detector output, not printed ground truth.
They make an unstable rescue route visible even when the receipt is still OCR-readable.
"""
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from App.geometry import service
from App.ocr.debug_performance import get_debug

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    digest: str
    method: str
    xs: tuple
    ys: tuple
    selection: str = None


CASES = (
    Case('short_51040', '796ae1bd2834d1cd', 'Physical Grid Graph V3',
         (20.07, 95.08, 427.16, 532.53, 685.75),
         (16.87, 90.71, 198.10, 254.63), 'strong_short_grid'),
    Case('fold_36981', 'eeccd07a7e4f0c8d', 'AutoGrid GridRefine V3',
         (36.05, 98.91, 408.92, 501.71, 641.81),
         (35.22, 96.71, 159.63, 224.91, 287.72, 389.91, 526.74,
          596.66, 670.59, 746.41, 824.03, 874.21)),
    Case('long_538936', '94fa3bf024dc6a45', 'Physical Grid Graph V3',
         (19.20, 42.64, 163.38, 200.77, 254.74),
         (24.31, 51.36, 77.48, 103.82, 129.65, 166.77, 204.30,
          231.04, 268.63, 306.84, 333.40, 360.03, 386.23, 412.07,
          450.78, 477.06, 515.83, 554.57, 580.99, 619.27, 646.31,
          698.51, 725.34, 752.74, 780.58, 808.55, 836.53, 889.88,
          917.72, 944.92, 972.97, 993.86), 'multiscale_joint_graph'),
    Case('problem_29464', 'ac99221713a3cf92', 'AutoGrid GridRefine V3',
         (23.04, 69.28, 306.50, 378.46, 481.20),
         (19.29, 68.58, 141.26, 189.14, 262.27, 313.16, 362.57,
          413.57, 453.04, 556.64, 638.04, 696.10, 782.85, 840.72, 885.97)),
    Case('problem_32596', '9647ff736a27acfb', 'AutoGrid GridRefine V3',
         (26.87, 91.70, 447.99, 561.71, 729.12),
         (18.61, 96.48, 173.46, 328.51, 563.42, 627.42)),
    Case('six_columns', 'fe857f29c3d3ae01', 'AutoGrid GridRefine V3',
         (31.68, 60.60, 236.90, 315.39, 386.55, 453.59, 527.10),
         (25.35, 82.56, 182.77, 259.68, 358.14, 455.09)),
    Case('wide_2142830460', 'e160fdaff47fd42b',
         'AutoGrid GridRefine V3 + Physical Grid Graph V3',
         (17.47, 42.68, 176.81, 219.84, 281.75),
         (23.61, 53.91, 84.13, 114.72, 137.63, 160.51, 182.79,
          213.11, 256.56, 286.24, 316.13, 403.02, 462.82, 495.01,
          541.88, 566.51)),
    Case('fallback_2142830502', '93a51fb9d0df824a', 'Physical Grid Graph V3',
         (37.82, 74.65, 266.65, 326.31, 412.39),
         (33.51, 77.35, 141.30, 183.55, 265.27, 326.92, 388.49,
          430.79, 494.41, 538.66, 572.30), 'multiscale_joint_graph'),
)

PATHS = {
    'short_51040': 'App/tests/fixtures/receipt_short_51040.jpg',
    'fold_36981': 'App/tests/fixtures/receipt_fold_36981.jpg',
    'long_538936': 'App/tests/fixtures/receipt_long_grid.png',
    'problem_29464': 'App/tests/fixtures/receipt_29464.jpg',
    'problem_32596': 'App/tests/fixtures/receipt_32596.jpg',
    'six_columns': 'App/tests/receipt_six_columns.jpg',
    'wide_2142830460': 'App/tests/receipt_2142830460.jpg',
    'fallback_2142830502': 'App/tests/receipt_2142830502.jpg',
}

CHILD = """
import json, sys
root, path = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
import cv2, numpy as np
from App.geometry.service import build_grid
image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
grid, _, geometry = build_grid(image)
print(json.dumps({
    'xs': [round(float(x), 2) for x in grid.xs],
    'ys': [round(float(y), 2) for y in grid.ys],
    'method': geometry['method'],
    'selection': geometry.get('physical_detection_selection'),
}))
"""


@pytest.fixture(autouse=True)
def info_logging(monkeypatch):
    monkeypatch.setenv('LOG_LEVEL', 'INFO')
    monkeypatch.setenv('OCR_LOCALGRID_MODE', 'on')
    monkeypatch.setenv('OCR_PAPER_RESCUE_V12', '1')
    monkeypatch.setenv('OCR_SOURCE_GRID_V16', '1')


def _image(case):
    path = ROOT / PATHS[case.name]
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _fingerprint(grid, geometry):
    return {
        'xs': [round(float(x), 2) for x in grid.xs],
        'ys': [round(float(y), 2) for y in grid.ys],
        'method': geometry['method'],
        'selection': geometry.get('physical_detection_selection'),
    }


def test_grid_fixtures_are_distinct_images():
    digests = [hashlib.sha256(_image(case).tobytes()).hexdigest()[:16] for case in CASES]
    assert len(set(digests)) == len(CASES)
    assert digests == [case.digest for case in CASES]


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.name)
def test_grid_search_is_frozen_and_repeatable(case):
    image = _image(case)
    grid, _, geometry = service.build_grid(image)
    assert get_debug() is None
    assert grid.method == 'original_cells'
    assert geometry['rectification'] == 'original_cells'
    assert geometry['method'] == case.method
    assert geometry.get('physical_detection_selection') == case.selection
    assert geometry['safety']['status'] == 'MEASURED'
    assert geometry['safety']['folded_fraction'] == 0
    assert geometry['safety']['source_valid_fraction'] >= 0.995
    assert list(grid.xs) == pytest.approx(case.xs, abs=0.75)
    assert list(grid.ys) == pytest.approx(case.ys, abs=0.75)
    assert np.all(np.diff(grid.xs) >= 5)
    assert np.all(np.diff(grid.ys) >= 5)
    assert len(grid.info['source_grid']['vertical']) == len(grid.xs)
    assert len(grid.info['source_grid']['horizontal']) == len(grid.ys)
    bands = np.diff(np.asarray(grid.ys, dtype=float))
    assert bands[0] <= max(90.0, 3.0 * float(np.median(bands[1:])))

    repeated, _, repeated_geometry = service.build_grid(image)
    assert _fingerprint(repeated, repeated_geometry) == _fingerprint(grid, geometry)
    assert np.array_equal(repeated.map_x, grid.map_x)
    assert np.array_equal(repeated.map_y, grid.map_y)


@pytest.mark.parametrize('seed', ['1', '314159'])
@pytest.mark.parametrize('name', ['short_51040', 'fallback_2142830502'])
def test_grid_search_is_stable_in_fresh_process(name, seed):
    case = next(item for item in CASES if item.name == name)
    env = dict(os.environ, LOG_LEVEL='INFO', PYTHONHASHSEED=seed,
               OCR_LOCALGRID_MODE='on', OCR_PAPER_RESCUE_V12='1',
               OCR_SOURCE_GRID_V16='1')
    completed = subprocess.run(
        [sys.executable, '-c', CHILD, str(ROOT), str(ROOT / PATHS[name])],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout) == {
        'xs': list(case.xs), 'ys': list(case.ys),
        'method': case.method, 'selection': case.selection,
    }


@pytest.mark.parametrize(('name', 'total'), [
    ('short_51040', '51040.00'),
    ('problem_32596', '32596.00'),
])
def test_pipeline_keeps_verified_grid_at_info_level(name, total):
    from App.pipeline import process
    case = next(item for item in CASES if item.name == name)
    result = process((ROOT / PATHS[name]).read_bytes())
    assert result['geometry']['method'] == case.method
    assert result['geometry']['safety']['status'] == 'MEASURED'
    assert result['total'] == total
    assert result['math_match']
