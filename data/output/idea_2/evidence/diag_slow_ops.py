import glob
import json
import os
import sys
import time
from io import BytesIO

ROOT = r'C:\Work\Projects\Prj_18_OCR_2'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
from PIL import Image, ImageOps

from App.geometry.autogrid_gridrefine_v3 import detector as ag
from App.geometry.autogrid_gridrefine_v3.fitter import fit_grid
from App.geometry.geometry_lab.physical_grid_graph_local import detect_physical_graph as v2
from App.geometry.geometry_lab.physical_grid_graph_v3 import detect_physical_graph as v3
from App.geometry.lab_adapter_gridrefine_v3 import detect_for_lab
from App.geometry.service import build_grid, cell_polygon
from App.ocr.engine import read_full, read_line

REFERENCE = os.path.join('data/output', '0251327d5d384287926c40adfb259768')
JOB = sys.argv[1] if len(sys.argv) > 1 else REFERENCE
print('job:', os.path.basename(JOB))

with Image.open(BytesIO(open(os.path.join(JOB, 'input.bin'), 'rb').read())) as photo:
    rgb = ImageOps.exif_transpose(photo).convert('RGB')
image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
cv2.setNumThreads(2)
print('image:', image.shape)


def timed(label, fn):
    t0 = time.perf_counter()
    result = fn()
    print(f'{label:<48} {(time.perf_counter() - t0) * 1000:8.1f} ms')
    return result


gray, segments, lsd_masks = timed('  AutoGrid: LSD segments', lambda: ag._h_v_segments(image))
morph_masks = timed('  AutoGrid: adaptive threshold + morphology', lambda: ag._morph_masks(gray))
candidates = timed('  AutoGrid: connected components + ranking', lambda: ag._rank_candidates(gray, lsd_masks, morph_masks))
fit = timed('  AutoGrid: fit_grid on best candidate', lambda: fit_grid(image, candidates[0].box)) if candidates else None
if fit:
    print('    fit status:', fit['status'], 'h:', fit['horizontal_boundaries'], 'v:', fit['vertical_boundaries'])

data, report = timed('AutoGrid total (detect_for_lab)', lambda: detect_for_lab(image))
print('  status:', report.get('status'), 'vertical:', len(data['vertical']) if data else None,
      'horizontal:', len(data['horizontal']) if data else None)

base, base_report = timed('PhysicalGraph V2 (called inside V3)', lambda: v2(image))
print('  status:', base_report.get('status'), 'horizontal:', len(base['horizontal']) if base else None)

phys, phys_report = timed('PhysicalGraph V3 total', lambda: v3(image))
print('  status:', phys_report.get('status'), 'selection:', phys_report.get('selection'))
title = timed('build_grid total (as in pipeline)', lambda: build_grid(image, rectify=False))
rect, overlay, geometry = title
print('  method:', geometry['method'], 'xs:', len(rect.xs), 'ys:', len(rect.ys))

# ---------------- footer / printed total ----------------
xs = [int(round(x)) for x in rect.xs]
ys = [int(round(y)) for y in rect.ys]
last_y = min(rect.map_y.shape[0] - 1, max(0, ys[-1]))
if rect.method == 'original_cells':
    y_bottom = int(max(np.max(cell_polygon(rect, len(ys) - 2, k)[34:51, 1])
                       for k in range(len(xs) - 1)) + rect.info['source_offset'][1])
else:
    y_bottom = int(np.max(rect.map_y[last_y, max(0, xs[0]):min(rect.map_y.shape[1], xs[-1])]))
footer_top = max(0, y_bottom - 20)
footer = image[footer_top:min(image.shape[0], y_bottom + max(55, int(image.shape[1] * .09))), :]
footer_left = int(image.shape[1] * .45)
total_crop = footer[:, footer_left:]
print('footer:', footer.shape, 'y_bottom:', y_bottom)

fast_x = int(image.shape[1] * .50)
fast_top, fast_bottom = max(0, y_bottom + 2), min(image.shape[0], y_bottom + int(image.shape[1] * .05))
d = timed('footer: direct read_line', lambda: read_line(image[fast_top:fast_bottom, fast_x:]))
print('  ', d)
for frac, offset, height in ((.45, -20, .06), (.50, -25, .08)):
    left = int(image.shape[1] * frac)
    top = max(0, y_bottom + offset)
    bottom = min(image.shape[0], top + int(image.shape[1] * height))
    line = timed(f'footer: compact read_line ({frac},{offset},{height})', lambda: read_line(image[top:bottom, left:]))
    print('  ', line)

for attempt in (1, 2):
    lines = timed(f'footer: read_full scale 1 (attempt {attempt})', lambda: read_full(total_crop))
    print('  ', [(x['text'], round(x['confidence'], 3)) for x in lines])
sample2 = cv2.resize(total_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
lines2 = timed('footer: read_full scale 2', lambda: read_full(sample2))
print('  ', [(x['text'], round(x['confidence'], 3)) for x in lines2])
