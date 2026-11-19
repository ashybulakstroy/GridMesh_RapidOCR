import glob
import os
import sys
import time

ROOT = r'C:\Work\Projects\Prj_18_OCR_2'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
from PIL import Image, ImageOps

from App.geometry.service import build_grid, cell_polygon
from App.geometry.geometry_lab.physical_grid_graph_v3 import multiscale_segments, fit_segments
from App.geometry.geometry_lab.physical_grid_graph_local import ridge_response
from App.ocr.engine import engine

REFERENCE = os.path.join('data/output', '0251327d5d384287926c40adfb259768')
JOB = sys.argv[1] if len(sys.argv) > 1 else REFERENCE
with Image.open(os.path.join(JOB, 'input.bin')) as photo:
    rgb = ImageOps.exif_transpose(photo).convert('RGB')
image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
cv2.setNumThreads(2)

rect, overlay, geometry = build_grid(image, rectify=False)
xs = [int(round(x)) for x in rect.xs]
ys = [int(round(y)) for y in rect.ys]
y_bottom = int(max(np.max(cell_polygon(rect, len(ys) - 2, k)[34:51, 1])
                   for k in range(len(xs) - 1)) + rect.info['source_offset'][1])
footer_top = max(0, y_bottom - 20)
footer = image[footer_top:min(image.shape[0], y_bottom + max(55, int(image.shape[1] * .09))), :]
total_crop = footer[:, int(image.shape[1] * .45):]

ocr = engine()
prep = ocr._load_det_model().get_preprocess()
print('footer crop:', total_crop.shape)
print('detector input tensor:', prep(total_crop).shape, '(N,C,H,W)')
print('=> detector processes', prep(total_crop).shape[2] * prep(total_crop).shape[3], 'pixels')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
t0 = time.perf_counter(); response, dark = ridge_response(gray)
print(f'V3: ridge_response             {(time.perf_counter()-t0)*1000:8.1f} ms')
t0 = time.perf_counter(); parts = multiscale_segments(gray)
print(f'V3: multiscale_segments (LSDx2){(time.perf_counter()-t0)*1000:8.1f} ms')
t0 = time.perf_counter(); v = fit_segments(parts['v'], 'v', gray.shape)
h = fit_segments(parts['h'], 'h', gray.shape)
print(f'V3: fit_segments v+h           {(time.perf_counter()-t0)*1000:8.1f} ms',
      'vertical:', len(v), 'horizontal:', len(h))
