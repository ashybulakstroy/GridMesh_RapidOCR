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
from App.ocr.engine import engine, read_full

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
print('crop:', total_crop.shape)

ocr = engine()
# warm everything
ocr(total_crop, use_det=True, use_cls=False)

prepared, record = ocr.preprocess_img(total_crop)
print('after preprocess:', prepared.shape, record)
det = ocr._load_det_model()

for i in range(3):
    t0 = time.perf_counter()
    det_res = det(prepared)
    t1 = time.perf_counter()
    boxes = det_res.boxes
    print(f'det run {i}: {(t1-t0)*1000:.1f} ms, boxes:', None if boxes is None else len(boxes))
    crops = ocr.crop_text_regions(prepared, boxes)
    print('  crop shapes:', [c.shape for c in crops])
    t2 = time.perf_counter()
    rec_res = ocr.recognize_txt(crops)
    t3 = time.perf_counter()
    print(f'  rec: {(t3-t2)*1000:.1f} ms', list(zip(rec_res.txts, [round(float(s), 3) for s in rec_res.scores])))

for i in range(3):
    t0 = time.perf_counter()
    out = ocr(total_crop, use_det=True, use_cls=False)
    print(f'full call {i}: {(time.perf_counter()-t0)*1000:.1f} ms')
