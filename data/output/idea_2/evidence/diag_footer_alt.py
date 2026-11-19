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
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, OCRVersion, ModelType

from App.geometry.service import build_grid, cell_polygon
from App.ocr.engine import read_line

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

common = {
    'Rec.lang_type': LangRec.CYRILLIC, 'Rec.ocr_version': OCRVersion.PPOCRV5,
    'Rec.model_type': ModelType.MOBILE, 'Rec.rec_batch_num': 8,
    'Det.ocr_version': OCRVersion.PPOCRV5, 'Det.model_type': ModelType.MOBILE,
    'Global.use_cls': False, 'Global.text_score': 0.25,
    'EngineConfig.onnxruntime.intra_op_num_threads': 2,
    'EngineConfig.onnxruntime.inter_op_num_threads': 1,
}

alt = RapidOCR(params={**common, 'Det.limit_type': 'max', 'Det.limit_side_len': 736})
print('alt detector input:', alt._load_det_model().get_preprocess()(total_crop).shape)
alt(total_crop, use_det=True, use_cls=False)  # warm
for i in range(3):
    t0 = time.perf_counter(); r = alt(total_crop, use_det=True, use_cls=False)
    print(f'alt det=min->max full read_full run {i}: {(time.perf_counter()-t0)*1000:.1f} ms',
          list(zip(r.txts, [round(float(s), 3) for s in r.scores])))

for scale in (1, 2, 3):
    crop = total_crop if scale == 1 else cv2.resize(total_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    read_line(crop)
    t0 = time.perf_counter(); r = read_line(crop)
    print(f'read_line scale {scale}: {(time.perf_counter()-t0)*1000:.1f} ms',
          r['text'].strip(), round(r['confidence'], 3))
