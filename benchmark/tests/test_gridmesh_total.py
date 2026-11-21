import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark.core.gridmesh import total_crop


class TotalMeshTests(unittest.TestCase):
    def test_coordinates_reused_without_detection(self):
        from PIL import Image, ImageOps
        from benchmark.core.gridmesh import extract_gridmesh
        from App.geometry.service import build_grid
        root=Path(__file__).resolve().parents[1]/'data'
        ann=json.loads((root/'annotations/short_01_2142701072.json').read_text(encoding='utf-8'))
        with Image.open(root/'images'/ann['image_file']) as im:
            image=np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(im).convert('RGB'))[:,:,::-1])
        with tempfile.TemporaryDirectory() as cache:
            with patch('App.geometry.service.build_grid', wraps=build_grid) as detect:
                first,_=extract_gridmesh(image,ann,include_total=True,cache_dir=cache)
                second,_=extract_gridmesh(image,ann,include_total=True,cache_dir=cache)
                self.assertEqual(detect.call_count,1)
                changed=image.copy(); changed[0,0]=255-changed[0,0]
                extract_gridmesh(changed,ann,include_total=True,cache_dir=cache)
                self.assertEqual(detect.call_count,2)
            for (fa,a),(fb,b) in zip(first,second):
                self.assertEqual(fa,fb)
                np.testing.assert_array_equal(a,b)

    def test_separate_total_preserves_annotated_pixels(self):
        image=np.random.default_rng(18).integers(0,256,(100,160,3),dtype=np.uint8)
        ann={'rois':[{'field_id':'printed_total','bbox':[.25,.6,.75,.9]}]}
        crop,corners=total_crop(image,ann)
        self.assertEqual(corners,[[40,60],[120,60],[120,90],[40,90]])
        np.testing.assert_array_equal(crop,image[60:90,40:120])

    def test_invalid_total_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            total_crop(np.zeros((100,160,3),dtype=np.uint8),
                       {'rois':[{'field_id':'printed_total','bbox':[.3,.8,1.2,.9]}]})
