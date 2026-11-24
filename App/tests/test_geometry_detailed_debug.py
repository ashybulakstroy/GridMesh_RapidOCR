"""Regression test for opt-in geometry profiling on a real fallback case."""
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from App.geometry.service import build_grid
from App.ocr.debug_performance import activate_telemetry, deactivate_telemetry, get_debug


class GeometryDetailedDebugTests(unittest.TestCase):
    def test_fallback_records_nested_steps_only_in_debug(self):
        path=Path(__file__).resolve().parents[2]/'data/output/c551341577ab4bfda8884d2e65f95009/original.png'
        if not path.is_file():
            self.skipTest('Local user receipt is not present')
        image=cv2.imread(str(path))
        with patch.dict('os.environ',{'LOG_LEVEL':'DEBUG','OCR_LOCALGRID_MODE':'off'}):
            token=activate_telemetry('geometry-debug-test')
            try:
                debug=get_debug()
                grid,_,info=build_grid(image)
                profile=debug.finish()
            finally:
                deactivate_telemetry(token)
        steps=profile['geometry_substeps']
        by_name={item['step']:item for item in steps}
        self.assertEqual(info['method'],'Physical Grid Graph V3')
        self.assertIn('reason',by_name['autogrid_gridrefine_v3']['detail'])
        self.assertEqual(by_name['geometry_fallback_trigger']['detail']['reason'],'autogrid_rejected')
        self.assertIn('physical_v3_baseline_v2',by_name)
        self.assertIn('physical_v3_score_hypotheses',by_name)
        self.assertEqual({x['detail']['scale'] for x in steps if x['step']=='physical_v3_scale_line_detection'}, {1.0,1.5})
        with patch.dict('os.environ',{'LOG_LEVEL':'INFO','OCR_LOCALGRID_MODE':'off'}):
            token=activate_telemetry('geometry-info-test')
            try:
                self.assertIsNone(get_debug())
                info_grid,_,_=build_grid(image)
            finally:
                deactivate_telemetry(token)
        self.assertEqual(grid.xs,info_grid.xs)
        self.assertEqual(grid.ys,info_grid.ys)


if __name__=='__main__':
    unittest.main()
