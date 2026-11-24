import os
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from App.geometry import service
from App.geometry.grid_completeness import audit_printed_rows
from App.geometry.geometry_lab.algorithms import intersection


class LocalGridRouteTest(unittest.TestCase):
    def test_disabled_mode_uses_original_detector(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('App.config.setting', return_value='off'), \
                patch.object(service, '_build_grid_core', return_value=('old', None, {})) as core, \
                patch('App.geometry.local_grid_runtime.try_local_grid') as local:
            self.assertEqual(service.build_grid(np.zeros((20, 20, 3), np.uint8))[0], 'old')
            local.assert_not_called()
            core.assert_called_once()

    def test_local_failure_uses_original_detector(self):
        with patch('App.config.setting', return_value='on'), \
                patch('App.geometry.local_grid_runtime.try_local_grid', side_effect=ValueError('failed')), \
                patch.object(service, '_build_grid_core', return_value=('old', None, {})) as core:
            result, _, info = service.build_grid(np.zeros((20, 20, 3), np.uint8))
            self.assertEqual(result, 'old')
            self.assertEqual(info['local_grid_roi']['status'], 'fallback_to_smart')
            core.assert_called_once()

    def test_guard_rejects_missing_upper_partial_rows(self):
        path = Path('data/output/0fd0075dff8643389f260ed2ed78999a/original.png')
        if not path.exists():
            self.skipTest('receipt image not available')
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        with patch('App.config.setting', return_value='off'):
            result, _, _ = service.build_grid(image)
        graph = result.info['source_grid']
        nodes = np.array([[intersection(v, h, image.shape) for v in graph['vertical']]
                          for h in graph['horizontal']])
        self.assertEqual(len(nodes), 6)
        accepted, audit = audit_printed_rows(image, nodes[2:])
        self.assertFalse(accepted)
        self.assertEqual(audit['reason'], 'suspicious_partial_unmatched_line')


if __name__ == '__main__':
    unittest.main()
