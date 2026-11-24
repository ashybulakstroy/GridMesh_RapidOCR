"""Regression checks for evidence-backed repair of a distorted grid line."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from App.geometry.geometry_lab.detection import Rule
from App.geometry.service import _repair_extreme_horizontal_curves, build_grid


class HorizontalRepairTests(unittest.TestCase):
    def test_extreme_curve_without_ink_is_not_invented(self):
        image=np.full((300,500,3),255,np.uint8)
        vertical=[Rule('v',np.array([0.,float(x)]),float(x),0.,(0,300))
                  for x in (40,80,260,340,460)]
        horizontal=[Rule('h',np.array(c,dtype=float),float(c[-1]),0.,(40,460))
                    for c in ([0,50],[0,100],[1000,0,150],[0,220])]
        data={'vertical':vertical,'horizontal':horizontal,'shape':image.shape[:2]}
        result,repaired,unresolved=_repair_extreme_horizontal_curves(image,data)
        self.assertEqual((repaired,unresolved),(0,1))
        self.assertIs(result,data)

    def test_occluded_receipt_lines_when_local_photo_is_available(self):
        path=Path(__file__).resolve().parents[2]/'data/output/5ce529fe31bc494fbfc0751adeab7ad1/original.png'
        if not path.is_file():
            self.skipTest('Local user receipt is not present')
        image=cv2.imread(str(path))
        grid,_,info=build_grid(image)
        self.assertEqual(info['repaired_extreme_horizontal'],1)
        self.assertEqual(len(grid.ys)-2,13)
        horizontal=grid.info['source_grid']['horizontal']
        self.assertTrue(any(470 < rule.position < 480 for rule in horizontal))
        self.assertTrue(any(500 < rule.position < 510 for rule in horizontal))


if __name__=='__main__':
    unittest.main()
