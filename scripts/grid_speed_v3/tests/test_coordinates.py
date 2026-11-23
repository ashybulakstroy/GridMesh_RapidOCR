"""Offline check of Rule/PolylineRule coordinate scaling with minimal fake imports."""
import sys
import types
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grid_speed_experiment import scale_graph


class Rule:
    def __init__(self,axis,coeff,position,support,domain):
        self.axis=axis
        self.coeff=np.asarray(coeff,float)
        self.position=position
        self.support=support
        self.domain=domain
    def eval(self,coord):
        return np.polyval(self.coeff, coord)


class PolylineRule:
    def __init__(self,points,axis,shape):
        self.points=np.asarray(points,float)
        self.axis=axis
        along=1 if axis=='v' else 0
        self.extent=shape[0 if axis=='v' else 1]
        idx=np.argsort(self.points[:,along])
        self.t=self.points[idx,along]
        self.values=self.points[idx,1-along]
        self.domain=(self.t[0],self.t[-1])
        self.position=float(self.eval(0))
        self.support=(self.t[-1]-self.t[0])/self.extent
    def eval(self,coord):
        return np.interp(np.asarray(coord)*self.extent+self.extent/2,self.t,self.values)


class ScaleRuleTests(unittest.TestCase):
    def setUp(self):
        self.restore={}
        for name in ['App','App.geometry','App.geometry.geometry_lab',
                     'App.geometry.geometry_lab.detection',
                     'App.geometry.geometry_lab.structure_diagnostics']:
            self.restore[name]=sys.modules.get(name)
            sys.modules[name]=types.ModuleType(name)
        sys.modules['App.geometry.geometry_lab.detection'].Rule=Rule
        sys.modules['App.geometry.geometry_lab.structure_diagnostics'].PolylineRule=PolylineRule

    def tearDown(self):
        for name,val in self.restore.items():
            if val is None:sys.modules.pop(name,None)
            else:sys.modules[name]=val

    def test_coefficients_transform_to_original_pixel_space(self):
        v=Rule('v',[2,80],80,1,(10,200))
        h=Rule('h',[-4,95],95,1,(20,300))
        data=scale_graph({'vertical':[v], 'horizontal':[h]},(250,400),(500,800))
        self.assertAlmostEqual(float(data['vertical'][0].eval(.2)),2*float(v.eval(.2)))
        self.assertAlmostEqual(float(data['horizontal'][0].eval(.3)),2*float(h.eval(.3)))
        self.assertEqual(data['vertical'][0].domain,(20,400))
        self.assertEqual(data['horizontal'][0].domain,(40,600))

    def test_polyline_rule_scales_both_coordinate_axes(self):
        p=PolylineRule([[20,30],[22,120],[25,200]],'v',(250,400))
        data=scale_graph({'vertical':[p], 'horizontal':[]},(250,400),(500,800))
        np.testing.assert_allclose(data['vertical'][0].points, np.array([[40,60],[44,240],[50,400]]))


if __name__=='__main__':
    unittest.main()
