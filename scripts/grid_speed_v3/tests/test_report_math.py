import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grid_speed_experiment import compare_grids, add_truth, summary_stats, report_html, write_csv


def fake_grid(rows=2, offset=0):
    return {
        'vertical_lines': 5,
        'horizontal_lines': rows + 2,
        'body_rows': rows,
        'intersections': [[[50 + j*40 + offset, 10 + i*32 + offset]
                           for j in range(5)] for i in range(rows + 2)],
        'last_boundary_y_px': 10+(rows+1)*32,
    }


class GridMathTests(unittest.TestCase):
    def test_same_grid_accepted(self):
        x = compare_grids(fake_grid(),fake_grid(),median_limit=3,max_limit=8)
        self.assertTrue(x['accepted'])
        self.assertEqual(x['status'],'geometry_match')

    def test_lost_row_always_rejected(self):
        x = compare_grids(fake_grid(),fake_grid(rows=1),median_limit=999,max_limit=999)
        self.assertFalse(x['accepted'])
        self.assertEqual(x['status'],'different_topology')

    def test_shift_rejected(self):
        x = compare_grids(fake_grid(),fake_grid(offset=12),median_limit=3,max_limit=8)
        self.assertFalse(x['accepted'])
        self.assertEqual(x['status'],'position_mismatch')

    def test_baseline_missing_never_treated_as_truth(self):
        x = compare_grids(None,fake_grid(),median_limit=3,max_limit=8)
        self.assertEqual(x['status'],'unverified_no_baseline')
        self.assertFalse(x['accepted'])

    def test_labeled_row_count(self):
        q={'status':'geometry_match','accepted':True}
        self.assertFalse(add_truth(q,'x.jpg',fake_grid(),{'x.jpg':{'body_rows':3}})['accepted'])
        self.assertTrue(add_truth(q,'x.jpg',fake_grid(),{'x.jpg':{'body_rows':2}})['accepted'])

    def test_write_reports_offline(self):
        q={'status':'reference_baseline','accepted':True,'truth_gate':'pass'}
        rec={'photo':'receipt_short_51040.jpg','orientation':'original','mode':'baseline',
             'repetition':1,'status':'ok','wall_ms':780.0,'cpu_ms':2000.0,
             'grid':fake_grid(),'substeps':{'auto_ms':200.0,'physical_ms':500.0},
             'quality':q,'overlay_file':None,'error':None}
        s=summary_stats([rec],['baseline'])
        self.assertEqual(s[0]['median_wall_ms'],780.0)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            write_csv(dest/'results.csv',[rec])
            report_html(dest,{'authored_commit':'test','local_commit':None,
                               'date':'2026-11-23','photos_count':1,
                               'results':[rec],'summary':s})
            self.assertIn('780', (dest/'results.csv').read_text('utf-8-sig'))
            self.assertIn('Grid Speed V3', (dest/'summary.html').read_text('utf-8'))


if __name__=='__main__':
    unittest.main()
