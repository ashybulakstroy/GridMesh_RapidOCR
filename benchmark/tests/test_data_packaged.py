import unittest
from pathlib import Path
from benchmark.core.runner import read_ground_truth, check_inputs
class DataPackagedTests(unittest.TestCase):
    def test_three_originals_and_matching_annotations(self):
        records=read_ground_truth()
        self.assertEqual(len(records),3)
        _,problems=check_inputs(list(records.values()),show=False)
        self.assertEqual(problems,[])
    def test_pending_not_loaded(self):
        loaded=read_ground_truth()
        self.assertNotIn('long_11_2142822722',loaded)
if __name__=='__main__': unittest.main()
