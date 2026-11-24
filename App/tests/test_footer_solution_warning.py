import unittest

from App.pipeline import has_multiple_consistent_footer_values


class FooterSolutionWarningTest(unittest.TestCase):
    def test_unmatched_ocr_variants_do_not_trigger_warning(self):
        matrix = {'footer_ambiguous': True, 'footer_variants': ['41250', '41280'],
                  'solutions': [{'footer': '41250'}]}
        self.assertFalse(has_multiple_consistent_footer_values(matrix))

    def test_distinct_consistent_totals_trigger_warning(self):
        matrix = {'solutions': [{'footer': '100'}, {'footer': '200'}]}
        self.assertTrue(has_multiple_consistent_footer_values(matrix))

    def test_several_solutions_with_same_total_do_not_trigger_footer_warning(self):
        matrix = {'solutions': [{'footer': '100'}, {'footer': '100.00'}]}
        self.assertFalse(has_multiple_consistent_footer_values(matrix))

    def test_no_solution_does_not_claim_consistent_alternatives(self):
        self.assertFalse(has_multiple_consistent_footer_values({'solutions': []}))


if __name__ == '__main__':
    unittest.main()
