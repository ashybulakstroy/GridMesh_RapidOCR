import unittest

from App.ocr import warmup


class WarmupLoggingTest(unittest.TestCase):
    def test_success_logs_model_and_elapsed_ms(self):
        with self.assertLogs('uvicorn.error', level='INFO') as logs:
            self.assertTrue(warmup._track('test_model / lane', lambda: None))
        self.assertIn('Прогрев модели завершён: test_model / lane', logs.output[0])
        self.assertIn('мс', logs.output[0])
        self.assertTrue(warmup.snapshot()['steps'][-1]['success'])

    def test_failure_logs_error_and_not_success(self):
        def fail():
            raise ValueError('ошибка теста')
        with self.assertLogs('uvicorn.error', level='WARNING') as logs:
            self.assertFalse(warmup._track('broken_model / lane', fail))
        self.assertIn('Ошибка прогрева модели: broken_model / lane', logs.output[0])
        self.assertIn('ValueError', logs.output[0])
        self.assertIn('мс', logs.output[0])
        self.assertFalse(warmup.snapshot()['steps'][-1]['success'])


if __name__ == '__main__':
    unittest.main()
