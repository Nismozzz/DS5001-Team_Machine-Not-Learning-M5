import unittest

from modeling.pipeline import export_splits


class ExportModeTests(unittest.TestCase):
    def setUp(self):
        self.splits = [{'split_id': name, 'origin_d': origin} for name, origin in
                       [('validation_1', 1829), ('validation_2', 1857),
                        ('validation_3', 1885), ('test', 1913), ('forecast', 1941)]]

    def test_backtest_excludes_test_and_forecast(self):
        result = export_splits('backtest', self.splits, [1745])
        self.assertEqual([s['origin_d'] for s in result], [1829, 1857, 1885])

    def test_missing_validation_fold_fails(self):
        with self.assertRaises(ValueError):
            export_splits('backtest', self.splits[1:], [1745])

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            export_splits('invalid', self.splits, [])

    def test_test_exports_only_holdout(self):
        self.assertEqual(export_splits('test', self.splits, []), [self.splits[3]])


if __name__ == '__main__':
    unittest.main()
