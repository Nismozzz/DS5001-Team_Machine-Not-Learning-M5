import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling.baselines import predict_baseline, seasonal_naive, tsb
from modeling.common import repair_quantiles
from modeling.features import aggregate_features, aggregate_history, period_features, period_training_origins
from modeling.learners import Encoder, fit_predict


class ModelTests(unittest.TestCase):
    def test_seasonal_repeat_and_zero(self):
        np.testing.assert_array_equal(seasonal_naive(np.tile(np.arange(7), 53)), np.tile(np.arange(7), 4))
        for model in ['ets', 'tsb', 'seasonal_naive']:
            result = predict_baseline(model, np.zeros(365))
            self.assertEqual(result['status'], 'ok')
            np.testing.assert_array_equal(result['prediction'], np.zeros(28))

    def test_tsb_known_update(self):
        # First demand size=4,p=1; next day zero -> p=.5; next size=2 -> p=.75,size=3.
        np.testing.assert_allclose(tsb([4, 0, 2], alpha=.5, beta=.5), np.full(28, 2.25))

    def test_ets_failure_preserves_series(self):
        with patch('modeling.baselines.ets', side_effect=RuntimeError('forced failure')):
            result = predict_baseline('ets', np.arange(365))
        self.assertEqual(result['status'], 'fallback')
        self.assertEqual(result['method_used'], 'seasonal_naive')
        self.assertIn('forced failure', result['fallback_reason'])
        np.testing.assert_array_equal(result['prediction'], seasonal_naive(np.arange(365)))

    def test_unknown_category_is_missing(self):
        encoder = Encoder().fit(pd.DataFrame({'item': ['a', 'b'], 'value': [1, 2]}))
        encoded = encoder.transform(pd.DataFrame({'item': ['b', 'new'], 'value': [2, 3]}))
        self.assertEqual(encoded.item.cat.codes.tolist(), [1, -1])

    def test_quantile_repairs_reported(self):
        fixed, stats = repair_quantiles([[-1, 4, 2, 3], [1, 2, 3, 4]])
        np.testing.assert_array_equal(fixed[0], [0, 2, 3, 4])
        self.assertEqual(stats['negative_values'], 1)
        self.assertEqual(stats['crossing_rows'], 1)

    def test_features_ignore_future_sales(self):
        nodes = pd.DataFrame({'node_id': ['n1', 'n2']})
        y = np.random.default_rng(5001).integers(0, 10, size=(2, 220)).astype(float)
        calendar = pd.DataFrame({'d_num': np.arange(1, 221), 'wm_yr_wk': 1, 'wday': 1,
                                 'month': 1, 'year': 2016, 'event_name_1': None,
                                 'event_type_1': None, 'event_name_2': None, 'event_type_2': None,
                                 'snap_CA': 0, 'snap_TX': 0, 'snap_WI': 0})
        before = aggregate_features(nodes, y, calendar, 180)
        period = period_features(nodes, y, calendar, 180)
        y[:, 180:] = 100000
        pd.testing.assert_frame_equal(before, aggregate_features(nodes, y, calendar, 180))
        pd.testing.assert_frame_equal(period, period_features(nodes, y, calendar, 180))

    def test_period_labels_entirely_in_training_window(self):
        split = {'origin_d': 1829, 'train_start_d': 1465, 'train_end_d': 1829}
        origins = period_training_origins(range(1353, 1942, 28), split)
        self.assertEqual(origins, list(range(1465, 1802, 28)))
        self.assertTrue(all(o + 1 >= 1465 and o + 28 <= 1829 for o in origins))

    def test_repeated_fit_and_serialized_model(self):
        config = {'seed': 5001, 'threads': 2,
                  'lightgbm': {'min_data_in_leaf': 2, 'verbosity': -1, 'deterministic': True,
                               'force_col_wise': True}}
        x = pd.DataFrame({'x': np.arange(60), 'item': ['a', 'b'] * 30})
        y = np.arange(60) % 7
        with tempfile.TemporaryDirectory() as temp:
            a = fit_predict(x, y, x, config, {'num_leaves': 7, 'rounds': 10}, 'poisson', Path(temp) / 'a')
            b = fit_predict(x, y, x, config, {'num_leaves': 7, 'rounds': 10}, 'poisson', Path(temp) / 'b')
            np.testing.assert_allclose(a, b, rtol=0, atol=0)
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(Path(temp) / 'a/model.txt'))
            xp = Encoder().fit(x).transform(x)
            np.testing.assert_allclose(a, model.predict(xp, num_threads=2), rtol=0, atol=0)

    def test_induced_aggregation_uses_selected_leaves(self):
        sales = pd.DataFrame({'series_id': ['a', 'b'], **{f'd_{d}': [2, 3] for d in range(1, 1942)}})
        mapping = pd.DataFrame({'series_id': ['a', 'b'], 'node_id': ['total', 'total'], 'level': 1})
        nodes, y = aggregate_history(sales, mapping, 1)
        self.assertEqual(nodes.node_id.tolist(), ['total'])
        np.testing.assert_array_equal(y, np.full((1, 1941), 5))


if __name__ == '__main__':
    unittest.main()
