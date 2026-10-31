"""Boundary and future-perturbation tests; no M5 files required."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("data01", Path(__file__).resolve().parents[1] / "01_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.origin = 180
        self.y = np.random.default_rng(5001).integers(0, 10, (4, 240), dtype="int16")
        self.ends = np.arange(7, 246, 7, dtype="int16")
        self.prices = np.random.default_rng(8).uniform(1, 10, (4, len(self.ends))).astype("float32")

    def test_future_sales_and_prices_cannot_change_features_or_strata(self):
        before, strata_before = data.origin_statistics(self.y, self.origin)
        price_before = data.price_features(self.prices, self.ends, self.origin)
        poisoned = self.y.copy()
        poisoned[:, self.origin:] = 30000
        future_prices = self.prices.copy()
        future_prices[:, self.ends > self.origin] = 99999
        after, strata_after = data.origin_statistics(poisoned, self.origin)
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(strata_before, strata_after)
        pd.testing.assert_frame_equal(price_before, data.price_features(future_prices, self.ends, self.origin))
        truncated, strata_truncated = data.origin_statistics(self.y[:, :self.origin], self.origin)
        pd.testing.assert_frame_equal(before, truncated)
        pd.testing.assert_frame_equal(strata_before, strata_truncated)

    def test_origin_inclusive_lags_and_rolling(self):
        f, _ = data.origin_statistics(self.y, self.origin)
        np.testing.assert_array_equal(f.sales_lag_1, self.y[:, 179])
        np.testing.assert_array_equal(f.sales_lag_7, self.y[:, 173])
        np.testing.assert_allclose(f.sales_mean_28, self.y[:, 152:180].mean(axis=1), rtol=1e-6)
        np.testing.assert_allclose(f.sales_std_28, self.y[:, 152:180].std(axis=1), rtol=1e-6)

    def test_complete_multihorizon_features_ignore_future(self):
        dates = pd.date_range("2011-01-29", periods=240)
        calendar = pd.DataFrame({"d_num": np.arange(1, 241), "date": dates.strftime("%Y-%m-%d"),
                                 "wm_yr_wk": np.arange(240) // 7, "wday": np.arange(240) % 7 + 1,
                                 "month": dates.month, "year": dates.year,
                                 "event_name_1": None, "event_name_2": None,
                                 "event_type_1": None, "event_type_2": None,
                                 "snap_CA": 1, "snap_TX": 0, "snap_WI": 0})
        meta = pd.DataFrame({"series_id": [f"item{i}_CA_1" for i in range(4)],
                             "item_id": [f"item{i}" for i in range(4)], "dept_id": "dept",
                             "cat_id": "cat", "store_id": "CA_1", "state_id": "CA"})
        before = data.build_features(meta, self.y, calendar, self.prices, self.ends, self.origin)
        poisoned_y = self.y.copy()
        poisoned_y[:, self.origin:] = 30000
        poisoned_prices = self.prices.copy()
        poisoned_prices[:, self.ends > self.origin] = 99999
        after = data.build_features(meta, poisoned_y, calendar, poisoned_prices, self.ends, self.origin)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(len(before), 4 * 28)
        self.assertNotIn("units", before)
        self.assertEqual(before.target_d.min(), 181)
        self.assertEqual(before.target_d.max(), 208)
        self.assertEqual(before.snap.unique().tolist(), [1])

    def test_missing_prices_and_completed_week_boundary(self):
        prices = np.full((2, 3), np.nan, dtype="float32")
        prices[0] = [4, 2, 100]
        f = data.price_features(prices, np.array([7, 14, 21]), 20)
        self.assertEqual(f.price_last.iloc[0], 2)
        self.assertEqual(f.discount_proxy.iloc[0], 0.5)
        self.assertEqual(f.price_change.iloc[0], -0.5)
        self.assertEqual(f.price_age_days.iloc[0], 6)
        self.assertTrue(np.isnan(f.price_last.iloc[1]))
        self.assertTrue(np.isnan(f.promotion_proxy.iloc[1]))

    def test_zero_new_dormant_demand(self):
        y = np.zeros((3, 400), dtype="int16")
        y[1, 395] = 2
        y[2, :100] = 1
        _, s = data.origin_statistics(y, 400)
        self.assertEqual(s.lifecycle.tolist(), ["not_yet_sold", "new", "dormant"])
        self.assertEqual(s.intermittency.iloc[0], "no_demand")
        self.assertTrue(np.isnan(s.adi.iloc[0]))
        self.assertEqual(s.adi.iloc[1], 5)

    def test_training_filter_excludes_future_origin_and_labels(self):
        keys = pd.DataFrame({"origin_d": [100, 100, 128, 156], "target_d": [127, 129, 129, 157]})
        split = {"origin_d": 128, "train_start_d": 1, "train_end_d": 128}
        result = data.select_training_rows(keys, split)
        self.assertEqual(result.index.tolist(), [0])

    def test_sample_is_independent_of_input_order(self):
        meta = pd.DataFrame({"series_id": [f"item_{i}" for i in range(10)], "state_id": "CA"})
        expected = data.sample_ids(meta, "CA", 5, 5001)
        self.assertEqual(expected, data.sample_ids(meta.sample(frac=1, random_state=8), "CA", 5, 5001))
        self.assertNotEqual(expected, data.sample_ids(meta, "CA", 5, 5002))


if __name__ == "__main__":
    unittest.main()
