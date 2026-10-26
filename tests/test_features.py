import datetime as dt
import unittest

import pandas as pd

from app.features import build_feature_plan, compute_daily_features, hourly_variables
from tests.helpers import FEATURES, synthetic_hourly

DAY = dt.date(2026, 7, 15)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_feature_plan(FEATURES)

    def test_plan_covers_every_model_feature(self):
        self.assertEqual(set(self.plan), set(FEATURES))
        self.assertEqual(len(hourly_variables(self.plan)), 19)

    def test_unknown_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            build_feature_plan(["mystery_value"])

    def test_aggregations(self):
        f = compute_daily_features(synthetic_hourly(DAY, wet=True), DAY, self.plan)
        self.assertEqual(f["precipitation_sum"], 34.0)
        self.assertEqual(f["precipitation_max"], 12.0)
        self.assertEqual(f["hours_ge_5mm"], 4.0)      # 12, 8, 6, 5
        self.assertEqual(f["hours_ge_10mm"], 1.0)     # 12
        self.assertEqual((f["month"], f["day_of_year"], f["year"]), (7.0, 196.0, 2026.0))
        self.assertAlmostEqual(f["relative_humidity_2m_max"], 95.0)
        self.assertTrue(all(v is not None for v in f.values()))

    def test_incomplete_day_gives_none_not_zero(self):
        hourly = synthetic_hourly(DAY).iloc[:10]           # only 10 of 24 hours
        f = compute_daily_features(hourly, DAY, self.plan, min_hours=20)
        self.assertIsNone(f["precipitation_sum"])
        self.assertEqual(f["year"], 2026.0)                # calendar features still present

    def test_missing_variable_column_gives_none(self):
        f = compute_daily_features(synthetic_hourly(DAY, drop=["boundary_layer_height"]), DAY, self.plan)
        self.assertIsNone(f["boundary_layer_height_mean"])
        self.assertIsNotNone(f["temperature_2m_mean"])

    def test_only_requested_day_is_used(self):
        two_days = pd.concat([synthetic_hourly(DAY - dt.timedelta(days=1), wet=True), synthetic_hourly(DAY)])
        f = compute_daily_features(two_days, DAY, self.plan)
        self.assertEqual(f["precipitation_sum"], 0.0)

    def test_empty_input(self):
        f = compute_daily_features(None, DAY, self.plan)
        self.assertIsNone(f["temperature_2m_mean"])


if __name__ == "__main__":
    unittest.main()
