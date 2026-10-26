import unittest
import warnings

from app.config import BASE_DIR
from app.features import build_feature_plan, compute_daily_features
from app.model_service import InsufficientDataError, ModelService
from tests.helpers import FEATURES, synthetic_hourly
import datetime as dt

warnings.filterwarnings("ignore")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = ModelService(BASE_DIR / "model" / "extreme_rainfall_model.pkl",
                             BASE_DIR / "model" / "model_metadata.json",
                             allow_version_mismatch=True)
        cls.plan = build_feature_plan(FEATURES)
        cls.median = {f: float(v) for f, v in zip(FEATURES, cls.m.pipeline[0].statistics_)}

    def test_loads_and_matches_metadata(self):
        self.assertEqual(len(self.m.features), 36)
        self.assertEqual(self.m.threshold, 0.875)

    def test_median_input_is_low_risk(self):
        r = self.m.predict(self.median)
        self.assertTrue(0.0 <= r["predicted_probability"] <= 1.0)
        self.assertEqual(r["prediction"], 0)
        self.assertEqual(r["missing_features"], [])
        self.assertEqual(len(r["top_factors"]), 5)

    def test_wetter_day_scores_higher_than_dry_day(self):
        day = dt.date(2024, 7, 15)
        dry = self.m.predict(compute_daily_features(synthetic_hourly(day), day, self.plan))
        wet = self.m.predict(compute_daily_features(synthetic_hourly(day, wet=True), day, self.plan))
        self.assertGreater(wet["predicted_probability"], dry["predicted_probability"])

    def test_missing_values_are_imputed_within_limit(self):
        feats = dict(self.median); feats["boundary_layer_height_max"] = None; feats["cloud_cover_mean"] = float("nan")
        r = self.m.predict(feats)
        self.assertEqual(sorted(r["missing_features"]), ["boundary_layer_height_max", "cloud_cover_mean"])
        self.assertTrue(r["warnings"])

    def test_too_many_missing_is_refused(self):
        with self.assertRaises(InsufficientDataError):
            self.m.predict({"year": 2024})

    def test_unknown_feature_is_refused(self):
        with self.assertRaises(ValueError):
            self.m.predict({**self.median, "bogus": 1})

    def test_year_extrapolation_warning(self):
        r = self.m.predict({**self.median, "year": 2026})
        self.assertTrue(any("outside the training period" in w for w in r["warnings"]))

    def test_matches_original_predict_py(self):
        """My wrapper must give the same probability as the supplied predict.py."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("orig_predict", "/mnt/user-data/uploads/predict.py")
        import shutil, tempfile, os
        tmp = tempfile.mkdtemp()
        for f in ("predict.py",):
            shutil.copy("/mnt/user-data/uploads/" + f, tmp)
        shutil.copy(BASE_DIR / "model" / "extreme_rainfall_model.pkl", tmp)
        shutil.copy(BASE_DIR / "model" / "model_metadata.json", tmp)
        spec = importlib.util.spec_from_file_location("orig_predict", os.path.join(tmp, "predict.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        feats = dict(self.median); feats["temperature_2m_mean"] = 30.0; feats["relative_humidity_2m_mean"] = 95.0
        self.assertEqual(mod.predict_extreme_rainfall(feats)["predicted_probability"],
                         self.m.predict(feats)["predicted_probability"])


if __name__ == "__main__":
    unittest.main()


class VersionGuardTests(unittest.TestCase):
    def test_refuses_mismatched_sklearn_unless_allowed(self):
        import sklearn
        paths = (BASE_DIR / "model" / "extreme_rainfall_model.pkl", BASE_DIR / "model" / "model_metadata.json")
        if sklearn.__version__.split(".")[:2] == ["1", "4"]:
            self.skipTest("pinned version installed")
        with self.assertRaisesRegex(RuntimeError, "ALLOW_SKLEARN_MISMATCH"):
            ModelService(*paths, expected_sklearn_version="1.4.2")
        ModelService(*paths, expected_sklearn_version="1.4.2", allow_version_mismatch=True)
