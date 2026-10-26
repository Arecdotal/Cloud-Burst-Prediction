import datetime as dt
import tempfile
import unittest
import warnings

import requests

from app.alerts import Notifier
from app.bootstrap import build_service
from app.storage import Store
from app.weather import OpenMeteoClient, WeatherError
from tests.helpers import make_settings, synthetic_hourly

warnings.filterwarnings("ignore")
DAY = dt.date(2026, 7, 15)


class FakeWeather:
    def __init__(self, df, fail=False):
        self.df, self.fail, self.calls = df, fail, []

    def fetch_hourly(self, day, source="auto", today=None):
        self.calls.append((day, source))
        if self.fail:
            raise WeatherError("provider down")
        return self.df, "forecast"


class FakeNotifier:
    def __init__(self, ok=True): self.sent, self.ok = [], ok
    def send(self, rec): self.sent.append(rec["target_date"]); return self.ok


class StoreTests(unittest.TestCase):
    def test_outcomes_and_metrics(self):
        s = Store(tempfile.mkdtemp() + "/t.db")
        def rec(target, pred):
            return dict(created_at="2026-07-01T00:00:00+00:00", location="X", feature_date="d", target_date=target,
                        predicted_probability=0.9 if pred else 0.1, threshold=0.875, prediction=pred, risk_label="r")
        for t, p in [("2026-07-02", 1), ("2026-07-03", 1), ("2026-07-04", 0), ("2026-07-05", 0)]:
            s.save_prediction(rec(t, p))
        for t, mm in [("2026-07-02", 80), ("2026-07-03", 10), ("2026-07-04", 60), ("2026-07-05", 0)]:
            s.record_outcome(t, mm, 50)
        m = s.live_metrics()
        self.assertEqual((m["true_positives"], m["false_positives"], m["false_negatives"], m["true_negatives"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)

    def test_history_returns_latest_run_per_day(self):
        s = Store(tempfile.mkdtemp() + "/t.db")
        base = dict(created_at="x", location="X", feature_date="d", target_date="2026-07-02",
                    threshold=0.875, prediction=0, risk_label="r")
        s.save_prediction({**base, "predicted_probability": 0.2})
        s.save_prediction({**base, "predicted_probability": 0.4})
        rows = s.list_predictions()
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["predicted_probability"], 0.4)


class WeatherClientTests(unittest.TestCase):
    class Resp:
        def __init__(self, code, body=None): self.status_code, self._b, self.text = code, body, ""
        def json(self): return self._b

    class Session:
        def __init__(self, responses): self.responses, self.requests = list(responses), []
        def get(self, url, params=None, timeout=None):
            self.requests.append((url, params))
            r = self.responses.pop(0)
            if isinstance(r, Exception): raise r
            return r

    def client(self, responses):
        from app.features import build_feature_plan, hourly_variables
        from tests.helpers import FEATURES
        sess = self.Session(responses)
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(d, http_retries=2)
        return OpenMeteoClient(st, hourly_variables(build_feature_plan(FEATURES)), session=sess, sleep=lambda s: None), sess

    def payload(self):
        df = synthetic_hourly(DAY)
        return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in df.index], **{c: df[c].tolist() for c in df.columns}}}

    def test_parses_and_pins_units(self):
        c, sess = self.client([self.Resp(200, self.payload())])
        df, used = c.fetch_hourly(DAY, today=DAY)
        self.assertEqual(used, "forecast"); self.assertEqual(len(df), 24)
        url, params = sess.requests[0]
        self.assertEqual(params["wind_speed_unit"], "kmh"); self.assertEqual(params["timezone"], "Asia/Kolkata")
        self.assertIn("total_column_integrated_water_vapour", params["hourly"])

    def test_old_dates_use_archive(self):
        c, sess = self.client([self.Resp(200, self.payload())])
        _, used = c.fetch_hourly(dt.date(2024, 7, 15), today=DAY)
        self.assertEqual(used, "archive"); self.assertIn("archive", sess.requests[0][0])

    def test_retries_then_succeeds(self):
        c, sess = self.client([self.Resp(503), requests.ConnectionError("x"), self.Resp(200, self.payload())])
        c.fetch_hourly(DAY, today=DAY); self.assertEqual(len(sess.requests), 3)

    def test_gives_up_after_retries(self):
        c, _ = self.client([self.Resp(500)] * 3)
        with self.assertRaises(WeatherError): c.fetch_hourly(DAY, today=DAY)

    def test_client_error_is_not_retried(self):
        c, sess = self.client([self.Resp(400, {"reason": "bad variable"})])
        with self.assertRaisesRegex(WeatherError, "bad variable"): c.fetch_hourly(DAY, today=DAY)
        self.assertEqual(len(sess.requests), 1)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.svc = build_service(make_settings(self.tmp))
        self.svc.weather = FakeWeather(synthetic_hourly(DAY, wet=True))
        self.svc.notifier = FakeNotifier()
        self.svc.today = lambda: DAY

    def test_run_stores_prediction_for_next_day(self):
        rec = self.svc.run()
        self.assertEqual(rec["feature_date"], "2026-07-15"); self.assertEqual(rec["target_date"], "2026-07-16")
        self.assertEqual(self.svc.store.latest_prediction()["target_date"], "2026-07-16")
        self.assertEqual(rec["missing_features"], [])

    def test_future_date_rejected(self):
        with self.assertRaises(ValueError): self.svc.run(DAY + dt.timedelta(days=1))

    def test_alert_sent_once_per_target_date(self):
        self.svc.model.threshold = 0.0                       # force a positive prediction
        a = self.svc.run(); b = self.svc.run()
        self.assertEqual((a["prediction"], a["alert_sent"], b["alert_sent"]), (1, True, False))
        self.assertEqual(self.svc.notifier.sent, ["2026-07-16"])

    def test_failed_alert_is_retried_next_run(self):
        self.svc.model.threshold = 0.0; self.svc.notifier = FakeNotifier(ok=False)
        self.svc.run(); self.svc.notifier.ok = True
        self.assertTrue(self.svc.run()["alert_sent"])

    def test_no_alert_below_threshold(self):
        self.svc.model.threshold = 1.1
        self.assertFalse(self.svc.run()["alert_sent"]); self.assertEqual(self.svc.notifier.sent, [])

    def test_job_records_failure_without_raising(self):
        self.svc.weather = FakeWeather(None, fail=True)
        self.svc.run_job()
        last = self.svc.store.last_run()
        self.assertEqual(last["ok"], 0); self.assertIn("provider down", last["error"])
        self.assertIsNone(self.svc.store.latest_prediction())

    def test_job_records_success(self):
        self.svc.run_job(); self.assertEqual(self.svc.store.last_run()["ok"], 1)

    def test_too_much_missing_weather_data_is_refused(self):
        from app.model_service import InsufficientDataError
        self.svc.weather = FakeWeather(synthetic_hourly(DAY).iloc[:5])
        with self.assertRaises(InsufficientDataError): self.svc.run()

    def test_webhook_notifier(self):
        class S:
            def __init__(s): s.calls = []
            def post(s, url, json=None, timeout=None):
                s.calls.append((url, json))
                class R:
                    def raise_for_status(self): pass
                return R()
        s = S(); n = Notifier("https://hooks.example/x", session=s)
        rec = dict(location="Dehradun", feature_date="a", target_date="b", predicted_probability=0.95, threshold=0.875)
        self.assertTrue(n.send(rec)); self.assertEqual(s.calls[0][1]["event"], "extreme_rainfall_alert")
        self.assertNotIn("cloudburst will", s.calls[0][1]["text"])


if __name__ == "__main__":
    unittest.main()
