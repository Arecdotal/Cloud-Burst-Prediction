"""Orchestration: weather -> features -> model -> storage -> alert."""
from __future__ import annotations

import datetime as dt
import logging
import threading
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from .alerts import Notifier
from .config import Settings
from .features import FeaturePlan, compute_daily_features
from .model_service import ModelService
from .storage import Store, utcnow
from .weather import OpenMeteoClient

logger = logging.getLogger(__name__)


class ForecastService:
    def __init__(self, settings: Settings, model: ModelService, plan: FeaturePlan,
                 store: Store, weather: OpenMeteoClient, notifier: Notifier):
        self.settings = settings
        self.model = model
        self.plan = plan
        self.store = store
        self.weather = weather
        self.notifier = notifier
        self._lock = threading.Lock()      # scheduled and manual runs never overlap

    def today(self) -> dt.date:
        return dt.datetime.now(ZoneInfo(self.settings.timezone)).date()

    def _features_for(self, day: dt.date, source: str):
        today = self.today()
        if day > today:
            raise ValueError(f"feature_date {day} is in the future (today is {today}).")
        hourly, used = self.weather.fetch_hourly(day, source, today=today)
        feats = compute_daily_features(hourly, day, self.plan, self.settings.min_hours_per_day)
        return feats, used

    def preview_features(self, day: Optional[dt.date] = None, source: str = "auto") -> Dict[str, Any]:
        day = day or self.today()
        feats, used = self._features_for(day, source)
        missing = [k for k, v in feats.items() if v is None]
        return {"feature_date": day.isoformat(), "data_source": used,
                "missing_features": missing, "features": feats}

    def run(self, feature_date: Optional[dt.date] = None, source: str = "auto",
            persist: bool = True, notify: bool = True, trigger: str = "manual") -> Dict[str, Any]:
        """Predict next-day extreme-rainfall risk from the weather of `feature_date` (default today)."""
        with self._lock:
            day = feature_date or self.today()
            feats, used = self._features_for(day, source)
            result = self.model.predict(feats)

            record: Dict[str, Any] = {
                **result,
                "created_at": utcnow(),
                "location": self.settings.location_name,
                "feature_date": day.isoformat(),
                "target_date": (day + dt.timedelta(days=1)).isoformat(),
                "data_source": used,
                "trigger": trigger,
                "features": feats,
                "alert_sent": False,
            }
            if persist:
                record["id"] = self.store.save_prediction(record)
                if notify and record["prediction"] == 1:
                    record["alert_sent"] = self._maybe_alert(record)
            return record

    def _maybe_alert(self, rec: Dict[str, Any]) -> bool:
        """At most one delivered alert per target date, so repeated runs don't spam."""
        if self.store.alert_sent(rec["target_date"]):
            return False
        if self.notifier.send(rec):
            self.store.mark_alert(rec["target_date"], rec["predicted_probability"])
            return True
        return False

    def run_job(self, trigger: str = "scheduler") -> None:
        """Entry point for the scheduler: never raises, records the outcome."""
        run_id = self.store.start_run(trigger)
        try:
            rec = self.run(trigger=trigger)
            self.store.finish_run(run_id, True, feature_date=rec["feature_date"])
            logger.info("Run OK: target %s score %.3f prediction %d",
                        rec["target_date"], rec["predicted_probability"], rec["prediction"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scheduled run failed")
            self.store.finish_run(run_id, False, error=f"{type(exc).__name__}: {exc}")
