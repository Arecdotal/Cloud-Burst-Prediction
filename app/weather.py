"""Open-Meteo client: downloads one local day of hourly data."""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import Settings

logger = logging.getLogger(__name__)

# ERA5 archive lags real time by several days; recent dates come from the forecast API.
ARCHIVE_LAG_DAYS = 5


class WeatherError(RuntimeError):
    """Weather provider unreachable or returned unusable data."""


class OpenMeteoClient:
    def __init__(self, settings: Settings, variables: List[str],
                 session: Optional[requests.Session] = None, sleep=time.sleep):
        self.s = settings
        self.variables = list(variables)
        self.session = session or requests.Session()
        self._sleep = sleep

    def choose_source(self, day: dt.date, source: str, today: dt.date) -> str:
        if source in ("forecast", "archive"):
            return source
        return "forecast" if day >= today - dt.timedelta(days=ARCHIVE_LAG_DAYS) else "archive"

    def fetch_hourly(self, day: dt.date, source: str = "auto",
                     today: Optional[dt.date] = None) -> Tuple[pd.DataFrame, str]:
        """Return (hourly DataFrame indexed by naive local time, source actually used)."""
        today = today or dt.datetime.now(ZoneInfo(self.s.timezone)).date()
        used = self.choose_source(day, source, today)
        url = self.s.openmeteo_forecast_url if used == "forecast" else self.s.openmeteo_archive_url

        params = {
            "latitude": self.s.latitude,
            "longitude": self.s.longitude,
            "hourly": ",".join(self.variables),
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": self.s.timezone,
            # Pin units to the Open-Meteo defaults the model's training medians imply.
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }
        if self.s.openmeteo_models:
            params["models"] = self.s.openmeteo_models

        payload = self._get_json(url, params)
        hourly = payload.get("hourly") or {}
        times = hourly.get("time")
        if not times:
            raise WeatherError("Open-Meteo response contained no hourly data.")

        columns = {k: v for k, v in hourly.items() if k != "time"}
        df = pd.DataFrame(columns, index=pd.to_datetime(times)).apply(pd.to_numeric, errors="coerce")
        absent = [v for v in self.variables if v not in df.columns]
        if absent:
            logger.warning("Open-Meteo (%s) did not return: %s", used, absent)
        return df, used

    def _get_json(self, url: str, params: dict) -> dict:
        last_error = "unknown error"
        for attempt in range(self.s.http_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.s.http_timeout)
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise WeatherError(f"Invalid JSON from Open-Meteo: {exc}") from exc
                if resp.status_code not in (429, 500, 502, 503, 504):
                    reason = ""
                    try:
                        reason = resp.json().get("reason", "")
                    except Exception:  # noqa: BLE001
                        reason = (resp.text or "")[:200]
                    raise WeatherError(f"Open-Meteo rejected the request ({resp.status_code}): {reason}")
                last_error = f"HTTP {resp.status_code}"
            if attempt < self.s.http_retries:
                delay = 2 ** attempt
                logger.warning("Open-Meteo attempt %d failed (%s); retrying in %ds", attempt + 1, last_error, delay)
                self._sleep(delay)
        raise WeatherError(f"Open-Meteo unavailable after retries: {last_error}")
