"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> Tuple[str, ...]:
    return tuple(p.strip() for p in os.getenv(name, default).split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    # --- model ---
    model_path: Path
    metadata_path: Path
    expected_sklearn_version: str      # version the pickle was trained with
    max_missing_fraction: float        # refuse to predict if more features than this are missing
    allow_sklearn_mismatch: bool       # start even if scikit-learn major.minor differs from training

    # --- location / weather source ---
    location_name: str
    latitude: float
    longitude: float
    timezone: str                      # defines the 00:00-23:59 "day" used for daily aggregates
    openmeteo_forecast_url: str
    openmeteo_archive_url: str
    openmeteo_models: str              # optional `models=` parameter (e.g. "era5"); empty = default
    http_timeout: float
    http_retries: int
    min_hours_per_day: int             # a daily aggregate needs at least this many valid hours

    # --- storage / scheduling ---
    db_path: Path
    enable_scheduler: bool
    schedule_hours: str                # cron "hour" field, in `timezone`
    schedule_minute: int
    run_on_startup: bool
    stale_after_hours: float

    # --- API ---
    admin_api_key: str
    cors_origins: Tuple[str, ...]
    log_level: str

    # --- alerting / evaluation ---
    alert_webhook_url: str
    extreme_threshold_mm: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "extreme_rainfall_model.pkl")),
            metadata_path=Path(os.getenv("METADATA_PATH", BASE_DIR / "model" / "model_metadata.json")),
            expected_sklearn_version=os.getenv("EXPECTED_SKLEARN_VERSION", "1.4.2"),
            max_missing_fraction=float(os.getenv("MAX_MISSING_FRACTION", "0.25")),
            allow_sklearn_mismatch=_bool("ALLOW_SKLEARN_MISMATCH", False),
            location_name=os.getenv("LOCATION_NAME", "Dehradun"),
            latitude=float(os.getenv("LATITUDE", "30.3165")),
            longitude=float(os.getenv("LONGITUDE", "78.0322")),
            timezone=os.getenv("WEATHER_TIMEZONE", "Asia/Kolkata"),
            openmeteo_forecast_url=os.getenv("OPENMETEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"),
            openmeteo_archive_url=os.getenv("OPENMETEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive"),
            openmeteo_models=os.getenv("OPENMETEO_MODELS", ""),
            http_timeout=float(os.getenv("HTTP_TIMEOUT", "30")),
            http_retries=int(os.getenv("HTTP_RETRIES", "3")),
            min_hours_per_day=int(os.getenv("MIN_HOURS_PER_DAY", "20")),
            db_path=Path(os.getenv("DB_PATH", BASE_DIR / "data" / "predictions.db")),
            enable_scheduler=_bool("ENABLE_SCHEDULER", True),
            schedule_hours=os.getenv("SCHEDULE_HOURS", "2,8,14,20"),
            schedule_minute=int(os.getenv("SCHEDULE_MINUTE", "10")),
            run_on_startup=_bool("RUN_ON_STARTUP", True),
            stale_after_hours=float(os.getenv("STALE_AFTER_HOURS", "12")),
            admin_api_key=os.getenv("ADMIN_API_KEY", ""),
            cors_origins=_csv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
            extreme_threshold_mm=float(os.getenv("EXTREME_THRESHOLD_MM", "50")),
        )
