import datetime as dt
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import BASE_DIR, Settings

METADATA = json.loads((BASE_DIR / "model" / "model_metadata.json").read_text())
FEATURES = METADATA["features"]


def make_settings(tmpdir: str, **overrides) -> Settings:
    base = Settings.from_env()
    data = {**base.__dict__, "db_path": Path(tmpdir) / "test.db", "enable_scheduler": False,
            "alert_webhook_url": "", "http_retries": 1, "allow_sklearn_mismatch": True, **overrides}
    return Settings(**data)


def synthetic_hourly(day: dt.date, wet: bool = False, drop=()) -> pd.DataFrame:
    """24 hours of plausible monsoon-ish data for `day`, using the real Open-Meteo variable names."""
    from app.features import build_feature_plan, hourly_variables
    idx = pd.date_range(pd.Timestamp(day), periods=24, freq="h")
    rng = np.random.default_rng(0)
    base = {
        "temperature_2m": 24 + 4 * np.sin(np.linspace(0, 2 * np.pi, 24)),
        "relative_humidity_2m": np.linspace(95, 80, 24) if wet else np.linspace(70, 50, 24),
        "dew_point_2m": np.full(24, 22.0 if wet else 10.0),
        "precipitation": np.r_[np.zeros(10), [12, 8, 6, 5, 3], np.zeros(9)] if wet else np.zeros(24),
        "rain": np.r_[np.zeros(10), [12, 8, 6, 5, 3], np.zeros(9)] if wet else np.zeros(24),
        "pressure_msl": np.full(24, 1004.0 if wet else 1012.0),
        "surface_pressure": np.full(24, 935.0 if wet else 940.0),
        "cloud_cover": np.full(24, 95.0 if wet else 20.0),
        "cloud_cover_low": np.full(24, 70.0 if wet else 0.0),
        "cloud_cover_mid": np.full(24, 60.0 if wet else 5.0),
        "cloud_cover_high": np.full(24, 50.0 if wet else 10.0),
        "wind_speed_10m": rng.uniform(3, 9, 24),
        "wind_gusts_10m": rng.uniform(15, 40, 24),
        "wind_direction_10m": rng.uniform(100, 200, 24),
        "vapour_pressure_deficit": np.full(24, 0.2 if wet else 1.1),
        "soil_moisture_0_to_7cm": np.full(24, 0.38 if wet else 0.2),
        "soil_temperature_0_to_7cm": np.full(24, 24.0),
        "boundary_layer_height": rng.uniform(100, 1500, 24),
        "total_column_integrated_water_vapour": np.full(24, 55.0 if wet else 18.0),
    }
    df = pd.DataFrame(base, index=idx).drop(columns=list(drop))
    needed = set(hourly_variables(build_feature_plan(FEATURES)))
    assert needed <= set(base), needed - set(base)
    return df
