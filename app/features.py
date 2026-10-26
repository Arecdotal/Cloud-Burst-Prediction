"""Hourly weather -> the 36 daily features the model was trained on.

The feature *names* drive everything. Each name is parsed into a rule:

    temperature_2m_mean  -> mean of hourly `temperature_2m` over the day
    precipitation_max    -> max of hourly `precipitation`
    precipitation_sum    -> sum of hourly `precipitation`
    hours_ge_5mm         -> number of hours with `precipitation` >= 5 mm
    month/day_of_year/year -> calendar values of the feature date

ASSUMPTIONS TO VERIFY AGAINST YOUR TRAINING NOTEBOOK (a mismatch silently
degrades predictions):
  * hours_ge_* counts hours of `precipitation` (not `rain`).
  * wind_direction_10m_mean is a plain arithmetic mean of hourly directions.
  * "day" = 00:00-23:00 in the configured timezone.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CALENDAR_FEATURES = ("month", "day_of_year", "year")
HOURS_GE_FEATURES = {"hours_ge_5mm": 5.0, "hours_ge_10mm": 10.0}
HOURS_GE_SOURCE = "precipitation"
AGG_SUFFIXES = {"_mean": "mean", "_min": "min", "_max": "max", "_sum": "sum"}

# feature -> (kind, hourly source variable, argument)
FeaturePlan = Dict[str, Tuple[str, Optional[str], object]]


def build_feature_plan(features: Iterable[str]) -> FeaturePlan:
    plan: FeaturePlan = {}
    for name in features:
        if name in CALENDAR_FEATURES:
            plan[name] = ("calendar", None, None)
        elif name in HOURS_GE_FEATURES:
            plan[name] = ("hours_ge", HOURS_GE_SOURCE, HOURS_GE_FEATURES[name])
        else:
            for suffix, agg in AGG_SUFFIXES.items():
                if name.endswith(suffix):
                    plan[name] = ("agg", name[: -len(suffix)], agg)
                    break
            else:
                raise ValueError(
                    f"Don't know how to compute feature '{name}'. "
                    "Add a rule in app/features.py."
                )
    return plan


def hourly_variables(plan: FeaturePlan) -> List[str]:
    """Open-Meteo hourly variables that must be downloaded to build the plan."""
    return sorted({src for _, src, _ in plan.values() if src})


def compute_daily_features(
    hourly: Optional[pd.DataFrame],
    day: dt.date,
    plan: FeaturePlan,
    min_hours: int = 20,
) -> Dict[str, Optional[float]]:
    """Aggregate one local day of hourly data. Unavailable features are None
    (the model's imputer fills them; the caller decides how many are tolerable)."""
    if hourly is None or hourly.empty:
        day_df = pd.DataFrame()
    else:
        idx = pd.DatetimeIndex(hourly.index)
        day_df = hourly.loc[idx.normalize() == pd.Timestamp(day)]

    out: Dict[str, Optional[float]] = {}
    for name, (kind, src, arg) in plan.items():
        if kind == "calendar":
            out[name] = float(
                {"month": day.month, "day_of_year": day.timetuple().tm_yday, "year": day.year}[name]
            )
            continue

        if src not in day_df.columns:
            out[name] = None
            continue

        series = pd.to_numeric(day_df[src], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(series.notna().sum()) < min_hours:
            out[name] = None
            continue

        if kind == "hours_ge":
            value = float((series >= float(arg)).sum())
        else:
            value = float(getattr(series, str(arg))())
        out[name] = None if np.isnan(value) else value
    return out
