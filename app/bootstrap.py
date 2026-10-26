"""Builds the object graph. Free of web-framework imports so it is easy to test."""
from __future__ import annotations

from .alerts import Notifier
from .config import Settings
from .features import build_feature_plan, hourly_variables
from .model_service import ModelService
from .service import ForecastService
from .storage import Store
from .weather import OpenMeteoClient


def build_service(settings: Settings) -> ForecastService:
    model = ModelService(
        settings.model_path, settings.metadata_path,
        max_missing_fraction=settings.max_missing_fraction,
        expected_sklearn_version=settings.expected_sklearn_version,
        allow_version_mismatch=settings.allow_sklearn_mismatch,
    )
    plan = build_feature_plan(model.features)          # raises if a feature has no rule
    weather = OpenMeteoClient(settings, hourly_variables(plan))
    return ForecastService(
        settings=settings, model=model, plan=plan, store=Store(settings.db_path),
        weather=weather, notifier=Notifier(settings.alert_webhook_url),
    )
