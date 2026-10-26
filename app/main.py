"""FastAPI application.

Run with:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000

NOTE: do not add `from __future__ import annotations` here. FastAPI must resolve the
locally-built FeatureInput request model from the real (non-string) annotation.
"""
import datetime as dt
import hmac
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .bootstrap import build_service
from .config import Settings
from .model_service import InsufficientDataError
from .schemas import OutcomeRequest, RunRequest, make_feature_model
from .service import ForecastService
from .weather import WeatherError

logger = logging.getLogger("cloudburst")


def create_app(settings: Optional[Settings] = None,
               service: Optional[ForecastService] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    svc = service or build_service(settings)
    FeatureInput = make_feature_model(svc.model.features)
    state = {"scheduler": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.enable_scheduler:
            from .scheduler import start_scheduler
            state["scheduler"] = start_scheduler(settings, svc.run_job)
        yield
        if state["scheduler"]:
            state["scheduler"].shutdown(wait=False)

    app = FastAPI(
        title="Extreme Rainfall Risk API",
        version="1.0.0",
        description="Next-day extreme-rainfall (>=50 mm/day) risk screening for the Dehradun region. "
                    "A prototype statistical model, not a verified cloudburst detector.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST", "PUT"], allow_headers=["*"])

    def require_admin(x_api_key: Optional[str] = Header(default=None)) -> None:
        if not settings.admin_api_key:
            raise HTTPException(403, "Admin endpoints are disabled: set ADMIN_API_KEY.")
        if not x_api_key or not hmac.compare_digest(x_api_key, settings.admin_api_key):
            raise HTTPException(401, "Invalid or missing X-API-Key header.")

    def age_info(rec: dict) -> dict:
        created = dt.datetime.fromisoformat(rec["created_at"])
        age_min = (dt.datetime.now(dt.timezone.utc) - created).total_seconds() / 60
        return {"age_minutes": round(age_min, 1), "stale": age_min > settings.stale_after_hours * 60}

    def run_pipeline(day, source, persist, notify, trigger):
        try:
            return svc.run(day, source, persist=persist, notify=notify, trigger=trigger)
        except InsufficientDataError as exc:
            raise HTTPException(422, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except WeatherError as exc:
            logger.error("Weather provider error: %s", exc)
            raise HTTPException(502, f"Weather data unavailable: {exc}")

    # ------------------------------------------------------------- public
    @app.get("/health", tags=["system"])
    def health():
        latest = svc.store.latest_prediction()
        return {
            "status": "ok" if svc.store.ping() else "degraded",
            "model_loaded": True,
            "database": svc.store.ping(),
            "scheduler_running": bool(state["scheduler"]),
            "last_run": svc.store.last_run(),
            "latest_prediction": ({"target_date": latest["target_date"], **age_info(latest)} if latest else None),
        }

    @app.get("/model-info", tags=["system"])
    def model_info():
        return {**svc.model.info(), "location": settings.location_name,
                "coordinates": [settings.latitude, settings.longitude], "timezone": settings.timezone}

    @app.get("/predictions/latest", tags=["predictions"])
    def latest_prediction():
        """Most recent stored prediction (what the frontend should poll)."""
        rec = svc.store.latest_prediction()
        if rec is None:
            raise HTTPException(404, "No prediction yet. The first scheduled run has not completed.")
        return {**rec, **age_info(rec)}

    @app.get("/predictions", tags=["predictions"])
    def prediction_history(limit: int = Query(30, ge=1, le=366),
                           start: Optional[dt.date] = None, end: Optional[dt.date] = None):
        """History, one row (the latest run) per target date, newest first."""
        return svc.store.list_predictions(limit, start.isoformat() if start else None,
                                          end.isoformat() if end else None)

    @app.post("/predict", tags=["predictions"])
    def predict(payload: FeatureInput):  # type: ignore[valid-type]
        """Score a manually supplied feature set (nothing is stored)."""
        try:
            return svc.model.predict(payload.model_dump())
        except InsufficientDataError as exc:
            raise HTTPException(422, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/metrics/live", tags=["evaluation"])
    def live_metrics():
        """Precision/recall of past predictions against outcomes recorded via PUT /outcomes."""
        return svc.store.live_metrics()

    # -------------------------------------------------------------- admin
    @app.post("/predictions/run", tags=["admin"], dependencies=[Depends(require_admin)])
    def run_now(body: RunRequest = RunRequest()):
        """Fetch weather and run the model now."""
        return run_pipeline(body.feature_date, body.source, body.persist, body.notify, "manual")

    @app.get("/features/preview", tags=["admin"], dependencies=[Depends(require_admin)])
    def preview_features(feature_date: Optional[dt.date] = None,
                         source: str = Query("auto", pattern="^(auto|forecast|archive)$")):
        """Show the 36 computed features for a day, to compare with your training data."""
        try:
            return svc.preview_features(feature_date, source)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except WeatherError as exc:
            raise HTTPException(502, f"Weather data unavailable: {exc}")

    @app.put("/outcomes/{target_date}", tags=["evaluation"], dependencies=[Depends(require_admin)])
    def record_outcome(target_date: dt.date, body: OutcomeRequest):
        """Record what actually happened on `target_date` so live metrics can be computed."""
        return svc.store.record_outcome(target_date.isoformat(), body.observed_max_rainfall_mm,
                                        settings.extreme_threshold_mm, body.source)

    return app

