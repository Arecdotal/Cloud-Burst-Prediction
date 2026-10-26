"""Background scheduler (APScheduler). Imported lazily so tests don't need it."""
from __future__ import annotations

import logging

from .config import Settings

logger = logging.getLogger(__name__)


def start_scheduler(settings: Settings, job):
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BackgroundScheduler(timezone=settings.timezone)
    scheduler.add_job(
        job,
        CronTrigger(hour=settings.schedule_hours, minute=settings.schedule_minute,
                    timezone=settings.timezone),
        id="forecast", max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
    if settings.run_on_startup:
        scheduler.add_job(job, id="startup-run")      # no trigger => runs once, immediately
    scheduler.start()
    logger.info("Scheduler started: hours=%s minute=%s tz=%s",
                settings.schedule_hours, settings.schedule_minute, settings.timezone)
    return scheduler
