"""Alert delivery. Always logs; optionally POSTs JSON to a webhook (Slack/Discord/custom)."""
from __future__ import annotations

import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, webhook_url: str = "", timeout: float = 10.0,
                 session: "requests.Session | None" = None):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.session = session or requests.Session()

    @staticmethod
    def message(rec: Dict[str, Any]) -> str:
        return (
            f"Elevated extreme-rainfall risk for {rec['target_date']} ({rec['location']} region). "
            f"Model score {rec['predicted_probability']:.2f} is at or above the alert threshold "
            f"{rec['threshold']}. This is a statistical screening signal for >=50 mm/day rainfall, "
            "not a confirmed cloudburst forecast; false alarms are common."
        )

    def send(self, rec: Dict[str, Any]) -> bool:
        """Return True if the alert counts as delivered (log, plus webhook if configured)."""
        text = self.message(rec)
        logger.warning("ALERT: %s", text)
        if not self.webhook_url:
            return True
        payload = {"text": text, "content": text, "event": "extreme_rainfall_alert",
                   "data": {k: rec[k] for k in ("location", "feature_date", "target_date",
                                                "predicted_probability", "threshold")}}
        try:
            resp = self.session.post(self.webhook_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("Alert webhook failed: %s", exc)
            return False
