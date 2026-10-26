"""Loads the pickled sklearn Pipeline and turns a feature dict into a risk result."""
from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

logger = logging.getLogger(__name__)

TARGET = "Next-day extreme rainfall"
SCORE_NOTE = (
    "The score comes from a class-balanced logistic regression and is NOT a calibrated "
    "probability of extreme rainfall. Validation precision was ~0.29, so most alerts "
    "will be false alarms. This is not a verified cloudburst detector."
)


class InsufficientDataError(ValueError):
    """Too many input features are missing to trust a prediction."""


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


class ModelService:
    def __init__(self, model_path: Path, metadata_path: Path,
                 max_missing_fraction: float = 0.25,
                 expected_sklearn_version: Optional[str] = None,
                 allow_version_mismatch: bool = False):
        with open(metadata_path, "r", encoding="utf-8") as fh:
            self.metadata: Dict[str, Any] = json.load(fh)
        self.features: List[str] = list(self.metadata["features"])
        self.threshold = float(self.metadata["threshold"])
        self.max_missing_fraction = max_missing_fraction
        self.train_end_year = int(str(self.metadata.get("date_end", "2024"))[:4])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.pipeline = joblib.load(model_path)
        mismatch = any(issubclass(w.category, InconsistentVersionWarning) for w in caught)
        if expected_sklearn_version:
            mismatch = mismatch or (
                sklearn.__version__.split(".")[:2] != expected_sklearn_version.split(".")[:2]
            )
        if mismatch:
            msg = (
                f"scikit-learn {sklearn.__version__} is installed but the model was trained with "
                f"{expected_sklearn_version or 'a different version'}. Unpickling across versions can "
                "crash at prediction time or silently change results. Install the pinned version from "
                "requirements.txt (or re-train/re-export the model), or set ALLOW_SKLEARN_MISMATCH=true "
                "to start anyway."
            )
            if not allow_version_mismatch:
                raise RuntimeError(msg)
            logger.warning(msg)

        # Fail fast if the pickle and the metadata disagree about the inputs.
        trained_names = list(getattr(self.pipeline[0], "feature_names_in_", self.features))
        if trained_names != self.features:
            raise RuntimeError("Feature list in model_metadata.json does not match the pickled model.")
        self.coef = np.asarray(self.pipeline[-1].coef_[0], dtype=float)

    # ------------------------------------------------------------------ info
    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name"),
            "target": self.metadata.get("target"),
            "positive_label_definition": self.metadata.get("positive_label_definition"),
            "threshold": self.threshold,
            "n_features": len(self.features),
            "features": self.features,
            "training_rows": self.metadata.get("training_rows"),
            "positive_labels": self.metadata.get("positive_labels"),
            "training_period": [self.metadata.get("date_start"), self.metadata.get("date_end")],
            "validation": {
                "f1": self.metadata.get("validation_f1"),
                "precision": self.metadata.get("validation_precision"),
                "recall": self.metadata.get("validation_recall"),
            },
            "caveat": self.metadata.get("caveat"),
            "score_note": SCORE_NOTE,
            "runtime_sklearn_version": sklearn.__version__,
        }

    # --------------------------------------------------------------- predict
    def predict(self, features: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = sorted(set(features) - set(self.features))
        if unknown:
            raise ValueError(f"Unknown features: {unknown}")

        values = {f: _to_float(features.get(f)) for f in self.features}
        missing = [f for f, v in values.items() if v is None]
        fraction = len(missing) / len(self.features)
        if fraction > self.max_missing_fraction:
            raise InsufficientDataError(
                f"{len(missing)} of {len(self.features)} features are missing "
                f"(limit {self.max_missing_fraction:.0%}): {missing}"
            )

        row = pd.DataFrame(
            [[np.nan if values[f] is None else values[f] for f in self.features]],
            columns=self.features,
        )
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        prediction = int(probability >= self.threshold)

        warns: List[str] = []
        if missing:
            warns.append(f"{len(missing)} feature(s) were missing and filled with training medians.")
        year = values.get("year")
        if year is not None and year > self.train_end_year + 1:
            warns.append(
                f"'year' ({int(year)}) is outside the training period (ends {self.train_end_year}); "
                "the model extrapolates its year trend."
            )

        return {
            "predicted_probability": round(probability, 4),
            "threshold": self.threshold,
            "prediction": prediction,
            "risk_label": "Elevated extreme-rainfall risk" if prediction else "Below model alert threshold",
            "target": TARGET,
            "missing_features": missing,
            "top_factors": self._top_factors(row, values, missing),
            "warnings": warns,
            "note": SCORE_NOTE,
        }

    def _top_factors(self, row: pd.DataFrame, values: Dict[str, Optional[float]],
                     missing: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """Per-feature contribution to the logit: coefficient x standardised value."""
        standardised = np.asarray(self.pipeline[:-1].transform(row))[0]
        contributions = standardised * self.coef
        skip = set(missing)
        order = np.argsort(-np.abs(contributions))
        factors: List[Dict[str, Any]] = []
        for i in order:
            name = self.features[i]
            if name in skip:
                continue
            factors.append({
                "feature": name,
                "value": round(values[name], 4),
                "contribution": round(float(contributions[i]), 3),
                "effect": "raises risk score" if contributions[i] > 0 else "lowers risk score",
            })
            if len(factors) == k:
                break
        return factors
