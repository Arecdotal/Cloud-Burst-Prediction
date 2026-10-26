"""Pydantic request models."""
import datetime as dt
from typing import Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, create_model


def make_feature_model(feature_names) -> Type[BaseModel]:
    """Request body for POST /predict, built from model_metadata.json.
    Every feature is optional (the model imputes), unknown names are rejected."""
    fields = {name: (Optional[float], None) for name in feature_names}
    return create_model("FeatureInput", __config__=ConfigDict(extra="forbid"), **fields)


class RunRequest(BaseModel):
    feature_date: Optional[dt.date] = Field(
        default=None, description="Day whose weather is used as input (default: today, local time). "
                                  "The prediction is for the following day.")
    source: Literal["auto", "forecast", "archive"] = "auto"
    persist: bool = True
    notify: bool = True


class OutcomeRequest(BaseModel):
    observed_max_rainfall_mm: float = Field(
        ge=0, description="Observed maximum daily rainfall (mm) across the region on the target date, "
                          "e.g. from IMD gridded data.")
    source: str = "IMD"
