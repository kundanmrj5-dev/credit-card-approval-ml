"""Prediction helpers for trained credit approval models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURE_COLUMNS


def load_model(model_path: str | Path = "models/credit_card_approval_pipeline.joblib"):
    """Load a trained pipeline."""

    return joblib.load(model_path)


def predict_application(application: dict[str, object], model_path: str | Path = "models/credit_card_approval_pipeline.joblib") -> dict[str, float | int | str]:
    """Predict approval for one application represented as a dictionary."""

    missing_columns = sorted(set(FEATURE_COLUMNS) - set(application))
    if missing_columns:
        raise ValueError(f"Application is missing columns: {missing_columns}")

    model = load_model(model_path)
    row = pd.DataFrame([application], columns=FEATURE_COLUMNS)
    approval_probability = float(model.predict_proba(row)[0, 1])
    prediction = int(approval_probability >= 0.5)

    return {
        "prediction": prediction,
        "decision": "approved" if prediction else "rejected",
        "approval_probability": approval_probability,
    }
