"""Data loading helpers for the UCI Credit Approval dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

FEATURE_COLUMNS = [f"A{i}" for i in range(1, 16)]
TARGET_COLUMN = "approved"
RAW_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

CATEGORICAL_COLUMNS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
NUMERIC_COLUMNS = ["A2", "A3", "A8", "A11", "A14", "A15"]


def load_credit_approval_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the credit approval dataset from a local path or the UCI URL."""

    source = Path(data_path) if data_path else DATA_URL
    df = pd.read_csv(source, header=None, names=RAW_COLUMNS, na_values="?")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"+": 1, "-": 0})
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains unknown labels. Expected '+' and '-'.")

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target vector."""

    return df[FEATURE_COLUMNS], df[TARGET_COLUMN].astype(int)
