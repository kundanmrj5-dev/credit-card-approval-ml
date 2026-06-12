"""Streamlit app for credit card approval prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.credit_card_approval.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

MODEL_PATH = Path("models/credit_card_approval_pipeline.joblib")

CATEGORY_OPTIONS = {
    "A1": ["b", "a"],
    "A4": ["u", "y", "l", "t"],
    "A5": ["g", "p", "gg"],
    "A6": ["c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"],
    "A7": ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"],
    "A9": ["t", "f"],
    "A10": ["t", "f"],
    "A12": ["t", "f"],
    "A13": ["g", "p", "s"],
}

DEFAULT_NUMERIC_VALUES = {
    "A2": 30.0,
    "A3": 4.0,
    "A8": 2.0,
    "A11": 1.0,
    "A14": 160.0,
    "A15": 100.0,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_form() -> dict[str, object]:
    values: dict[str, object] = {}

    st.subheader("Application Features")
    left, right = st.columns(2)

    for index, column in enumerate(FEATURE_COLUMNS):
        container = left if index % 2 == 0 else right
        if column in NUMERIC_COLUMNS:
            values[column] = container.number_input(
                column,
                value=float(DEFAULT_NUMERIC_VALUES[column]),
                step=1.0,
            )
        elif column in CATEGORICAL_COLUMNS:
            values[column] = container.selectbox(column, CATEGORY_OPTIONS[column])

    return values


def main() -> None:
    st.set_page_config(page_title="Credit Card Approval Predictor", page_icon=":credit_card:", layout="wide")
    st.title("Credit Card Approval Predictor")

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python -m src.credit_card_approval.train` first.")
        st.stop()

    model = load_model()
    application = build_input_form()

    if st.button("Predict Approval", type="primary"):
        row = pd.DataFrame([application], columns=FEATURE_COLUMNS)
        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(probability >= 0.5)
        decision = "Approved" if prediction else "Rejected"

        st.metric("Decision", decision)
        st.metric("Approval Probability", f"{probability:.1%}")
        st.progress(probability)


if __name__ == "__main__":
    main()
