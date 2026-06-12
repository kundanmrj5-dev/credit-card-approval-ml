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

FIELD_LABELS = {
    "A1": "Applicant profile type",
    "A2": "Applicant age",
    "A3": "Debt amount",
    "A4": "Account status",
    "A5": "Customer category",
    "A6": "Employment category",
    "A7": "Residence category",
    "A8": "Years employed",
    "A9": "Prior credit approved",
    "A10": "Currently employed",
    "A11": "Credit history count",
    "A12": "Owns property",
    "A13": "Application region",
    "A14": "Area code",
    "A15": "Income",
}

FIELD_HELP = {
    "A1": "Original anonymized column: A1",
    "A2": "Original anonymized column: A2",
    "A3": "Original anonymized column: A3",
    "A4": "Original anonymized column: A4",
    "A5": "Original anonymized column: A5",
    "A6": "Original anonymized column: A6",
    "A7": "Original anonymized column: A7",
    "A8": "Original anonymized column: A8",
    "A9": "Original anonymized column: A9",
    "A10": "Original anonymized column: A10",
    "A11": "Original anonymized column: A11",
    "A12": "Original anonymized column: A12",
    "A13": "Original anonymized column: A13",
    "A14": "Original anonymized column: A14",
    "A15": "Original anonymized column: A15",
}

OPTION_LABELS = {
    "A1": {"b": "Profile type B", "a": "Profile type A"},
    "A4": {"u": "Active account", "y": "Secondary account", "l": "Legacy account", "t": "Temporary account"},
    "A5": {"g": "Standard customer", "p": "Premium customer", "gg": "Extended customer"},
    "A6": {
        "c": "Employment group C",
        "d": "Employment group D",
        "cc": "Employment group CC",
        "i": "Employment group I",
        "j": "Employment group J",
        "k": "Employment group K",
        "m": "Employment group M",
        "r": "Employment group R",
        "q": "Employment group Q",
        "w": "Employment group W",
        "x": "Employment group X",
        "e": "Employment group E",
        "aa": "Employment group AA",
        "ff": "Employment group FF",
    },
    "A7": {
        "v": "Residence group V",
        "h": "Residence group H",
        "bb": "Residence group BB",
        "j": "Residence group J",
        "n": "Residence group N",
        "z": "Residence group Z",
        "dd": "Residence group DD",
        "ff": "Residence group FF",
        "o": "Residence group O",
    },
    "A9": {"t": "Yes", "f": "No"},
    "A10": {"t": "Yes", "f": "No"},
    "A12": {"t": "Yes", "f": "No"},
    "A13": {"g": "Region G", "p": "Region P", "s": "Region S"},
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_form() -> dict[str, object]:
    values: dict[str, object] = {}

    st.subheader("Applicant Details")
    left, right = st.columns(2)

    for index, column in enumerate(FEATURE_COLUMNS):
        container = left if index % 2 == 0 else right
        label = FIELD_LABELS[column]
        help_text = FIELD_HELP[column]
        if column in NUMERIC_COLUMNS:
            values[column] = container.number_input(
                label,
                value=float(DEFAULT_NUMERIC_VALUES[column]),
                step=1.0,
                help=help_text,
            )
        elif column in CATEGORICAL_COLUMNS:
            values[column] = container.selectbox(
                label,
                CATEGORY_OPTIONS[column],
                format_func=lambda option, feature=column: OPTION_LABELS[feature].get(option, option),
                help=help_text,
            )

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
