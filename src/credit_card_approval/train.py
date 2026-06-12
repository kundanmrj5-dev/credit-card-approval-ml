"""Train a Logistic Regression model for credit card approval prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_credit_approval_data, split_features_target


def build_pipeline() -> Pipeline:
    """Build the preprocessing and modeling pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", random_state=42)),
        ]
    )


def tune_model(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, cv: int) -> GridSearchCV:
    """Tune Logistic Regression hyperparameters with cross-validation."""

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__penalty": ["l1", "l2"],
        "model__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Compute holdout metrics."""

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    report = classification_report(y_test, predictions, target_names=["rejected", "approved"])
    matrix = confusion_matrix(y_test, predictions)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix(matrix: list[list[int]], reports_dir: Path) -> None:
    """Save a confusion matrix image."""

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=["rejected", "approved"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Credit Approval Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a credit card approval classifier.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional local raw dataset path.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory for model artifacts.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory for evaluation reports.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size.")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_credit_approval_data(args.data_path)
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline()
    search = tune_model(pipeline, x_train, y_train, args.cv)
    best_model = search.best_estimator_

    metrics = evaluate_model(best_model, x_test, y_test)
    metrics["best_params"] = search.best_params_
    metrics["best_cv_f1"] = search.best_score_
    metrics["train_rows"] = len(x_train)
    metrics["test_rows"] = len(x_test)

    joblib.dump(best_model, args.models_dir / "credit_card_approval_pipeline.joblib")

    metrics_for_json = {key: value for key, value in metrics.items() if key != "classification_report"}
    (args.reports_dir / "metrics.json").write_text(json.dumps(metrics_for_json, indent=2), encoding="utf-8")
    (args.reports_dir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    save_confusion_matrix(metrics["confusion_matrix"], args.reports_dir)

    print("Training complete")
    print(f"Best parameters: {search.best_params_}")
    print(f"Holdout F1: {metrics['f1']:.3f}")
    print(f"Holdout ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Saved model: {args.models_dir / 'credit_card_approval_pipeline.joblib'}")


if __name__ == "__main__":
    main()
