import pandas as pd

from src.credit_card_approval.data import FEATURE_COLUMNS, split_features_target
from src.credit_card_approval.train import build_pipeline


def test_pipeline_fits_with_missing_and_categorical_values():
    rows = [
        ["b", 30.83, 0.0, "u", "g", "w", "v", 1.25, "t", "t", 1, "f", "g", 202.0, 0, 1],
        ["a", 58.67, 4.46, "u", "g", "q", "h", 3.04, "t", "t", 6, "f", "g", 43.0, 560, 1],
        ["a", None, 0.5, "u", "g", "q", "h", 1.5, "t", "f", 0, "f", "g", 280.0, 824, 0],
        ["b", 27.83, 1.54, None, None, "w", "v", 3.75, "t", "t", 5, "t", "g", 100.0, 3, 1],
        [None, 20.17, 5.62, "u", "g", None, None, 1.71, "f", "f", 0, "f", "s", None, 0, 0],
        ["b", 32.08, 4.0, "y", "p", "cc", "v", 2.5, "f", "t", 2, "t", "g", 360.0, 1000, 0],
    ]
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["approved"])

    x, y = split_features_target(df)
    pipeline = build_pipeline()
    pipeline.fit(x, y)

    predictions = pipeline.predict(x)

    assert len(predictions) == len(df)
    assert set(predictions).issubset({0, 1})
