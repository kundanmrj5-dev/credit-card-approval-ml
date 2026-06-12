# Predicting Credit Card Approvals

An end-to-end machine learning project for predicting whether a credit card application should be approved. It follows the classic DataCamp-style workflow: missing value handling, categorical encoding, feature scaling, class imbalance handling, and hyperparameter tuning with `GridSearchCV` and Logistic Regression.

## Project Highlights

- Downloads the UCI Credit Approval dataset automatically.
- Handles missing values marked as `?`.
- Processes categorical and numerical columns with a single reproducible `Pipeline`.
- Scales numeric features and one-hot encodes categorical features.
- Tunes Logistic Regression hyperparameters with cross-validation.
- Compares regular and class-balanced Logistic Regression settings.
- Saves the trained model and evaluation reports.
- Includes an optional Streamlit prediction app.

## Dataset

The project uses the UCI Credit Approval dataset:

- 690 credit card applications
- 15 anonymized input attributes
- 1 target column: `+` for approved and `-` for rejected
- Mixed categorical and continuous features
- Missing values represented by `?`

Source files:

- Raw data: https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data
- Dataset notes: https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names

If you open only `https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening` in a browser, UCI may show `NOT FOUND`. Use the direct file links above.

## Project Structure

```text
credit-card-approval-ml/
  app.py
  requirements.txt
  README.md
  data/
    README.md
  src/
    credit_card_approval/
      __init__.py
      data.py
      train.py
      predict.py
  tests/
    test_pipeline.py
```

## Setup

```bash
cd credit-card-approval-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Train the Model

```bash
python -m src.credit_card_approval.train
```

The training script will:

1. Load the dataset from UCI, unless `--data-path` is provided.
2. Split the data with stratification.
3. Run `GridSearchCV` over Logistic Regression settings.
4. Evaluate the best model on a holdout test set.
5. Save artifacts under `models/` and `reports/`.

Useful options:

```bash
python -m src.credit_card_approval.train --data-path data/crx.data
python -m src.credit_card_approval.train --cv 10 --test-size 0.25
```

## Run the App

Train the model first, then run:

```bash
streamlit run app.py
```

The app exposes all 15 anonymized application fields and returns approval probability plus the predicted decision.

## Run Tests

```bash
pytest
```

## Model Notes

Because the original feature names and values are anonymized, this project is suitable for practicing ML workflow and deployment patterns, not for making real lending decisions. Real credit approval systems require stronger fairness, compliance, explainability, monitoring, and human review controls.
