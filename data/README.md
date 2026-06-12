# Data Folder

The training script downloads the UCI Credit Approval dataset automatically from the direct raw data file:

https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data

Dataset notes are available at:

https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names

Opening the folder URL without a filename can show `NOT FOUND`, so use the direct file links above.

If you want to train offline, place the raw file here as:

```text
data/crx.data
```

Then run:

```bash
python -m src.credit_card_approval.train --data-path data/crx.data
```

The raw file has no header row. Missing values are represented by `?`.
