# Freight Rate Prediction

Predicting the posted rate for truckload freight, for the Spotter ML Engineer assessment.

The final model is a gradient-boosted tree model (scikit-learn `HistGradientBoostingRegressor`) that predicts rate per mile. On a time-based holdout (train Jan–Aug 2025, test Sep–Oct 2025) it gets a mean absolute error of **$43 (1.8%)** on normal loads, against $80 (3.5%) for a rate-per-mile lookup baseline.

The write-up with the data findings, validation approach and the December chart is in [REPORT.md](REPORT.md) (also as `report.pdf`).

## How to run

Needs Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-model.txt

python run.py        # train, write both prediction files, run score.py
python evaluate.py   # holdout metrics and month-by-month backtest
```

`run.py` takes a few seconds. It writes:

- `validation_predictions.csv`: `load_id,predicted_rate` for the 12,000 validation loads
- `data/december_chart_inputs.csv`: `predicted_rate` filled in for the 31 December rows
- `scorer_results/candidate_december.png`: the chart produced by `score.py`

`requirements.txt` is the scorer's original file and is unchanged. `requirements-model.txt` adds scikit-learn and pins everything.

## Layout

| File | What it does |
|---|---|
| `clean.py` | Loads the CSVs and fixes data-quality issues (negative and missing weights, missing market index, label outliers) |
| `features.py` | Builds the feature matrix. The docstring explains why the market features are left out |
| `baseline.py` | Median rate per mile by lane and equipment, falling back to equipment and distance band |
| `model.py` | The gradient boosting model |
| `evaluate.py` | Time-based holdout, unseen-city check and monthly backtest |
| `run.py` | End-to-end pipeline |
| `score.py` | Provided scorer (unchanged) |

The provided CSVs are in `data/`, renamed to the underscore names `score.py` and the instructions use (`train-test.csv` became `data/train_test.csv`, and so on).
