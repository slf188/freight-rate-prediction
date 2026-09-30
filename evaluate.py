"""Time-based validation of candidate models.

Split: train on Jan-Aug 2025, test on Sep-Oct 2025. validation.csv is Nov-Dec,
so the model is always judged on months after the ones it learned from.
Cleaning fill values and outlier removal are fit on the training fold only.

Two test views are reported:
  - "all":   every Sep-Oct row, raw labels included (closest to Spotter's scoring)
  - "clean": label outliers removed (how well we price normal loads)
A second check hides CITY_HOLDOUT random cities from training, because ~12% of
validation.csv rows involve cities never seen in train_test.csv.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import clean
from baseline import RatePerMileBaseline

SPLIT_DATE = "2025-09-01"
CITY_HOLDOUT = 8
SEED = 0


def time_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = clean.load("train_test.csv")
    train_raw = raw[raw["date"] < SPLIT_DATE]
    test_raw = raw[raw["date"] >= SPLIT_DATE]
    fills = clean.fit_fill_values(train_raw)
    train = clean.clean_features(train_raw, fills)
    train = train[~clean.outlier_mask(train)].reset_index(drop=True)
    test = clean.clean_features(test_raw, fills).reset_index(drop=True)
    return train, test


def score(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = (y - pred).abs()
    return {"MAE": err.mean(), "MAPE%": 100 * (err / y).mean(), "rows": len(y)}


def evaluate(make_model: Callable[[], object], name: str) -> pd.DataFrame:
    train, test = time_split()
    ok = ~clean.outlier_mask(test)
    pred = make_model().fit(train).predict(test)

    cities = np.random.default_rng(SEED).choice(
        sorted(set(train["pickup"]) | set(train["delivery"])), CITY_HOLDOUT, replace=False
    )
    touches = lambda df: df["pickup"].isin(cities) | df["delivery"].isin(cities)
    unseen_pred = make_model().fit(train[~touches(train)]).predict(test[touches(test) & ok])

    rows = {
        "all": score(test["posted_rate"], pred),
        "clean": score(test.loc[ok, "posted_rate"], pred[ok]),
        "unseen cities (clean)": score(test.loc[touches(test) & ok, "posted_rate"], unseen_pred),
    }
    return pd.DataFrame(rows).T.assign(model=name)


if __name__ == "__main__":
    results = evaluate(RatePerMileBaseline, "baseline")
    print(results.round(2).to_string())
