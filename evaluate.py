"""Time-based validation of candidate models.

Split: train on Jan-Aug 2025, test on Sep-Oct 2025. validation.csv is Nov-Dec,
so the model is always judged on months after the ones it learned from.
Cleaning fill values and outlier removal are fit on the training fold only.

Two test views are reported:
  - "all":   every Sep-Oct row, raw labels included (closest to Spotter's scoring)
  - "clean": label outliers removed (how well we price normal loads)
A second check hides CITY_HOLDOUT random cities from training, because ~12% of
validation.csv rows involve cities never seen in train_test.csv.
A month-by-month backtest (train on all earlier months, test on the next one)
shows how stable each model is across market regimes. August is the most
relevant month: its quote_signal pattern matches Nov-Dec in validation.csv.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import clean
from baseline import RatePerMileBaseline
from model import RateModel

SPLIT_DATE = "2025-09-01"
CITY_HOLDOUT = 8
SEED = 0


BACKTEST_MONTHS = [6, 7, 8, 9, 10]


def time_split(
    train_rows: Callable[[pd.DataFrame], pd.Series] = lambda df: df["date"] < SPLIT_DATE,
    test_rows: Callable[[pd.DataFrame], pd.Series] = lambda df: df["date"] >= SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = clean.load("train_test.csv")
    train_raw = raw[train_rows(raw)]
    test_raw = raw[test_rows(raw)]
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


def backtest(make_model: Callable[[], object]) -> pd.Series:
    """Clean-row MAE per month, training on every earlier month."""
    mae = {}
    for month in BACKTEST_MONTHS:
        train, test = time_split(
            lambda df: df["date"].dt.month < month, lambda df: df["date"].dt.month == month
        )
        test = test[~clean.outlier_mask(test)]
        mae[month] = (test["posted_rate"] - make_model().fit(train).predict(test)).abs().mean()
    return pd.Series(mae, name="MAE by test month")


if __name__ == "__main__":
    models = {"baseline": RatePerMileBaseline, "gbm": RateModel}
    print(pd.concat([evaluate(make, name) for name, make in models.items()]).round(2).to_string())
    print("\nClean-row MAE by test month (expanding window):")
    print(pd.DataFrame({name: backtest(make) for name, make in models.items()}).round(1).to_string())
