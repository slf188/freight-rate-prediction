"""Load and clean the raw CSVs.

Data-quality issues found during profiling, and how each is handled:
- Negative weights (~0.6% of rows) look like sign flips -> take the absolute value.
- Missing weight (~0.6%) -> fill with the training median.
- Missing market_index (~0.8%) -> fill with that day's mean (it barely varies
  within a day), falling back to the training median.
- Label outliers (~1.4% of training rows): rate-per-mile about 3x or 0.2x the
  typical value for its equipment and distance band. There is a clean empty gap
  in the ratio distribution (nothing between 0.5-0.7 or 1.5-2.0), so they are
  treated as entry errors and dropped from training only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
DISTANCE_BANDS = [0, 250, 500, 1000, 2000, float("inf")]
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.75


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, parse_dates=["date"])


def fit_fill_values(train: pd.DataFrame) -> dict[str, float]:
    """Fill values are learned from training data only, then reused everywhere."""
    return {
        "weight": train["weight"].abs().median(),
        "market_index": train["market_index"].median(),
    }


def clean_features(df: pd.DataFrame, fills: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["weight"].abs().fillna(fills["weight"])
    daily_mi = df.groupby("date")["market_index"].transform("mean")
    df["market_index"] = df["market_index"].fillna(daily_mi).fillna(fills["market_index"])
    return df


def outlier_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose rate-per-mile is far outside its equipment/distance band."""
    rpm = df["posted_rate"] / df["distance"]
    band = pd.cut(df["distance"], DISTANCE_BANDS)
    typical = rpm.groupby([df["equipment"], band], observed=True).transform("median")
    ratio = rpm / typical
    return (ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH)


def load_train(fills: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    raw = load("train_test.csv")
    fills = fills or fit_fill_values(raw)
    df = clean_features(raw, fills)
    df = df[~outlier_mask(df)].reset_index(drop=True)
    return df, fills
