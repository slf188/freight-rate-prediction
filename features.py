"""Turn cleaned rows into model features.

Calendar features are limited to day of week: month/day-of-year would ask the
model to extrapolate to Nov-Dec, which it never saw in training. Seasonality is
instead carried by market_index and quote_signal, which exist for every date.
City names are categoricals; coordinates are kept as a secondary signal so the
model can place the 8 cities that only appear in validation.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = [
    "distance",
    "log_distance",
    "weight",
    "market_index",
    "quote_signal",
    "day_of_week",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]
FEATURES = CATEGORICAL + NUMERIC


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["log_distance"] = np.log(X["distance"])
    X["day_of_week"] = X["date"].dt.dayofweek
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    return X[FEATURES]
