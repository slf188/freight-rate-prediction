"""Gradient-boosted trees predicting rate-per-mile.

scikit-learn's HistGradientBoostingRegressor is the same algorithm family as
LightGBM (histogram-based boosting, native categoricals) with no system
dependencies. The target is rate-per-mile rather than the raw rate: distance
explains most of the price, so the trees spend their splits on everything else.
Predicting rpm also beat log(rate) and raw rate in validation.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build_features


class RateModel:
    def __init__(self) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.05,
            categorical_features="from_dtype",
            random_state=0,
        )

    def fit(self, df: pd.DataFrame) -> "RateModel":
        self.model.fit(build_features(df), df["posted_rate"] / df["distance"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        rpm = self.model.predict(build_features(df))
        return pd.Series(rpm * df["distance"].to_numpy(), index=df.index, name="predicted_rate")
