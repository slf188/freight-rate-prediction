"""Baseline: median rate-per-mile x distance.

Rate-per-mile is looked up at the most specific level with enough history:
  1. lane + equipment      (pickup, delivery, equipment), if >= MIN_LOADS loads
  2. equipment + distance band, always available
This is how a broker would quote from history, so any ML model has to beat it.
"""
from __future__ import annotations

import pandas as pd

from clean import DISTANCE_BANDS

MIN_LOADS = 5
LANE = ["pickup", "delivery", "equipment"]


def _band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["distance"], DISTANCE_BANDS).astype(str)


class RatePerMileBaseline:
    def fit(self, df: pd.DataFrame) -> "RatePerMileBaseline":
        rpm = df["posted_rate"] / df["distance"]
        lane = rpm.groupby([df[c] for c in LANE]).agg(["median", "size"])
        self.lane_rpm = lane.loc[lane["size"] >= MIN_LOADS, "median"].rename("lane_rpm")
        self.band_rpm = rpm.groupby([df["equipment"], _band(df)]).median().rename("band_rpm")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        keys = df[LANE].assign(band=_band(df))
        out = keys.join(self.lane_rpm, on=LANE).join(self.band_rpm, on=["equipment", "band"])
        rpm = out["lane_rpm"].fillna(out["band_rpm"])
        return (rpm * df["distance"]).rename("predicted_rate")
