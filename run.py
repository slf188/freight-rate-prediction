"""End-to-end pipeline: train on all of train_test.csv, write both prediction
files, then run the official scorer.

    python run.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

import clean
from model import RateModel

ROOT = Path(__file__).parent
PREDICTIONS = ROOT / "validation_predictions.csv"
DECEMBER = clean.DATA / "december_chart_inputs.csv"


def city_coords(train: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon per city (each city has a single fixed coordinate in the data)."""
    frames = [
        train[[f"{side}", f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for side in ("pickup", "delivery")
    ]
    return pd.concat(frames).drop_duplicates("city").set_index("city")


def december_rows(template: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """The chart file has no coordinates or market data; coordinates come from training cities."""
    rows = template.assign(date=pd.to_datetime(template["date"]))
    for side in ("pickup", "delivery"):
        rows[f"{side}_lat"] = rows[side].map(coords["lat"])
        rows[f"{side}_lon"] = rows[side].map(coords["lon"])
    return rows


def main() -> None:
    train, fills = clean.load_train()
    print(f"Training on {len(train):,} cleaned loads ({train['date'].min():%b %d} - {train['date'].max():%b %d})")
    model = RateModel().fit(train)

    validation = clean.clean_features(clean.load("validation.csv"), fills)
    predicted = model.predict(validation).round(2)
    template = pd.read_csv(clean.DATA / "validation_predictions_template.csv")
    template["predicted_rate"] = template["load_id"].map(dict(zip(validation["load_id"], predicted)))
    template.to_csv(PREDICTIONS, index=False)
    print(f"Wrote {PREDICTIONS.name}: {len(template):,} rows, median ${template['predicted_rate'].median():,.0f}")

    december = pd.read_csv(DECEMBER)
    december["predicted_rate"] = model.predict(december_rows(december, city_coords(train))).round(2)
    december.to_csv(DECEMBER, index=False)
    print(f"Filled {DECEMBER.name}: ${december['predicted_rate'].min():,.0f} - ${december['predicted_rate'].max():,.0f}")

    sys.stdout.flush()  # keep our log lines ahead of the scorer's output
    subprocess.run(
        [sys.executable, "score.py", "--predictions", str(PREDICTIONS), "--december-predictions", str(DECEMBER)],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
