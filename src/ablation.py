"""Validation R^2 with and without one feature, on identical rows.

Adding cnt_roll_28 also raised LAG_WARMUP, which dropped training rows, so a
before/after comparison against the starter mixes two changes. Here both fits
use the same rows of the current feature table; only the column differs.
Validation only: this module never scores the test split.

Run: python -m src.ablation <feature>
"""
import json
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import ABLATION, PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split


def score(model, train, valid, cols) -> dict:
    fitted = clone(model).fit(train[cols], train[TARGET])
    pred = fitted.predict(valid[cols])
    return {
        "r2": round(float(r2_score(valid[TARGET], pred)), 4),
        "mae": round(float(mean_absolute_error(valid[TARGET], pred)), 1),
    }


def ablate(feature: str) -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    if feature not in cols:
        raise SystemExit(f"{feature!r} is not a model input: {cols}")
    without = [c for c in cols if c != feature]
    train, valid, _ = temporal_split(df)
    res = {"feature": feature, "n_train": len(train), "n_valid": len(valid), "valid": {}}
    for name, model in MODELS.items():
        with_f = score(model, train, valid, cols)
        without_f = score(model, train, valid, without)
        res["valid"][name] = {
            "with": with_f,
            "without": without_f,
            "delta_r2": round(with_f["r2"] - without_f["r2"], 4),
        }
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.ablation <feature>")
    res = ablate(sys.argv[1])
    ABLATION.write_text(json.dumps(res, indent=2) + "\n")
    deltas = {k: v["delta_r2"] for k, v in res["valid"].items()}
    print(f"OK ablation: {res['feature']} on {res['n_train']} training rows, valid delta R^2 {deltas} -> {ABLATION.name}")
