"""Exercise 5: is ridge worth deploying over "yesterday again"?

Compares the pipeline's ridge (untransformed, as in reports/metrics.json) with
the persistence baseline day by day on the test split. Both models are fixed
beforehand; nothing here is tuned.

Run: python -m src.exercise5
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import EXERCISE5, PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split

SANDY = [pd.Timestamp("2012-10-29"), pd.Timestamp("2012-10-30")]
WORST_N = 10


def profile(err: pd.Series) -> dict:
    """Absolute-error summary for one model, in rides."""
    return {
        "mae": round(float(err.mean()), 1),
        "median_ae": round(float(err.median()), 1),
        "p90_ae": round(float(err.quantile(0.9)), 1),
    }


def compare() -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, _, test = temporal_split(df)
    preds = {
        name: pd.Series(
            clone(MODELS[name]).fit(train[cols], train[TARGET]).predict(test[cols]),
            index=test.dteday,
        )
        for name in ("ridge", "baseline_persistence")
    }
    actual = pd.Series(test[TARGET].to_numpy(dtype=float), index=test.dteday)
    errs = {name: (p - actual).abs() for name, p in preds.items()}
    ridge, pers = errs["ridge"], errs["baseline_persistence"]

    no_sandy = ~actual.index.isin(SANDY)
    worst = np.maximum(ridge, pers).nlargest(WORST_N).index
    out = {"n_test": len(actual)}
    for name, err in errs.items():
        out[name] = {
            **profile(err),
            "mae_without_sandy": round(float(err[no_sandy].mean()), 1),
            "mae_on_worst_days": round(float(err[worst].mean()), 1),
            "bias": round(float((preds[name] - actual).mean()), 1),
        }
    out["share_of_days_closer"] = {
        "ridge": round(float((ridge < pers).mean()), 3),
        "baseline_persistence": round(float((pers < ridge).mean()), 3),
    }
    out["worst_days"] = [d.date().isoformat() for d in worst]
    return out


if __name__ == "__main__":
    res = compare()
    EXERCISE5.write_text(json.dumps(res, indent=2) + "\n")
    print(f"OK exercise5: ridge vs persistence on {res['n_test']} test days -> {EXERCISE5.name}")
