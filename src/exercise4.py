"""Exercise 4: transform the target so the tree model can compete.

The level is cnt_roll_28, the mean of cnt over the 28 days before d, so the
forward transform and its inverse use only the past. `make all` does not run
this module; reports/metrics.json stays the untransformed pipeline.

Run: python -m src.exercise4          validation only -> reports/exercise4.json
     python -m src.exercise4 --final  CHOSEN on test, once -> reports/exercise4_final.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import EXERCISE4, EXERCISE4_FINAL, PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split

LEVEL = "cnt_roll_28"

# name -> (forward(y, level), inverse(prediction, level))
TRANSFORMS = {
    "cnt": (lambda y, lv: y, lambda p, lv: p),
    "log": (lambda y, lv: np.log(y), lambda p, lv: np.exp(p)),
    "diff_roll28": (lambda y, lv: y - lv, lambda p, lv: p + lv),
    "ratio_roll28": (lambda y, lv: y / lv, lambda p, lv: p * lv),
}
BASELINES = ("baseline_mean", "baseline_persistence")
MODEL_NAMES = ("ridge", "hgb")

# Set from the validation comparison and committed before --final is run.
CHOSEN = "diff_roll28"


def score(y_true, y_pred) -> dict:
    """Same rounding as src/train.py."""
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
    }


def fit_transformed(name: str, transform: str, train, part, cols) -> dict:
    """Fit on train with the transformed target; score `part` in rides."""
    forward, inverse = TRANSFORMS[transform]
    model = clone(MODELS[name]).fit(train[cols], forward(train[TARGET], train[LEVEL]))
    return score(part[TARGET], inverse(model.predict(part[cols]), part[LEVEL]))


def fit_baselines(train, part, cols) -> dict:
    out = {}
    for name in BASELINES:
        model = clone(MODELS[name]).fit(train[cols], train[TARGET])
        out[name] = score(part[TARGET], model.predict(part[cols]))
    return out


def run(split: str, transforms) -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    part = {"valid": valid, "test": test}[split]
    return {
        "split": split,
        "level": LEVEL,
        "n_train": len(train),
        f"n_{split}": len(part),
        "baselines": fit_baselines(train, part, cols),
        "transforms": {
            t: {m: fit_transformed(m, t, train, part, cols) for m in MODEL_NAMES}
            for t in transforms
        },
    }


if __name__ == "__main__":
    if "--final" in sys.argv:
        if CHOSEN is None:
            raise SystemExit("Set CHOSEN from the validation results and commit it before --final.")
        res = run("test", ["cnt", CHOSEN])
        res["chosen"] = CHOSEN
        out = EXERCISE4_FINAL
    else:
        res = run("valid", list(TRANSFORMS))
        out = EXERCISE4
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"OK exercise4: {len(res['transforms'])} transforms on {res['split']} -> {out.name}")
