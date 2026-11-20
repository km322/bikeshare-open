"""Guard rails on the ingest contract and the trained model's behaviour."""
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.config import (DATE_COL, ENCODED_COLS, FORBIDDEN, LAG_WARMUP,
                         RAW_FEATURE_COLS, REDUNDANT_COLS, TARGET)
from src.features import build_features, feature_columns, reindex_to_calendar
from src.ingest import load_raw
from src.train import MODELS, temporal_split


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: cnt_roll_28, the longest window, is NaN on the 28 days after
    each absent day m and its NaNs cover every shorter feature's, so each
    absent day costs LAG_WARMUP rows; here 28 warm-up + 28 after 2011-03-10
    = 56. Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_beats_the_mean_baseline(feats):
    """Scores on VALIDATION, not TEST; tuning on test leaks information and inflates R^2."""
    cols = feature_columns(feats)
    train, valid, _ = temporal_split(feats)
    ridge = MODELS["ridge"]
    ridge.fit(train[cols], train[TARGET])
    assert r2_score(valid[TARGET], ridge.predict(valid[cols])) > 0.4


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS


def test_every_raw_column_is_classified(raw, feats):
    """A new raw column must not enter the model silently.

    feature_columns() is a denylist, so any unclassified numeric column would
    become a feature. Every raw column must sit in exactly one class in
    src/config.py (FORBIDDEN, RAW_FEATURE_COLS, ENCODED_COLS, REDUNDANT_COLS,
    DATE_COL), and the raw columns that reach the model must be exactly
    RAW_FEATURE_COLS."""
    classified_sets = [frozenset(FORBIDDEN), frozenset(RAW_FEATURE_COLS), frozenset(ENCODED_COLS),
                       frozenset(REDUNDANT_COLS), frozenset({DATE_COL})]

    # Assert no column appears in two classes
    total_size = sum(len(s) for s in classified_sets)
    union_size = len(set().union(*classified_sets))
    assert total_size == union_size, "a column is in two classes"

    # Assert all raw columns are covered by a classification
    union_cols = set().union(*classified_sets)
    unclassified = set(raw.columns) - union_cols
    assert not unclassified, f"unclassified columns: {sorted(unclassified)}"

    # Assert only RAW_FEATURE_COLS reach the model (feature_columns excludes FORBIDDEN and non-numeric)
    reaching_model = set(feature_columns(feats)) & set(raw.columns)
    assert reaching_model == RAW_FEATURE_COLS, f"raw columns that reach the model: {sorted(reaching_model)}"
