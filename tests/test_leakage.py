"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import pandas as pd
import pytest

from src.config import ENGINEERED_COLS, FORBIDDEN, RAW_FEATURE_COLS, TARGET
from src.features import build_features, feature_columns, encode_weather
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_every_model_input_is_classified(feats):
    """The model input must be exactly the classified features. A leaked column
    copied under an innocent name (out["members"] = out["registered"]) passed
    every other test."""
    cols = set(feature_columns(feats))
    expected = RAW_FEATURE_COLS | ENGINEERED_COLS
    assert cols == expected, f"unclassified model inputs: {sorted(cols - expected)}; missing: {sorted(expected - cols)}"


def test_weather_encoder_rejects_unknown_labels():
    """The encoder must raise on an unknown label, not silently produce NaN."""
    # The raw file has "GOOD" / "MISTY" / "BAD"; an attacker or corrupted data
    # might introduce "SNOW". If encode_weather just produces NaN for SNOW, a
    # dropna() would hide it without anyone noticing. The raise makes the
    # problem visible during feature building rather than at training time.
    s = pd.Series(["GOOD", "MISTY", "SNOW"])
    with pytest.raises(ValueError, match="SNOW"):
        encode_weather(s)


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_lags_are_true_calendar_lags(feats):

    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_7_is_a_true_7_day_lag(feats):
    """cnt_lag_7 for date d must equal cnt on d-7 calendar days, not 28.

    No other test checks cnt_lag_7, which is how a window tied to
    LAG_WARMUP could silently become a 28-day lag.
    """
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_7.items():
        prev = date - pd.Timedelta(days=7)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag_7 at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified for cnt_lag_7"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """For every row whose 7-day window is complete, cnt_roll_7 equals that window's mean."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_roll_7.items():
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
            assert lag == pytest.approx(expected), f"roll_7 mismatch at {date.date()}"
            checked += 1
    assert checked > 600, f"only {checked} rows had a complete 7-day window"


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique


def test_cnt_roll_28_value_correctness():
    """cnt_roll_28 on one date must equal the mean of cnt over the 28 calendar
    days strictly before it. We pick a date whose 28-day window is complete in
    the raw file and verify the computed value matches hand-calculated from load_raw()."""
    # Use a date where all 28 previous days exist in the raw data: 2011-05-01
    # Its 28-day window is 2011-04-03 .. 2011-04-30 (all complete, no gap)
    target_date = pd.Timestamp("2011-05-01")

    raw_df = load_raw()

    # Index by dteday so we can check membership
    raw_by_date = raw_df.set_index("dteday")

    # The 28-day window strictly before the target date
    window_dates = [target_date - pd.Timedelta(days=k) for k in range(1, 29)]

    # Assert all 28 dates exist in the raw file (the gap is at 2011-03-10,
    # which is before this window, so this date should be safe)
    missing_in_raw = [d for d in window_dates if d not in raw_by_date.index]
    assert not missing_in_raw, f"Some dates in the window are missing from raw data: {missing_in_raw}"

    # Compute expected mean over all 28 days
    expected = raw_by_date.loc[window_dates, "cnt"].mean()

    # Run features to get the feature table
    feats = build_features(raw_df)

    # Check that cnt_roll_28 exists
    assert "cnt_roll_28" in feats.columns, "cnt_roll_28 column does not exist"

    actual = feats.set_index("dteday").loc[target_date, "cnt_roll_28"]
    assert actual == pytest.approx(expected), f"roll_28 mismatch at {target_date.date()}: expected {expected}, got {actual}"


def test_cnt_roll_28_excludes_gapped_windows():
    """No row may be kept whose 28-day window includes the missing day 2011-03-10.

    build_features drops rows by NaN, the 7-day features recover a week after the missing 2011-03-10, so without a 28-day feature the starter keeps dates from mid-March to early April 2011 whose 28-day window contains the gap."""
    missing_date = pd.Timestamp("2011-03-10")

    feats = build_features(load_raw())

    # Find every date d such that d - 28 <= missing_date <= d - 1
    offending_dates = []
    for idx, row in feats.iterrows():
        dteday = pd.Timestamp(row["dteday"])
        start_of_window = dteday - pd.Timedelta(days=28)
        end_of_window = dteday - pd.Timedelta(days=1)
        if start_of_window <= missing_date <= end_of_window:
            offending_dates.append(dteday.date())

    assert len(offending_dates) == 0, f"{len(offending_dates)} dates kept with 28-day windows containing the gap: first={offending_dates[0] if offending_dates else 'N/A'}, last={offending_dates[-1] if offending_dates else 'N/A'}"

