# Seeded faults

## Exercise 1: failing tests for `cnt_roll_28`, before the feature exists

The two tests check that cnt_roll_28 on 2011-05-01 equals the mean of cnt over the 28 raw days before it, and that no kept date has the missing 2011-03-10 in its 28-day window.

```text
$ python -m pytest tests/test_leakage.py -q --tb=line -k roll_28
FF                                                                       [100%]
=================================== FAILURES ===================================
E   AssertionError: cnt_roll_28 column does not exist
    assert 'cnt_roll_28' in Index(['dteday', 'instant', 'season', 'yr', 'mnth', 'holiday', 'weekday',\n       'temp', 'atemp', 'hum', 'windspeed', ...kday', 'month_sin', 'month_cos',\n       'weather_severity', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_7'],\n      dtype='str')
     +  where Index(['dteday', 'instant', 'season', 'yr', 'mnth', 'holiday', 'weekday',\n       'temp', 'atemp', 'hum', 'windspeed', ...kday', 'month_sin', 'month_cos',\n       'weather_severity', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_7'],\n      dtype='str') =         dteday  instant  season  ...  cnt_lag_1 cnt_lag_7   cnt_roll_7\n0   2011-01-10     10.0  WINTER  ...      822.0...    1787.0  1530.142857\n713 2012-12-31    731.0  WINTER  ...     1796.0     920.0  1531.428571\n\n[714 rows x 28 columns].columns
/Users/ketan/Downloads/DSC_180A/bikeshare-open/tests/test_leakage.py:97: AssertionError: cnt_roll_28 column does not exist
E   AssertionError: 21 dates kept with 28-day windows containing the gap: first=2011-03-18, last=2011-04-07
    assert 21 == 0
     +  where 21 = len([datetime.date(2011, 3, 18), datetime.date(2011, 3, 19), datetime.date(2011, 3, 20), datetime.date(2011, 3, 21), datetime.date(2011, 3, 22), datetime.date(2011, 3, 23), ...])
/Users/ketan/Downloads/DSC_180A/bikeshare-open/tests/test_leakage.py:120: AssertionError: 21 dates kept with 28-day windows containing the gap: first=2011-03-18, last=2011-04-07
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_cnt_roll_28_value_correctness - AssertionE...
FAILED tests/test_leakage.py::test_cnt_roll_28_excludes_gapped_windows - Asse...
2 failed, 7 deselected in 0.37s
```

- `test_cnt_roll_28_value_correctness` fails because build_features does not make a cnt_roll_28 column yet.
- `test_cnt_roll_28_excludes_gapped_windows` fails because the starter keeps 21 dates, 2011-03-18 to 2011-04-07, whose 28-day window contains 2011-03-10: its 7-day features recover a week after the gap.

## Fault: leaked columns copied under innocent names

Seeded with `Edit` (the rubric fault "a leaked column renamed to something innocent"); a copy, not a rename, so the raw columns stay intact.

```diff
diff --git a/src/features.py b/src/features.py
index 35fbd8f..8a15f7f 100644
--- a/src/features.py
+++ b/src/features.py
@@ -76,6 +76,9 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     out["cnt_roll_7"] = out[TARGET].shift(1).rolling(7).mean()
     out["cnt_roll_28"] = out[TARGET].shift(1).rolling(28).mean()
 
+    out["walkups"] = out["casual"]
+    out["members"] = out["registered"]
+
     return out.dropna().reset_index(drop=True)
 
 
```

Hook: silent. The hook script, fed the Edit's payload, exited 0 with no output: it only runs tests/test_leakage.py, and every test in it still passed.

Full suite:

```text
$ python -m pytest tests/ -q --tb=line -p no:cacheprovider
...................                                                      [100%]
19 passed in 0.98s
```

The pipeline with the fault in place:

```text
$ make features train
python -m src.features
OK features: 672 rows, 17 features -> features.parquet
  temp, atemp, hum, windspeed, dayofweek, is_weekend, is_holiday, is_workday, month_sin, month_cos, weather_severity, cnt_lag_1, cnt_lag_7, cnt_roll_7, cnt_roll_28, walkups, members
python -m src.train
{
  "n_features": 17,
  "n_train": 306,
  "n_valid": 182,
  "n_test": 184,
  "baseline_mean": {
    "valid": {
      "r2": -0.6894,
      "mae": 1877.3
    },
    "test": {
      "r2": -1.7717,
      "mae": 2659.6
    }
  },
  "baseline_persistence": {
    "valid": {
      "r2": 0.4507,
      "mae": 900.6
    },
    "test": {
      "r2": 0.5099,
      "mae": 840.1
    }
  },
  "ridge": {
    "valid": {
      "r2": 1.0,
      "mae": 7.9
    },
    "test": {
      "r2": 1.0,
      "mae": 9.7
    }
  },
  "hgb": {
    "valid": {
      "r2": 0.6716,
      "mae": 697.3
    },
    "test": {
      "r2": 0.2378,
      "mae": 1277.1
    }
  }
}
```

Every test green and the hook silent, while ridge scores R^2 1.0 on validation and test (MAE 7.9 and 9.7 rides). feature_columns() is a denylist and test_every_raw_column_is_classified only checks raw columns, so a new column built inside build_features reaches the model unchecked. This is a gap in the suite. The two lines were deleted again with Edit; the test that closes the gap follows.
