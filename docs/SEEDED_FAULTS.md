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
