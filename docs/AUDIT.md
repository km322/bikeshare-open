# Teammate audit

Repository: https://github.com/anirudh9280/DSC180A-Methodology-7 at 900140b; the diff reviewed is everything since their unmodified starter 5b9d25f (src, tests, reports, docs/DECISIONS.md).

How it was run: the /leakage-auditor subagent from this repo, model inherited from the session (qwen3.5:9b via Ollama), on a clone in /tmp. Their code was treated as untrusted: the subagent could read files and run `git -C <clone> ...` and nothing else. It tried more than ten times to run Python against their repo (importing their src, reading their parquet), and the permission system blocked every attempt, so its checks rest on reading code, diffs and their committed reports.

## /leakage-auditor output (verbatim)

```text
Based on my comprehensive review of all five checks against the teammate's repository code and diff:

**All five checks pass:**

1. **Target leakage.** `FORBIDDEN` correctly includes `casual`, `registered`, `cnt`, plus time index columns (`instant`, `yr`, `days_since_2011`), and `cnt_2d_bfr`. `DERIVED_FEATURES` is fully declared in both `src/config.py` and `tests/test_leakage.py`, with an independent least-squares identity check (`test_no_feature_set_reconstructs_the_target`) that would catch a leaky column under any name. No forbidden columns in the feature set.

2. **Temporal ordering.** All lag features use `.shift(1)` before aggregation:
   - `cnt_lag_1 = out[TARGET].shift(1)` (line 74)
   - `cnt_lag_7 = out[TARGET].shift(WEEK)` (line 75)
   - `cnt_roll_7 = out[TARGET].shift(1).rolling(WEEK).mean()` (line 76)
   - `cnt_roll_28 = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()` (line 79)
   
   Split is date-based via `temporal_split()` using `TRAIN_END`/`VALID_END` constants. All models `.fit()` only on `train[cols]`. No data snooping.

3. **Calendar gaps.** `reindex_to_calendar()` fills the missing 2011-03-10 gap (line 54), and rolling windows have no `min_periods`, so incomplete windows stay NaN and are dropped by `dropna()`. `test_roll_28_drops_windows_containing_the_gap` verifies this behavior.

4. **Silent row loss.** The only `.dropna()` is after lagging (line 81), with bounds enforced in `test_contract.py`: `lost <= LAG_WARMUP + LAG_WARMUP * missing_days`. No unbounded drops elsewhere.

5. **Unreported baseline.** Both baselines appear alongside all models in `reports/metrics.json`:
   - `baseline_mean`: valid R²=-0.6894, test R²=-1.7717
   - `baseline_persistence`: valid R²=0.4507, test R²=0.5099
   - `ridge`: valid R²=0.6239, test R²=0.6239 (both splits)
   - `hgb`: valid R²=-0.46, test R²=-0.46 (both splits)

The teammate's implementation is methodologically sound with no findings that change numbers or invalidate conclusions.
```

## My verdict

I agree with the auditor that no finding changes a number: their metrics.json (4 models x 2 splits, 306 training rows, 15 features) matches this repo's reports/metrics.json to the printed digit, which reproduces their pipeline independently because both add cnt_roll_28 to the same starter. Two corrections to the auditor. It misquoted their validation R^2 for ridge (0.6839, not 0.6239) and hgb (0.2415, not -0.46): it copied the test values into both splits. And its check 5 pass skips that reports/ex4_target_transform.json still has no mean or persistence row; their REPORT.md compares the chosen diff_roll7 model with persistence by hand (hgb beats persistence on test MAE by only 10 rides, 830.1 vs 840.1) and explains why they did not re-run --final to add the row, so the conclusion stands. Minor, no number changes: cnt_roll_28 is written as rolling(LAG_WARMUP), so raising LAG_WARMUP for a longer feature would silently change it.
