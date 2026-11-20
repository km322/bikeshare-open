# Methodology 7: bikeshare demand, open

Daily Capital Bikeshare counts, 2011-2012. Train = 2011, validation = H1 2012,
test = H2 2012. Reproduce the "after" rows with
`make all && git diff --exit-code reports/metrics.json`; the "before" rows are
`reports/metrics_starter.json`, generated from the unmodified starter
(`44f7edb`) with `python -m src.features` and `python -m src.train`.

## Results

Feature added with `/new-feature`: `cnt_roll_28`, the mean of `cnt` over
d-28..d-1. "Before" is the starter (14 features, 348 training days); "after"
adds `cnt_roll_28` (15 features, 306 training days: the 28-day warm-up drops
42 more).

| Model | Feature set | Valid R^2 | Valid MAE | Test R^2 | Test MAE |
|---|---|---:|---:|---:|---:|
| baseline_mean | before | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| baseline_persistence | before | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | before | 0.6475 | 832.9 | 0.5833 | 947.1 |
| hgb | before | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| baseline_mean | after | -0.6894 | 1877.3 | -1.7717 | 2659.6 |
| baseline_persistence | after | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | after | 0.6839 | 777.5 | 0.6239 | 879.3 |
| hgb | after | 0.2415 | 1296.6 | -0.46 | 1952.4 |

The feature was chosen and kept on validation only: ridge +0.0364 and hgb
+0.0230 R^2. On the same 306 training rows without the feature
(`python -m src.ablation cnt_roll_28`, `reports/ablation.json`) the gains are
+0.0377 and +0.0486, so they come from the feature, not from the shorter
training set. The mean baseline moved only because its training rows changed;
persistence uses no training rows and did not move.

### Checks behind these numbers

- `make test`: 22 tests. Every raw column and every model input is classified
  in `src/config.py`, so an appended raw column or a leaked column copied
  under an innocent name turns a test red. Before that classification, the
  copied leak (fault 5) left every test green with ridge R^2 1.0.
- Faults 1-3 each turn a test red and fire the hook; fresh qwen sessions fixed
  them in 5, 4 and 4 turns from the test output alone. The fault 3 fix put the
  `raise` back with a different message, which the test accepts.
- Fault 4: Claude Code 2.1.321 refused the `cp` into `data/raw/` (the
  `Edit(./data/raw/**)` deny rule now covers shell writes), but a
  `python -c` copy went through with the hook silent and all 20 tests green
  while the test set shrank from 184 to 155 days. A SHA-256 check in
  `make data` and `test_raw_file_is_the_validated_one` now stop it.
- Fault 5 also showed the limit of a loud test: the first fresh session made
  it green by registering the leaked columns in `ENGINEERED_COLS`. That fix was
  reverted, and CLAUDE.md now forbids registering a column computed from a
  `FORBIDDEN` one.
- `reports/audit.jsonl` (`agent/audit.py` on the finished repo, qwen3.5:9b):
  leakage none, temporal split, baselines and reproducibility all pass, no
  findings, confidence high; an identical result on the earlier run. In that
  earlier run the auditor tried to Edit and Write `src/features.py` and was
  denied, so `allowed_tools` held as the boundary. An empty list from a 9B
  auditor is a screen, not proof, so I checked three weak spots by hand:
  Exercise 4's test run writes its own file, so a validation rerun cannot
  erase it; the ablation reports both baselines; the README's test count is
  now 22. Same-day weather is discussed under Exercise 5.

## Headline finding

The naive evaluation is perfect and invalid: `casual + registered` equals
`cnt` on every row, so a model given them scores R^2 1.000
(`python -m src.leakage_demo`) and forecasts nothing. Under the temporal split
the 2011 mean scores test R^2 -1.77, because mean daily ridership was 63%
higher in 2012. Ridge reaches 0.62 against a persistence baseline of 0.51, but
loses to persistence on MAE (879 vs 840) because it under-forecasts by 427
rides a day. Gradient boosting scores -0.46: far better than the mean baseline,
worse than persistence, because a tree cannot predict above the 2011 maximum
of 6,043 rides and 56.5% of test days exceed it. Moving the level out of the
target fixes the tree: hgb on `cnt - cnt_roll_28` scores test R^2 0.63 and
MAE 732, 108 rides a day better than persistence (Exercise 4). My feature,
`cnt_roll_28`, raised validation R^2 by 0.036 for ridge and 0.023 for hgb, and
I kept it.

## Exercise 4: target transformation

`python -m src.exercise4`, `reports/exercise4.json`. Four targets for ridge
and hgb, compared on validation only, with both baselines (validation R^2:
mean -0.6894, persistence 0.4507):

| Target | ridge valid R^2 / MAE | hgb valid R^2 / MAE |
|---|---|---|
| `cnt` | 0.6839 / 777.5 | 0.2415 / 1296.6 |
| `log(cnt)` | 0.374 / 983.3 | 0.184 / 1325.5 |
| `cnt - cnt_roll_28` | 0.6932 / 761.3 | **0.695 / 716.2** |
| `cnt / cnt_roll_28` | 0.4039 / 1069.0 | 0.6754 / 732.4 |

Chosen: hgb on `cnt - cnt_roll_28`, then add `cnt_roll_28` back. The choice
was committed (`9b8e799`) before the test set was scored, once, with
`--final` (`reports/exercise4_final.json`):

| Model | Test R^2 | Test MAE |
|---|---:|---:|
| baseline_mean | -1.7717 | 2659.6 |
| baseline_persistence | 0.5099 | 840.1 |
| hgb, untransformed | -0.46 | 1952.4 |
| hgb, `cnt - cnt_roll_28` | 0.6335 | 731.8 |
| ridge, `cnt - cnt_roll_28` | 0.635 | 856.7 |

**Defensible.** The transform uses only `cnt_roll_28`, the mean of d-28..d-1,
which is known before day d. Building the target and inverting the prediction
back to rides both use that one past-only quantity, and its alignment and its
handling of the 2011-03-10 gap are tested in `tests/test_leakage.py`. The
model never sees day d's count. What the transform does is let the trees
predict the deviation from the recent level, which is roughly stable across
years, instead of the level itself, which is not.

Caveats: the choice is the best of four on 182 validation days, and only 0.02
ahead of the ratio. Before choosing, the Exercise 3 audit had already shown a
teammate's test result for a similar transform (`cnt - cnt_roll_7`). On test,
the transformed hgb ties ridge on R^2 (0.6335 vs 0.635) and wins on MAE (731.8
vs 856.7), so the case for it is MAE.

## Exercise 5: ridge vs persistence

`python -m src.exercise5`, `reports/exercise5.json`. The pipeline's
untransformed ridge against persistence, per test day (184 days):

| Test | ridge | persistence |
|---|---:|---:|
| Share of days that model is closer | 42.9% | 57.1% |
| Median absolute error | 806.5 | 506.5 |
| MAE | 879.3 | 840.1 |
| MAE without Sandy and the day after | 854.3 | 819.0 |
| 90th percentile absolute error | 1711.1 | 1960.2 |
| MAE on the 10 worst days (either model) | 2216.0 | 3499.0 |
| Bias, prediction minus actual | -426.7 | +16.1 |

**No: do not deploy the 2011-trained ridge over persistence for day-ahead
forecasting.** Persistence is the better forecast on most days (57%) and on the
typical day (median error 507 vs 807), and Hurricane Sandy does not explain the
MAE gap: without it persistence still wins, 819 vs 854. Ridge's higher R^2
(0.624 vs 0.510) comes from a handful of extreme days, where it is about 1,280
rides better, because R^2 squares errors; it pays for those wins with a level
bias of -427 rides on ordinary days, the same 2011-to-2012 growth as the
headline. Its large-day wins also come from the weather features, which are
the day's observed weather rather than a forecast made the day before, so a
deployed ridge would be worse exactly where it now wins. If anything is
deployed, Exercise 4's detrended model is the candidate: it beats persistence
on test MAE by 108 rides (731.8 vs 840.1), not just on R^2. The
counter-case is an operator who cares only about the largest days, such as
storm rebalancing; there ridge's tail advantage is real.

## Attribution

Model: qwen3.5:9b via Ollama (`bikeshare-coder`, 64k context) as the Claude Code agent, the `/leakage-auditor` subagent and the `agent/audit.py` auditor.

Written in qwen3.5:9b Claude Code sessions:

- `src/features.py`: the `cnt_roll_28` feature and the literal 7-day windows.
- `src/config.py`: the column classes (`RAW_FEATURE_COLS`, `ENCODED_COLS`,
  `REDUNDANT_COLS`, `ENGINEERED_COLS`), `RAW_SHA256`, report paths.
- `src/ingest.py`: `verify_checksum`.
- `tests/test_leakage.py` and `tests/test_contract.py`: every new or changed
  test.
- `src/exercise4.py`, `src/exercise5.py`, `src/ablation.py`.
- `.claude/skills/seed-fault/SKILL.md`; the seeding and the fixes of faults
  1-5; the setup fixes to `Makefile`, `model/check.sh` and
  `.claude/settings.json`.
- Drafts of the documents, including CLAUDE.md additions and DECISIONS lines.

My decisions (each a line in `docs/DECISIONS.md`):

- What to predict, what counts as leakage, how to split, which baselines:
  kept the starter's choices (the five starter lines).
- Add `cnt_roll_28` and keep it, on validation and same-rows evidence.
- Classify every raw column and every model input, and accept that a new
  feature fails loudly until it is registered.
- Gate `test_beats_the_mean_baseline` on validation, not test.
- Add the SHA-256 check as the catcher for fault 4.
- The difference transform is fair and is the Exercise 4 answer, kept out of
  `make all`.
- Persistence over ridge in Exercise 5.
