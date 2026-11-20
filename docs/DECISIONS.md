# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-19: Agent model setup. bikeshare-coder uses qwen3.5:9b via Ollama with num_ctx=65536; make model writes Modelfile to /tmp then invokes ollama create because Ollama 0.40 no longer reads from stdin; check-model warned falsely about tools because, under pipefail, grep -q exited at the first match and ollama show then failed with SIGPIPE; it now captures the output in a variable and greps that; CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 in .claude/settings.json to prevent Claude Code from misreading an unrecognized model's 200k default and missing Ollama's 64k truncation.
- 2026-11-19: added and kept cnt_roll_28 (mean of cnt over d-28..d-1, shift(1).rolling(28) on calendar index); the availability argument (every input is observed before d); validation R^2 before -> after for ridge 0.6475 -> 0.6839 (+0.0364), hgb 0.2185 -> 0.2415 (+0.0230); LAG_WARMUP 7 -> 28 costs training rows (348 -> 306), so cnt_lag_7 and cnt_roll_7 now use literal windows, guarded by the new cnt_lag_7 test.
