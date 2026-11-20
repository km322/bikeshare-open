---
name: seed-fault
description: Seeds one deliberate fault in src/, records the change, the hook's stderr and the red test run in docs/SEEDED_FAULTS.md, then restores and confirms green; to be used when asked to break something on purpose or test the verification suite.
allowed-tools: Read, Edit, Bash(python -m pytest*), Bash(make test), Bash(git diff*), Bash(git status*), Bash(printf*), Bash(.claude/hooks/check_leakage.sh*)
---

# Starting state

!`git status --short src/ tests/`

# Seeding the fault: $ARGUMENTS

1. **Baseline:** The git status above must be empty and `make test` green; otherwise stop and say so, because a red test you cannot attribute to the fault proves nothing.

2. **Seed:** One `Edit` to one file under src/ (Edit, not Bash, so the PostToolUse hook sees it), the smallest change matching $ARGUMENTS. Never touch data/raw/ or tests/.

3. **Record,** before fixing anything, by appending to docs/SEEDED_FAULTS.md with shell commands so every pasted block is real output and never retyped: a heading `## Fault: $ARGUMENTS` (printf), then in ```diff fences the output of `git diff src/ >> docs/SEEDED_FAULTS.md`, then in ```text fences the hook's stderr appended by re-running the hook on the same file, which is deterministic and avoids retyping: `printf '{"tool_input":{"file_path":"<edited file>"}}' | .claude/hooks/check_leakage.sh 2>> docs/SEEDED_FAULTS.md` (exit code 2 means the hook fired; no output means "hook silent", because it only runs tests/test_leakage.py), which must match the message that came back after the Edit, then in ```text fences the red run `python -m pytest tests/ -q --tb=line -p no:cacheprovider >> docs/SEEDED_FAULTS.md 2>&1` (exit code 1 is expected). If no test went red, write that plainly: it is a gap in the suite and the finding to report.

4. **Fix from the evidence:** Work from the failing test output, not from what you remember changing, and do not restore with git. Count the turns: each response that makes a tool call counts as one, from the first red result to the first fully green `make test`. If the user says the fix will be done in a fresh session, stop here and resume at step 5 when asked.

5. **Confirm the restore:** `git diff src/` is empty and `make test` is green.

6. **Append which test caught the fault and the turn count** to the section. Do not commit; say what you wrote.
