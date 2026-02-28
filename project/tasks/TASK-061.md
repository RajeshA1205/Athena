# TASK-061: Convert remaining f-string logger calls in olmoe.py and grpo.py

## Status
- **State:** Queued
- **Priority:** ⚪ Nit
- **Depends on:** None
- **Created:** 2026-02-25

## Objective
Convert all remaining f-string `logger.*()` calls in `models/olmoe.py` and `training/stage2_agemem/grpo.py` to `%s`-style lazy formatting. TASK-045 covered the agent files and TASK-051 covered `finetune.py` and `trainer.py`; these two files were missed.

## Context
Python's logging module defers string formatting until the record is actually emitted. Using f-strings forces formatting at call time even when the log level is disabled, wasting CPU. This is a pure style/performance nit — no behaviour change. Approximate locations:
- `models/olmoe.py`: ~lines 103, 161, 177, 215-216
- `training/stage2_agemem/grpo.py`: ~lines 141, 292

## Scope & Constraints
- **May modify:** `models/olmoe.py`, `training/stage2_agemem/grpo.py`
- **Must NOT modify:** any other file
- Only change the logger call style — do not alter logic, variable names, or log message content

## Input
- `models/olmoe.py` — all `logger.*(f"...")` calls
- `training/stage2_agemem/grpo.py` — all `logger.*(f"...")` calls

## Expected Output
Each occurrence changed from:
```python
logger.info(f"Loading model from {path}")
```
to:
```python
logger.info("Loading model from %s", path)
```

## Acceptance Criteria
- [ ] Zero f-string logger calls remain in `models/olmoe.py`
- [ ] Zero f-string logger calls remain in `training/stage2_agemem/grpo.py`
- [ ] No logic, variable names, or message text changed beyond formatting style
- [ ] `python3 -m pytest tests/ -q` stays green (173 passed, 4 skipped)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
