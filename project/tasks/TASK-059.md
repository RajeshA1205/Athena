# TASK-059: Log debug message on model-driven selection failure

## Status
- **State:** Queued
- **Priority:** 🟡 Minor
- **Depends on:** None
- **Created:** 2026-02-25

## Objective
Replace the bare `except Exception: pass` in `_select_operation_unified` inside `training/stage2_agemem/trainer.py` with a debug-level log statement so failures from the model-driven operation selection path are observable.

## Context
`_select_operation_unified` wraps a model inference call to pick the next AgeMem operation. The current bare `pass` silently swallows all exceptions (import errors, shape mismatches, OOM, etc.), making it impossible to diagnose why the model-driven path fails during development or training runs.

## Scope & Constraints
- **May modify:** `training/stage2_agemem/trainer.py`
- **Must NOT modify:** any other file
- One-line change inside the except clause

## Input
- `training/stage2_agemem/trainer.py` — `_select_operation_unified` method

## Expected Output
Change:
```python
except Exception:
    pass
```
to:
```python
except Exception as e:
    self.logger.debug("Model-driven op selection failed: %s", e)
```

## Acceptance Criteria
- [ ] `except Exception: pass` replaced with the debug log form shown above
- [ ] No other lines in `trainer.py` changed beyond this one substitution
- [ ] `python3 -m pytest tests/ -q` stays green (173 passed, 4 skipped)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
