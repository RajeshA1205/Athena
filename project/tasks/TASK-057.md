# TASK-057: Replace `hash()` with `_stable_hash()` in trainer delete path

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** None
- **Created:** 2026-02-25

## Objective
Replace the non-deterministic `str(hash(content))` call in `training/stage2_agemem/trainer.py`'s `_execute_operation()` with a stable, deterministic alternative. TASK-054 already fixed the same pattern in `training/data/scrapers/market.py` but missed this second occurrence.

## Context
Python's built-in `hash()` is randomized per-process (PYTHONHASHSEED). Using it to generate a delete ID means the same content produces a different ID on every run, making delete operations non-reproducible and untestable. TASK-027 introduced `_stable_hash` in `trading/order_management.py` and TASK-054 used `hashlib.sha256` directly in the scraper. Either approach is acceptable here — prefer `hashlib.sha256` to avoid a cross-module import.

## Scope & Constraints
- **May modify:** `training/stage2_agemem/trainer.py`
- **Must NOT modify:** any other file
- Add `import hashlib` at the top of the file if not already present

## Input
- `training/stage2_agemem/trainer.py` — locate `str(hash(content))` in `_execute_operation()`

## Expected Output
Replace:
```python
episode_id = str(hash(content))
```
with:
```python
episode_id = hashlib.sha256(content.encode()).hexdigest()[:16]
```
(or equivalent stable hash producing a short string). Add `import hashlib` if missing.

## Acceptance Criteria
- [ ] No `hash()` call remains in `trainer.py`'s delete path
- [ ] `hashlib` (or another deterministic hash) is used instead
- [ ] `python3 -m pytest tests/ -q` stays green (173 passed, 4 skipped)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
