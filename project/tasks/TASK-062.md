# TASK-062: Fix timezone-naive datetime.now() in ExecutionAgent

## Status
- **State:** Queued
- **Priority:** Critical
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Replace all `datetime.now()` calls in `agents/execution_agent.py` with `datetime.now(timezone.utc)` so that every timestamp produced by ExecutionAgent is timezone-aware and consistent with `trading/order_management.py`, which already uses UTC-aware datetimes.

## Context
`Order.__post_init__` and other callsites in `execution_agent.py` use bare `datetime.now()` which returns a naive datetime (no tzinfo). `trading/order_management.py` uses `datetime.now(timezone.utc)` throughout. When orders from both modules are compared or sorted by timestamp, the naive/aware mismatch raises a `TypeError` at runtime and produces incorrect timestamp ordering — directly impacting P&L and fill sequencing.

Reference: `project/context.md` for project conventions. Sprint 8 (TASK-040) fixed this for `core/utils.py` and `memory/operations.py`; this task extends the fix to `execution_agent.py`.

## Scope & Constraints
- **May modify:** `agents/execution_agent.py` only
- **Must NOT modify:** Any other file
- Follow existing `from datetime import datetime, timezone` import pattern used elsewhere in the codebase
- Do not change any method signatures or public API

## Input
- `agents/execution_agent.py` — current source with naive `datetime.now()` calls

## Expected Output
- `agents/execution_agent.py` — all `datetime.now()` occurrences replaced with `datetime.now(timezone.utc)`; `timezone` imported from `datetime` if not already present

## Acceptance Criteria
- [ ] `from datetime import datetime, timezone` (or equivalent) is present in `execution_agent.py`
- [ ] Zero occurrences of bare `datetime.now()` remain in `execution_agent.py` (grep clean)
- [ ] All occurrences replaced with `datetime.now(timezone.utc)`
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
